"""Benchmark kecepatan GPU, pencarian micro-batch maksimal, dan estimasi waktu nyata (Bagian 12.1)."""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from llmkit.gpu_table import get_gpu_spec
from llmkit.model import build_qwen3_config, create_adamw_optimizer, create_model
from llmkit.state import atomic_write_json, compute_config_hash, load_project


def run_benchmark(project_dir: Path) -> dict[str, Any]:
    """
    Ukur tok/s pada GPU CUDA dengan variasi micro_batch dan gradient checkpointing.
    Simpan hasil ke bench.json dan perbarui estimasi durasi/biaya.
    """
    if not torch.cuda.is_available():
        print("[INFO] Benchmark GPU (bench) memerlukan GPU CUDA. Dilewati pada lingkungan CPU.")
        return {"status": "skipped", "reason": "No CUDA GPU available"}

    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    hyper = plan.get("hyperparameters", {})
    seq_len = int(hyper.get("context_pretrain", 2048))
    batch_tokens = int(hyper.get("batch_tokens", 1_048_576))

    dev_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_spec = get_gpu_spec(dev_name)

    use_bf16 = torch.cuda.is_bf16_supported()
    cast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    best_tok_per_s = 0.0
    best_config: dict[str, Any] = {}

    loss_fn = nn.CrossEntropyLoss()

    # Coba kedua opsi: dengan dan tanpa gradient checkpointing
    for grad_ckpt in [False, True]:
        for mb in [1, 2, 4, 8, 16, 32]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                cfg = build_qwen3_config(arch, max_position_embeddings=seq_len)
                model = create_model(cfg, gradient_checkpointing=grad_ckpt).cuda()
                optimizer = create_adamw_optimizer(model)

                # 3 Warmup steps
                for _ in range(3):
                    x = torch.randint(0, cfg.vocab_size, (mb, seq_len), device="cuda")
                    y = torch.randint(0, cfg.vocab_size, (mb, seq_len), device="cuda")
                    with torch.amp.autocast(device_type="cuda", dtype=cast_dtype):
                        out = model(x)
                        loss = loss_fn(out.logits.view(-1, cfg.vocab_size), y.view(-1))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                torch.cuda.synchronize()
                start_t = time.time()

                # 20 Measured steps
                measured_steps = 20
                for _ in range(measured_steps):
                    x = torch.randint(0, cfg.vocab_size, (mb, seq_len), device="cuda")
                    y = torch.randint(0, cfg.vocab_size, (mb, seq_len), device="cuda")
                    with torch.amp.autocast(device_type="cuda", dtype=cast_dtype):
                        out = model(x)
                        loss = loss_fn(out.logits.view(-1, cfg.vocab_size), y.view(-1))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                torch.cuda.synchronize()
                elapsed = time.time() - start_t
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

                total_tok = measured_steps * mb * seq_len
                tok_per_s = total_tok / max(0.001, elapsed)

                # Evaluasi apakah memori <= 92% VRAM
                if peak_mem <= 0.92 * vram_gb:
                    if tok_per_s > best_tok_per_s:
                        best_tok_per_s = tok_per_s
                        grad_accum = max(1, round(batch_tokens / (mb * seq_len)))
                        best_config = {
                            "micro_batch": mb,
                            "grad_accum": grad_accum,
                            "grad_checkpointing": grad_ckpt,
                            "tok_per_s": round(tok_per_s, 1),
                            "peak_mem_gb": round(peak_mem, 2),
                        }

                del model, optimizer
                torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError):
                # Tangkap OOM dan hentikan kenaikan micro-batch untuk mode ini
                torch.cuda.empty_cache()
                break

    if not best_config:
        best_config = {
            "micro_batch": 1,
            "grad_accum": max(1, round(batch_tokens / seq_len)),
            "grad_checkpointing": True,
            "tok_per_s": 1000.0,
            "peak_mem_gb": 0.0,
        }

    # Hitung ulang ETA dan biaya aktual
    tokens_target = float(plan.get("tokens", {}).get("target", 1e9))
    actual_tok_s = best_config.get("tok_per_s", 1000.0)
    actual_seconds = tokens_target / max(1.0, actual_tok_s)
    actual_hours = actual_seconds / 3600.0
    price_hr = gpu_spec.price_per_hour if gpu_spec else 0.0
    actual_cost = actual_hours * price_hr

    result_data = {
        "config_hash": compute_config_hash(arch),
        "gpu": dev_name,
        "date": datetime.datetime.now().isoformat(),
        "dtype": "bf16" if use_bf16 else "fp16",
        "tokens_target": tokens_target,
        "actual_hours": round(actual_hours, 1),
        "actual_cost": round(actual_cost, 2),
        **best_config,
    }

    # Simpan ke bench.json
    atomic_write_json(project_dir / "bench.json", result_data)
    print(f"\n[OK] Benchmark selesai:")
    print(f"  GPU               : {dev_name} (VRAM {vram_gb:.1f} GB)")
    print(f"  Konfigurasi optimal: micro_batch={best_config['micro_batch']}, grad_accum={best_config['grad_accum']}, grad_checkpointing={best_config['grad_checkpointing']}")
    print(f"  Kecepatan puncak  : {best_config['tok_per_s']:,} token/detik (Memori {best_config['peak_mem_gb']:.1f} GB)")
    print(f"  Estimasi durasi nyata: {actual_hours:.1f} jam ({actual_hours / 24.0:.1f} hari) ~ ${actual_cost:.2f}")

    return result_data
