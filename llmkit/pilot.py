"""Pilot: Validasi pipeline murah dengan model kecil (0.05B), sweep LR, dan uji resume (Bagian 12.2)."""

from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from llmkit.checkpoint import load_checkpoint, save_checkpoint
from llmkit.dataset import PretrainDataLoader
from llmkit.model import (
    build_qwen3_config,
    create_adamw_optimizer,
    create_model,
)
from llmkit.planner import PRESETS, calculate_params
from llmkit.state import atomic_write_json, load_project


def run_pilot(
    project_dir: Path,
    pilot_tokens: int = 100_000,
    steps_per_sweep: int = 20
) -> dict[str, Any]:
    """
    Jalankan pilot validation pada model kecil (0.05B) dengan 3 nilai LR (0.5x, 1.0x, 2.0x).
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})

    pilot_dir = project_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preset kecil 0.05B
    p05 = PRESETS[0.05]
    cfg = build_qwen3_config(p05, max_position_embeddings=1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Base LR untuk model 0.05B
    N_B = p05["total"] / 1e9
    base_lr = min(2e-3, max(2e-4, 8e-4 * ((0.125 / N_B) ** 0.25)))

    lr_candidates = [0.5 * base_lr, 1.0 * base_lr, 2.0 * base_lr]
    sweep_results: list[dict[str, Any]] = []

    print("\n=== PILOT: SWEEP LEARNING RATE ===")
    print(f"Menjalankan model uji coba 0.05B ({p05['total']:,} params) pada data asli...")

    best_loss = float("inf")
    best_lr = base_lr

    loss_fn = nn.CrossEntropyLoss()

    for idx, lr in enumerate(lr_candidates, start=1):
        print(f"\n[Run {idx}/3] Menguji LR: {lr:.2e} ...")
        model = create_model(cfg, seed=1337 + idx).to(device)
        optimizer = create_adamw_optimizer(model, lr=lr)
        loader = PretrainDataLoader(project_dir, plan, seed=1337)

        losses = []
        for s in range(steps_per_sweep):
            x, y = loader.next_batch(micro_batch=2, seq_len=256, device=device)
            out = model(x)
            loss = loss_fn(out.logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        final_loss = float(losses[-1]) if losses else 999.0
        sweep_results.append({
            "lr": lr,
            "multiplier": f"{[0.5, 1.0, 2.0][idx-1]}x",
            "initial_loss": round(losses[0], 4),
            "final_loss": round(final_loss, 4),
            "loss_drop": round(losses[0] - final_loss, 4)
        })

        if final_loss < best_loss:
            best_loss = final_loss
            best_lr = lr

    # 2. Uji Resume
    print("\n=== PILOT: UJI RESUME CHECKPOINT ===")
    test_ckpt_dir = pilot_dir / "test_ckpt"
    model_a = create_model(cfg, seed=42).to(device)
    optim_a = create_adamw_optimizer(model_a, lr=best_lr)
    loader_a = PretrainDataLoader(project_dir, plan, seed=42)

    # Latih 10 step
    for _ in range(10):
        x, y = loader_a.next_batch(micro_batch=2, seq_len=128, device=device)
        loss = loss_fn(model_a(x).logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        optim_a.step()
        optim_a.zero_grad()

    # Simpan
    t_state = {"step": 10, "tokens_seen": 10 * 2 * 128}
    d_state = loader_a.get_state()
    save_checkpoint(test_ckpt_dir, 10, model_a, optim_a, t_state, d_state)

    # Muat ke model_b dan verifikasi step ke-11
    model_b = create_model(cfg, seed=999).to(device)
    optim_b = create_adamw_optimizer(model_b, lr=best_lr)
    loader_b = PretrainDataLoader(project_dir, plan, seed=42)

    load_checkpoint(test_ckpt_dir / "step_00000010", model_b, optim_b, device=device)
    loader_b.load_state(d_state)

    x11_a, y11_a = loader_a.next_batch(micro_batch=2, seq_len=128, device=device)
    x11_b, y11_b = loader_b.next_batch(micro_batch=2, seq_len=128, device=device)

    is_resume_identical = torch.equal(x11_a, x11_b) and torch.equal(y11_a, y11_b)
    print(f"Uji resume batch identik: {'[OK] Berhasil' if is_resume_identical else '[GAGAL] Mismatch'}")

    # 3. Laporan Fertilitas
    tok_meta_path = project_dir / "tokenizer" / "tokenizer_meta.json"
    fertility_data = {}
    if tok_meta_path.exists():
        with open(tok_meta_path, "r", encoding="utf-8") as f:
            fertility_data = json.load(f).get("fertility", {})

    result = {
        "date": datetime.datetime.now().isoformat(),
        "pilot_model": "0.05B",
        "best_lr": best_lr,
        "sweep_results": sweep_results,
        "resume_test_passed": is_resume_identical,
        "fertility": fertility_data,
        "note": "Hasil sweep LR pilot adalah panduan awal. Model besar umumnya membutuhkan LR yang sedikit lebih rendah."
    }

    atomic_write_json(pilot_dir / "result.json", result)
    print(f"\n[OK] Pilot selesai. Rekomendasi LR terbaik: {best_lr:.2e}")
    return result
