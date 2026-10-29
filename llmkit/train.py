"""Loop Pretraining dengan LR Schedule WSD, Deteksi Spike, Rollback, dan Evaluasi Berkala (Bagian 11)."""

from __future__ import annotations

import datetime
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from llmkit.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from llmkit.dataset import PretrainDataLoader
from llmkit.model import (
    build_qwen3_config,
    count_parameters,
    create_adamw_optimizer,
    create_model,
)
from llmkit.state import (
    atomic_write_json,
    compute_config_hash,
    load_project,
    log_history,
)


def get_wsd_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    lr_scale: float = 1.0
) -> float:
    """
    LR Schedule WSD (Warmup -> Stable -> Decay) sesuai Bagian 6.5:
    - 0 .. warmup_steps: linier naik ke peak_lr
    - warmup_steps .. 80% total_steps: konstan di peak_lr
    - 80% .. 100%: peluruhan linier ke 10% peak_lr
    """
    effective_peak = peak_lr * lr_scale
    min_lr = 0.1 * effective_peak

    if step < warmup_steps:
        return effective_peak * float(step) / max(1.0, float(warmup_steps))

    decay_start_step = int(0.8 * total_steps)
    if step <= decay_start_step:
        return effective_peak

    # Final 20% decay
    decay_steps = total_steps - decay_start_step
    progress = float(step - decay_start_step) / max(1.0, float(decay_steps))
    progress = min(1.0, max(0.0, progress))
    return effective_peak - progress * (effective_peak - min_lr)


def evaluate_validation(
    model: nn.Module,
    project_dir: Path,
    device: str = "cpu",
    max_batches: int = 40,
    seq_len: int = 2048,
    use_autocast: bool = False,
    cast_dtype: torch.dtype = torch.bfloat16
) -> dict[str, Any]:
    """
    Evaluasi loss dan perplexity pada file-file validasi per sumber di tokenized/_val/ (Bagian 11.1).
    """
    val_dir = project_dir / "tokenized" / "_val"
    if not val_dir.exists():
        return {"val_loss": 0.0, "perplexity": 1.0, "sources": {}}

    model.eval()
    sources_res = {}
    all_losses = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for val_file in val_dir.glob("*.bin"):
            src_name = val_file.stem
            try:
                data = np.memmap(val_file, dtype=np.uint16, mode="r")
                needed_tokens = (seq_len + 1)
                total_sequences = len(data) // needed_tokens
                batches_to_run = min(max_batches, total_sequences)

                if batches_to_run <= 0:
                    continue

                src_losses = []
                for b_idx in range(batches_to_run):
                    start = b_idx * needed_tokens
                    end = start + needed_tokens
                    chunk = data[start:end]
                    x = torch.from_numpy(chunk[:-1].astype(np.int64)).unsqueeze(0).to(device)
                    y = torch.from_numpy(chunk[1:].astype(np.int64)).unsqueeze(0).to(device)

                    if use_autocast:
                        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=cast_dtype):
                            outputs = model(x)
                            loss = loss_fn(outputs.logits.float().view(-1, outputs.logits.size(-1)), y.view(-1))
                    else:
                        outputs = model(x)
                        loss = loss_fn(outputs.logits.float().view(-1, outputs.logits.size(-1)), y.view(-1))

                    src_losses.append(loss.item())

                if src_losses:
                    avg_src = float(np.mean(src_losses))
                    sources_res[src_name] = avg_src
                    all_losses.extend(src_losses)
            except Exception:
                pass

    model.train()
    overall_loss = float(np.mean(all_losses)) if all_losses else 0.0
    ppl = math.exp(min(20.0, overall_loss))
    bpb = (overall_loss / math.log(2.0)) * 0.5  # Perkiraan BPB (uint16 = 2 bytes)

    return {
        "val_loss": overall_loss,
        "perplexity": ppl,
        "bits_per_byte": bpb,
        "sources": sources_res
    }


def generate_sample_completions(
    model: nn.Module,
    project_dir: Path,
    device: str = "cpu"
) -> list[str]:
    """Generasi teks sederhana dari 3 prompt tetap (EN, ID, kode) untuk inspeksi berkala."""
    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        return []

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        prompts = [
            "The universe is composed of",
            "Indonesia memiliki kekayaan alam",
            "def quicksort(arr):",
        ]
        results = []
        model.eval()

        with torch.no_grad():
            for p in prompts:
                input_ids = torch.tensor([tokenizer.encode(p).ids], dtype=torch.long).to(device)
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=11
                )
                gen_text = tokenizer.decode(outputs[0].tolist())
                results.append(f"Prompt: {p}\nHasil: {gen_text}\n")

        model.train()
        return results
    except Exception:
        return []


def run_pretrain(
    project_dir: Path,
    resume: bool = False,
    new_run: bool = False,
    run_id: Optional[str] = None,
    specific_step: Optional[int] = None,
    max_hours: Optional[float] = None,
    override_lr: Optional[float] = None,
    custom_out_dir: Optional[str] = None,
    compile_model: bool = False,
    seed: int = 1337
) -> dict[str, Any]:
    """
    Eksekusi loop pretraining penuh sesuai Bagian 11.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    hyper = plan.get("hyperparameters", {})
    settings = plan.get("settings", {})

    runs_base_dir = Path(custom_out_dir) if custom_out_dir else project_dir / "runs"
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tentukan Run ID
    if resume and not new_run:
        if run_id:
            curr_run_dir = runs_base_dir / run_id
        else:
            existing_runs = sorted(list(runs_base_dir.glob("run_*")), key=lambda p: p.name)
            if not existing_runs:
                raise FileNotFoundError("Tidak ditemukan run sebelumnya untuk di-resume.")
            curr_run_dir = existing_runs[-1]
    else:
        existing_runs = sorted(list(runs_base_dir.glob("run_*")), key=lambda p: p.name)
        next_idx = len(existing_runs) + 1
        curr_run_dir = runs_base_dir / f"run_{next_idx:03d}"

    curr_run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = curr_run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Simpan config run
    atomic_write_json(curr_run_dir / "config.json", plan)

    # 2. Setup Device & Autocast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_autocast = (device == "cuda")

    # 3. Model & Optimizer
    seq_len = int(hyper.get("context_pretrain", 2048))
    config = build_qwen3_config(arch, max_position_embeddings=seq_len)
    grad_checkpointing = bool(settings.get("grad_checkpointing", {}).get("value", False))

    model = create_model(config, seed=seed, gradient_checkpointing=grad_checkpointing)
    model.to(device)

    if compile_model and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    peak_lr = override_lr if override_lr else float(hyper.get("peak_lr", 5.4e-4))
    optimizer = create_adamw_optimizer(model, lr=peak_lr, weight_decay=float(hyper.get("weight_decay", 0.1)))
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not use_bf16) else None

    # 4. Dataloader
    loader = PretrainDataLoader(project_dir, plan, seed=seed)

    # State training
    step = 0
    tokens_seen = 0
    lr_scale = 1.0
    rollback_count = 0
    loss_ema: Optional[float] = None
    spike_consecutive = 0
    best_val_loss = float("inf")

    # Resume jika diminta
    if resume:
        target_ckpt = None
        if specific_step is not None:
            target_ckpt = ckpt_dir / f"step_{specific_step:08d}"
        else:
            target_ckpt = find_latest_checkpoint(ckpt_dir)

        if target_ckpt and target_ckpt.exists():
            res_data = load_checkpoint(target_ckpt, model, optimizer, scaler, device=device)
            t_state = res_data.get("train_state", {})
            step = t_state.get("step", 0)
            tokens_seen = t_state.get("tokens_seen", 0)
            lr_scale = t_state.get("lr_scale", 1.0)
            rollback_count = t_state.get("rollback_count", 0)
            loss_ema = t_state.get("loss_ema")

            d_state = res_data.get("data_state", {})
            loader.load_state(d_state)
            print(f"Resume berhasil dari {target_ckpt.name} (step {step:,}, {tokens_seen / 1e9:.2f}B token).")
        else:
            raise FileNotFoundError("Checkpoint tidak ditemukan untuk resume.")

    total_steps = int(hyper.get("total_steps", 50000))
    tokens_target = float(plan.get("tokens", {}).get("target", 1e9))
    grad_accum = int(hyper.get("grad_accum", 16))
    micro_batch = int(hyper.get("micro_batch", 4))
    warmup_steps = int(hyper.get("warmup_steps", 500))
    eval_every = int(hyper.get("eval_every_tokens", 500_000_000))
    ckpt_every_tokens = int(hyper.get("ckpt_every_tokens", 500_000_000))

    last_eval_tokens = tokens_seen
    last_ckpt_tokens = tokens_seen
    last_ckpt_time = time.time()
    last_heartbeat_time = time.time()
    start_time = time.time()

    loss_fn = nn.CrossEntropyLoss()
    metrics_file = curr_run_dir / "metrics.jsonl"
    run_file = curr_run_dir / "run.json"
    samples_file = curr_run_dir / "samples.txt"

    # Penangan sinyal jeda / Ctrl+C
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        print("\nSinyal berhenti diterima. Menyimpan checkpoint sebelum keluar...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    model.train()

    # 5. Loop Training
    while step < total_steps and tokens_seen < tokens_target and not stop_requested:
        step_start_time = time.time()
        optimizer.zero_grad()

        step_loss = 0.0

        for _ in range(grad_accum):
            x, y = loader.next_batch(micro_batch, seq_len, device=device)
            if use_autocast:
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=cast_dtype):
                    outputs = model(x)
                    loss = loss_fn(outputs.logits.float().view(-1, outputs.logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(x)
                loss = loss_fn(outputs.logits.float().view(-1, outputs.logits.size(-1)), y.view(-1))
                loss = loss / grad_accum
                loss.backward()

            step_loss += loss.item() * grad_accum

        # Deteksi spike & rollback otomatis (Bagian 11.1)
        is_nan = math.isnan(step_loss) or math.isinf(step_loss)
        is_spike = False
        if loss_ema is not None and not is_nan:
            if step_loss > 2.0 * loss_ema:
                is_spike = True

        if is_nan or is_spike:
            spike_consecutive += 1
            if spike_consecutive >= 5 or is_nan:
                if rollback_count < 3:
                    rollback_count += 1
                    spike_consecutive = 0
                    print(f"\n[!] Deteksi lonjakan loss (spike). Rollback #{rollback_count} ke checkpoint sehat...")
                    latest_healthy = find_latest_checkpoint(ckpt_dir)
                    if latest_healthy:
                        res = load_checkpoint(latest_healthy, model, optimizer, scaler, device=device)
                        loader.load_state(res.get("data_state", {}))
                        # Lewati 2000 sequence
                        for _ in range(2000 // micro_batch):
                            loader.next_batch(micro_batch, seq_len, device="cpu")
                        lr_scale *= 0.8
                        loss_ema = None
                        print(f"Rollback selesai. LR diturunkan 20% (lr_scale={lr_scale:.2f}). Melanjutkan...")
                        continue
                else:
                    print("\n[ERROR] Lonjakan loss berulang setelah 3 kali rollback. Training dihentikan.")
                    atomic_write_json(run_file, {
                        "kind": "pretrain", "status": "stopped_spike", "step": step,
                        "tokens_seen": tokens_seen, "finished": False
                    })
                    return {"status": "stopped_spike", "step": step}
        else:
            spike_consecutive = 0
            if loss_ema is None:
                loss_ema = step_loss
            else:
                loss_ema = 0.98 * loss_ema + 0.02 * step_loss

        # Optimizer step & clipping
        if scaler:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

        step += 1
        current_lr = get_wsd_lr(step, total_steps, peak_lr, warmup_steps, lr_scale)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Metrik
        now = time.time()
        elapsed = now - step_start_time
        batch_tok = micro_batch * seq_len * grad_accum
        tokens_seen += batch_tok
        tok_per_s = batch_tok / max(0.001, elapsed)

        # Logging berkala tiap 10 step
        if step % 10 == 0 or step == 1:
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            metric_entry = {
                "step": step, "tokens": tokens_seen, "loss": round(step_loss, 4),
                "lr": current_lr, "grad_norm": round(grad_norm, 4),
                "tok_per_s": round(tok_per_s, 1), "gpu_mem_gb": round(mem_gb, 2),
                "time": datetime.datetime.now().isoformat()
            }
            with open(metrics_file, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(metric_entry) + "\n")

        # Update run.json tiap 30 detik
        if now - last_heartbeat_time >= 30.0:
            atomic_write_json(run_file, {
                "kind": "pretrain",
                "status": "running",
                "pid": os.getpid(),
                "heartbeat": datetime.datetime.now().isoformat(),
                "step": step,
                "tokens_seen": tokens_seen,
                "tokens_target": tokens_target,
                "last_loss": round(step_loss, 4),
                "tok_per_s": round(tok_per_s, 1),
                "finished": False
            })
            last_heartbeat_time = now

        # Eval Validasi berkala
        if (tokens_seen - last_eval_tokens) >= eval_every or step == total_steps:
            val_res = evaluate_validation(model, project_dir, device=device, seq_len=seq_len, use_autocast=use_autocast, cast_dtype=cast_dtype)
            curr_val_loss = val_res.get("val_loss", 0.0)
            is_best = curr_val_loss < best_val_loss
            if is_best:
                best_val_loss = curr_val_loss

            samples = generate_sample_completions(model, project_dir, device=device)
            if samples:
                with open(samples_file, "a", encoding="utf-8") as sf:
                    sf.write(f"\n--- EVAL STEP {step} (Loss {curr_val_loss:.4f}) ---\n" + "".join(samples))

            last_eval_tokens = tokens_seen

        # Checkpoint berkala (berdasarkan token atau 30 menit)
        time_elapsed_ckpt = now - last_ckpt_time
        should_ckpt = ((tokens_seen - last_ckpt_tokens) >= ckpt_every_tokens) or (time_elapsed_ckpt >= 1800.0)

        # Milestone 10%
        pct_progress = int((tokens_seen / tokens_target) * 10) * 10
        milestone_tag = f"milestone_{pct_progress}" if pct_progress > 0 and (tokens_seen - batch_tok) < (pct_progress * tokens_target / 100) else None

        if should_ckpt or milestone_tag:
            t_state = {
                "step": step, "tokens_seen": tokens_seen, "lr_scale": lr_scale,
                "rollback_count": rollback_count, "loss_ema": loss_ema,
                "config_hash": compute_config_hash(arch)
            }
            d_state = loader.get_state()
            save_checkpoint(
                ckpt_dir, step, model, optimizer, t_state, d_state,
                scaler=scaler, is_milestone=milestone_tag
            )
            last_ckpt_tokens = tokens_seen
            last_ckpt_time = now

        # Batas jam (--max-hours)
        if max_hours is not None and (now - start_time) >= (max_hours * 3600.0):
            print(f"\nBatas waktu {max_hours} jam tercapai. Menyimpan dan jeda...")
            stop_requested = True

    # 6. Penyelesaian atau Jeda
    status_str = "finished" if (step >= total_steps or tokens_seen >= tokens_target) else "paused"

    # Simpan checkpoint final/terakhir
    t_state = {
        "step": step, "tokens_seen": tokens_seen, "lr_scale": lr_scale,
        "rollback_count": rollback_count, "loss_ema": loss_ema,
        "config_hash": compute_config_hash(arch)
    }
    d_state = loader.get_state()
    save_checkpoint(ckpt_dir, step, model, optimizer, t_state, d_state, scaler=scaler)

    if status_str == "finished":
        final_dir = ckpt_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        with open(final_dir / "COMPLETE", "w", encoding="utf-8") as f:
            f.write("OK\n")

    atomic_write_json(run_file, {
        "kind": "pretrain",
        "status": status_str,
        "pid": os.getpid(),
        "heartbeat": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_seen": tokens_seen,
        "tokens_target": tokens_target,
        "finished": (status_str == "finished")
    })

    log_history(project_dir, "pretrain", f"--step {step}", True, f"Status: {status_str}, tokens: {tokens_seen:,}")
    return {"status": status_str, "step": step, "tokens_seen": tokens_seen}
