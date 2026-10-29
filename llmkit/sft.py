"""Supervised Fine-Tuning (SFT) untuk chat, mode think, dan tool calling (Bagian 12.3)."""

from __future__ import annotations

import datetime
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM
from llmkit.chat_template import get_default_system, render_conversation
from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from llmkit.dataset import SFTDataset, sft_collate_fn
from llmkit.i18n import tr
from llmkit.model import (
    build_qwen3_config,
    create_adamw_optimizer,
    create_model,
)
from llmkit.state import atomic_write_json, compute_config_hash, load_project, log_history
from llmkit.ui import print_info, print_ok, print_warn


def get_sft_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1
) -> float:
    """Learning rate schedule untuk SFT: linear warmup + cosine decay ke min_lr."""
    min_lr = peak_lr * min_lr_ratio
    if total_steps <= 0:
        return peak_lr
    if step <= warmup_steps:
        if warmup_steps <= 0:
            return peak_lr
        return peak_lr * float(step) / float(warmup_steps)
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return min_lr
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def load_sft_samples(data_dir: Path) -> list[dict[str, Any]]:
    """Muat seluruh data SFT dari semua kategori di data/sft/."""
    samples = []
    sft_root = data_dir / "sft"
    if not sft_root.exists():
        return samples

    for cat_dir in sft_root.iterdir():
        if cat_dir.is_dir():
            for jsonl_file in cat_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    item.setdefault("category", cat_dir.name)
                                    item["_category"] = cat_dir.name
                                    samples.append(item)
                                except Exception:
                                    pass
                except Exception:
                    pass
    return samples


def hashlib_id_val(s_id: str) -> int:
    """Hash deterministik untuk memisahkan eval split SFT."""
    import hashlib
    return int(hashlib.md5(s_id.encode("utf-8")).hexdigest()[:8], 16)


def split_sft_samples(
    samples: list[dict[str, Any]],
    val_ratio: float = 0.01,
    max_val: int = 5000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Pisahkan hold-out untuk evaluasi (deterministik berdasarkan hash ID).
    Sampel identity selalu 100% masuk train_samples dan tidak pernah dialihkan ke val.
    """
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    val_mod = max(1, int(round(1.0 / val_ratio))) if val_ratio > 0 else 100

    for s in samples:
        is_identity = (
            s.get("category") == "identity"
            or s.get("_category") == "identity"
            or (
                s.get("category") in (None, "", "identity")
                and (
                    str(s.get("id", "")).startswith("ident")
                    or str(s.get("id", "")).startswith("identity")
                    or str(s.get("id", "")).startswith("id-")
                )
            )
        )
        if is_identity:
            train_samples.append(s)
            continue
        s_id = str(s.get("id", ""))
        h = int(hashlib_id_val(s_id))
        if h % val_mod == 0 and len(val_samples) < max_val:
            val_samples.append(s)
        else:
            train_samples.append(s)

    if not val_samples:
        non_identity_train = [
            s for s in train_samples
            if not (
                s.get("category") == "identity"
                or s.get("_category") == "identity"
                or (
                    s.get("category") in (None, "", "identity")
                    and (
                        str(s.get("id", "")).startswith("ident")
                        or str(s.get("id", "")).startswith("identity")
                        or str(s.get("id", "")).startswith("id-")
                    )
                )
            )
        ]
        val_source = non_identity_train if non_identity_train else train_samples
        val_samples = val_source[:min(len(val_source), 10)]

    return train_samples, val_samples


def run_sft(
    project_dir: Path,
    from_checkpoint: Optional[str] = None,
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed: int = 1337,
    default_system: Optional[str] = None
) -> dict[str, Any]:
    """
    Eksekusi training Supervised Fine-Tuning (SFT) dengan loss mask pada giliran assistant.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    sft_plan = plan.get("sft_data", {})
    settings = plan.get("settings", {})
    answers = project_data.get("answers", {})

    model_name = answers.get("model_name", "Bahasa")
    creator = answers.get("creator", "Pengguna")
    if default_system is None:
        default_system = get_default_system(model_name, creator)

    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError("Tokenizer tidak ditemukan. Jalankan 'tokenizer' terlebih dahulu.")

    tokenizer = Tokenizer.from_file(str(tok_path))

    # Muat data SFT
    samples = load_sft_samples(project_dir / "data")
    if not samples:
        raise ValueError("Data SFT tidak ditemukan di data/sft/. Siapkan data terlebih dahulu.")

    # Hold-out 1% untuk evaluasi (deterministik berdasarkan hash ID)
    # Sampel identity selalu 100% masuk train_samples dan tidak dialihkan ke val
    train_samples, val_samples = split_sft_samples(samples)

    vram_gb = 24.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    context_final = int(settings.get("context_final", {}).get("value", 4096))
    max_seq_len = min(context_final, 4096) if vram_gb < 24.0 else context_final

    train_dataset = SFTDataset(train_samples, tokenizer, max_seq_len=max_seq_len, default_system=default_system)
    val_dataset = SFTDataset(val_samples, tokenizer, max_seq_len=max_seq_len, default_system=default_system)

    if len(train_dataset) == 0:
        raise ValueError(tr(
            f"Semua sampel latih SFT ({len(train_samples)}) melebihi max_seq_len ({max_seq_len}).",
            f"All SFT training samples ({len(train_samples)}) exceed max_seq_len ({max_seq_len})."
        ))

    total_skipped = train_dataset.skipped_overlong + val_dataset.skipped_overlong
    if total_skipped > 0:
        print_info(tr(
            f"Sampel melebihi max_seq_len ({max_seq_len}) dilewati: {total_skipped} sampel (train: {train_dataset.skipped_overlong}, val: {val_dataset.skipped_overlong})",
            f"Samples exceeding max_seq_len ({max_seq_len}) skipped: {total_skipped} samples (train: {train_dataset.skipped_overlong}, val: {val_dataset.skipped_overlong})"
        ))

    # Setup Run SFT
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(list(runs_dir.glob("run_*")), key=lambda p: p.name)
    next_idx = len(existing_runs) + 1
    curr_run_dir = runs_dir / f"run_{next_idx:03d}"
    curr_run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = curr_run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not use_bf16) else None

    # Inisialisasi Model
    cfg = build_qwen3_config(arch, max_position_embeddings=max_seq_len)
    model = create_model(cfg, seed=seed).to(device)

    # Muat bobot pretrain
    pretrain_ckpt_loaded = False
    target_ckpt_path = None
    if from_checkpoint:
        cand = Path(from_checkpoint)
        if cand.exists():
            target_ckpt_path = cand
        elif ":" in from_checkpoint:
            r_part, s_part = from_checkpoint.split(":", 1)
            step_int = int(s_part) if s_part.isdigit() else int(s_part.replace("step_", ""))
            cand = runs_dir / r_part / "ckpt" / f"step_{step_int:08d}"
            if cand.exists():
                target_ckpt_path = cand
        else:
            cand_run = runs_dir / from_checkpoint / "ckpt"
            if (cand_run / "final").exists():
                target_ckpt_path = cand_run / "final"
            else:
                target_ckpt_path = find_latest_checkpoint(cand_run)
    else:
        # Cari dari run pretrain terakhir
        for r_dir in reversed(existing_runs):
            r_json = r_dir / "run.json"
            if r_json.exists():
                try:
                    with open(r_json, "r", encoding="utf-8") as f:
                        rj = json.load(f)
                    if rj.get("kind") == "pretrain":
                        fin_cand = r_dir / "ckpt" / "final"
                        if fin_cand.exists() and (fin_cand / "COMPLETE").exists():
                            target_ckpt_path = fin_cand
                            break
                        cand_latest = find_latest_checkpoint(r_dir / "ckpt")
                        if cand_latest:
                            target_ckpt_path = cand_latest
                            break
                except Exception:
                    pass

    if target_ckpt_path and target_ckpt_path.exists():
        load_checkpoint(target_ckpt_path, model, device=device)
        pretrain_ckpt_loaded = True
        print(tr(f"[OK] Bobot pretrain berhasil dimuat dari: {target_ckpt_path}", f"[OK] Pretrain weights loaded from: {target_ckpt_path}"))
    else:
        raise FileNotFoundError(
            "SFT membutuhkan checkpoint pretrain selesai (final/) atau run pretrain yang valid.\n"
            "Solusi:\n"
            "  1. Jalankan pretraining terlebih dahulu: python main.py pretrain\n"
            "  2. Atau tentukan checkpoint tertentu: python main.py sft --from <run>[:step]"
        )

    # Cetak 3 sampel hasil render + mask (Bagian 12.3)
    print("\n" + "=" * 50)
    print(tr("PREVIEW FORMAT SFT (3 Sampel Render & Loss Masking):", "SFT FORMAT PREVIEW (3 Rendered Samples & Loss Masking):"))
    for s_idx, samp in enumerate(train_samples[:3], start=1):
        rendered_text, spans = render_conversation(samp, default_system=default_system)
        print(f"\n--- {tr('Sampel', 'Sample')} #{s_idx} ({samp.get('id', '-')}, mode: {samp.get('mode', '-')}, lang: {samp.get('lang', '-')}) ---")
        curr_pos = 0
        input_tag = tr("[INPUT/TIDAK DILATIH]", "[INPUT/UNMASKED]")
        asst_tag = tr("[ASSISTANT/DILATIH MASK-LOSS]", "[ASSISTANT/TRAINED LOSS MASK]")
        for start, end in spans:
            if start > curr_pos:
                print(f"{input_tag}\n{rendered_text[curr_pos:start]}", end="")
            print(f"{asst_tag}\n{rendered_text[start:end]}\n", end="")
            curr_pos = end
        if curr_pos < len(rendered_text):
            print(f"{input_tag}\n{rendered_text[curr_pos:]}", end="")
        print("-" * 40)
    print("=" * 50 + "\n")

    epochs = epochs_override if epochs_override else int(sft_plan.get("epochs", 3))
    batch_size = 4
    grad_accum = max(1, 128 // batch_size)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    run_file = curr_run_dir / "run.json"
    metrics_file = curr_run_dir / "metrics.jsonl"

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum) if len(train_loader) > 0 else 1
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_ratio = float(sft_plan.get("warmup", 0.03))
    warmup_steps = max(1, int(round(total_steps * warmup_ratio))) if total_steps > 1 else 0

    lr = lr_override if lr_override else float(sft_plan.get("lr", 5e-5))
    initial_sft_lr = get_sft_lr(1, total_steps, lr, warmup_steps, min_lr_ratio=0.1)
    optimizer = create_adamw_optimizer(model, lr=initial_sft_lr, weight_decay=0.0)

    print("\n" + tr(
        f"=== MEMULAI SFT: {len(train_dataset):,} sampel latih, {len(val_dataset):,} sampel evaluasi ===",
        f"=== STARTING SFT: {len(train_dataset):,} train samples, {len(val_dataset):,} validation samples ==="
    ))
    print(f"Epochs: {epochs}, Max Seq Len: {max_seq_len}, Peak LR: {lr:.2e}, Warmup Steps: {warmup_steps}/{total_steps}")

    step = 0
    best_val_loss = float("inf")
    model.train()

    for epoch in range(1, epochs + 1):
        accum_loss = 0.0
        accum_tokens = 0
        optimizer.zero_grad()

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=cast_dtype):
                    outputs = model(input_ids)
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fn(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
            else:
                outputs = model(input_ids)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))

            valid_tokens = (shift_labels != -100).sum().item()
            if valid_tokens > 0:
                normalized_loss = loss / max(1, valid_tokens)
                accum_loss += loss.item()
                accum_tokens += valid_tokens
                scaled_loss = normalized_loss / grad_accum
                if scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                target_step = step + 1
                current_lr = get_sft_lr(target_step, total_steps, lr, warmup_steps, min_lr_ratio=0.1)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                step += 1

                step_loss = accum_loss / max(1, accum_tokens)
                accum_loss = 0.0
                accum_tokens = 0

                if step % 5 == 0 or step == 1:
                    metric = {
                        "epoch": epoch, "step": step, "loss": round(step_loss, 4),
                        "lr": current_lr, "time": datetime.datetime.now().isoformat()
                    }
                    with open(metrics_file, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(metric) + "\n")

        # Evaluasi akhir epoch
        model.eval()
        val_loss_sum = 0.0
        val_tokens_sum = 0
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=sft_collate_fn)
            for v_in, _, v_lbl in val_loader:
                v_in = v_in.to(device)
                v_lbl = v_lbl.to(device)
                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=cast_dtype):
                        out = model(v_in)
                        s_log = out.logits[..., :-1, :].contiguous()
                        s_lbl = v_lbl[..., 1:].contiguous()
                        v_loss = loss_fn(s_log.view(-1, cfg.vocab_size), s_lbl.view(-1))
                else:
                    out = model(v_in)
                    s_log = out.logits[..., :-1, :].contiguous()
                    s_lbl = v_lbl[..., 1:].contiguous()
                    v_loss = loss_fn(s_log.view(-1, cfg.vocab_size), s_lbl.view(-1))
                v_toks = (s_lbl != -100).sum().item()
                val_loss_sum += v_loss.item()
                val_tokens_sum += v_toks

        avg_val_loss = val_loss_sum / max(1, val_tokens_sum)
        print(tr(f"Epoch {epoch}/{epochs} Selesai - Val Loss: {avg_val_loss:.4f}", f"Epoch {epoch}/{epochs} Completed - Val Loss: {avg_val_loss:.4f}"))

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Checkpoint per epoch
        t_state = {"epoch": epoch, "step": step, "val_loss": avg_val_loss}
        save_checkpoint(ckpt_dir, step, model, optimizer, t_state, {}, scaler=scaler, is_best=is_best)
        model.train()

    # Finalize SFT
    final_dir = ckpt_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    with open(final_dir / "COMPLETE", "w", encoding="utf-8") as f:
        f.write("OK\n")

    atomic_write_json(run_file, {
        "kind": "sft",
        "status": "finished",
        "step": step,
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 4),
        "skipped_overlong": total_skipped,
        "finished": True
    })

    log_history(project_dir, "sft", f"--epochs {epochs}", True, f"SFT selesai, best val loss: {best_val_loss:.4f}")
    return {"status": "finished", "step": step, "best_val_loss": best_val_loss, "skipped_overlong": total_skipped}
