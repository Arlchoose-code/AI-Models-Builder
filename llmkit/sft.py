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
from llmkit.chat_template import render_conversation
from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from llmkit.dataset import SFTDataset, sft_collate_fn
from llmkit.i18n import tr
from llmkit.model import (
    build_qwen3_config,
    create_adamw_optimizer,
    create_model,
)
from llmkit.state import atomic_write_json, compute_config_hash, load_project, log_history


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
                                    samples.append(json.loads(line))
                                except Exception:
                                    pass
                except Exception:
                    pass
    return samples


def run_sft(
    project_dir: Path,
    from_checkpoint: Optional[str] = None,
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed: int = 1337
) -> dict[str, Any]:
    """
    Eksekusi training Supervised Fine-Tuning (SFT) dengan loss mask pada giliran assistant.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    sft_plan = plan.get("sft_data", {})
    settings = plan.get("settings", {})

    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError("Tokenizer tidak ditemukan. Jalankan 'tokenizer' terlebih dahulu.")

    tokenizer = Tokenizer.from_file(str(tok_path))

    # Muat data SFT
    samples = load_sft_samples(project_dir / "data")
    if not samples:
        raise ValueError("Data SFT tidak ditemukan di data/sft/. Siapkan data terlebih dahulu.")

    # Hold-out 1% untuk evaluasi (deterministik berdasarkan hash ID)
    train_samples = []
    val_samples = []
    for s in samples:
        s_id = str(s.get("id", ""))
        h = int(hashlib_id_val(s_id))
        if h % 100 == 0 and len(val_samples) < 5000:
            val_samples.append(s)
        else:
            train_samples.append(s)

    if not val_samples:
        val_samples = train_samples[:min(len(train_samples), 10)]

    vram_gb = 24.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    context_final = int(settings.get("context_final", {}).get("value", 4096))
    max_seq_len = min(context_final, 4096) if vram_gb < 24.0 else context_final

    train_dataset = SFTDataset(train_samples, tokenizer, max_seq_len=max_seq_len)
    val_dataset = SFTDataset(val_samples, tokenizer, max_seq_len=max_seq_len)

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
        rendered_text, spans = render_conversation(samp)
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

    lr = lr_override if lr_override else float(sft_plan.get("lr", 5e-5))
    optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=0.0)

    epochs = epochs_override if epochs_override else int(sft_plan.get("epochs", 3))
    batch_size = 4
    grad_accum = max(1, 128 // batch_size)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    run_file = curr_run_dir / "run.json"
    metrics_file = curr_run_dir / "metrics.jsonl"

    print("\n" + tr(
        f"=== MEMULAI SFT: {len(train_dataset):,} sampel latih, {len(val_dataset):,} sampel evaluasi ===",
        f"=== STARTING SFT: {len(train_dataset):,} train samples, {len(val_dataset):,} validation samples ==="
    ))
    print(f"Epochs: {epochs}, Max Seq Len: {max_seq_len}, LR: {lr:.2e}")

    step = 0
    best_val_loss = float("inf")
    model.train()

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )

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
                (normalized_loss / grad_accum).backward()
                accum_loss += loss.item()
                accum_tokens += valid_tokens

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
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
                        "lr": lr, "time": datetime.datetime.now().isoformat()
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
        save_checkpoint(ckpt_dir, step, model, optimizer, t_state, {}, is_best=is_best)
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
        "finished": True
    })

    log_history(project_dir, "sft", f"--epochs {epochs}", True, f"SFT selesai, best val loss: {best_val_loss:.4f}")
    return {"status": "finished", "step": step, "best_val_loss": best_val_loss}


def hashlib_id_val(s_id: str) -> int:
    """Hash deterministik untuk memisahkan eval split SFT."""
    import hashlib
    return int(hashlib.md5(s_id.encode("utf-8")).hexdigest()[:8], 16)
