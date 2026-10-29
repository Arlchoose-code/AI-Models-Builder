"""Manajemen checkpoint atomik, rotasi, milestone, best, dan pemulihan state RNG & Dataloader (Bagian 11.3)."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from llmkit.state import atomic_write_json


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: dict[str, Any],
    data_state: dict[str, Any],
    scaler: Optional[torch.amp.GradScaler] = None,
    is_milestone: Optional[str] = None,
    is_best: bool = False,
    keep_recent: int = 3
) -> Path:
    """
    Simpan checkpoint secara atomik ke <nama>.tmp lalu rename menjadi folder akhir.
    Menulis file COMPLETE terakhir sebagai penanda integritas.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    folder_name = is_milestone if is_milestone else f"step_{step:08d}"
    final_path = ckpt_dir / folder_name
    tmp_path = ckpt_dir / f"{folder_name}.tmp"

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # 1. model.pt (fp32)
    model_cpu = {k: v.cpu().float() for k, v in model.state_dict().items()}
    torch.save(model_cpu, tmp_path / "model.pt")

    # 2. optim.pt
    optim_data = {"optimizer": optimizer.state_dict()}
    if scaler is not None:
        optim_data["scaler"] = scaler.state_dict()
    torch.save(optim_data, tmp_path / "optim.pt")

    # 3. train_state.json & data_state.json
    with open(tmp_path / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(train_state, f, indent=2)
    with open(tmp_path / "data_state.json", "w", encoding="utf-8") as f:
        json.dump(data_state, f, indent=2)

    # 4. rng_state.pt
    rng_dict = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_dict["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_dict, tmp_path / "rng_state.pt")

    # 5. COMPLETE marker
    with open(tmp_path / "COMPLETE", "w", encoding="utf-8") as f:
        f.write("OK\n")
        f.flush()
        os.fsync(f.fileno())

    # Rename atomik
    if final_path.exists():
        shutil.rmtree(final_path, ignore_errors=True)
    os.replace(tmp_path, final_path)

    # Update latest.json jika bukan sekadar milestone terpisah
    if not is_milestone:
        atomic_write_json(ckpt_dir / "latest.json", {"path": folder_name, "step": step})

    # Best checkpoint
    if is_best:
        best_path = ckpt_dir / "best"
        if best_path.exists():
            shutil.rmtree(best_path, ignore_errors=True)
        shutil.copytree(final_path, best_path)

    # Rotasi: simpan keep_recent step_* terakhir, hapus yang lebih lama
    rotate_checkpoints(ckpt_dir, keep=keep_recent)

    return final_path


def rotate_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    """Hapus folder step_* yang lebih lama dari `keep` terakhir."""
    step_folders = []
    for item in ckpt_dir.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            complete_file = item / "COMPLETE"
            if complete_file.exists():
                step_folders.append(item)
            else:
                # Bersihkan checkpoint yang rusak / tidak selesai
                shutil.rmtree(item, ignore_errors=True)
        elif item.is_dir() and item.name.endswith(".tmp"):
            # Bersihkan sisa temporary folder
            shutil.rmtree(item, ignore_errors=True)

    step_folders.sort(key=lambda p: p.name)
    if len(step_folders) > keep:
        to_delete = step_folders[:-keep]
        for folder in to_delete:
            shutil.rmtree(folder, ignore_errors=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Muat checkpoint lengkap: model, optimizer, scaler, train_state, data_state, dan RNG.
    """
    path = Path(path)
    if not (path / "COMPLETE").exists() and not (path / "model.safetensors").exists():
        raise ValueError(f"Checkpoint di {path} rusak atau belum selesai (file COMPLETE tidak ada).")

    # 1. Model
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_file
        model_state = load_file(str(path / "model.safetensors"), device=device)
    elif (path / "model.pt").exists():
        model_state = torch.load(path / "model.pt", map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"Tidak ada model.pt atau model.safetensors di {path}")
    model.load_state_dict(model_state, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # 2. Optimizer & Scaler
    if optimizer is not None and (path / "optim.pt").exists():
        optim_state = torch.load(path / "optim.pt", map_location=device, weights_only=False)
        if "optimizer" in optim_state:
            optimizer.load_state_dict(optim_state["optimizer"])
        if scaler is not None and "scaler" in optim_state:
            scaler.load_state_dict(optim_state["scaler"])

    # 3. Train State & Data State
    train_state = {}
    if (path / "train_state.json").exists():
        with open(path / "train_state.json", "r", encoding="utf-8") as f:
            train_state = json.load(f)

    data_state = {}
    if (path / "data_state.json").exists():
        with open(path / "data_state.json", "r", encoding="utf-8") as f:
            data_state = json.load(f)

    # 4. RNG
    if (path / "rng_state.pt").exists():
        rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return {
        "train_state": train_state,
        "data_state": data_state,
        "step": train_state.get("step", 0),
        "tokens_seen": train_state.get("tokens_seen", 0),
    }


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Cari checkpoint valid terbaru dari latest.json atau direktori ckpt/."""
    ckpt_dir = Path(ckpt_dir)
    latest_file = ckpt_dir / "latest.json"
    if latest_file.exists():
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidate = ckpt_dir / data.get("path", "")
            if candidate.exists() and (candidate / "COMPLETE").exists():
                return candidate
        except Exception:
            pass

    # Fallback cari folder step_* terbesar yang memiliki COMPLETE
    step_folders = [
        d for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_") and (d / "COMPLETE").exists()
    ]
    if step_folders:
        step_folders.sort(key=lambda p: p.name)
        return step_folders[-1]

    final_cand = ckpt_dir / "final"
    if final_cand.exists() and (
        (final_cand / "COMPLETE").exists()
        or (final_cand / "model.safetensors").exists()
        or (final_cand / "model.pt").exists()
    ):
        return final_cand

    return None
