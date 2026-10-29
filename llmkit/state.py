"""Manajemen project.json, logging riwayat (history.log), dan deteksi status disk."""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Tulis file JSON secara atomik (.tmp -> os.replace)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)


def atomic_write_text(file_path: Path, text: str) -> None:
    """Tulis file teks secara atomik (.tmp -> os.replace)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)


def log_history(project_dir: Path, command: str, args_str: str, success: bool, summary: str) -> None:
    """Catat perintah ke logs/history.log."""
    try:
        log_file = project_dir / "logs" / "history.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.datetime.now().isoformat()
        status_str = "ok" if success else "gagal"
        line = f"{now_iso} | {command} | {args_str} | {status_str} | {summary}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def compute_config_hash(arch_config: dict[str, Any]) -> str:
    """Hash representasi konfigurasi arsitektur model."""
    keys = ["d", "L", "Hq", "Hkv", "hd", "ffn", "V"]
    norm = {k: arch_config.get(k) for k in keys}
    s = json.dumps(norm, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def compute_data_fingerprint(data_dir: Path) -> list[dict[str, Any]]:
    """Daftar path, ukuran, dan mtime semua file data pretrain."""
    entries: list[dict[str, Any]] = []
    if not data_dir.exists():
        return entries

    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
                p = Path(root) / f
                rel = str(p.relative_to(data_dir)).replace("\\", "/")
                try:
                    stat = p.stat()
                    entries.append({
                        "path": rel,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except OSError:
                    pass
    return sorted(entries, key=lambda x: x["path"])


def is_process_alive(pid: int) -> bool:
    """Periksa apakah proses dengan PID tertentu masih aktif di sistem."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass

    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if handle != 0:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def load_project(project_dir: Path) -> dict[str, Any]:
    """Muat project.json dari direktori proyek."""
    proj_file = project_dir / "project.json"
    if not proj_file.exists():
        raise FileNotFoundError(f"File {proj_file} tidak ditemukan.")
    with open(proj_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project_dir: Path, data: dict[str, Any]) -> None:
    """Simpan project.json secara atomik."""
    atomic_write_json(project_dir / "project.json", data)


def detect_stage_status(project_dir: Path) -> dict[str, Any]:
    """
    Deteksi status proyek langsung dari kondisi file di disk (Bagian 5.2).
    Mengembalikan dict rincian status per tahap, run aktif, dan rekomendasi berikutnya.
    """
    stages: dict[str, dict[str, Any]] = {}
    next_step: Optional[str] = None
    next_command: Optional[str] = None
    warning_note: Optional[str] = None

    # 1. init
    proj_file = project_dir / "project.json"
    if not proj_file.exists():
        return {
            "stages": {"init": {"status": "[ ]", "label": "belum"}},
            "next_step": "init",
            "next_command": "python main.py init",
            "warning": "Proyek belum diinisialisasi."
        }

    try:
        project_data = load_project(project_dir)
        stages["init"] = {"status": "[x]", "label": "selesai", "data": project_data}
    except Exception as e:
        return {
            "stages": {"init": {"status": "[!]", "label": "rusak", "error": str(e)}},
            "next_step": "init",
            "next_command": "python main.py init",
            "warning": "project.json rusak."
        }

    plan = project_data.get("plan", {})
    answers = project_data.get("answers", {})

    # 2. data
    data_dir = project_dir / "data"
    pretrain_data_dir = data_dir / "pretrain"
    has_pretrain_files = False
    if pretrain_data_dir.exists():
        for root, _, files in os.walk(pretrain_data_dir):
            if any(f.endswith(".jsonl") for f in files):
                has_pretrain_files = True
                break

    if has_pretrain_files:
        stages["data"] = {"status": "[x]", "label": "ada file data"}
    else:
        stages["data"] = {"status": "[ ]", "label": "belum diisi"}
        if not next_step:
            next_step = "data"
            next_command = "python main.py check"

    # 3. tokenizer
    tok_json = project_dir / "tokenizer" / "tokenizer.json"
    tok_meta = project_dir / "tokenizer" / "tokenizer_meta.json"
    if tok_json.exists() and tok_meta.exists():
        stages["tokenizer"] = {"status": "[x]", "label": "selesai"}
    else:
        stages["tokenizer"] = {"status": "[ ]", "label": "belum"}
        if not next_step and has_pretrain_files:
            next_step = "tokenizer"
            next_command = "python main.py tokenizer"

    # 4. prep
    manifest_file = project_dir / "tokenized" / "_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            recorded_fp = manifest.get("data_fingerprint", [])
            current_fp = compute_data_fingerprint(pretrain_data_dir)
            if recorded_fp == current_fp:
                stages["prep"] = {"status": "[x]", "label": "selesai"}
            else:
                stages["prep"] = {"status": "[!]", "label": "STALE (data berubah)", "stale": True}
                if not next_step:
                    next_step = "prep"
                    next_command = "python main.py prep"
                    warning_note = "Data mentah berubah sejak terakhir di-prep. Perlu prep ulang."
        except Exception:
            stages["prep"] = {"status": "[!]", "label": "manifest rusak"}
    else:
        stages["prep"] = {"status": "[ ]", "label": "belum"}
        if not next_step and stages.get("tokenizer", {}).get("status") == "[x]":
            next_step = "prep"
            next_command = "python main.py prep"

    # 5. bench
    bench_file = project_dir / "bench.json"
    curr_cfg_hash = compute_config_hash(plan.get("architecture", {}))
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            if bench_data.get("config_hash") == curr_cfg_hash:
                stages["bench"] = {"status": "[x]", "label": f"selesai ({bench_data.get('tok_per_s', 0):,} tok/s)"}
            else:
                stages["bench"] = {"status": "[!]", "label": "STALE (arsitektur berubah)"}
        except Exception:
            stages["bench"] = {"status": "[!]", "label": "bench.json rusak"}
    else:
        stages["bench"] = {"status": "[ ]", "label": "belum"}
        if not next_step and stages.get("prep", {}).get("status") == "[x]":
            next_step = "bench"
            next_command = "python main.py bench"

    # 6. pilot
    pilot_file = project_dir / "pilot" / "result.json"
    if pilot_file.exists():
        stages["pilot"] = {"status": "[x]", "label": "selesai"}
    else:
        stages["pilot"] = {"status": "[ ]", "label": "belum"}
        if not next_step and stages.get("bench", {}).get("status") == "[x]":
            next_step = "pilot"
            next_command = "python main.py pilot"

    # 7. pretrain & deteksi run.json
    runs_dir = Path(answers.get("runs_dir", project_dir / "runs"))
    if not runs_dir.is_absolute():
        runs_dir = project_dir / runs_dir

    latest_pretrain_run = None
    pretrain_finished = False
    pretrain_crashed = False

    if runs_dir.exists():
        for run_path in sorted(runs_dir.glob("run_*")):
            r_json = run_path / "run.json"
            if r_json.exists():
                try:
                    with open(r_json, "r", encoding="utf-8") as f:
                        r_data = json.load(f)
                    if r_data.get("kind") == "pretrain":
                        latest_pretrain_run = (run_path.name, r_data)
                        if r_data.get("status") == "finished":
                            pretrain_finished = True
                        elif r_data.get("status") == "running":
                            # Cek heartbeat & pid
                            hb_str = r_data.get("heartbeat")
                            pid = r_data.get("pid", 0)
                            hb_age = 999999
                            if hb_str:
                                try:
                                    hb_time = datetime.datetime.fromisoformat(hb_str)
                                    hb_age = (datetime.datetime.now() - hb_time).total_seconds()
                                except Exception:
                                    pass
                            if hb_age > 180 or not is_process_alive(pid):
                                pretrain_crashed = True
                        elif r_data.get("status") in ("paused", "crashed", "stopped_spike"):
                            pretrain_crashed = True
                except Exception:
                    pass

    if pretrain_finished:
        stages["pretrain"] = {"status": "[x]", "label": "selesai"}
    elif pretrain_crashed:
        stages["pretrain"] = {"status": "[!]", "label": "terhenti / crash"}
        next_step = "pretrain"
        next_command = "python main.py pretrain --resume"
        warning_note = "Training berhenti tiba-tiba (sesi putus/crash?). Silakan resume."
    elif latest_pretrain_run and latest_pretrain_run[1].get("status") == "running":
        stages["pretrain"] = {"status": "[~]", "label": "sedang berjalan"}
    else:
        stages["pretrain"] = {"status": "[ ]", "label": "belum"}
        if not next_step and stages.get("pilot", {}).get("status") == "[x]":
            next_step = "pretrain"
            next_command = "python main.py pretrain"

    # 8. sft
    latest_sft_run = None
    sft_finished = False
    if runs_dir.exists():
        for run_path in sorted(runs_dir.glob("run_*")):
            r_json = run_path / "run.json"
            if r_json.exists():
                try:
                    with open(r_json, "r", encoding="utf-8") as f:
                        r_data = json.load(f)
                    if r_data.get("kind") == "sft":
                        latest_sft_run = (run_path.name, r_data)
                        if r_data.get("status") == "finished":
                            sft_finished = True
                except Exception:
                    pass

    if sft_finished:
        stages["sft"] = {"status": "[x]", "label": "selesai"}
    elif latest_sft_run and latest_sft_run[1].get("status") == "running":
        stages["sft"] = {"status": "[~]", "label": "sedang berjalan"}
    else:
        stages["sft"] = {"status": "[ ]", "label": "belum"}
        if not next_step and pretrain_finished:
            next_step = "sft"
            next_command = "python main.py sft"

    # 9. eval
    eval_done = False
    if runs_dir.exists():
        for run_path in sorted(runs_dir.glob("run_*")):
            if (run_path / "eval.json").exists():
                eval_done = True
                break

    if eval_done:
        stages["eval"] = {"status": "[x]", "label": "selesai"}
    else:
        stages["eval"] = {"status": "[ ]", "label": "belum"}
        if not next_step and sft_finished:
            next_step = "eval"
            next_command = "python main.py eval"

    # 10. export
    export_dir = project_dir / "export"
    has_export = False
    if export_dir.exists():
        for p in export_dir.rglob("model.safetensors"):
            has_export = True
            break

    if has_export:
        stages["export"] = {"status": "[x]", "label": "selesai"}
    else:
        stages["export"] = {"status": "[ ]", "label": "belum"}
        if not next_step and (sft_finished or eval_done):
            next_step = "export"
            next_command = "python main.py export"

    return {
        "stages": stages,
        "next_step": next_step or "selesai",
        "next_command": next_command or "python main.py chat",
        "warning": warning_note,
        "latest_pretrain_run": latest_pretrain_run,
        "latest_sft_run": latest_sft_run,
    }
