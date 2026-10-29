"""Diagnostik lingkungan sistem, hardware, versi library, dan uji forward/backward GPU (Bagian 12.7).
Mendukung dwibahasa (Bahasa Indonesia & English).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional
from llmkit.i18n import get_lang, tr
from llmkit.ui import print_error, print_header, print_info, print_ok, print_warn


def run_doctor(project_dir: Optional[Path] = None) -> bool:
    """
    Periksa seluruh kesiapan lingkungan dan tampilkan laporan OK/WARN/ERROR.
    Mengembalikan True jika tidak ada ERROR kritis.
    """
    print_header(tr("DIAGNOSTIK LINGKUNGAN (DOCTOR)", "ENVIRONMENT DIAGNOSTICS (DOCTOR)"))
    has_error = False

    # 1. Versi Python
    py_ver = sys.version.split()[0]
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 13:
        print_ok(tr(f"Versi Python: {py_ver} (>= 3.13 didukung penuh)", f"Python version: {py_ver} (>= 3.13 fully supported)"))
    elif major == 3 and minor >= 10:
        print_warn(tr(f"Versi Python: {py_ver} (Direkomendasikan Python 3.13 untuk performa terbaik)", f"Python version: {py_ver} (Python 3.13 recommended for optimal performance)"))
    else:
        print_error(tr(f"Versi Python {py_ver} terlalu usang. Butuh minimal Python 3.10+ (disarankan 3.13).", f"Python version {py_ver} is too old. Requires at least Python 3.10+ (3.13 recommended)."))
        has_error = True

    # 2. PyTorch & CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_tag = f"CUDA {torch.version.cuda}" if cuda_avail else tr("CPU saja", "CPU only")
        print_ok(f"PyTorch: {torch.__version__} ({cuda_tag})")

        if cuda_avail:
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            bf16_ok = torch.cuda.is_bf16_supported()
            bf16_str = tr("Didukung", "Supported") if bf16_ok else tr("Tidak didukung", "Not supported")
            print_ok(tr(f"GPU Terdeteksi: {dev_name} (VRAM {vram_gb:.1f} GB, bf16: {bf16_str})", f"GPU Detected: {dev_name} (VRAM {vram_gb:.1f} GB, bf16: {bf16_str})"))
        else:
            print_warn(tr("CUDA GPU tidak terdeteksi. Training akan berjalan sangat lambat pada CPU.", "CUDA GPU not detected. Training will run very slowly on CPU."))
    except ImportError:
        print_error(tr("Library 'torch' belum terpasang. Jalankan: pip install torch", "Library 'torch' is not installed. Run: pip install torch"))
        has_error = True

    # 3. Transformers
    try:
        import transformers
        trans_ver = transformers.__version__
        has_qwen3 = hasattr(transformers, "Qwen3ForCausalLM")
        if has_qwen3:
            print_ok(tr(f"Transformers: {trans_ver} (Arsitektur Qwen3 didukung)", f"Transformers: {trans_ver} (Qwen3 architecture supported)"))
        else:
            print_error(tr(f"Transformers {trans_ver} belum mendukung Qwen3. Butuh >= 4.51.0 (penyebab: KeyError: 'qwen3').", f"Transformers {trans_ver} does not support Qwen3 yet. Requires >= 4.51.0 (cause: KeyError: 'qwen3')."))
            has_error = True
    except ImportError:
        print_error(tr("Library 'transformers' belum terpasang. Jalankan: pip install transformers>=4.51.0", "Library 'transformers' is not installed. Run: pip install transformers>=4.51.0"))
        has_error = True

    # 4. Dependensi Pendukung
    deps = ["tokenizers", "safetensors", "numpy", "tqdm"]
    for d in deps:
        try:
            mod = __import__(d)
            ver = getattr(mod, "__version__", tr("terpasang", "installed"))
            print_ok(tr(f"Dependensi '{d}': {ver}", f"Dependency '{d}': {ver}"))
        except ImportError:
            print_error(tr(f"Dependensi '{d}' belum terpasang. Jalankan: pip install {d}", f"Dependency '{d}' is not installed. Run: pip install {d}"))
            has_error = True

    # Rich (Opsional)
    try:
        import rich
        try:
            import importlib.metadata
            r_ver = importlib.metadata.version("rich")
        except Exception:
            r_ver = tr("terpasang", "installed")
        print_ok(tr(f"Rich UI: {r_ver} (Tampilan berwarna aktif)", f"Rich UI: {r_ver} (Colored display active)"))
    except ImportError:
        print_info(tr("Rich UI tidak terpasang (menggunakan fallback teks biasa).", "Rich UI not installed (using plain text fallback)."))

    # 5. OS & Perangkat Keras
    cpu_count = os.cpu_count() or 1
    os_name = sys.platform
    print_info(tr(f"Sistem Operasi: {os_name} | CPU: {cpu_count} core", f"Operating System: {os_name} | CPU: {cpu_count} cores"))
    if os_name == "win32":
        print_info(tr(
            "Catatan Windows: torch.compile dimatikan secara default. Untuk performa maksimal, pertimbangkan WSL2.",
            "Windows Note: torch.compile is disabled by default. For maximum performance, consider WSL2."
        ))

    # Ruang Disk
    check_path = project_dir or Path.cwd()
    try:
        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024 ** 3)
        if free_gb < 20.0:
            print_warn(tr(
                f"Sisa ruang disk di {check_path.drive or check_path}: {free_gb:.1f} GB (Saran: sediakan minimal 50 GB+)",
                f"Free disk space on {check_path.drive or check_path}: {free_gb:.1f} GB (Tip: provide at least 50 GB+)"
            ))
        else:
            print_ok(tr(f"Sisa ruang disk di direktori kerja: {free_gb:.1f} GB", f"Free disk space in working directory: {free_gb:.1f} GB"))
    except Exception:
        pass

    # 6. Uji Kecil Forward & Backward
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(2, 64, device=dev, requires_grad=True)
        w = torch.randn(64, 64, device=dev, requires_grad=True)
        y = (x @ w).sum()
        y.backward()
        print_ok(tr(f"Uji tensor alokasi dan autograd pada '{dev}': Berhasil", f"Tensor allocation and autograd test on '{dev}': Success"))
    except Exception as e:
        print_error(tr(f"Uji alokasi tensor gagal: {e}", f"Tensor allocation test failed: {e}"))
        has_error = True

    status_tag = tr(
        "[ERROR DITEMUKAN]" if has_error else "[SEMUA SISTEM SIAP]",
        "[ERRORS FOUND]" if has_error else "[ALL SYSTEMS READY]"
    )
    lbl = tr("Status Diagnostik: ", "Diagnostic Status: ")
    print(f"\n{lbl}{status_tag}")
    return not has_error
