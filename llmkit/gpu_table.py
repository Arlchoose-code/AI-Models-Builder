"""Tabel GPU (TFLOPs efektif, VRAM, harga per jam, presisi) dan deteksi otomatis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSpec:
    name: str
    vram_gb: float
    effective_tflops: float
    price_per_hour: float
    precision: str  # "bf16" atau "fp16"
    aliases: list[str]


GPU_DATABASE: list[GPUSpec] = [
    GPUSpec("H100", 80.0, 300.0, 2.5, "bf16", ["h100", "nvidia h100"]),
    GPUSpec("A100", 40.0, 120.0, 1.5, "bf16", ["a100", "nvidia a100"]),
    GPUSpec("L4", 24.0, 35.0, 0.8, "bf16", ["l4", "nvidia l4"]),
    GPUSpec("RTX 4090", 24.0, 80.0, 0.0, "bf16", ["4090", "rtx 4090", "geforce rtx 4090"]),
    GPUSpec("RTX 4060 Laptop", 8.0, 12.0, 0.0, "bf16", ["4060 laptop", "4060 mobile", "rtx 4060 laptop gpu"]),
    GPUSpec("T4", 16.0, 15.0, 0.0, "fp16", ["t4", "tesla t4", "nvidia t4"]),
]


def get_gpu_spec(name: str) -> Optional[GPUSpec]:
    """Cari GPUSpec berdasarkan nama atau alias (case-insensitive)."""
    norm = name.strip().lower()
    for spec in GPU_DATABASE:
        if norm == spec.name.lower():
            return spec
        for alias in spec.aliases:
            if alias in norm or norm in alias:
                return spec
    return None


def detect_gpu() -> tuple[Optional[str], Optional[GPUSpec]]:
    """Deteksi GPU sistem menggunakan torch jika tersedia."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0).strip()
            # Cek kecocokan
            for spec in GPU_DATABASE:
                for alias in spec.aliases:
                    if alias in dev_name.lower():
                        return dev_name, spec
            # Fallback jika ada CUDA tapi tidak di daftar standar
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / (1024 ** 3), 1)
            # Default perkiraan
            custom_spec = GPUSpec(
                name=dev_name,
                vram_gb=vram,
                effective_tflops=30.0,
                price_per_hour=0.0,
                precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
                aliases=[]
            )
            return dev_name, custom_spec
    except Exception:
        pass
    return None, None
