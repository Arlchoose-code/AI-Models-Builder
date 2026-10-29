"""Manajemen Settings (Bagian 18) dengan mode auto/manual dan aturan kalkulasi."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional
from llmkit.gpu_table import GPUSpec, detect_gpu, get_gpu_spec


SETTING_KEYS = [
    "goal",
    "profile",
    "languages",
    "vision",
    "think",
    "tool_call",
    "think_max_tokens",
    "context_pretrain",
    "context_final",
    "code_focus",
    "vocab_size",
    "arch_style",
    "precision",
    "grad_checkpointing",
    "micro_batch",
    "ckpt_every",
    "dedup",
    "leak_filter",
    "optimizer",
]


def detect_system_profile(gpu_spec: Optional[GPUSpec]) -> str:
    """Deteksi profil lingkungan: colab, laptop, atau cloud."""
    if "google.colab" in sys.modules or "COLAB_GPU" in os.environ:
        return "colab"
    if gpu_spec is not None:
        name_lower = gpu_spec.name.lower()
        if "laptop" in name_lower or "mobile" in name_lower or gpu_spec.vram_gb <= 12.0:
            return "laptop"
    return "cloud"


def compute_auto_settings(answers: dict[str, Any], current_settings: Optional[dict[str, Any]] = None) -> dict[str, dict[str, Any]]:
    """
    Hitung nilai auto untuk semua settings sesuai urutan dependensi (18.1 & 18.2).
    Mempertahankan setting yang di-set 'manual' oleh pengguna.
    """
    current_settings = current_settings or {}
    result: dict[str, dict[str, Any]] = {}

    size_b = float(answers.get("size_b", 0.6))
    params_est = size_b * 1e9
    gpu_name = answers.get("gpu", "H100")
    gpu_spec = get_gpu_spec(gpu_name)
    vram_gb = gpu_spec.vram_gb if gpu_spec else 24.0

    # 1. goal
    goal_mode = current_settings.get("goal", {}).get("mode", "auto")
    if goal_mode == "manual":
        goal_val = current_settings["goal"]["value"]
        goal_reason = "(manual)"
    else:
        goal_val = answers.get("goal", "produk")
        goal_reason = "pilihan utama alur kerja"
    result["goal"] = {"value": goal_val, "mode": goal_mode, "reason": goal_reason}

    # 2. profile
    prof_mode = current_settings.get("profile", {}).get("mode", "auto")
    if prof_mode == "manual":
        prof_val = current_settings["profile"]["value"]
        prof_reason = "(manual)"
    else:
        prof_val = answers.get("profile") or detect_system_profile(gpu_spec)
        prof_reason = f"deteksi perangkat: {gpu_name} (VRAM {vram_gb} GB)"
    result["profile"] = {"value": prof_val, "mode": prof_mode, "reason": prof_reason}

    # 3. languages
    lang_mode = current_settings.get("languages", {}).get("mode", "auto")
    if lang_mode == "manual":
        lang_val = current_settings["languages"]["value"]
        lang_reason = "(manual)"
    else:
        lang_val = answers.get("languages", "en+id")
        lang_reason = "dwibahasa Indonesia dan Inggris"
    result["languages"] = {"value": lang_val, "mode": lang_mode, "reason": lang_reason}

    # 4. vision
    vis_mode = current_settings.get("vision", {}).get("mode", "auto")
    if vis_mode == "manual":
        vis_val = current_settings["vision"]["value"]
        vis_reason = "(manual)"
    else:
        vis_val = False
        vis_reason = "ukuran/VRAM belum memadai; kerjakan setelah teks selesai"
    result["vision"] = {"value": vis_val, "mode": vis_mode, "reason": vis_reason}

    # 5. think & tool_call
    thk_mode = current_settings.get("think", {}).get("mode", "auto")
    if thk_mode == "manual":
        thk_val = current_settings["think"]["value"]
        thk_reason = "(manual)"
    else:
        thk_val = False if size_b < 0.1 else answers.get("features", {}).get("think", True)
        thk_reason = "ukuran < 0.1B penalaran dimatikan" if size_b < 0.1 else f"ukuran {size_b}B >= 0.1B"
    result["think"] = {"value": thk_val, "mode": thk_mode, "reason": thk_reason}

    tc_mode = current_settings.get("tool_call", {}).get("mode", "auto")
    if tc_mode == "manual":
        tc_val = current_settings["tool_call"]["value"]
        tc_reason = "(manual)"
    else:
        tc_val = answers.get("features", {}).get("tool_call", True)
        tc_reason = "dukungan pemanggilan tool eksternal aktif"
    result["tool_call"] = {"value": tc_val, "mode": tc_mode, "reason": tc_reason}

    # 6. think_max_tokens
    tmt_mode = current_settings.get("think_max_tokens", {}).get("mode", "auto")
    if tmt_mode == "manual":
        tmt_val = current_settings["think_max_tokens"]["value"]
        tmt_reason = "(manual)"
    else:
        if size_b <= 0.15:
            tmt_val = 512
            tmt_reason = "model <= 0.15B"
        elif size_b <= 0.8:
            tmt_val = 1024
            tmt_reason = "model 0.15B - 0.8B"
        else:
            tmt_val = 2048
            tmt_reason = "model > 0.8B"
    result["think_max_tokens"] = {"value": tmt_val, "mode": tmt_mode, "reason": tmt_reason}

    # 7. vocab_size
    voc_mode = current_settings.get("vocab_size", {}).get("mode", "auto")
    if voc_mode == "manual":
        voc_val = current_settings["vocab_size"]["value"]
        voc_reason = "(manual)"
    else:
        if size_b <= 0.2:
            voc_val = 32000
            voc_reason = "ukuran <= 0.2B"
        elif lang_val != "en+id" and size_b <= 0.6:
            voc_val = 32000
            voc_reason = f"bahasa tunggal ({lang_val}) dan ukuran <= 0.6B"
        else:
            voc_val = 64000
            voc_reason = "dwibahasa en+id, ukuran > 0.2B"
    result["vocab_size"] = {"value": voc_val, "mode": voc_mode, "reason": voc_reason}

    # 8. arch_style
    arch_mode = current_settings.get("arch_style", {}).get("mode", "auto")
    if arch_mode == "manual":
        arch_val = current_settings["arch_style"]["value"]
        arch_reason = "(manual)"
    else:
        arch_val = "preset"
        arch_reason = "arsitektur standar preset Qwen3"
    result["arch_style"] = {"value": arch_val, "mode": arch_mode, "reason": arch_reason}

    # 9. context_pretrain
    cp_mode = current_settings.get("context_pretrain", {}).get("mode", "auto")
    if cp_mode == "manual":
        cp_val = current_settings["context_pretrain"]["value"]
        cp_reason = "(manual)"
    else:
        if size_b <= 0.15:
            cp_val = 2048
            cp_reason = "N <= 0.15B"
        elif size_b > 0.15 and vram_gb >= 24.0:
            cp_val = 4096
            cp_reason = f"N={size_b}B dan VRAM {vram_gb:.0f} GB >= 24 GB"
        else:
            cp_val = 2048
            cp_reason = f"VRAM {vram_gb:.0f} GB < 24 GB"
    result["context_pretrain"] = {"value": cp_val, "mode": cp_mode, "reason": cp_reason}

    # 10. context_final
    cf_mode = current_settings.get("context_final", {}).get("mode", "auto")
    if cf_mode == "manual":
        cf_val = current_settings["context_final"]["value"]
        cf_reason = "(manual)"
    else:
        tokens_target_b = float(answers.get("tokens_target_b", 59.8) or 59.8)
        long_ctx_requested = bool(answers.get("long_context", False))

        if long_ctx_requested and tokens_target_b >= 20.0 and vram_gb >= 24.0:
            cf_val = 32768
            cf_reason = "long-context diminta, token >= 20B, dan VRAM >= 24 GB"
        elif tc_val:
            cf_val = 8192
            cf_reason = "tool call aktif: hasil search butuh konteks panjang"
        else:
            cf_val = 4096
            cf_reason = "standar post-training"
        if cf_val < cp_val:
            cf_val = cp_val
            cf_reason += f" (disesuaikan agar >= context_pretrain {cp_val})"
    result["context_final"] = {"value": cf_val, "mode": cf_mode, "reason": cf_reason}

    # 11. code_focus
    code_mode = current_settings.get("code_focus", {}).get("mode", "auto")
    if code_mode == "manual":
        code_val = current_settings["code_focus"]["value"]
        code_reason = "(manual)"
    else:
        code_val = "sedang"
        code_reason = "fokus kode proporsional seimbang"
    result["code_focus"] = {"value": code_val, "mode": code_mode, "reason": code_reason}

    # 12. precision
    prec_mode = current_settings.get("precision", {}).get("mode", "auto")
    if prec_mode == "manual":
        prec_val = current_settings["precision"]["value"]
        prec_reason = "(manual)"
    else:
        prec_val = gpu_spec.precision if gpu_spec else "bf16"
        prec_reason = f"dukungan hardware {gpu_name}"
    result["precision"] = {"value": prec_val, "mode": prec_mode, "reason": prec_reason}

    # 13. grad_checkpointing
    # Estimasi statis: params * 16 bytes
    mem_static_gb = (params_est * 16) / (1024 ** 3)
    gc_mode = current_settings.get("grad_checkpointing", {}).get("mode", "auto")
    if gc_mode == "manual":
        gc_val = current_settings["grad_checkpointing"]["value"]
        gc_reason = "(manual)"
    else:
        # Estimasi kasar tanpa grad-checkpoint
        # d ≈ 1024, L ≈ 24
        est_act_gb = (1 * cp_val * 1024 * 24 * 34) / (1024 ** 3)
        est_total = mem_static_gb + est_act_gb + 1.5
        if est_total > 0.85 * vram_gb:
            gc_val = True
            gc_reason = f"estimasi memori {est_total:.1f} GB > 85% VRAM ({0.85 * vram_gb:.1f} GB)"
        else:
            gc_val = False
            gc_reason = f"estimasi memori {est_total:.1f} GB aman dalam VRAM"
    result["grad_checkpointing"] = {"value": gc_val, "mode": gc_mode, "reason": gc_reason}

    # 14. micro_batch
    mb_mode = current_settings.get("micro_batch", {}).get("mode", "auto")
    if mb_mode == "manual":
        mb_val = current_settings["micro_batch"]["value"]
        mb_reason = "(manual)"
    else:
        if vram_gb >= 80:
            mb_val = 16
        elif vram_gb >= 40:
            mb_val = 8
        elif vram_gb >= 24:
            mb_val = 4
        elif vram_gb >= 16:
            mb_val = 2
        else:
            mb_val = 1
        mb_reason = f"estimasi awal untuk VRAM {vram_gb:.0f} GB (final via bench)"
    result["micro_batch"] = {"value": mb_val, "mode": mb_mode, "reason": mb_reason}

    # 15. ckpt_every
    ckpt_mode = current_settings.get("ckpt_every", {}).get("mode", "auto")
    if ckpt_mode == "manual":
        ckpt_val = current_settings["ckpt_every"]["value"]
        ckpt_reason = "(manual)"
    else:
        if prof_val == "laptop":
            ckpt_val = {"tokens": None, "menit": 15, "keep": 3}
            ckpt_reason = "profil laptop: tiap 15 menit"
        elif prof_val == "colab":
            ckpt_val = {"tokens": None, "menit": 10, "keep": 3}
            ckpt_reason = "profil colab: tiap 10 menit (cegah timeout)"
        else:
            ckpt_val = {"tokens": "auto", "menit": 30, "keep": 3}
            ckpt_reason = "profil cloud: berbasis token atau 30 menit"
    result["ckpt_every"] = {"value": ckpt_val, "mode": ckpt_mode, "reason": ckpt_reason}

    # 16. dedup & leak_filter
    result["dedup"] = {"value": True, "mode": current_settings.get("dedup", {}).get("mode", "auto"), "reason": "kualitas data latih"}
    result["leak_filter"] = {"value": True, "mode": current_settings.get("leak_filter", {}).get("mode", "auto"), "reason": "cegah kebocoran benchmark"}
    result["optimizer"] = {"value": "adamw", "mode": current_settings.get("optimizer", {}).get("mode", "auto"), "reason": "standar transformer decoder-only"}

    return result
