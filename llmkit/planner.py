"""Planner: kalkulasi parameter, preset arsitektur, solver, compute, dan komposisi data."""

from __future__ import annotations

import math
from typing import Any, Optional
from llmkit.gpu_table import GPU_DATABASE, GPUSpec, get_gpu_spec
from llmkit.settings import compute_auto_settings


PRESETS = {
    0.05: {"d": 512, "L": 12, "Hq": 8, "Hkv": 2, "hd": 64, "ffn": 1408, "V": 32000, "total": 50_214_912},
    0.1:  {"d": 640, "L": 18, "Hq": 10, "Hkv": 2, "hd": 64, "ffn": 1728, "V": 32000, "total": 97_920_384},
    0.3:  {"d": 1024, "L": 20, "Hq": 16, "Hkv": 4, "hd": 64, "ffn": 2816, "V": 64000, "total": 291_024_384},
    0.6:  {"d": 1280, "L": 30, "Hq": 20, "Hkv": 4, "hd": 64, "ffn": 3456, "V": 64000, "total": 598_097_920},
    1.0:  {"d": 1792, "L": 28, "Hq": 28, "Hkv": 4, "hd": 64, "ffn": 4864, "V": 64000, "total": 1_052_482_816},
}


def calculate_params(d: int, L: int, Hq: int, Hkv: int, hd: int, ffn: int, V: int, tied: bool = True) -> dict[str, int]:
    """Hitung parameter Qwen3 berdasarkan rumus spesifikasi Bagian 6.1."""
    attn_per_layer = d * (Hq * hd) + d * (Hkv * hd) + d * (Hkv * hd) + (Hq * hd) * d
    ffn_per_layer = 3 * d * ffn
    norm_per_layer = 2 * d + 2 * hd
    per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
    total = L * per_layer + V * d + d
    total_untied = total + V * d
    non_embedding = L * per_layer + d
    return {
        "attn_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "norm_per_layer": norm_per_layer,
        "per_layer": per_layer,
        "total": total,
        "total_untied": total_untied,
        "non_embedding": non_embedding,
    }


def solve_architecture(
    target_params: int,
    vocab_size: int,
    arch_style: str = "preset"
) -> dict[str, Any]:
    """
    Cari arsitektur optimal untuk ukuran parameter di luar preset (Bagian 6.2).
    Mengembalikan dict config dan kandidat teratas.
    """
    V = vocab_size
    candidates: list[dict[str, Any]] = []
    target_ratio = 36.0 if arch_style == "dalam_ramping" else 44.0

    for d in range(256, 4097, 64):
        hd = 64
        Hq = d // 64
        # Divisor Hq yang paling dekat ke Hq / 5, minimal 1
        divisors = [i for i in range(1, Hq + 1) if Hq % i == 0]
        ideal_hkv = max(1.0, Hq / 5.0)
        Hkv = min(divisors, key=lambda x: abs(x - ideal_hkv))
        ffn = int(round(2.7 * d / 64.0) * 64)

        for L in range(8, 49):
            p = calculate_params(d, L, Hq, Hkv, hd, ffn, V)
            total = p["total"]
            rel_error = abs(total - target_params) / target_params
            candidates.append({
                "d": d, "L": L, "Hq": Hq, "Hkv": Hkv, "hd": hd,
                "ffn": ffn, "V": V, "total": total,
                "rel_error": rel_error, "dl_ratio": d / L
            })

    # Urutkan berdasarkan rel_error terkecil, lalu selisih d/L dengan target_ratio
    candidates.sort(key=lambda c: (c["rel_error"], abs(c["dl_ratio"] - target_ratio)))
    best = candidates[0]
    return {
        "best": best,
        "top3": candidates[:3],
        "is_within_3pct": best["rel_error"] <= 0.03
    }


def get_architecture_config(
    size_b: float,
    vocab_size: int = 64000,
    arch_style: str = "preset"
) -> dict[str, Any]:
    """Dapatkan konfigurasi arsitektur dari preset atau solver."""
    target_params = int(round(size_b * 1e9))

    # Cek kecocokan preset (+- 5%) jika arch_style == 'preset'
    if arch_style == "preset":
        for p_size, p_conf in PRESETS.items():
            if abs(size_b - p_size) / p_size <= 0.05:
                # Sesuaikan vocab jika diminta berbeda
                conf = dict(p_conf)
                if conf["V"] != vocab_size:
                    conf["V"] = vocab_size
                    recalc = calculate_params(conf["d"], conf["L"], conf["Hq"], conf["Hkv"], conf["hd"], conf["ffn"], vocab_size)
                    conf["total"] = recalc["total"]
                return {
                    "config": conf,
                    "is_preset": True,
                    "preset_size": p_size,
                    "error": abs(conf["total"] - target_params) / target_params
                }

    # Gunakan solver
    solved = solve_architecture(target_params, vocab_size, arch_style)
    best = solved["best"]
    return {
        "config": best,
        "is_preset": False,
        "preset_size": None,
        "error": best["rel_error"],
        "top3": solved["top3"],
        "is_within_3pct": solved["is_within_3pct"]
    }


def calculate_tokens(
    total_params: int,
    mode: str = "balanced",
    tokens_override: Optional[float] = None,
    budget_usd: Optional[float] = None,
    hours: Optional[float] = None,
    gpu_spec: Optional[GPUSpec] = None
) -> tuple[float, str, float]:
    """
    Hitung jumlah token, mengembalikan (tokens, label_mode, ratio).
    Tokens dibulatkan ke 0.1B terdekat.
    """
    if tokens_override is not None and tokens_override > 0:
        tokens = tokens_override
        ratio = tokens / total_params
        return tokens, "custom", ratio

    if budget_usd is not None and budget_usd > 0 and gpu_spec is not None:
        if gpu_spec.price_per_hour > 0:
            gpu_hours = budget_usd / gpu_spec.price_per_hour
        else:
            gpu_hours = budget_usd  # anggap jam jika GPU gratis
        tflops = gpu_spec.effective_tflops
        raw_tokens = (gpu_hours * 3600.0 * tflops * 1e12) / (6.0 * total_params)
        tokens = round(raw_tokens / 1e8) * 1e8  # round to 0.1B
        ratio = tokens / total_params
        label = f"budget_${budget_usd:.0f}"
        return tokens, label, ratio

    if hours is not None and hours > 0 and gpu_spec is not None:
        tflops = gpu_spec.effective_tflops
        raw_tokens = (hours * 3600.0 * tflops * 1e12) / (6.0 * total_params)
        tokens = round(raw_tokens / 1e8) * 1e8
        ratio = tokens / total_params
        label = f"hours_{hours:.0f}h"
        return tokens, label, ratio

    ratios = {
        "chinchilla": 20.0,
        "balanced": 100.0,
        "overtrain": 500.0,
        "max": 3000.0,
    }
    ratio = ratios.get(mode, 100.0)
    raw_tokens = ratio * total_params
    tokens = round(raw_tokens / 1e8) * 1e8
    return tokens, mode, ratio


def compute_flops_and_cost(
    params: int,
    tokens: float,
    gpu_spec: Optional[GPUSpec] = None
) -> dict[str, Any]:
    """Hitung FLOPs, durasi dalam jam, dan biaya dolar."""
    flops = 6.0 * float(params) * float(tokens)
    if gpu_spec is None:
        return {"flops": flops, "seconds": 0.0, "hours": 0.0, "cost": 0.0}

    seconds = flops / (gpu_spec.effective_tflops * 1e12)
    hours = seconds / 3600.0
    cost = hours * gpu_spec.price_per_hour
    return {
        "flops": flops,
        "seconds": seconds,
        "hours": hours,
        "cost": cost,
    }


def compute_gpu_comparison(params: int, tokens: float) -> list[dict[str, Any]]:
    """Bandingkan waktu dan estimasi biaya di semua GPU terdaftar."""
    comparison = []
    for spec in GPU_DATABASE:
        res = compute_flops_and_cost(params, tokens, spec)
        comparison.append({
            "gpu": spec.name,
            "vram_gb": spec.vram_gb,
            "hours": res["hours"],
            "days": res["hours"] / 24.0,
            "cost": res["cost"],
            "precision": spec.precision,
        })
    return comparison


def get_batch_tokens(total_tokens: float) -> int:
    """Tentukan batch_tokens per optimizer step (Bagian 6.5)."""
    if total_tokens < 5e9:
        return 262_144
    if total_tokens <= 30e9:
        return 524_288
    if total_tokens <= 150e9:
        return 1_048_576
    if total_tokens <= 600e9:
        return 2_097_152
    return 4_194_304


def compute_memory_estimate(
    params: int,
    micro_batch: int,
    seq_len: int,
    d: int,
    L: int,
    V: int,
    grad_checkpointing: bool = False
) -> dict[str, float]:
    """Hitung estimasi kebutuhan memori VRAM (Bagian 6.7)."""
    mem_static = float(params) * 16.0
    c = 4.0 if grad_checkpointing else 34.0
    mem_act = float(micro_batch) * float(seq_len) * float(d) * float(L) * c
    mem_logits = float(micro_batch) * float(seq_len) * float(V) * 4.0 * 2.0
    overhead = 1.5 * (1024 ** 3)
    total_bytes = mem_static + mem_act + mem_logits + overhead

    return {
        "mem_static_gb": mem_static / (1024 ** 3),
        "mem_act_gb": mem_act / (1024 ** 3),
        "mem_logits_gb": mem_logits / (1024 ** 3),
        "overhead_gb": overhead / (1024 ** 3),
        "total_gb": total_bytes / (1024 ** 3),
    }


def compute_pretrain_composition(
    total_tokens: float,
    context_pretrain: int,
    context_final: int,
    languages: str = "en+id",
    code_focus: str = "sedang",
    think: bool = True
) -> dict[str, Any]:
    """
    Hitung komposisi data pretrain per tahap (Stage A, Stage B, Stage C)
    dan token yang dibutuhkan per folder. Sesuai Bagian 6.8 & 18.2.
    """
    stage_c_active = context_final > context_pretrain
    if stage_c_active:
        pct_A, pct_B, pct_C = 75.0, 20.0, 5.0
    else:
        pct_A, pct_B, pct_C = 80.0, 20.0, 0.0

    tokens_A = total_tokens * (pct_A / 100.0)
    tokens_B = total_tokens * (pct_B / 100.0)
    tokens_C = total_tokens * (pct_C / 100.0)

    # Base weights Stage A
    w_A: dict[str, float] = {
        "en_web": 36.0,
        "code": 25.0,
        "id_web": 12.0,
        "math": 8.0,
        "books_wiki": 8.0,
        "synthetic": 9.0,
        "dialog": 2.0,
    }
    # Base weights Stage B
    w_B: dict[str, float] = {
        "code": 30.0,
        "math": 25.0,
        "reasoning": 15.0,
        "en_hq": 12.0,
        "id_hq": 8.0,
        "instruct_text": 10.0,
    }
    # Base weights Stage C
    if context_final >= 32768:
        w_C: dict[str, float] = {
            "long_docs": 50.0,
            "long_code": 25.0,
            "en_web": 15.0,
            "id_web": 10.0,
        }
    else:
        # Jika context_final < 32768, pakai bobot yang sama dengan Stage A
        w_C = dict(w_A)

    # Penyesuaian think=false: reasoning dipindah ke math
    if not think and "reasoning" in w_B:
        r_val = w_B.pop("reasoning")
        w_B["math"] = w_B.get("math", 0.0) + r_val

    # Penyesuaian code_focus
    if code_focus == "ringan":
        delta = 15.0  # 25 -> 10
        w_A["code"] = 10.0
        others = [k for k in w_A if k != "code"]
        sum_others = sum(w_A[k] for k in others)
        for k in others:
            w_A[k] += delta * (w_A[k] / sum_others)
    elif code_focus == "berat":
        delta = 10.0  # 25 -> 35
        w_A["code"] = 35.0
        others = [k for k in w_A if k != "code"]
        sum_others = sum(w_A[k] for k in others)
        for k in others:
            w_A[k] -= delta * (w_A[k] / sum_others)

    # Penyesuaian languages
    def filter_and_normalize(w_dict: dict[str, float]) -> dict[str, float]:
        if languages == "id":
            cleaned = {k: v for k, v in w_dict.items() if not k.startswith("en_")}
        elif languages == "en":
            cleaned = {k: v for k, v in w_dict.items() if not k.startswith("id_")}
        else:
            cleaned = dict(w_dict)
        total_w = sum(cleaned.values())
        if total_w > 0:
            return {k: round(v * 100.0 / total_w, 4) for k, v in cleaned.items()}
        return cleaned

    w_A = filter_and_normalize(w_A)
    w_B = filter_and_normalize(w_B)
    w_C = filter_and_normalize(w_C) if stage_c_active else {}

    # Pastikan jumlah tiap stage genap 100.0
    def adjust_to_100(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return d
        diff = 100.0 - sum(d.values())
        first_key = next(iter(d))
        d[first_key] = round(d[first_key] + diff, 4)
        return d

    w_A = adjust_to_100(w_A)
    w_B = adjust_to_100(w_B)
    if stage_c_active:
        w_C = adjust_to_100(w_C)

    # Hitung token yang dibutuhkan per folder
    all_folders = sorted(list(set(list(w_A.keys()) + list(w_B.keys()) + list(w_C.keys()))))
    needed: dict[str, float] = {}
    for f in all_folders:
        tok = (
            tokens_A * (w_A.get(f, 0.0) / 100.0)
            + tokens_B * (w_B.get(f, 0.0) / 100.0)
            + tokens_C * (w_C.get(f, 0.0) / 100.0)
        )
        needed[f] = tok

    # Total porsi bahasa Indonesia dan kode
    id_tokens = sum(tok for f, tok in needed.items() if f.startswith("id_"))
    code_tokens = sum(tok for f, tok in needed.items() if "code" in f)
    pct_id = (id_tokens / total_tokens) * 100.0 if total_tokens > 0 else 0.0
    pct_code = (code_tokens / total_tokens) * 100.0 if total_tokens > 0 else 0.0

    return {
        "stage_c_active": stage_c_active,
        "stages": {
            "A": {"pct": pct_A, "tokens": tokens_A, "weights": w_A},
            "B": {"pct": pct_B, "tokens": tokens_B, "weights": w_B},
            "C": {"pct": pct_C, "tokens": tokens_C, "weights": w_C} if stage_c_active else None,
        },
        "needed": needed,
        "pct_id": pct_id,
        "pct_code": pct_code,
    }


def compute_sft_composition(
    size_b: float,
    think: bool = True,
    tool_call: bool = True,
    code_focus: str = "sedang"
) -> dict[str, Any]:
    """Hitung komposisi data SFT (Bagian 6.9 & 18.2)."""
    # Total sampel SFT: n_sft = clamp(round(400_000 * N_B), 20_000, 1_000_000) round to 1k
    raw_n = min(1_000_000, max(20_000, int(round(400_000 * size_b))))
    n_sft = int(round(raw_n / 1000.0) * 1000)

    # Base weights
    w = {
        "chat": 30.0,
        "think": 20.0,
        "tool_call": 15.0,
        "multiturn": 15.0,
        "code": 10.0,
        "format": 5.0,
        "refusal": 3.0,
        "identity": 2.0,
    }

    # Penyesuaian think=false
    if not think:
        w["think"] = 0.0
        w["chat"] += 10.0
        w["code"] += 5.0
        w["multiturn"] += 5.0

    # Penyesuaian tool_call=false
    if not tool_call:
        w["tool_call"] = 0.0
        w["chat"] += 8.0
        w["format"] += 4.0
        w["multiturn"] += 3.0

    # Penyesuaian code_focus
    if code_focus == "ringan":
        w["code"] = 5.0
        w["chat"] += 5.0
    elif code_focus == "berat":
        w["code"] = 20.0
        w["chat"] -= 10.0

    # Pastikan total 100.0
    diff = 100.0 - sum(w.values())
    w["chat"] = round(w["chat"] + diff, 2)

    categories: dict[str, dict[str, Any]] = {}
    for cat, pct in w.items():
        if pct > 0:
            samples = int(round(n_sft * (pct / 100.0)))
            categories[cat] = {"pct": pct, "samples": samples}

    # Hyperparameter SFT
    if size_b <= 0.15:
        lr = 1e-4
    elif size_b <= 0.8:
        lr = 5e-5
    else:
        lr = 3e-5
    epochs = 2 if n_sft > 500_000 else 3

    return {
        "n_sft": n_sft,
        "categories": categories,
        "lr": lr,
        "epochs": epochs,
        "batch_size": 128,
        "warmup": 0.03,
        "weight_decay": 0.0,
    }


def make_plan(answers: dict[str, Any], current_settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Fungsi perencanaan menyeluruh (Bagian 6 & 18).
    Menghitung arsitektur, parameter, token, compute, memory, komposisi pretrain & SFT.
    """
    settings = compute_auto_settings(answers, current_settings)
    size_b = float(answers.get("size_b", 0.6))
    vocab_size = int(settings["vocab_size"]["value"])
    arch_style = str(settings["arch_style"]["value"])

    # Arsitektur & Parameter
    arch_res = get_architecture_config(size_b, vocab_size, arch_style)
    cfg = arch_res["config"]
    p_info = calculate_params(cfg["d"], cfg["L"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["ffn"], cfg["V"])
    total_params = p_info["total"]
    non_embedding = p_info["non_embedding"]

    # Token
    token_mode = str(answers.get("token_mode", "balanced"))
    tokens_override = float(answers["tokens_override"]) if answers.get("tokens_override") else None
    budget_usd = float(answers["budget_usd"]) if answers.get("budget_usd") else None
    hours = float(answers["hours"]) if answers.get("hours") else None

    gpu_name = str(answers.get("gpu", "H100"))
    gpu_spec = get_gpu_spec(gpu_name)
    tokens, mode_label, ratio = calculate_tokens(
        total_params, token_mode, tokens_override, budget_usd, hours, gpu_spec
    )

    # Compute & Biaya
    compute_res = compute_flops_and_cost(total_params, tokens, gpu_spec)
    gpu_comparisons = compute_gpu_comparison(total_params, tokens)

    # Hyperparameter Pretrain
    context_pretrain = int(settings["context_pretrain"]["value"])
    context_final = int(settings["context_final"]["value"])
    batch_tokens = get_batch_tokens(tokens)
    total_steps = math.ceil(tokens / batch_tokens)

    micro_batch = int(settings["micro_batch"]["value"])
    grad_accum = max(1, round(batch_tokens / (micro_batch * context_pretrain)))

    N_B = total_params / 1e9
    peak_lr = min(2e-3, max(2e-4, 8e-4 * ((0.125 / N_B) ** 0.25)))
    warmup_steps = min(2000, max(100, int(round(0.01 * total_steps))))
    eval_every_tokens = min(2_000_000_000, max(50_000_000, int(round(tokens / 50.0))))
    ckpt_every_tokens = min(1_000_000_000, max(100_000_000, int(round(tokens / 100.0))))

    # Memory
    grad_checkpointing = bool(settings["grad_checkpointing"]["value"])
    mem_est = compute_memory_estimate(
        total_params, micro_batch, context_pretrain, cfg["d"], cfg["L"], cfg["V"], grad_checkpointing
    )

    # Data Composition
    languages = str(settings["languages"]["value"])
    code_focus = str(settings["code_focus"]["value"])
    think = bool(settings["think"]["value"])
    tool_call = bool(settings["tool_call"]["value"])

    pretrain_comp = compute_pretrain_composition(
        tokens, context_pretrain, context_final, languages, code_focus, think
    )
    sft_comp = compute_sft_composition(N_B, think, tool_call, code_focus)

    # Storage
    tokenized_gb = (tokens * 2.0) / 1e9
    raw_est_gb = tokenized_gb * 2.0
    suggested_workspace_gb = 3.0 * raw_est_gb + tokenized_gb

    plan = {
        "architecture": {
            "d": cfg["d"],
            "L": cfg["L"],
            "Hq": cfg["Hq"],
            "Hkv": cfg["Hkv"],
            "hd": cfg["hd"],
            "ffn": cfg["ffn"],
            "V": cfg["V"],
            "total_params": total_params,
            "non_embedding": non_embedding,
            "is_preset": arch_res["is_preset"],
            "preset_size": arch_res.get("preset_size"),
            "error": arch_res["error"],
        },
        "tokens": {
            "target": tokens,
            "mode": mode_label,
            "ratio": ratio,
        },
        "compute": {
            "flops": compute_res["flops"],
            "hours": compute_res["hours"],
            "cost": compute_res["cost"],
            "gpu": gpu_name,
            "gpu_spec": gpu_spec.__dict__ if gpu_spec else None,
            "comparisons": gpu_comparisons,
        },
        "hyperparameters": {
            "context_pretrain": context_pretrain,
            "context_final": context_final,
            "batch_tokens": batch_tokens,
            "total_steps": total_steps,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "peak_lr": peak_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "eval_every_tokens": eval_every_tokens,
            "ckpt_every_tokens": ckpt_every_tokens,
        },
        "memory": mem_est,
        "pretrain_data": pretrain_comp,
        "sft_data": sft_comp,
        "storage": {
            "bytes_per_token": 2,
            "tokenized_gb": tokenized_gb,
            "suggested_workspace_gb": suggested_workspace_gb,
            "ckpt_weight_gb": (total_params * 2.0) / (1024 ** 3),
            "ckpt_full_gb": (total_params * 16.0) / (1024 ** 3),
        },
        "settings": settings,
    }

    return plan
