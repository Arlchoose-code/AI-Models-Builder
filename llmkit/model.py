"""Pembangunan model Qwen3ForCausalLM dari konfigurasi dan pemisahan kelompok parameter optimizer (Bagian 3 & 6.1)."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM


def build_qwen3_config(
    arch_plan: dict[str, Any],
    max_position_embeddings: int = 4096,
    rope_theta: float = 1000000.0,
    attn_implementation: str = "sdpa"
) -> Qwen3Config:
    """Bangun objek Qwen3Config dari spesifikasi rencana toolkit."""
    config = Qwen3Config(
        vocab_size=int(arch_plan.get("V", 64000)),
        hidden_size=int(arch_plan.get("d", 1280)),
        intermediate_size=int(arch_plan.get("ffn", 3456)),
        num_hidden_layers=int(arch_plan.get("L", 30)),
        num_attention_heads=int(arch_plan.get("Hq", 20)),
        num_key_value_heads=int(arch_plan.get("Hkv", 4)),
        head_dim=int(arch_plan.get("hd", 64)),
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        attention_bias=False,
        use_sliding_window=False,
        attn_implementation=attn_implementation,
    )
    return config


def create_model(
    config: Qwen3Config,
    seed: int = 1337,
    torch_dtype: torch.dtype = torch.float32,
    gradient_checkpointing: bool = False
) -> Qwen3ForCausalLM:
    """Inisialisasi model Qwen3ForCausalLM secara deterministik dengan bobot acak."""
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    if torch_dtype != torch.float32:
        model = model.to(torch_dtype)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Hitung parameter unik (tied embedding dihitung satu kali) dan non-embedding."""
    total_params = sum(p.numel() for p in model.parameters())
    # Cari embedding untuk menghitung non-embedding
    embed_params = 0
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            embed_params = sum(p.numel() for p in emb.parameters())
    non_embedding = total_params - embed_params
    return total_params, non_embedding


def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 5.4e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8
) -> torch.optim.AdamW:
    """
    Buat optimizer AdamW dengan decay hanya untuk tensor matriks 2D (selain embedding),
    tanpa decay (0.0) untuk embedding dan layer norm (Bagian 3).
    """
    decay_params = []
    nodecay_params = []

    # Nama modul embedding
    embed_names = ["embed_tokens", "lm_head"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = any(en in name for en in embed_names)
        if param.dim() >= 2 and not is_embedding:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps)
