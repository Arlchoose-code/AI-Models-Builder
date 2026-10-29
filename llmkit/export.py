"""Export model ke format folder Hugging Face standar dan petunjuk konversi GGUF / Ollama (Bagian 12.6)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM
from llmkit.chat_template import (
    EOS_TOKEN,
    PAD_TOKEN,
    PRETRAIN_EOS_TOKEN,
    SPECIAL_TOKENS,
    get_default_system,
    get_jinja_template,
)
from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint
from llmkit.i18n import tr
from llmkit.model import build_qwen3_config, create_model
from llmkit.state import load_project, log_history
from llmkit.ui import print_info, print_ok, print_warn


def export_hf_model(
    project_dir: Path,
    run_id: Optional[str] = None,
    out_name: Optional[str] = None,
    device: str = "cpu"
) -> Path:
    """
    Ekspor checkpoint model dan tokenizer ke folder Hugging Face lengkap.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    answers = project_data.get("answers", {})
    model_name = answers.get("model_name", "Bahasa")
    creator = answers.get("creator", "Pengguna")

    export_name = out_name or project_data.get("name", "model-export")
    export_dir = project_dir / "export" / export_name
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cari Checkpoint Sumber
    runs_dir = project_dir / "runs"
    if run_id:
        target_run = runs_dir / run_id
    else:
        existing_runs = sorted(list(runs_dir.glob("run_*")), key=lambda p: p.name)
        if not existing_runs:
            raise FileNotFoundError("Belum ada run training untuk diekspor.")
        target_run = existing_runs[-1]

    ckpt_dir = target_run / "ckpt"
    source_ckpt = ckpt_dir / "final"
    if not source_ckpt.exists():
        source_ckpt = find_latest_checkpoint(ckpt_dir)

    if not source_ckpt or not source_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di {ckpt_dir}.")

    print_info(tr(f"Mengekspor bobot dari: {source_ckpt} ke {export_dir}", f"Exporting weights from: {source_ckpt} to {export_dir}"))

    # 2. Muat dan Simpan Bobot Model
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    export_dtype = torch.bfloat16 if use_bf16 else torch.float32

    cfg = build_qwen3_config(arch, max_position_embeddings=4096)
    model = create_model(cfg, torch_dtype=export_dtype).to(device)

    if (source_ckpt / "model.safetensors").exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(source_ckpt / "model.safetensors"), device=device), strict=False)
        model.tie_weights()
    elif (source_ckpt / "model.pt").exists() and not (source_ckpt / "train_state.json").exists():
        state_d = torch.load(source_ckpt / "model.pt", map_location=device, weights_only=False)
        model.load_state_dict(state_d, strict=False)
        model.tie_weights()
    else:
        load_checkpoint(source_ckpt, model, device=device)

    # 3. Simpan Model, Config & Generation Config via save_pretrained
    model.save_pretrained(str(export_dir))

    gen_cfg = GenerationConfig(
        bos_token_id=None,
        eos_token_id=cfg.eos_token_id or 2,
        pad_token_id=11,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.05
    )
    gen_cfg.save_pretrained(str(export_dir))

    # 4. Tokenizer & Chat Template Jinja
    tok_src = project_dir / "tokenizer" / "tokenizer.json"
    if tok_src.exists():
        shutil.copy(tok_src, export_dir / "tokenizer.json")

    jinja_template = get_jinja_template(get_default_system(model_name, creator))
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "chat_template": jinja_template,
        "clean_up_tokenization_spaces": False,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(export_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    special_tokens_map = {
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": SPECIAL_TOKENS,
    }
    with open(export_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    print_ok(tr(f"Export format Hugging Face berhasil di: {export_dir}", f"Hugging Face format export successful at: {export_dir}"))

    # 5. Petunjuk Konversi GGUF & Ollama (Bagian 12.6)
    print("\n" + "=" * 60)
    print(tr("PETUNJUK KONVERSI KE GGUF / OLLAMA:", "CONVERSION INSTRUCTIONS TO GGUF / OLLAMA:"))
    print(tr(
        "Catatan: Periksa dokumentasi llama.cpp/Ollama terbaru karena argumen perintah dapat diperbarui.",
        "Note: Check latest llama.cpp/Ollama documentation as command syntax may evolve."
    ))
    print("\n" + tr("[Jalur A - Disarankan (llama.cpp)]", "[Route A - Recommended (llama.cpp)]"))
    print(tr("  1. Konversi ke GGUF:", "  1. Convert to GGUF:"))
    print(f"     python <llama.cpp>/convert_hf_to_gguf.py export/{export_name} --outfile {export_name}.gguf --outtype bf16")
    print(tr("  2. Kuantisasi (opsional, misal Q8_0):", "  2. Quantize (optional, e.g. Q8_0):"))
    print(f"     llama-quantize {export_name}.gguf {export_name}-q8_0.gguf Q8_0")
    print("\n" + tr("[Ollama Modelfile]", "[Ollama Modelfile]"))
    print(tr("  Buat file bernama 'Modelfile':", "  Create a file named 'Modelfile':"))
    print(f"     FROM ./{export_name}-q8_0.gguf")
    print(f'     PARAMETER stop "<|im_end|>"')
    print(tr("  Lalu jalankan:", "  Then run:"))
    print(f"     ollama create {export_name} -f Modelfile")
    print("\n" + tr("[PERINGATAN WAJIB]", "[MANDATORY WARNING]"))
    print(tr(
        "Penyebab utama hasil generasi rusak atau ngelantur setelah dikonversi adalah\nchat template atau token EOS yang tidak sama dengan saat training.",
        "The primary reason converted generations degrade or hallucinate is\nchat template or EOS token mismatches with training time settings."
    ))
    print("=" * 60 + "\n")

    log_history(project_dir, "export", f"--run {target_run.name}", True, f"Berhasil diexport ke {export_dir}")
    return export_dir
