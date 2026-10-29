"""Validator data pretrain dan SFT (Bagian 9)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from llmkit.data_spec import PRETRAIN_FOLDERS, SFT_CATEGORIES, extract_think


BYTES_PER_TOKEN_ESTIMATES = {
    "en_web": 4.3,
    "en_hq": 4.3,
    "id_web": 3.9,
    "id_hq": 3.9,
    "code": 3.6,
    "long_code": 3.6,
    "math": 3.5,
    "books_wiki": 4.0,
    "synthetic": 4.0,
    "dialog": 4.0,
    "reasoning": 3.8,
    "instruct_text": 4.0,
    "long_docs": 4.0,
}


def estimate_tokens_for_file(
    file_path: Path,
    folder_name: str,
    tokenizer: Optional[Any] = None,
    cache_dict: Optional[dict[str, Any]] = None
) -> int:
    """Estimasi jumlah token untuk file berdasarkan sampling atau rasio bytes_per_token."""
    try:
        stat = file_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        return 0

    cache_key = f"{file_path.resolve()}_{file_size}_{mtime}"
    if cache_dict is not None and cache_key in cache_dict:
        return cache_dict[cache_key]

    if tokenizer is not None and file_size > 0:
        # Sampling 1MB dari awal, tengah, dan akhir
        sample_texts = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Awal
                sample_texts.append(f.read(300_000))
                # Tengah
                if file_size > 1_000_000:
                    f.seek(file_size // 2)
                    sample_texts.append(f.read(400_000))
                # Akhir
                if file_size > 2_000_000:
                    f.seek(max(0, file_size - 300_000))
                    sample_texts.append(f.read(300_000))
        except Exception:
            pass

        full_sample = "".join(sample_texts)
        if len(full_sample) > 1000:
            sample_bytes = len(full_sample.encode("utf-8"))
            tok_count = len(tokenizer.encode(full_sample).ids)
            bpt = sample_bytes / max(1, tok_count)
            estimated = int(file_size / max(1.0, bpt))
            if cache_dict is not None:
                cache_dict[cache_key] = estimated
            return estimated

    # Fallback tanpa tokenizer
    bpt = BYTES_PER_TOKEN_ESTIMATES.get(folder_name, 4.0)
    estimated = int(file_size / bpt)
    if cache_dict is not None:
        cache_dict[cache_key] = estimated
    return estimated


def validate_tool_call(call: dict[str, Any], tools_schema: list[dict[str, Any]]) -> tuple[bool, Optional[str]]:
    """Validasi apakah format panggilan tool cocok dengan skema fungsi."""
    fn_name = call.get("name")
    args = call.get("arguments")
    if not fn_name or not isinstance(fn_name, str):
        return False, "Nama fungsi tidak ada atau bukan string"
    if not isinstance(args, dict):
        return False, f"Argumen tool '{fn_name}' harus berupa JSON object/dict"

    # Cari skema fungsi
    matched_schema = None
    for t in tools_schema:
        fn = t.get("function", {})
        if fn.get("name") == fn_name:
            matched_schema = fn
            break

    if not matched_schema:
        return False, f"Fungsi '{fn_name}' tidak terdaftar dalam daftar tools"

    params_schema = matched_schema.get("parameters", {})
    required = params_schema.get("required", [])
    for r in required:
        if r not in args:
            return False, f"Parameter wajib '{r}' tidak ada pada panggilan '{fn_name}'"

    # Validasi tipe dasar
    props = params_schema.get("properties", {})
    for k, v in args.items():
        if k in props:
            expected_type = props[k].get("type")
            if expected_type == "string" and not isinstance(v, str):
                return False, f"Tipe '{k}' harus string"
            elif expected_type == "integer" and not isinstance(v, int):
                return False, f"Tipe '{k}' harus integer"
            elif expected_type == "number" and not isinstance(v, (int, float)):
                return False, f"Tipe '{k}' harus number"
            elif expected_type == "boolean" and not isinstance(v, bool):
                return False, f"Tipe '{k}' harus boolean"
            elif expected_type == "array" and not isinstance(v, list):
                return False, f"Tipe '{k}' harus array/list"
            elif expected_type == "object" and not isinstance(v, dict):
                return False, f"Tipe '{k}' harus object/dict"

    return True, None


def check_dataset(
    project_dir: Path,
    quick: bool = False,
    fix_hints: bool = False
) -> tuple[bool, dict[str, Any]]:
    """
    Validasi seluruh data pretrain dan SFT sesuai Bagian 9.
    Menghasilkan ringkasan dan menulis file logs/check_errors.txt jika ada error.
    """
    proj_file = project_dir / "project.json"
    if not proj_file.exists():
        return False, {"error": "project.json tidak ditemukan."}

    with open(proj_file, "r", encoding="utf-8") as f:
        project_data = json.load(f)

    plan = project_data.get("plan", {})
    settings = plan.get("settings", {})
    think_max_tokens = int(settings.get("think_max_tokens", {}).get("value", 1024))
    pretrain_needed = plan.get("pretrain_data", {}).get("needed", {})
    sft_plan = plan.get("sft_data", {})
    sft_cats = sft_plan.get("categories", {})

    data_dir = project_dir / "data"
    pretrain_dir = data_dir / "pretrain"
    sft_dir = data_dir / "sft"

    errors: list[str] = []
    warnings: list[str] = []

    # Muat tokenizer jika ada
    tokenizer = None
    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if tok_path.exists():
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tok_path))
        except Exception:
            pass

    # Cache index
    cache_file = project_dir / ".cache" / "data_index.json"
    cache_dict = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
        except Exception:
            pass

    pretrain_summary: dict[str, Any] = {}
    total_pretrain_est_tokens = 0

    # 1. Cek Pretrain
    for folder_name in PRETRAIN_FOLDERS:
        f_dir = pretrain_dir / folder_name
        files = list(f_dir.glob("*.jsonl")) if f_dir.exists() else []
        target_tokens = pretrain_needed.get(folder_name, 0.0)

        total_size = sum(p.stat().st_size for p in files if p.exists())
        folder_est_tokens = sum(
            estimate_tokens_for_file(p, folder_name, tokenizer, cache_dict)
            for p in files
        )
        total_pretrain_est_tokens += folder_est_tokens

        doc_count = 0
        folder_errors = []
        dup_hashes = set()
        dup_count = 0

        if not quick and files:
            for p in files:
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            doc_count += 1
                            try:
                                obj = json.loads(line)
                            except Exception as e:
                                folder_errors.append(f"{p.name} baris {line_num}: JSON rusak ({e})")
                                continue

                            text = obj.get("text")
                            if text is None or not isinstance(text, str) or not text.strip():
                                folder_errors.append(f"{p.name} baris {line_num}: field 'text' kosong/tidak valid")
                            elif len(text.strip()) < 20:
                                warnings.append(f"{folder_name}/{p.name} baris {line_num}: dokumen sangat pendek (<20 karakter)")

                            # Duplikat sampel hingga 2000
                            if doc_count <= 2000 and text:
                                norm = re.sub(r"\s+", " ", text.lower().strip())
                                h = hashlib.md5(norm.encode("utf-8")).hexdigest()
                                if h in dup_hashes:
                                    dup_count += 1
                                else:
                                    dup_hashes.add(h)
                except Exception as e:
                    folder_errors.append(f"Gagal membaca file {p.name}: {e}")

        for fe in folder_errors[:5]:
            errors.append(f"PRETRAIN [{folder_name}] {fe}")

        dup_pct = (dup_count / max(1, min(doc_count, 2000))) * 100.0
        if dup_pct > 1.0:
            warnings.append(f"{folder_name}: {dup_pct:.1f}% dokumen duplikat dari sampel {min(doc_count, 2000)}")

        pretrain_summary[folder_name] = {
            "files": len(files),
            "doc_count": doc_count,
            "size": total_size,
            "est_tokens": folder_est_tokens,
            "target_tokens": target_tokens,
            "status": "OK" if (target_tokens == 0 or folder_est_tokens >= 0.9 * target_tokens) else "KURANG"
        }

    # 2. Cek SFT
    sft_summary: dict[str, Any] = {}
    all_sft_ids = set()

    for cat_name in SFT_CATEGORIES:
        c_dir = sft_dir / cat_name
        files = list(c_dir.glob("*.jsonl")) if c_dir.exists() else []
        cat_info = sft_cats.get(cat_name, {})
        target_samples = cat_info.get("samples", 0)

        sample_count = 0
        en_count = 0
        id_count = 0
        think_mode_count = 0
        no_think_mode_count = 0
        invalid_tools_count = 0
        cat_errors = []

        for p in files:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        sample_count += 1
                        try:
                            sample = json.loads(line)
                        except Exception as e:
                            cat_errors.append(f"{p.name} baris {line_num}: JSON rusak ({e})")
                            continue

                        s_id = sample.get("id")
                        if not s_id:
                            cat_errors.append(f"{p.name} baris {line_num}: field 'id' tidak ada")
                        elif s_id in all_sft_ids:
                            cat_errors.append(f"{p.name} baris {line_num}: id '{s_id}' duplikat")
                        else:
                            all_sft_ids.add(s_id)

                        lang = sample.get("lang")
                        if lang == "en":
                            en_count += 1
                        elif lang == "id":
                            id_count += 1
                        else:
                            cat_errors.append(f"{p.name} baris {line_num}: 'lang' harus 'en' atau 'id'")

                        mode = sample.get("mode")
                        if mode == "think":
                            think_mode_count += 1
                        elif mode == "no_think":
                            no_think_mode_count += 1
                        else:
                            cat_errors.append(f"{p.name} baris {line_num}: 'mode' harus 'think' atau 'no_think'")

                        messages = sample.get("messages", [])
                        if not isinstance(messages, list) or not messages:
                            cat_errors.append(f"{p.name} baris {line_num}: 'messages' kosong")
                            continue

                        # Periksa pesan terakhir harus assistant
                        if messages[-1].get("role") != "assistant":
                            cat_errors.append(f"{p.name} baris {line_num}: percakapan harus diakhiri role 'assistant'")

                        # Periksa aturan role dan tool calls
                        tools = sample.get("tools", [])
                        last_human_idx = -1
                        for idx, m in enumerate(messages):
                            if m.get("role") == "user":
                                last_human_idx = idx

                        for idx, m in enumerate(messages):
                            r = m.get("role")
                            cnt = m.get("content", "")
                            tc = m.get("tool_calls", [])
                            rsn = m.get("reasoning", "")

                            # Aturan think
                            if mode == "no_think" and rsn:
                                cat_errors.append(f"{p.name} baris {line_num}: mode 'no_think' tidak boleh ada 'reasoning'")
                            if mode == "think" and rsn and idx <= last_human_idx:
                                cat_errors.append(f"{p.name} baris {line_num}: reasoning hanya diizinkan setelah user terakhir")

                            # Aturan assistant content kosong
                            if r == "assistant" and not cnt and not tc:
                                cat_errors.append(f"{p.name} baris {line_num}: assistant content kosong tanpa tool_calls")

                            # Validasi tool calls
                            if tc:
                                for call in tc:
                                    is_v, t_err = validate_tool_call(call, tools)
                                    if not is_v:
                                        invalid_tools_count += 1
                                        err_msg = f"{p.name} baris {line_num}: validasi tool gagal -> {t_err}"
                                        cat_errors.append(err_msg)

            except Exception as e:
                cat_errors.append(f"Gagal membaca file {p.name}: {e}")

        for ce in cat_errors[:5]:
            errors.append(f"SFT [{cat_name}] {ce}")

        # Cek proporsi bahasa SFT
        total_valid_lang = en_count + id_count
        if total_valid_lang > 0:
            pct_en = (en_count / total_valid_lang) * 100.0
            pct_id = (id_count / total_valid_lang) * 100.0
            if abs(pct_en - 55.0) > 10.0:
                warnings.append(f"SFT [{cat_name}]: Proporsi EN {pct_en:.0f}% / ID {pct_id:.0f}% menyimpang dari target 55/45 (> +-10 poin)")

        sft_summary[cat_name] = {
            "files": len(files),
            "samples": sample_count,
            "target": target_samples,
            "en": en_count,
            "id": id_count,
            "invalid_tools": invalid_tools_count,
            "status": "OK" if sample_count >= target_samples else "KURANG"
        }

    # Simpan cache data index
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f)
    except Exception:
        pass

    # Tulis logs/check_errors.txt jika ada error atau warning
    log_file = project_dir / "logs" / "check_errors.txt"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        from llmkit.i18n import tr
        if errors:
            lines.append(tr("=== DAFTAR ERROR ===", "=== ERROR LIST ==="))
            lines.extend(errors)
        if warnings:
            lines.append("\n" + tr("=== DAFTAR PERINGATAN ===", "=== WARNING LIST ==="))
            lines.extend(warnings)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass

    is_valid = len(errors) == 0
    return is_valid, {
        "pretrain": pretrain_summary,
        "sft": sft_summary,
        "errors": errors,
        "warnings": warnings,
        "total_pretrain_est_tokens": total_pretrain_est_tokens
    }
