"""Pipeline persiapan data: tokenisasi ke shard biner uint16, dedup, filter kebocoran, dan split validasi (Bagian 10.2)."""

from __future__ import annotations

import array
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional, Set
import numpy as np
from tokenizers import Tokenizer
from llmkit.i18n import tr
from llmkit.state import compute_data_fingerprint


def normalize_text_for_dedup(text: str) -> str:
    """Normalisasi teks: NFC, huruf kecil, dan spasi tunggal."""
    nfc = unicodedata.normalize("NFC", text)
    low = nfc.lower()
    return re.sub(r"\s+", " ", low).strip()


def compute_doc_hash(norm_text: str) -> int:
    """Hash cepat 64-bit untuk deduplikasi persis."""
    raw_bytes = norm_text.encode("utf-8")
    try:
        import xxhash
        return xxhash.xxh64_intdigest(raw_bytes)
    except ImportError:
        return int.from_bytes(hashlib.blake2b(raw_bytes, digest_size=8).digest(), "little")


def build_leak_ngram_hashes(leak_dir: Path) -> Set[int]:
    """Bangun set hash 13-word n-gram dari file-file di eval_leak/."""
    ngram_hashes: Set[int] = set()
    if not leak_dir.exists():
        return ngram_hashes

    for file_path in leak_dir.glob("*.jsonl"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        words = normalize_text_for_dedup(text).split()
                        if len(words) >= 13:
                            for idx in range(len(words) - 12):
                                chunk = " ".join(words[idx : idx + 13])
                                h = compute_doc_hash(chunk)
                                ngram_hashes.add(h)
                    except Exception:
                        pass
        except Exception:
            pass

    return ngram_hashes


def contains_leak(words: list[str], leak_set: Set[int]) -> bool:
    """Periksa apakah urutan kata mengandung salah satu 13-gram bocor."""
    if not leak_set or len(words) < 13:
        return False
    for idx in range(len(words) - 12):
        chunk = " ".join(words[idx : idx + 13])
        if compute_doc_hash(chunk) in leak_set:
            return True
    return False


def run_prep(
    project_dir: Path,
    tokens_per_shard: int = 100_000_000,
    val_max_tokens: int = 5_000_000
) -> dict[str, Any]:
    """
    Eksekusi pipeline data prep menyeluruh sesuai spesifikasi Bagian 10.2.
    """
    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer tidak ditemukan di {tok_path}. Jalankan 'tokenizer' terlebih dahulu.")

    tokenizer = Tokenizer.from_file(str(tok_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("Token '<|endoftext|>' tidak ditemukan di tokenizer.")

    pretrain_dir = project_dir / "data" / "pretrain"
    leak_dir = project_dir / "data" / "eval_leak"
    tokenized_dir = project_dir / "tokenized"
    val_dir = tokenized_dir / "_val"

    tokenized_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bangun leak filter
    leak_hashes = build_leak_ngram_hashes(leak_dir)

    # 2. Cek manifest lama untuk resumable
    manifest_path = tokenized_dir / "_manifest.json"
    old_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
        except Exception:
            pass

    current_data_fp = compute_data_fingerprint(pretrain_dir)
    tokenizer_fp = hashlib.md5(tok_path.read_bytes()).hexdigest()

    sources_manifest: dict[str, Any] = {}
    total_tokens_all = 0

    # Iterasi setiap sub-folder sumber
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"Folder pretrain tidak ditemukan di {pretrain_dir}.")

    source_folders = [d for d in pretrain_dir.iterdir() if d.is_dir()]
    if not source_folders:
        raise ValueError(f"Tidak ada subfolder data pretrain di {pretrain_dir}.")

    for src_folder in sorted(source_folders, key=lambda x: x.name):
        src_name = src_folder.name
        files = sorted(list(src_folder.glob("*.jsonl")))
        if not files:
            continue

        src_out_dir = tokenized_dir / src_name
        src_out_dir.mkdir(parents=True, exist_ok=True)

        # Cek apakah sumber ini sudah selesai dan inputnya tidak berubah
        old_src_info = old_manifest.get("sources", {}).get(src_name)
        src_fp = [e for e in current_data_fp if e["path"].startswith(f"{src_name}/")]
        if (
            old_src_info
            and old_src_info.get("fingerprint") == src_fp
            and old_manifest.get("tokenizer_hash") == tokenizer_fp
            and all((tokenized_dir / s).exists() for s in old_src_info.get("shards", []))
        ):
            sources_manifest[src_name] = old_src_info
            total_tokens_all += old_src_info.get("tokens", 0)
            continue

        # Inisialisasi proses per sumber
        seen_doc_hashes: Set[int] = set()
        discarded_short = 0
        discarded_dup = 0
        discarded_leak = 0

        shard_list: list[str] = []
        shard_idx = 0
        current_shard_tokens = 0
        current_shard_buffer: array.array = array.array("H")  # uint16

        val_tokens = 0
        val_buffer: array.array = array.array("H")

        total_src_tokens = 0

        for f_path in files:
            with open(f_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    raw_text = obj.get("text")
                    if not raw_text or not isinstance(raw_text, str):
                        continue

                    # 1. Normalisasi & filter pendek
                    if len(raw_text.strip()) < 20:
                        discarded_short += 1
                        continue

                    norm_text = normalize_text_for_dedup(raw_text)

                    # 2. Dedup
                    doc_h = compute_doc_hash(norm_text)
                    if doc_h in seen_doc_hashes:
                        discarded_dup += 1
                        continue
                    seen_doc_hashes.add(doc_h)

                    # 3. Filter kebocoran
                    words = norm_text.split()
                    if leak_hashes and contains_leak(words, leak_hashes):
                        discarded_leak += 1
                        continue

                    # 4. Tokenisasi
                    encoded = tokenizer.encode(raw_text)
                    token_ids = encoded.ids
                    if not token_ids:
                        continue

                    doc_tokens = token_ids + [eos_id]

                    # 5. Split validasi (0.1% dokumen)
                    if doc_h % 1000 == 0 and val_tokens < val_max_tokens:
                        val_buffer.extend(doc_tokens)
                        val_tokens += len(doc_tokens)
                    else:
                        current_shard_buffer.extend(doc_tokens)
                        current_shard_tokens += len(doc_tokens)
                        total_src_tokens += len(doc_tokens)

                        # Tulis shard jika sudah mencapai batas tokens_per_shard
                        if current_shard_tokens >= tokens_per_shard:
                            shard_filename = f"{src_name}/shard_{shard_idx:05d}.bin"
                            shard_path = tokenized_dir / shard_filename
                            with open(shard_path, "wb") as sf:
                                current_shard_buffer.tofile(sf)
                            shard_list.append(shard_filename)
                            shard_idx += 1
                            current_shard_buffer = array.array("H")
                            current_shard_tokens = 0

        # Tulis sisa shard terakhir
        if len(current_shard_buffer) > 0:
            shard_filename = f"{src_name}/shard_{shard_idx:05d}.bin"
            shard_path = tokenized_dir / shard_filename
            with open(shard_path, "wb") as sf:
                current_shard_buffer.tofile(sf)
            shard_list.append(shard_filename)

        # Tulis file validasi sumber
        val_file = val_dir / f"{src_name}.bin"
        with open(val_file, "wb") as vf:
            val_buffer.tofile(vf)

        sources_manifest[src_name] = {
            "shards": shard_list,
            "tokens": total_src_tokens,
            "val_tokens": val_tokens,
            "discarded": {
                "short": discarded_short,
                "dup": discarded_dup,
                "leak": discarded_leak,
            },
            "fingerprint": src_fp,
        }
        total_tokens_all += total_src_tokens

    manifest_data = {
        "tokenizer_hash": tokenizer_fp,
        "data_fingerprint": current_data_fp,
        "total_tokens": total_tokens_all,
        "sources": sources_manifest,
    }

    # Tulis manifest atomik
    tmp_man = manifest_path.with_suffix(".tmp")
    with open(tmp_man, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    os.replace(tmp_man, manifest_path)

    # 8. Tampilkan token aktual vs needed dan hitung ulang epochs (Bagian 10.2)
    proj_file = project_dir / "project.json"
    needed_dict: dict[str, float] = {}
    if proj_file.exists():
        try:
            with open(proj_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            needed_dict = p_data.get("plan", {}).get("pretrain_data", {}).get("needed", {})
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(tr("HASIL DATA PREPARATION (PREP):", "DATA PREPARATION (PREP) RESULTS:"))
    headers = [tr("Sumber", "Source"), tr("Token Aktual", "Actual Tokens"), tr("Target Needed", "Target Needed"), tr("Epochs", "Epochs"), tr("Status", "Status")]
    rows = []
    has_warn = False
    has_err = False

    for src_name, s_info in sources_manifest.items():
        act_tok = s_info.get("tokens", 0)
        need_tok = needed_dict.get(src_name, 0.0)
        if need_tok > 0:
            epochs = act_tok / need_tok
            if epochs < 0.125:
                status = tr("ERROR (>8x epoch)", "ERROR (>8x epochs)")
                has_err = True
            elif epochs < 0.25:
                status = tr("PERINGATAN (4-8x epoch)", "WARN (4-8x epochs)")
                has_warn = True
            else:
                status = "OK"
            ep_str = f"{(need_tok / max(1, act_tok)):.1f}x"
        else:
            ep_str = "1.0x"
            status = "OK"

        from llmkit.ui import format_number
        rows.append([
            src_name,
            format_number(act_tok),
            format_number(need_tok) if need_tok else "-",
            ep_str,
            status,
        ])

    from llmkit.ui import print_table, print_warn, print_error
    print_table(headers, rows)
    print(tr(f"Total Token Pretrain: {total_tokens_all:,}", f"Total Pretrain Tokens: {total_tokens_all:,}"))

    if has_err:
        print_error(tr("Ditemukan folder dengan pengulangan data > 8 epoch! Tambah data atau kurangi target.", "Folders found with data repetition > 8 epochs! Add more data or reduce targets."))
    elif has_warn:
        print_warn(tr("Ditemukan folder dengan pengulangan data antara 4-8 epoch.", "Folders found with data repetition between 4-8 epochs."))

    print("\n" + tr("Lanjut: python main.py bench", "Next: python main.py bench"))
    return manifest_data
