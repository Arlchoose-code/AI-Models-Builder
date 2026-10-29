"""Pelatihan Tokenizer Byte-level BPE dan Analisis Fertilitas (Bagian 10.1)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator, Optional
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from llmkit.chat_template import SPECIAL_TOKENS, EOS_TOKEN, PRETRAIN_EOS_TOKEN, PAD_TOKEN


def build_bpe_trainer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Konstruksi objek Tokenizer BPE dan Trainer dengan byte-level dan pemecahan digit."""
    if vocab_size > 65535:
        raise ValueError(f"Ukuran vocab ({vocab_size}) melebihi batas 65.535 (uint16).")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    return tok, trainer


def stream_training_texts(
    pretrain_dir: Path,
    max_bytes: int = int(1.5 * (1024 ** 3)),
    seed: int = 42
) -> Iterator[str]:
    """
    Iterator streaming teks untuk training tokenizer dengan proporsi target:
    en_* 35%, id_* 35%, code 22%, math 8%.
    """
    random.seed(seed)
    buckets: dict[str, list[Path]] = {
        "en": [],
        "id": [],
        "code": [],
        "math": [],
        "other": []
    }

    if pretrain_dir.exists():
        for sub in pretrain_dir.iterdir():
            if sub.is_dir():
                s_name = sub.name
                files = list(sub.glob("*.jsonl"))
                if s_name.startswith("en_"):
                    buckets["en"].extend(files)
                elif s_name.startswith("id_"):
                    buckets["id"].extend(files)
                elif "code" in s_name:
                    buckets["code"].extend(files)
                elif "math" in s_name:
                    buckets["math"].extend(files)
                else:
                    buckets["other"].extend(files)

    total_read = 0
    weights = [("en", 0.35), ("id", 0.35), ("code", 0.22), ("math", 0.08)]

    # Generator dokumen dari file-file
    file_handles = {}
    for b_name, b_files in buckets.items():
        if b_files:
            file_handles[b_name] = [open(f, "r", encoding="utf-8", errors="ignore") for f in b_files]

    if not any(file_handles.values()):
        # Tidak ada data, kembalikan teks dasar
        yield "Hello world. Ini adalah teks pembuka untuk pelatihan tokenizer dwibahasa."
        return

    try:
        while total_read < max_bytes:
            # Pilih bucket secara acak berbobot
            r = random.random()
            cum = 0.0
            chosen_bucket = "other"
            for b_name, w in weights:
                cum += w
                if r <= cum and b_name in file_handles and file_handles[b_name]:
                    chosen_bucket = b_name
                    break

            if chosen_bucket not in file_handles or not file_handles[chosen_bucket]:
                # Ambil bucket mana saja yang ada
                avail = [b for b, h in file_handles.items() if h]
                if not avail:
                    break
                chosen_bucket = random.choice(avail)

            handles = file_handles[chosen_bucket]
            h = random.choice(handles)
            line = h.readline()
            if not line:
                # File habis, tutup dan hapus
                h.close()
                handles.remove(h)
                continue

            try:
                obj = json.loads(line)
                txt = obj.get("text", "")
                if txt and len(txt.strip()) >= 20:
                    encoded_len = len(txt.encode("utf-8"))
                    total_read += encoded_len
                    yield txt
            except Exception:
                pass
    finally:
        for handles in file_handles.values():
            for h in handles:
                try:
                    h.close()
                except Exception:
                    pass


def compute_fertility_report(tokenizer: Tokenizer) -> dict[str, Any]:
    """
    Hitung laporan fertilitas (bytes_per_token, tokens_per_word) untuk teks sampel
    Bahasa Indonesia, Inggris, Kode, dan Matematika (Bagian 10.1).
    """
    samples = {
        "en": "Artificial intelligence and machine learning models continue to transform natural language processing across multiple domains.",
        "id": "Kecerdasan buatan dan model pembelajaran mesin terus mentransformasi pemrosesan bahasa alami di berbagai bidang.",
        "id_affix": "mempertanggungjawabkan ketidakpastian memperdayakan keikutsertaan gak banget wkwk",
        "code": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid",
        "math": "The quadratic formula states that for ax^2 + bx + c = 0, x = (-b +- sqrt(b^2 - 4ac)) / (2a)."
    }

    report: dict[str, Any] = {}
    for key, text in samples.items():
        encoded = tokenizer.encode(text)
        token_count = len(encoded.ids)
        byte_count = len(text.encode("utf-8"))
        word_count = len(text.split())

        bpt = byte_count / max(1, token_count)
        tpw = token_count / max(1, word_count)
        report[key] = {
            "tokens": token_count,
            "bytes": byte_count,
            "words": word_count,
            "bytes_per_token": round(bpt, 2),
            "tokens_per_word": round(tpw, 2),
        }

    # Cek rasio fertilitas ID vs EN
    en_tpw = report["en"]["tokens_per_word"]
    id_tpw = report["id"]["tokens_per_word"]
    ratio = id_tpw / max(0.1, en_tpw)
    report["ratio_id_en"] = round(ratio, 2)
    report["is_ratio_acceptable"] = ratio <= 1.6

    return report


def train_tokenizer(
    project_dir: Path,
    vocab_size: int = 64000,
    max_gb: float = 1.5,
    seed: int = 42
) -> tuple[Tokenizer, dict[str, Any]]:
    """Latih tokenizer, simpan ke tokenizer/ dan kembalikan objek Tokenizer serta meta."""
    tok, trainer = build_bpe_trainer(vocab_size)
    pretrain_dir = project_dir / "data" / "pretrain"
    max_bytes = int(max_gb * (1024 ** 3))

    text_iterator = stream_training_texts(pretrain_dir, max_bytes=max_bytes, seed=seed)
    tok.train_from_iterator(text_iterator, trainer=trainer)

    # Verifikasi tes wajib (Bagian 10.1):
    # 1. Digit terpisah
    test_digits = "12345"
    digit_ids = tok.encode(test_digits).ids
    if len(digit_ids) != 5:
        raise RuntimeError(f"Uji digit gagal: '12345' menghasilkan {len(digit_ids)} token, diharapkan 5 terpisah.")

    # 2. Special tokens ID
    special_ids = {token: tok.token_to_id(token) for token in SPECIAL_TOKENS}
    for token, tid in special_ids.items():
        if tid is None:
            raise RuntimeError(f"Special token '{token}' tidak ditemukan dalam tokenizer.")

    # 3. Laporan fertilitas
    fertility = compute_fertility_report(tok)

    # Simpan hasil ke disk
    out_dir = project_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_file = out_dir / "tokenizer.json"
    tok.save(str(tok_file))

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": special_ids,
        "fertility": fertility,
        "eos_token": EOS_TOKEN,
        "pretrain_eos_token": PRETRAIN_EOS_TOKEN,
        "pad_token": PAD_TOKEN,
    }
    meta_file = out_dir / "tokenizer_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return tok, meta
