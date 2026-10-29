"""Spesifikasi data pretrain & SFT, skema, contoh format, pembuat folder, dan konverter (Bagian 7)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


PRETRAIN_FOLDERS = [
    "en_web",
    "id_web",
    "code",
    "math",
    "books_wiki",
    "synthetic",
    "dialog",
    "reasoning",
    "en_hq",
    "id_hq",
    "instruct_text",
    "long_docs",
    "long_code",
]

SFT_CATEGORIES = [
    "chat",
    "think",
    "tool_call",
    "multiturn",
    "code",
    "format",
    "refusal",
    "identity",
]


PRETRAIN_FOLDER_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "en_web": {
        "deskripsi": "Teks web bahasa Inggris berkualitas tinggi (contoh: FineWeb-Edu, DCLM).",
        "contoh": '{"text": "The water cycle describes how water moves through evaporation, condensation, and precipitation."}'
    },
    "id_web": {
        "deskripsi": "Teks web bahasa Indonesia berkualitas (contoh: FineWeb-2 ind_Latn, berita terpercaya).",
        "contoh": '{"text": "Komputer memproses data menggunakan instruksi yang dijalankan oleh prosesor."}'
    },
    "code": {
        "deskripsi": "Kode pemrograman berbagai bahasa beserta dokumentasi dan penjelasannya.",
        "contoh": '{"text": "def binary_search(arr, x):\\n    lo, hi = 0, len(arr) - 1\\n    while lo <= hi:\\n        mid = (lo + hi) // 2\\n        if arr[mid] == x: return mid\\n        elif arr[mid] < x: lo = mid + 1\\n        else: hi = mid - 1\\n    return -1"}'
    },
    "math": {
        "deskripsi": "Teks matematika, penalaran kuantitatif, dan konten STEM.",
        "contoh": '{"text": "Teorema Pythagoras menyatakan bahwa a^2 + b^2 = c^2 untuk segitiga siku-siku dengan sisi miring c."}'
    },
    "books_wiki": {
        "deskripsi": "Artikel Wikipedia (EN & ID) serta buku domain publik bermutu tinggi.",
        "contoh": '{"text": "Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara."}'
    },
    "synthetic": {
        "deskripsi": "Teks sintetis terstruktur buatan model besar (textbook, terjemahan bermutu EN<->ID).",
        "contoh": '{"text": "Mari kita telaah konsep gravitasi dalam fisika klasik. Isaac Newton merumuskan hukum gravitasi universal..."}'
    },
    "dialog": {
        "deskripsi": "Teks percakapan multi-pihak, forum diskusi, dan tanya jawab alami.",
        "contoh": '{"text": "User1: Bagaimana cara kerja generator di Python?\\nUser2: Generator menggunakan kata kunci yield..."}'
    },
    "reasoning": {
        "deskripsi": "Teks penalaran langkah-demi-langkah (boleh mengandung tag <think>...</think>).",
        "contoh": '{"text": "<think>\\nMari kita uraikan langkah penyelesaian soal ini secara cermat...\\n</think>\\nSolusi akhirnya adalah x = 5."}'
    },
    "en_hq": {
        "deskripsi": "Subset teks bahasa Inggris paling bersih dan bermutu tinggi.",
        "contoh": '{"text": "Scientific inquiry requires hypothesis formulation, rigorous experimentation, and peer review."}'
    },
    "id_hq": {
        "deskripsi": "Subset teks bahasa Indonesia paling bersih (ensiklopedia, buku teks terverifikasi).",
        "contoh": '{"text": "Pancasila adalah dasar filsafat negara Republik Indonesia yang terdiri dari lima sila."}'
    },
    "instruct_text": {
        "deskripsi": "Teks bergaya instruksi, format terstruktur, dan dokumentasi API dalam bentuk teks mentah.",
        "contoh": '{"text": "Panduan Instalasi: 1. Pasang dependensi. 2. Konfigurasi environment variable. 3. Jalankan service."}'
    },
    "long_docs": {
        "deskripsi": "Dokumen panjang untuk latihan konteks panjang (makalah, novel domain publik).",
        "contoh": '{"text": "BAB I: PENDAHULUAN\\nLatar belakang penelitian ini berfokus pada analisis mendalam terhadap..."}'
    },
    "long_code": {
        "deskripsi": "Repositori kode lengkap dan file sumber panjang untuk penalaran multi-fungsi.",
        "contoh": '{"text": "// File: app/main.rs\\nuse std::sync::Arc;\\nstruct ServerConfig { port: u16 }\\n..."}'
    },
}


SFT_CATEGORY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "chat": {
        "deskripsi": "Tanya jawab umum dwibahasa, mode no_think.",
        "contoh": '{"id":"chat-0001","lang":"id","mode":"no_think","messages":[{"role":"user","content":"Apa fungsi CPU?"},{"role":"assistant","content":"CPU mengeksekusi instruksi program dan memproses data."}]}'
    },
    "think": {
        "deskripsi": "Penalaran matematika, kode, dan logika dengan mode think dan field reasoning.",
        "contoh": '{"id":"think-0001","lang":"en","mode":"think","messages":[{"role":"user","content":"A shirt costs $40 and is 25% off. What is the sale price?"},{"role":"assistant","reasoning":"25% of 40 is 10. Discount is $10. 40 - 10 = 30.","content":"The sale price is $30."}]}'
    },
    "tool_call": {
        "deskripsi": "Pemanggilan fungsi terstruktur (search, calculate, dll) atau penanganan error tool.",
        "contoh": '{"id":"tool-0001","lang":"id","mode":"no_think","tools":[{"type":"function","function":{"name":"calculate","description":"Hitung ekspresi","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],"messages":[{"role":"user","content":"Berapa 125 * 8?"},{"role":"assistant","content":"","tool_calls":[{"name":"calculate","arguments":{"expression":"125 * 8"}}]},{"role":"tool","name":"calculate","content":"1000"},{"role":"assistant","content":"125 * 8 = 1000."}]}'
    },
    "multiturn": {
        "deskripsi": "Percakapan multi-turn (3-6 giliran) bertema konsisten.",
        "contoh": '{"id":"multi-0001","lang":"id","mode":"no_think","messages":[{"role":"user","content":"Halo, saya mau belajar Python."},{"role":"assistant","content":"Halo! Senang mendengarnya. Kamu mau mulai dari dasar atau punya proyek tertentu?"},{"role":"user","content":"Mulai dari dasar variabel dan tipe data."},{"role":"assistant","content":"Bagus! Di Python, variabel dibuat langsung dengan assignment, misal x = 10."}]}'
    },
    "code": {
        "deskripsi": "Menulis, menjelaskan, dan memperbaiki kode program.",
        "contoh": '{"id":"code-0001","lang":"en","mode":"no_think","messages":[{"role":"user","content":"Write a python function to check if a number is prime."},{"role":"assistant","content":"```python\\ndef is_prime(n: int) -> bool:\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True\\n```"}]}'
    },
    "format": {
        "deskripsi": "Instruksi format keluaran ketat (JSON, Markdown table, CSV).",
        "contoh": '{"id":"fmt-0001","lang":"id","mode":"no_think","messages":[{"role":"user","content":"Tampilkan 2 planet terdekat ke matahari dalam JSON array."},{"role":"assistant","content":"[\\\"Merkurius\\\", \\\"Venus\\\"]"}]}'
    },
    "refusal": {
        "deskripsi": "Penolakan sopan untuk konten berbahaya atau ketidaktahuan faktual yang jujur.",
        "contoh": '{"id":"ref-0001","lang":"id","mode":"no_think","messages":[{"role":"user","content":"Bagaimana cara meretas akun orang lain?"},{"role":"assistant","content":"Maaf, saya tidak dapat membantu aktivitas peretasan atau akses tanpa izin ke akun orang lain."}]}'
    },
    "identity": {
        "deskripsi": "Respon pengenalan diri sesuai nama model dan nama pembuat.",
        "contoh": '{"id":"id-0001","lang":"id","mode":"no_think","messages":[{"role":"user","content":"Kamu siapa?"},{"role":"assistant","content":"Aku {model_name}, asisten AI dwibahasa buatan {creator}."}]}'
    },
}


def extract_think(content: str) -> tuple[str, str]:
    """
    Ekstrak tag <think>...</think> dari teks content.
    Mengembalikan (reasoning, clean_content).
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
        return reasoning, clean_content
    return "", content


def create_project_data_scaffold(project_dir: Path, plan: dict[str, Any], model_name: str, creator: str) -> None:
    """
    Buat folder-folder kosong data pretrain & SFT beserta README.txt panduan format.
    """
    data_dir = project_dir / "data"
    pretrain_dir = data_dir / "pretrain"
    sft_dir = data_dir / "sft"
    eval_leak_dir = data_dir / "eval_leak"

    eval_leak_dir.mkdir(parents=True, exist_ok=True)
    leak_readme = eval_leak_dir / "README.txt"
    if not leak_readme.exists():
        leak_readme.write_text(
            "Folder eval_leak (opsional):\n"
            "Letakkan file .jsonl dengan format {\"text\": \"...\"} berisi soal-soal benchmark.\n"
            "Proses 'prep' akan memfilter dan membuang dokumen pretrain yang memiliki n-gram 13 kata yang sama.\n",
            encoding="utf-8"
        )

    # Pretrain folders
    needed = plan.get("pretrain_data", {}).get("needed", {})
    for folder_name in PRETRAIN_FOLDERS:
        f_dir = pretrain_dir / folder_name
        f_dir.mkdir(parents=True, exist_ok=True)
        readme = f_dir / "README.txt"
        info = PRETRAIN_FOLDER_DESCRIPTIONS.get(folder_name, {})
        target_tok = needed.get(folder_name, 0.0)
        target_str = f"{target_tok / 1e9:.2f}B token" if target_tok >= 1e9 else f"{target_tok / 1e6:.1f}M token"

        contoh_pretrain = info.get("contoh", '{"text": "..."}')
        content = (
            f"=== DATA PRETRAIN: {folder_name} ===\n"
            f"Deskripsi: {info.get('deskripsi', '')}\n"
            f"Target alokasi proyek: {target_str}\n"
            f"Format wajib: file .jsonl (UTF-8), satu baris satu dokumen JSON:\n"
            f"{contoh_pretrain}\n\n"
            f"Field wajib: 'text' (string tidak kosong).\n"
            f"Field opsional: 'lang', 'source', 'id'.\n"
        )
        readme.write_text(content, encoding="utf-8")

    # SFT folders
    sft_cats = plan.get("sft_data", {}).get("categories", {})
    for cat_name in SFT_CATEGORIES:
        c_dir = sft_dir / cat_name
        c_dir.mkdir(parents=True, exist_ok=True)
        readme = c_dir / "README.txt"
        info = SFT_CATEGORY_DESCRIPTIONS.get(cat_name, {})
        cat_info = sft_cats.get(cat_name, {})
        samples_target = cat_info.get("samples", 0)

        example = info.get("contoh", "{}")
        if cat_name == "identity":
            example = example.replace("{model_name}", model_name).replace("{creator}", creator)

        content = (
            f"=== DATA SFT: {cat_name} ===\n"
            f"Deskripsi: {info.get('deskripsi', '')}\n"
            f"Target alokasi proyek: {samples_target:,} sampel\n"
            f"Format wajib: file .jsonl (UTF-8), satu baris satu percakapan JSON:\n"
            f"{example}\n\n"
            f"Aturan format:\n"
            f"- 'id': string unik\n"
            f"- 'lang': 'en' atau 'id'\n"
            f"- 'mode': 'think' atau 'no_think'\n"
            f"- 'messages': array giliran user/assistant/system/tool\n"
        )
        readme.write_text(content, encoding="utf-8")

        # Jika identity: buat sampel awal otomatis jika belum ada
        if cat_name == "identity":
            auto_file = c_dir / "auto.jsonl"
            if not auto_file.exists():
                lines = [
                    json.dumps({
                        "id": "ident-001", "lang": "id", "mode": "no_think",
                        "messages": [
                            {"role": "user", "content": "Siapa kamu?"},
                            {"role": "assistant", "content": f"Aku {model_name}, asisten AI dwibahasa (Indonesia dan Inggris) yang dibuat oleh {creator}."}
                        ]
                    }, ensure_ascii=False),
                    json.dumps({
                        "id": "ident-002", "lang": "en", "mode": "no_think",
                        "messages": [
                            {"role": "user", "content": "Who are you?"},
                            {"role": "assistant", "content": f"I am {model_name}, a bilingual AI assistant (Indonesian and English) created by {creator}."}
                        ]
                    }, ensure_ascii=False),
                    json.dumps({
                        "id": "ident-003", "lang": "id", "mode": "no_think",
                        "messages": [
                            {"role": "user", "content": "Siapa yang menciptakanmu?"},
                            {"role": "assistant", "content": f"Saya diciptakan oleh {creator}."}
                        ]
                    }, ensure_ascii=False),
                ]
                auto_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_file(
    source_file: Path,
    out_file: Path,
    from_format: str,
    text_column: str = "text"
) -> int:
    """
    Konversi file format umum (parquet/sharegpt/jsonl) ke format kanonik (Bagian 7.4).
    Mengembalikan jumlah baris yang berhasil dikonversi.
    """
    count = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as out_f:
        if from_format == "text_column":
            # JSONL dengan kolom teks yang berbeda atau Parquet
            if source_file.suffix == ".parquet":
                import pandas as pd
                df = pd.read_parquet(source_file)
                for _, row in df.iterrows():
                    val = str(row.get(text_column, "")).strip()
                    if val:
                        out_f.write(json.dumps({"text": val}, ensure_ascii=False) + "\n")
                        count += 1
            else:
                with open(source_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            val = str(obj.get(text_column, "")).strip()
                            if val:
                                out_f.write(json.dumps({"text": val}, ensure_ascii=False) + "\n")
                                count += 1
                        except Exception:
                            pass

        elif from_format == "sharegpt":
            # ShareGPT -> SFT format
            with open(source_file, "r", encoding="utf-8") as in_f:
                for idx, line in enumerate(in_f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        convs = obj.get("conversations", obj.get("messages", []))
                        norm_msgs = []
                        for m in convs:
                            r = m.get("from", m.get("role", ""))
                            if r in ("human", "user"):
                                role = "user"
                            elif r in ("gpt", "assistant", "chatgpt"):
                                role = "assistant"
                            elif r == "system":
                                role = "system"
                            else:
                                role = "user"
                            norm_msgs.append({"role": role, "content": str(m.get("value", m.get("content", "")))})

                        if norm_msgs and norm_msgs[-1]["role"] == "assistant":
                            canonical = {
                                "id": f"conv-{idx:06d}",
                                "lang": obj.get("lang", "en"),
                                "mode": "no_think",
                                "messages": norm_msgs
                            }
                            out_f.write(json.dumps(canonical, ensure_ascii=False) + "\n")
                            count += 1
                    except Exception:
                        pass

    return count


def make_identity_file(
    project_dir: Path,
    model_name: Optional[str] = None,
    creator: Optional[str] = None
) -> Path:
    """
    Buat atau perbarui file data/sft/identity/auto.jsonl berdasarkan identitas proyek (Fase 4 helper).
    """
    proj_file = project_dir / "project.json"
    if proj_file.exists():
        try:
            with open(proj_file, "r", encoding="utf-8") as f:
                pd = json.load(f)
            answers = pd.get("answers", {})
            if not model_name:
                model_name = answers.get("model_name", "Bahasa")
            if not creator:
                creator = answers.get("creator", "Pengguna")
        except Exception:
            pass

    model_name = model_name or "Bahasa"
    creator = creator or "Pengguna"

    ident_dir = project_dir / "data" / "sft" / "identity"
    ident_dir.mkdir(parents=True, exist_ok=True)
    auto_file = ident_dir / "auto.jsonl"

    lines = [
        json.dumps({
            "id": "ident-001", "lang": "id", "mode": "no_think",
            "messages": [
                {"role": "user", "content": "Siapa kamu?"},
                {"role": "assistant", "content": f"Aku {model_name}, asisten AI dwibahasa (Indonesia dan Inggris) yang dibuat oleh {creator}."}
            ]
        }, ensure_ascii=False),
        json.dumps({
            "id": "ident-002", "lang": "en", "mode": "no_think",
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": f"I am {model_name}, a bilingual AI assistant (Indonesian and English) created by {creator}."}
            ]
        }, ensure_ascii=False),
        json.dumps({
            "id": "ident-003", "lang": "id", "mode": "no_think",
            "messages": [
                {"role": "user", "content": "Siapa yang menciptakanmu?"},
                {"role": "assistant", "content": f"Saya diciptakan oleh {creator} sebagai asisten AI {model_name}."}
            ]
        }, ensure_ascii=False),
        json.dumps({
            "id": "ident-004", "lang": "en", "mode": "no_think",
            "messages": [
                {"role": "user", "content": "What is your name and who created you?"},
                {"role": "assistant", "content": f"My name is {model_name}, and I was developed by {creator}."}
            ]
        }, ensure_ascii=False)
    ]
    auto_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from llmkit.i18n import tr
    from llmkit.ui import print_ok
    print_ok(tr(f"File identitas berhasil dibuat di: {auto_file}", f"Identity file successfully created at: {auto_file}"))
    return auto_file

