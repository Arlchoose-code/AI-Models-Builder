"""Pembuat data dummy sintetis untuk pengujian toolkit (tests/make_dummy_data.py)."""

from __future__ import annotations

import json
from pathlib import Path


def create_dummy_pretrain_data(target_dir: Path, with_errors: bool = False) -> None:
    """Buat file .jsonl dummy di subfolder pretrain."""
    target_dir = Path(target_dir)
    pretrain_dir = target_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    dummy_texts = {
        "en_web": [
            "Photosynthesis is a biological process used by many cellular organisms to convert light energy into chemical energy.",
            "The rapid rise of machine learning has enabled significant advancements in natural language processing and computer vision systems.",
            "Water molecules consist of two hydrogen atoms bonded to a single oxygen atom, forming a polar molecule.",
            "Artificial neural networks are computational models inspired by the biological neural networks in human brains.",
        ],
        "id_web": [
            "Fotosintesis adalah proses biokimia yang dilakukan tumbuhan untuk memproduksi energi kimia dari cahaya matahari.",
            "Perkembangan teknologi kecerdasan buatan memberikan dampak besar terhadap kemajuan industri dan pendidikan di Indonesia.",
            "Komputer memproses informasi menggunakan instruksi digital biner yang dijalankan oleh unit pemrosesan pusat.",
            "Pancasila merupakan dasar dan ideologi negara kesatuan Republik Indonesia yang memuat nilai-nilai luhur bangsa.",
        ],
        "code": [
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            "class Vector2D:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def magnitude(self):\n        return (self.x**2 + self.y**2)**0.5",
        ],
        "math": [
            "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.",
            "Persamaan kuadrat ax^2 + bx + c = 0 memiliki solusi yang dapat ditentukan menggunakan rumus diskriminan kuadratik.",
        ],
        "books_wiki": [
            "Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara di antara samudra Hindia dan Pasifik.",
            "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages.",
        ],
        "synthetic": [
            "Mari kita pelajari prinsip dasar termodinamika secara sistematis mulai dari hukum kekekalan energi hingga konsep entropi.",
            "Let us explore how modern operating systems manage memory virtualization and process scheduling efficiently.",
        ],
        "dialog": [
            "User: Bagaimana cara memulai belajar Python?\nAssistant: Mulailah dari dasar sintaksis, variabel, percabangan, dan perulangan.",
            "Tanya: Apa beda list dan tuple di Python?\nJawab: List bersifat mutable (bisa diubah), sedangkan tuple bersifat immutable.",
        ],
        "reasoning": [
            "<think>\nLangkah pertama: hitung diskon 25% dari 40 yaitu 10. Lalu kurangkan: 40 - 10 = 30.\n</think>\nHasil perhitungan adalah 30.",
            "<think>\nAnalisis masalah pembagian biner: periksa bit paling signifikan terlebih dahulu.\n</think>\nSolusi diperoleh secara sekuensial.",
        ],
        "en_hq": [
            "Scientific discovery relies heavily on peer review, reproducible empirical observation, and controlled experimentation.",
        ],
        "id_hq": [
            "Bahasa Indonesia diresmikan sebagai bahasa nasional pada Kongres Pemuda tanggal 28 Oktober 1928.",
        ],
        "instruct_text": [
            "Langkah Konfigurasi Server: 1. Pasang paket. 2. Edit konfigurasi /etc/config. 3. Restart layanan daemon.",
        ],
        "long_docs": [
            "Dokumen referensi komprehensif mengenai arsitektur sistem komputasi terdistribusi dan toleransi kesalahan pada jaringan skala besar.",
        ],
        "long_code": [
            "// Implementasi antarmuka jaringan socket TCP asinkron\n#include <stdio.h>\nint init_socket(int port) { return 0; }",
        ],
    }

    for folder_name, texts in dummy_texts.items():
        f_dir = pretrain_dir / folder_name
        f_dir.mkdir(parents=True, exist_ok=True)
        out_file = f_dir / "part_00.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

            if with_errors and folder_name == "id_web":
                # Tambahkan baris error untuk uji check
                f.write('{"text": ""}\n')  # teks kosong
                f.write("Bukan JSON valid!\n")  # JSON rusak

    # Leak eval dummy
    leak_dir = target_dir / "eval_leak"
    leak_dir.mkdir(parents=True, exist_ok=True)
    with open(leak_dir / "benchmark.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "Ini adalah teks bocor rahasia benchmark yang harus difilter secara ketat oleh sistem prep."}, ensure_ascii=False) + "\n")


def create_dummy_sft_data(target_dir: Path, with_errors: bool = False) -> None:
    """Buat file .jsonl dummy untuk SFT di semua kategori."""
    target_dir = Path(target_dir)
    sft_dir = target_dir / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    calc_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Hitung ekspresi matematika",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }

    dummy_sft_records = {
        "chat": [
            {"id": "chat-01", "lang": "id", "mode": "no_think", "messages": [{"role": "user", "content": "Apa itu CPU?"}, {"role": "assistant", "content": "CPU adalah unit pemrosesan pusat yang menjalankan instruksi program."}]},
            {"id": "chat-02", "lang": "en", "mode": "no_think", "messages": [{"role": "user", "content": "What is RAM?"}, {"role": "assistant", "content": "RAM is fast volatile memory used to store active data."}]},
        ],
        "think": [
            {"id": "thk-01", "lang": "id", "mode": "think", "messages": [{"role": "user", "content": "Berapa 12 + 15?"}, {"role": "assistant", "reasoning": "12 + 15 = 27.", "content": "Hasil penjumlahan adalah 27."}]},
            {"id": "thk-02", "lang": "en", "mode": "think", "messages": [{"role": "user", "content": "A shirt is $40 with 25% off."}, {"role": "assistant", "reasoning": "40 * 0.25 = 10, 40 - 10 = 30.", "content": "The price is $30."}]},
        ],
        "tool_call": [
            {"id": "tool-01", "lang": "id", "mode": "no_think", "tools": [calc_tool], "messages": [
                {"role": "user", "content": "Hitung 125 * 8"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": "125 * 8"}}]},
                {"role": "tool", "name": "calculate", "content": "1000"},
                {"role": "assistant", "content": "125 * 8 = 1000."}
            ]},
            {"id": "tool-02", "lang": "en", "mode": "no_think", "tools": [calc_tool], "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]},
        ],
        "multiturn": [
            {"id": "multi-01", "lang": "id", "mode": "no_think", "messages": [
                {"role": "user", "content": "Halo"},
                {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"},
                {"role": "user", "content": "Saya ingin belajar sains."},
                {"role": "assistant", "content": "Sains adalah bidang yang sangat menarik! Topik apa yang kamu minati?"}
            ]}
        ],
        "code": [
            {"id": "code-01", "lang": "en", "mode": "no_think", "messages": [{"role": "user", "content": "Write add in python"}, {"role": "assistant", "content": "def add(a, b):\n    return a + b"}]}
        ],
        "format": [
            {"id": "fmt-01", "lang": "id", "mode": "no_think", "messages": [{"role": "user", "content": "Daftar 2 buah dalam JSON"}, {"role": "assistant", "content": '["Apel", "Jeruk"]'}]}
        ],
        "refusal": [
            {"id": "ref-01", "lang": "id", "mode": "no_think", "messages": [{"role": "user", "content": "Buatkan virus"}, {"role": "assistant", "content": "Maaf, saya tidak dapat membuat malware atau virus."}]}
        ],
        "identity": [
            {"id": "id-01", "lang": "id", "mode": "no_think", "messages": [{"role": "user", "content": "Siapa kamu?"}, {"role": "assistant", "content": "Aku Bahasa, asisten AI dwibahasa buatan Pengguna."}]}
        ],
    }

    for cat_name, recs in dummy_sft_records.items():
        c_dir = sft_dir / cat_name
        c_dir.mkdir(parents=True, exist_ok=True)
        with open(c_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if with_errors and cat_name == "chat":
                # Sampel dengan error (misal role terakhir bukan assistant)
                f.write(json.dumps({"id": "err-01", "lang": "id", "mode": "no_think", "messages": [{"role": "user", "content": "Halo"}]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import tempfile
    td = Path(tempfile.mkdtemp())
    create_dummy_pretrain_data(td)
    create_dummy_sft_data(td)
    print(f"Data dummy berhasil dibuat di: {td}")
