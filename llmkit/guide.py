"""Teks panduan lengkap dari nol sampai export (guide), bantuan perintah (help), dan daftar dataset kandidat (Bagian 13.4, 14, 17)."""

from __future__ import annotations

from typing import Optional


DATASET_CANDIDATES = """=== KANDIDAT DATASET PRETRAIN & SFT ===
Catatan: Verifikasi dulu bahwa dataset masih tersedia di Hugging Face Hub, dan periksa lisensinya sebelum digunakan!

Pretrain:
- en_web / en_hq : HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus
- id_web / id_hq : HuggingFaceFW/fineweb-2 (subset ind_Latn), wikimedia/wikipedia (subset id)
- code           : bigcode/the-stack-v2, subset Python-edu dari smollm-corpus
- math           : HuggingFaceTB/finemath, open-web-math/open-web-math
- books_wiki     : wikimedia/wikipedia (en & id)
- synthetic      : Cosmopedia (bagian dari smollm-corpus) + terjemahan EN->ID
- eval_leak      : hellaswag, mmlu, indommlu (soal evaluasi agar disaring dari data latih)

SFT:
- chat / multiturn : HuggingFaceTB/smoltalk
- tool_call        : NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE
- think            : open-thoughts/OpenThoughts-114k, open-r1/OpenR1-Math-220k
- data Indonesia   : Dibuat melalui terjemahan atau generasi sintetis model besar.
"""

HONESTY_NOTES = """=== CATATAN KEJUJURAN UNTUK PENGGUNA ===
1. Angka perkiraan (rasio token, LR, TFLOPs efektif, harga GPU) adalah TITIK AWAL, bukan kepastian mutlak.
   Perintah 'bench' dan 'pilot' yang akan menentukan angka kecepatan dan LR nyata.
2. Melatih model 0.6B dari nol dengan <= 100B token hampir pasti kalah cerdas dibanding model frontier seperti
   Qwen3-0.6B resmi (yang dilatih hingga puluhan triliun token). Jalur paling efisien untuk hasil produk adalah
   continued pretraining dari model dasar. Toolkit ini sangat tepat untuk eksperimen, riset mandiri, dan edukasi mendalam.
3. Kualitas data jauh lebih menentukan hasil akhir daripada sekadar kuantitas token mentah.
"""

STEP_BY_STEP_GUIDE = [
    {
        "langkah": 1,
        "judul": "Inisialisasi Proyek (init)",
        "perintah": "python main.py init",
        "deskripsi": "Jawab pertanyaan konfigurasi (ukuran, GPU, target token). Program akan menghitung seluruh rencana dan membuat struktur folder data.",
        "ekspektasi": "Terbentuk file project.json dan folder data/pretrain/ serta data/sft/ lengkap dengan README.txt.",
        "kalau_gagal": "Pastikan hak tulis direktori tersedia dan input angka valid."
    },
    {
        "langkah": 2,
        "judul": "Pengisian dan Validasi Data (check)",
        "perintah": "python main.py check",
        "deskripsi": "Isi folder data dengan file .jsonl sesuai format arahan, lalu jalankan validator data.",
        "ekspektasi": "Semua folder pretrain dan SFT terisi minimal 90% target tanpa error format.",
        "kalau_gagal": "Periksa file logs/check_errors.txt untuk melihat baris JSON yang bermasalah."
    },
    {
        "langkah": 3,
        "judul": "Pelatihan Tokenizer (tokenizer)",
        "perintah": "python main.py tokenizer",
        "deskripsi": "Melatih Byte-level BPE dwibahasa (EN + ID) dengan pemecahan digit dan token khusus.",
        "ekspektasi": "Terbentuk tokenizer/tokenizer.json dan laporan fertilitas token per kata.",
        "kalau_gagal": "Pastikan terdapat minimal 1-2 GB teks di data/pretrain/."
    },
    {
        "langkah": 4,
        "judul": "Persiapan & Tokenisasi Shard (prep)",
        "perintah": "python main.py prep",
        "deskripsi": "Tokenisasi seluruh teks mentah ke file biner uint16 (.bin), filtering n-gram kebocoran, dan dedup persis.",
        "ekspektasi": "Terbentuk tokenized/<sumber>/shard_*.bin dan tokenized/_manifest.json.",
        "kalau_gagal": "Periksa ruang disk kosong (butuh ~2 byte per token)."
    },
    {
        "langkah": 5,
        "judul": "Benchmark Kecepatan GPU (bench)",
        "perintah": "python main.py bench",
        "deskripsi": "Mencari micro-batch terbesar yang muat di VRAM tanpa OOM dan mengukur tok/detik nyata.",
        "ekspektasi": "Terbentuk bench.json dengan estimasi waktu dan biaya training riil.",
        "kalau_gagal": "Jika terjadi CUDA OOM, bench akan otomatis menurunkan ukuran batch."
    },
    {
        "langkah": 6,
        "judul": "Pilot Uji Coba Model Kecil (pilot)",
        "perintah": "python main.py pilot",
        "deskripsi": "Menjalankan model kecil 0.05B untuk sweep LR dan menguji mekanisme simpan/muat checkpoint.",
        "ekspektasi": "Terbentuk pilot/result.json dengan rekomendasi LR terbaik.",
        "kalau_gagal": "Periksa apakah data di tokenized/ dapat dimuat dengan baik."
    },
    {
        "langkah": 7,
        "judul": "Pretraining Penuh (pretrain)",
        "perintah": "python main.py pretrain",
        "deskripsi": "Loop pretraining utama dengan LR schedule WSD, auto-checkpoint, dan deteksi spike.",
        "ekspektasi": "Model dilatih hingga target token. Checkpoint tersimpan di runs/run_XXX/ckpt/.",
        "kalau_gagal": "Jika sesi terputus atau crash, lanjutkan dengan: python main.py pretrain --resume."
    },
    {
        "langkah": 8,
        "judul": "Supervised Fine-Tuning (sft)",
        "perintah": "python main.py sft",
        "deskripsi": "Melatih model menguasai percakapan, penalaran (mode think), dan pemanggilan tool.",
        "ekspektasi": "SFT selesai 2-3 epoch dengan loss mask pada jawaban assistant.",
        "kalau_gagal": "Pastikan data di data/sft/tool_call/ dan chat/ valid."
    },
    {
        "langkah": 9,
        "judul": "Evaluasi Komprehensif (eval)",
        "perintah": "python main.py eval",
        "deskripsi": "Menguji kepatuhan mode, sintaks tool call, kecocokan bahasa, dan degenerasi.",
        "ekspektasi": "Tercetak tabel skor kepatuhan dan tersimpan di runs/run_XXX/eval.json.",
        "kalau_gagal": "Periksa daftar failed_examples untuk melihat kegagalan respons model."
    },
    {
        "langkah": 10,
        "judul": "Percakapan & Ekspor (chat & export)",
        "perintah": "python main.py chat  /  python main.py export",
        "deskripsi": "Uji interaktif di terminal dengan tool sandbox, lalu ekspor model ke format Hugging Face / GGUF.",
        "ekspektasi": "Folder export/<nama>/ siap dijalankan atau dikonversi ke llama.cpp / Ollama.",
        "kalau_gagal": "Pastikan chat template Jinja identik dengan saat training."
    }
]


COMMAND_HELP_MAP = {
    "init": "Inisialisasi proyek baru: tanya jawab terminal, hitung rencana, buat folder scaffolding data.",
    "plan": "Kalkulator cepat parameter, token, waktu, dan biaya tanpa membuat proyek. Contoh: python main.py plan 0.6B --mode balanced --gpu H100",
    "status": "Deteksi otomatis progres proyek dari kondisi file di disk dan cetak rekomendasi langkah berikutnya.",
    "check": "Validasi file data mentah pretrain & SFT, estimasi token, dan periksa integritas format JSONL.",
    "tokenizer": "Latih tokenizer Byte-level BPE dwibahasa (EN+ID) dengan pemecahan digit dan special tokens.",
    "prep": "Tokenisasi dokumen mentah menjadi shard biner uint16, dedup persis, dan split validasi.",
    "bench": "Ukur kecepatan hardware GPU (token/detik) dan cari micro-batch optimal.",
    "pilot": "Jalankan model uji coba 0.05B pada data nyata untuk sweep LR dan validasi pipeline.",
    "pretrain": "Mulai atau lanjutkan pretraining utama dengan LR schedule WSD dan checkpointing atomik.",
    "sft": "Fine-tuning chat, penalaran (mode think), dan pemanggilan fungsi (tool calling).",
    "eval": "Evaluasi otomatis kepatuhan mode, akurasi pemanggilan tool, dan kualitas respons.",
    "chat": "REPL percakapan interaktif di terminal dengan eksekusi tool sandbox otomatis.",
    "export": "Ekspor checkpoint ke folder standar Hugging Face lengkap dengan template Jinja.",
    "runs": "Tampilkan tabel ringkasan semua run training yang pernah dijalankan.",
    "doctor": "Pemeriksaan kesiapan sistem, GPU, PyTorch, dan dependensi lingkungan.",
    "replan": "Ubah parameter proyek (ukuran, GPU, target token) dan hitung ulang rencana.",
    "config": "Kelola konfigurasi settings (show, set, auto) sesuai aturan Advisor Bagian 18.",
    "guide": "Tampilkan panduan urut langkah demi langkah dari awal hingga selesai.",
    "help": "Bantuan penggunaan perintah CLI.",
}


def print_command_help(command: Optional[str] = None) -> None:
    """Tampilkan bantuan perintah."""
    if command and command in COMMAND_HELP_MAP:
        print(f"\nPerintah: python main.py {command}")
        print(f"Penjelasan: {COMMAND_HELP_MAP[command]}")
    else:
        print("\n=== DAFTAR PERINTAH TOOLKIT LLMKIT ===")
        for cmd, desc in COMMAND_HELP_MAP.items():
            print(f"  {cmd.ljust(12)} : {desc}")
        print("\nKetik 'python main.py help <perintah>' untuk detail perintah tertentu.")


def print_full_guide() -> None:
    """Tampilkan panduan lengkap berurutan 1 sampai 10."""
    print("\n" + "=" * 70)
    print("PANDUAN ALUR LENGKAP: MEMBUAT LLM DARI AWAL (STEP 1 - 10)")
    print("=" * 70)

    for item in STEP_BY_STEP_GUIDE:
        print(f"\n[{item['langkah']}] {item['judul']}")
        print(f"    Perintah    : {item['perintah']}")
        print(f"    Deskripsi   : {item['deskripsi']}")
        print(f"    Hasil       : {item['ekspektasi']}")
        print(f"    Solusi Error: {item['kalau_gagal']}")

    print("\n" + HONESTY_NOTES)
    print(DATASET_CANDIDATES)
