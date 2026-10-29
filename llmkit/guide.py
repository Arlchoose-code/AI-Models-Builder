"""Teks panduan lengkap dari nol sampai export (guide), bantuan perintah (help), dan daftar dataset kandidat (Bagian 13.4, 14, 17).
Mendukung dwibahasa (Bahasa Indonesia & English).
"""

from __future__ import annotations

from typing import Any, Optional
from llmkit.i18n import get_lang, tr

DATASET_CANDIDATES_ID = """=== KANDIDAT DATASET PRETRAIN & SFT ===
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

DATASET_CANDIDATES_EN = """=== CANDIDATE DATASETS FOR PRETRAIN & SFT ===
Note: Verify that the datasets are currently available on Hugging Face Hub, and check licenses before use!

Pretrain:
- en_web / en_hq : HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus
- id_web / id_hq : HuggingFaceFW/fineweb-2 (ind_Latn subset), wikimedia/wikipedia (id subset)
- code           : bigcode/the-stack-v2, Python-edu subset from smollm-corpus
- math           : HuggingFaceTB/finemath, open-web-math/open-web-math
- books_wiki     : wikimedia/wikipedia (en & id)
- synthetic      : Cosmopedia (part of smollm-corpus) + EN->ID translations
- eval_leak      : hellaswag, mmlu, indommlu (evaluation benchmarks to filter out from training data)

SFT:
- chat / multiturn : HuggingFaceTB/smoltalk
- tool_call        : NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE
- think            : open-thoughts/OpenThoughts-114k, open-r1/OpenR1-Math-220k
- Indonesian data  : Created via translations or synthetic generation from frontier models.
"""

HONESTY_NOTES_ID = """=== CATATAN KEJUJURAN UNTUK PENGGUNA ===
1. Angka perkiraan (rasio token, LR, TFLOPs efektif, harga GPU) adalah TITIK AWAL, bukan kepastian mutlak.
   Perintah 'bench' dan 'pilot' yang akan menentukan angka kecepatan dan LR nyata.
2. Melatih model 0.6B dari nol dengan <= 100B token hampir pasti kalah cerdas dibanding model frontier seperti
   Qwen3-0.6B resmi (yang dilatih hingga puluhan triliun token). Jalur paling efisien untuk hasil produk adalah
   continued pretraining dari model dasar. Toolkit ini sangat tepat untuk eksperimen, riset mandiri, dan edukasi mendalam.
3. Kualitas data jauh lebih menentukan hasil akhir daripada sekadar kuantitas token mentah.
"""

HONESTY_NOTES_EN = """=== HONESTY NOTES FOR USERS ===
1. Estimates (token ratios, LR, effective TFLOPs, GPU prices) are STARTING POINTS, not absolute truths.
   The 'bench' and 'pilot' commands determine your actual speeds and realistic learning rates.
2. Training a 0.6B model from scratch with <= 100B tokens will almost certainly not match frontier models like
   official Qwen3-0.6B (which trained on tens of trillions of tokens). The most efficient route for production
   is continued pretraining from a base checkpoint. This toolkit is ideal for deep research, self-education, and full experiments.
3. Data quality matters significantly more than raw token volume.
"""

STEP_BY_STEP_GUIDE_ID = [
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

STEP_BY_STEP_GUIDE_EN = [
    {
        "langkah": 1,
        "judul": "Project Initialization (init)",
        "perintah": "python main.py init",
        "deskripsi": "Answer configuration prompts (model size, GPU, token target). Computes the full plan and scaffolds data directories.",
        "ekspektasi": "Creates project.json and folders data/pretrain/ and data/sft/ with detailed README.txt files.",
        "kalau_gagal": "Ensure write permissions in working directory and provide valid numerical inputs."
    },
    {
        "langkah": 2,
        "judul": "Dataset Ingestion & Validation (check)",
        "perintah": "python main.py check",
        "deskripsi": "Populate data directories with .jsonl files according to specifications, then validate volume and format.",
        "ekspektasi": "All pretrain and SFT folders reach >=90% allocation target with 0 JSON formatting errors.",
        "kalau_gagal": "Review logs/check_errors.txt to diagnose malformed JSON rows."
    },
    {
        "langkah": 3,
        "judul": "Tokenizer Training (tokenizer)",
        "perintah": "python main.py tokenizer",
        "deskripsi": "Train a custom Byte-level BPE tokenizer (EN + ID) with digit splitting and special tokens.",
        "ekspektasi": "Outputs tokenizer/tokenizer.json and token fertility reports per word.",
        "kalau_gagal": "Ensure at least 1-2 GB of clean text is available in data/pretrain/."
    },
    {
        "langkah": 4,
        "judul": "Shard Preparation & Tokenization (prep)",
        "perintah": "python main.py prep",
        "deskripsi": "Tokenize all raw JSONL documents into uint16 binary shards (.bin), filter benchmark leaks, and exact dedup.",
        "ekspektasi": "Generates tokenized/<source>/shard_*.bin and tokenized/_manifest.json.",
        "kalau_gagal": "Check free disk space (requires ~2 bytes per token)."
    },
    {
        "langkah": 5,
        "judul": "Hardware Speed Benchmark (bench)",
        "perintah": "python main.py bench",
        "deskripsi": "Benchmark GPU throughput (tok/sec) across micro-batch sizes without OOM and measure real peak speed.",
        "ekspektasi": "Outputs bench.json with realistic training duration and cost projections.",
        "kalau_gagal": "If CUDA OOM occurs, bench automatically scales down batch sizes."
    },
    {
        "langkah": 6,
        "judul": "Mini Model Pilot Run (pilot)",
        "perintah": "python main.py pilot",
        "deskripsi": "Train a mini 0.05B model on real shards for learning rate sweep and checkpoint resume verification.",
        "ekspektasi": "Outputs pilot/result.json with optimal learning rate recommendations.",
        "kalau_gagal": "Verify that binary shards in tokenized/ can be memory-mapped."
    },
    {
        "langkah": 7,
        "judul": "Full Pretraining (pretrain)",
        "perintah": "python main.py pretrain",
        "deskripsi": "Execute main pretraining loop with WSD learning rate schedule, atomic checkpoint rotation, and spike rollback.",
        "ekspektasi": "Trains model up to target tokens. Checkpoints saved under runs/run_XXX/ckpt/.",
        "kalau_gagal": "If disconnected or interrupted, resume seamlessly with: python main.py pretrain --resume."
    },
    {
        "langkah": 8,
        "judul": "Supervised Fine-Tuning (sft)",
        "perintah": "python main.py sft",
        "deskripsi": "Fine-tune pretrained weights on conversations, reasoning (<think> tags), and function calling.",
        "ekspektasi": "Completes SFT for 2-3 epochs with strict loss masking on assistant turns only.",
        "kalau_gagal": "Ensure valid entries in data/sft/tool_call/ and data/sft/chat/."
    },
    {
        "langkah": 9,
        "judul": "Comprehensive Evaluation (eval)",
        "perintah": "python main.py eval",
        "deskripsi": "Evaluate mode compliance, tool call schemas, language matching, and degeneration loops.",
        "ekspektasi": "Prints metrics score table and writes runs/run_XXX/eval.json.",
        "kalau_gagal": "Inspect failed_examples in eval.json to identify response degradation."
    },
    {
        "langkah": 10,
        "judul": "Chat & Model Export (chat & export)",
        "perintah": "python main.py chat  /  python main.py export",
        "deskripsi": "Interact live in terminal with tool sandbox execution, then export weights to Hugging Face and GGUF / Ollama.",
        "ekspektasi": "Folder export/<name>/ ready for local inference or llama.cpp conversion.",
        "kalau_gagal": "Ensure the Jinja chat template matches training special tokens."
    }
]

COMMAND_HELP_MAP_ID = {
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
    "lang": "Atur atau tampilkan preferensi bahasa antarmuka CLI (en/id).",
    "help": "Bantuan penggunaan perintah CLI.",
}

COMMAND_HELP_MAP_EN = {
    "init": "Initialize a new project: interactive prompts, calculate full plan, scaffold data folders.",
    "plan": "Quick calculator for parameters, tokens, duration, and cost without creating files. E.g.: python main.py plan 0.6B --mode balanced --gpu H100",
    "status": "Auto-detect project progress from files on disk and recommend next action.",
    "check": "Validate raw pretrain & SFT JSONL files, estimate tokens, and verify format integrity.",
    "tokenizer": "Train custom Byte-level BPE bilingual tokenizer (EN+ID) with digit splitting and special tokens.",
    "prep": "Tokenize raw documents into uint16 binary shards, exact dedup, and validation splits.",
    "bench": "Benchmark GPU hardware throughput (tokens/sec) and identify optimal micro-batch size.",
    "pilot": "Run mini 0.05B pilot model on real data for learning rate sweep and pipeline validation.",
    "pretrain": "Start or resume pretraining loop with WSD learning rate schedule and atomic checkpoints.",
    "sft": "Supervised Fine-Tuning for chat, reasoning (<think> mode), and function calling.",
    "eval": "Automated evaluation of mode compliance, tool call accuracy, and response quality.",
    "chat": "Interactive terminal chat REPL with automatic tool sandbox execution.",
    "export": "Export checkpoints to official Hugging Face standard directory with Jinja template.",
    "runs": "Display a summary table of all training runs recorded on disk.",
    "doctor": "Comprehensive environment check for GPU, CUDA, PyTorch, and dependencies.",
    "replan": "Modify project answers (size, GPU, tokens) and recompute plan with override persistence.",
    "config": "Manage project settings (show, set, auto) with Advisor rule engine.",
    "guide": "Display the comprehensive 10-step end-to-end training guide.",
    "lang": "View or switch CLI interface language (en/id).",
    "help": "Help and usage manual for CLI commands.",
}

# Compatibility aliases
DATASET_CANDIDATES = DATASET_CANDIDATES_ID
HONESTY_NOTES = HONESTY_NOTES_ID
STEP_BY_STEP_GUIDE = STEP_BY_STEP_GUIDE_ID
COMMAND_HELP_MAP = COMMAND_HELP_MAP_ID


def get_dataset_candidates() -> str:
    return DATASET_CANDIDATES_EN if get_lang() == "en" else DATASET_CANDIDATES_ID


def get_honesty_notes() -> str:
    return HONESTY_NOTES_EN if get_lang() == "en" else HONESTY_NOTES_ID


def get_step_by_step_guide() -> list[dict[str, Any]]:
    return STEP_BY_STEP_GUIDE_EN if get_lang() == "en" else STEP_BY_STEP_GUIDE_ID


def get_command_help_map() -> dict[str, str]:
    return COMMAND_HELP_MAP_EN if get_lang() == "en" else COMMAND_HELP_MAP_ID


def print_command_help(command: Optional[str] = None) -> None:
    """Tampilkan bantuan perintah dwibahasa."""
    cmd_map = get_command_help_map()
    if command and command in cmd_map:
        cmd_label = tr("Perintah", "Command")
        desc_label = tr("Penjelasan", "Description")
        print(f"\n{cmd_label}: python main.py {command}")
        print(f"{desc_label}: {cmd_map[command]}")
    else:
        title = tr("=== DAFTAR PERINTAH TOOLKIT LLMKIT ===", "=== LLMKIT TOOLKIT COMMAND LIST ===")
        footer = tr(
            "Ketik 'python main.py help <perintah>' untuk detail perintah tertentu.",
            "Type 'python main.py help <command>' for specific command details."
        )
        print(f"\n{title}")
        for cmd, desc in cmd_map.items():
            print(f"  {cmd.ljust(12)} : {desc}")
        print(f"\n{footer}")


def print_full_guide() -> None:
    """Tampilkan panduan lengkap berurutan 1 sampai 10 dalam bahasa aktif."""
    guide_list = get_step_by_step_guide()
    title = tr(
        "PANDUAN ALUR LENGKAP: MEMBUAT LLM DARI AWAL (STEP 1 - 10)",
        "FULL STEP-BY-STEP GUIDE: BUILDING AN LLM FROM SCRATCH (STEP 1 - 10)"
    )
    cmd_lbl = tr("Perintah    :", "Command     :")
    desc_lbl = tr("Deskripsi   :", "Description :")
    res_lbl = tr("Hasil       :", "Outcome     :")
    err_lbl = tr("Solusi Error:", "Error Fix   :")

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for item in guide_list:
        print(f"\n[{item['langkah']}] {item['judul']}")
        print(f"    {cmd_lbl} {item['perintah']}")
        print(f"    {desc_lbl} {item['deskripsi']}")
        print(f"    {res_lbl} {item['ekspektasi']}")
        print(f"    {err_lbl} {item['kalau_gagal']}")

    print("\n" + get_honesty_notes())
    print(get_dataset_candidates())
