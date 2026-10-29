# LLMKit

[English](#english) | [Bahasa Indonesia](#bahasa-indonesia)

---

<a name="english"></a>
## English

# LLMKit - End-to-End Small Language Model (SLM) Toolkit from Scratch

A comprehensive end-to-end Python & PyTorch toolkit for designing, pretraining, evaluating, and fine-tuning Small Language Models (SLM) based on the **Qwen3** architecture from scratch to Supervised Fine-Tuning (SFT) with reasoning (`<think>`) and function calling (`<tool_call>`) capabilities.

---

### 🌐 Bilingual CLI & Language Selection

LLMKit provides full bilingual support across all terminal interactions, guides, error diagnostics, and planning tables:
- **Interactive Startup Prompt**: On first interactive run, LLMKit asks for your language preference (English or Bahasa Indonesia) and saves it automatically.
- **Language Flag (`--lang`)**: You can specify `--lang en` or `--lang id` with any command:
  ```bash
  python main.py --lang en doctor
  python main.py --lang id plan 0.6B
  ```
- **Dedicated Language Subcommand (`lang`)**: Check or switch the active language preference anytime:
  ```bash
  python main.py lang        # Displays current language
  python main.py lang en     # Switches and persists language to English
  python main.py lang id     # Switches and persists language to Bahasa Indonesia
  ```
- **Environment Variable**: Set `LLMKIT_LANG=en` or `LLMKIT_LANG=id` to override preferences across scripts.

---

### 🚀 Key Features

- **Modern Qwen3 Architecture**: Grouped-Query Attention (GQA), RMSNorm, SwiGLU FFN, RoPE context extension ($\theta = 1,000,000$), and tied word embeddings.
- **Intelligent Planner & Advisor**:
  - Calculates parameter counts, static & activation VRAM allocation, FLOPs, training duration, and cloud GPU rental costs (A100, H100, RTX 4090, L4, etc.).
  - 18 automated Advisor rules (**A01 - A18**) to safeguard against common failure modes (OOM, LR mismatch, Chinchilla ratio violations, data leakage).
- **Bilingual Tokenizer (ID + EN)**:
  - Custom Byte-level BPE with digit splitting (`\d`) and special control tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `<tool_response>`).
- **High-Performance Data Pipeline**:
  - Raw JSONL text conversion to binary shards (`uint16` / `uint32`) with zero-copy memory mapping (`numpy.memmap`).
  - Strict n-gram test leakage filtering and exact pretrain document deduplication.
- **Resilient Pretraining Engine**:
  - Warmup-Stable-Decay (WSD) learning rate schedule.
  - Atomic checkpoint rotation with `COMPLETE` integrity markers.
  - Built-in supervisor auto-restart (`--auto`) for hardware faults and spot instance interruptions.
- **SFT with Strict Loss Masking**:
  - Loss calculated only on assistant tokens (reasoning `<think>` blocks and final answers); prompt tokens and tool responses are masked with `-100`.
  - Visual terminal preview of masking spans prior to training commencement.
- **Tool Calling Sandbox & Interactive Chat REPL**:
  - Interactive conversation REPL in terminal with real-time sandbox execution (safe math evaluator, knowledge retrieval, live weather mock).
- **Standard Hugging Face & GGUF/Ollama Export**:
  - Direct Hugging Face format export with `model.safetensors`, `tokenizer.json`, and Jinja chat templates.
  - Conversion guides for llama.cpp (`.gguf`) and Ollama `Modelfile`.

---

### 📋 System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu/Debian), or macOS.
- **Python**: 3.10 - 3.13.
- **GPU**: NVIDIA GPU with 8 GB+ VRAM recommended (RTX 3060/4060+ for local pilots, A100/H100 for full pretraining). CPU mode supported for local dry-runs and pipeline checks.
- **PyTorch**: 2.1+ with CUDA support.

---

### 🛠️ Installation & Verification

1. Install required Python packages:
   ```bash
   pip install torch transformers tokenizers safetensors numpy rich tqdm jinja2
   ```

2. Verify your hardware and software readiness:
   ```bash
   python main.py doctor
   ```
   If everything is set up properly, the system reports: `Diagnostic Status: [ALL SYSTEMS READY]`.

---

### 📖 Step-by-Step Usage Guide

#### 1. Plan Your Model (Quick Calculator)
Simulate parameter count, token budgets, memory requirements, and rental costs without initializing a project directory:
```bash
python main.py plan 0.6B --mode balanced --gpu H100
```
To test hyperparameter override impacts (delta simulation):
```bash
python main.py plan 0.6B --set context_final=8192
```

#### 2. Initialize a New Project
Start a new model project with the interactive questionnaire:
```bash
python main.py init
```
This generates `project.json` and scaffolds data directories:
- `data/pretrain/` (en_web, id_web, code, math, books_wiki, synthetic, dialog)
- `data/sft/` (chat, multiturn, think, tool_call, identity, code, refusal)

#### 3. Validate Dataset Integrity
After placing `.jsonl` data into the folders, validate format and quotas:
```bash
python main.py check
```
Generate bilingual identity tuning samples automatically:
```bash
python main.py make-identity
```

#### 4. Train the Tokenizer
Train a custom Byte-level BPE tokenizer optimized for Indonesian and English:
```bash
python main.py tokenizer
```

#### 5. Tokenize into Binary Shards (Prep)
Compile text documents into binary memory-mapped shards:
```bash
python main.py prep
```

#### 6. Benchmark GPU Hardware
Measure real-world throughput and identify optimal micro-batch size without OOM:
```bash
python main.py bench
```

#### 7. Pilot Run (0.05B Mini Model)
Execute a pilot run on real data for learning rate sweep and pipeline validation:
```bash
python main.py pilot
```

#### 8. Main Pretraining
Launch pretraining with automatic checkpointing:
```bash
# Pretraining with auto-restart supervisor
python main.py pretrain --auto

# Clean new run
python main.py pretrain --new
```

#### 9. Supervised Fine-Tuning (SFT)
Fine-tune on conversational, reasoning, and tool-calling data:
```bash
# SFT from the latest pretrain run checkpoint
python main.py sft

# SFT from a specific run
python main.py sft --from run_001
```

#### 10. Evaluation & Run Comparison
Benchmark loss, perplexity, and format compliance:
```bash
python main.py eval --run run_001
python main.py eval --compare run_001 run_002
```

#### 11. Interactive Chat REPL
Chat directly with your model in terminal with automatic tool execution:
```bash
python main.py chat
```

#### 12. Export to Hugging Face & GGUF/Ollama
Export weights to Hugging Face directory:
```bash
python main.py export --run run_002 --out export/my_model
```
Convert to GGUF format for llama.cpp / Ollama:
```bash
python <path-to-llama.cpp>/convert_hf_to_gguf.py export/my_model --outfile my_model.gguf --outtype bf16
ollama create my_model -f Modelfile
ollama run my_model
```

---

### 🗂️ CLI Command Reference

| Command | Description | Example |
|---|---|---|
| `lang` | View or switch CLI interface language (en/id) | `python main.py lang en` |
| `init` | Initialize a new project and scaffold data folders | `python main.py init` |
| `plan` | Fast calculator for parameters, tokens, duration, and cost | `python main.py plan 0.6B --mode balanced --gpu H100` |
| `status` | Auto-detect project progress from disk and advise next steps | `python main.py status` |
| `check` | Validate format integrity and token quotas of raw JSONL files | `python main.py check` |
| `make-identity` | Generate synthetic bilingual identity tuning samples | `python main.py make-identity` |
| `tokenizer` | Train custom Byte-level BPE bilingual tokenizer | `python main.py tokenizer` |
| `prep` | Tokenize raw text into memory-mapped uint16 shards | `python main.py prep` |
| `bench` | Benchmark GPU speed and determine optimal micro-batch size | `python main.py bench` |
| `pilot` | Run 0.05B pilot model for LR sweep and pipeline verification | `python main.py pilot` |
| `pretrain` | Start or resume main pretraining loop | `python main.py pretrain --auto` |
| `sft` | Supervised Fine-Tuning loop (chat, reasoning, tool-call) | `python main.py sft --from run_001` |
| `eval` | Automated evaluation and side-by-side run comparison | `python main.py eval --compare run_001 run_002` |
| `chat` | Interactive terminal chat REPL with tool sandbox execution | `python main.py chat` |
| `export` | Export trained model checkpoints to Hugging Face format | `python main.py export --out export/model` |
| `runs` | List summary table of all recorded training runs | `python main.py runs` |
| `doctor` | Diagnose hardware, CUDA, PyTorch, and dependency readiness | `python main.py doctor` |
| `replan` | Recompute project plan with parameter updates | `python main.py replan` |
| `config` | View, lock, or restore project settings via Advisor | `python main.py config show` |
| `guide` | Display sequential 10-step guide in terminal | `python main.py guide` |

---

<a name="bahasa-indonesia"></a>
## Bahasa Indonesia

# LLMKit - Toolkit Pembuatan Small Language Model (SLM) dari Nol

Toolkit end-to-end berbasis Python & PyTorch untuk merancang, melatih, mengevaluasi, dan mengekspor Small Language Model (SLM) berbasis arsitektur **Qwen3** dari tahap pretraining hingga Supervised Fine-Tuning (SFT) dengan mode penalaran (*reasoning*) dan pemanggilan fungsi (*tool calling*).

---

### 🌐 Antarmuka CLI Dwibahasa & Pengaturan Bahasa

LLMKit mendukung dwibahasa penuh (Bahasa Indonesia & English) pada seluruh output terminal, petunjuk, diagnostik error, dan tabel rencana:
- **Pilihan Bahasa Interaktif Awal**: Saat pertama kali dijalankan di terminal interaktif, LLMKit menampilkan opsi pemilihan bahasa dan menyimpannya secara otomatis.
- **Opsi Flag Bahasa (`--lang`)**: Anda dapat menyertakan `--lang id` atau `--lang en` pada perintah apa pun:
  ```bash
  python main.py --lang id doctor
  python main.py --lang en plan 0.6B
  ```
- **Subperintah Khusus Bahasa (`lang`)**: Periksa atau ubah bahasa aktif kapan saja:
  ```bash
  python main.py lang        # Menampilkan bahasa saat ini
  python main.py lang id     # Mengubah dan menyimpan bahasa ke Bahasa Indonesia
  python main.py lang en     # Mengubah dan menyimpan bahasa ke English
  ```
- **Environment Variable**: Tetapkan `LLMKIT_LANG=id` atau `LLMKIT_LANG=en` untuk kebutuhan automasi skrip.

---

### 🚀 Fitur Utama

- **Arsitektur Qwen3 Modern**: Grouped-Query Attention (GQA), RMSNorm, SwiGLU FFN, RoPE context extension ($\theta = 1,000,000$), dan tie word embeddings.
- **Planner & Advisor Cerdas**:
  - Menghitung jumlah parameter, alokasi VRAM statis & aktivasi, FLOPs, estimasi waktu, serta biaya sewa GPU (A100, H100, RTX 4090, L4, dll.).
  - 18 aturan Advisor otomatis (**A01 - A18**) untuk mencegah kesalahan umum (OOM, LR mismatch, rasio Chinchilla, data leakage).
- **Tokenizer Dwibahasa (ID + EN)**:
  - Byte-level BPE dengan pemecahan digit (`\d`) dan special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `<tool_response>`).
- **Pipeline Data Cepat & Efisien**:
  - Konversi teks mentah JSONL ke shard biner `uint16` / `uint32` dengan zero-copy memory mapping (`numpy.memmap`).
  - Filtering kebocoran n-gram dan dedup exact data pretrain.
- **Pretraining Tangguh (WSD Schedule)**:
  - Warmup-Stable-Decay (WSD) learning rate schedule.
  - Rotasi checkpoint atomik dengan integritas marker `COMPLETE`.
  - Supervisor auto-restart (`--auto`) untuk menangani gangguan hardware / OOM.
- **SFT dengan Strict Loss Masking**:
  - Loss hanya dihitung pada token assistant (di dalam blok `<think>` dan jawaban akhir); token prompt dan tool response dimaskir (`-100`).
  - Preview visual rentang masking sebelum pelatihan dimulai.
- **Tool Calling Sandbox & Chat REPL**:
  - REPL percakapan interaktif langsung di terminal dengan eksekusi alat otomatis (kalkulator aman, pencarian knowledge base, cuaca).
- **Ekspor Standar Hugging Face & GGUF/Ollama**:
  - Format output Hugging Face dengan `model.safetensors`, `tokenizer.json`, dan template Jinja.
  - Petunjuk konversi ke llama.cpp (GGUF) dan pembuatan Modelfile Ollama.

---

### 📋 Persyaratan Sistem

- **Sistem Operasi**: Windows 10/11, Linux (Ubuntu/Debian), atau macOS.
- **Python**: Versi 3.10 - 3.13.
- **GPU**: NVIDIA GPU dengan VRAM minimal 8 GB disarankan (RTX 3060/4060 ke atas untuk uji coba lokal, A100/H100 untuk pretraining skala penuh). Mode CPU didukung untuk dry-run lokal dan pengujian alur.
- **PyTorch**: 2.1+ dengan dukungan CUDA.

---

### 🛠️ Instalasi Dependensi & Verifikasi

1. Pasang dependensi Python yang dibutuhkan:
   ```bash
   pip install torch transformers tokenizers safetensors numpy rich tqdm jinja2
   ```

2. Lakukan verifikasi kesiapan sistem dan hardware:
   ```bash
   python main.py doctor
   ```
   Jika semua dependensi dan sistem siap, sistem akan menampilkan status: `Diagnostic Status: [ALL SYSTEMS READY]`.

---

### 📖 Panduan Penggunaan Lengkap (Step-by-Step)

#### 1. Perencanaan Model (Kalkulator Cepat)
Simulasikan arsitektur, kebutuhan data, memori GPU, dan biaya sewa tanpa membuat folder proyek:
```bash
python main.py plan 0.6B --mode balanced --gpu H100
```
Untuk menguji dampak perubahan parameter (override delta):
```bash
python main.py plan 0.6B --set context_final=8192
```

#### 2. Inisialisasi Proyek Baru
Mulai proyek baru dengan panduan interaktif terminal:
```bash
python main.py init
```
Perintah ini akan membuat file `project.json` dan struktur folder data:
- `data/pretrain/` (en_web, id_web, code, math, books_wiki, synthetic, dialog)
- `data/sft/` (chat, multiturn, think, tool_call, identity, code, refusal)

#### 3. Cek & Validasi Integritas Data Mentah
Setelah menaruh file `.jsonl` pada folder `data/`, validasi format dan kecukupan volumenya:
```bash
python main.py check
```
Untuk membuat data identitas sintetis model dwibahasa secara otomatis:
```bash
python main.py make-identity
```

#### 4. Pelatihan Tokenizer
Latih tokenizer Byte-level BPE khusus kosakata Indonesia dan Inggris:
```bash
python main.py tokenizer
```

#### 5. Konversi Data ke Shard Biner (Prep)
Lakukan tokenisasi data mentah ke format binary shard uint16:
```bash
python main.py prep
```

#### 6. Benchmark Hardware GPU
Ukur performa throughput GPU riil dan temukan micro-batch size optimal tanpa OOM:
```bash
python main.py bench
```

#### 7. Uji Coba Pilot (Model Mini 0.05B)
Jalankan model uji coba pada data riil untuk sweep learning rate dan verifikasi pipeline:
```bash
python main.py pilot
```

#### 8. Pretraining Utama
Mulai pelatihan pretraining model dari nol:
```bash
# Jalankan pretraining dengan supervisor auto-resume jika terjadi crash/OOM
python main.py pretrain --auto

# Memulai run baru secara bersih
python main.py pretrain --new
```

#### 9. Supervised Fine-Tuning (SFT)
Latih model pretrain dengan data instruksi percakapan, penalaran, dan tool calling:
```bash
# SFT dari checkpoint run pretrain terakhir
python main.py sft

# SFT dari run tertentu
python main.py sft --from run_001
```

#### 10. Evaluasi & Perbandingan Run
Uji performa dan kepatuhan format model yang telah dilatih:
```bash
python main.py eval --run run_001
python main.py eval --compare run_001 run_002
```

#### 11. Chat Interaktif di Terminal
Uji coba langsung respons model di terminal dengan eksekusi sandbox otomatis:
```bash
python main.py chat
```

#### 12. Ekspor ke Hugging Face & Format GGUF/Ollama
Ekspor bobot final model ke format standar Hugging Face:
```bash
python main.py export --run run_002 --out export/my_model
```
Konversi ke GGUF (llama.cpp) & jalankan di Ollama:
```bash
python <path-to-llama.cpp>/convert_hf_to_gguf.py export/my_model --outfile my_model.gguf --outtype bf16
ollama create my_model -f Modelfile
ollama run my_model
```

---

### 🗂️ Referensi Perintah CLI (`main.py`)

| Perintah | Deskripsi Singkat | Contoh Penggunaan |
|---|---|---|
| `lang` | Tampilkan atau ubah preferensi bahasa antarmuka CLI | `python main.py lang id` |
| `init` | Inisialisasi proyek baru dan buat scaffolding folder | `python main.py init` |
| `plan` | Kalkulator parameter, token, waktu, dan estimasi biaya | `python main.py plan 0.6B --mode balanced --gpu H100` |
| `status` | Cek status kemajuan proyek di disk dan saran langkah berikut | `python main.py status` |
| `check` | Validasi integritas dan volume dataset JSONL | `python main.py check` |
| `make-identity` | Buat 50 sampel data identitas sintetis dwibahasa | `python main.py make-identity` |
| `tokenizer` | Latih tokenizer Byte-level BPE dwibahasa (ID+EN) | `python main.py tokenizer` |
| `prep` | Tokenisasi data teks ke file biner uint16 (`.bin`) | `python main.py prep` |
| `bench` | Benchmark kecepatan GPU dan ukur micro-batch optimal | `python main.py bench` |
| `pilot` | Pilot run model 0.05B untuk sweep LR pada data riil | `python main.py pilot` |
| `pretrain` | Mulai atau lanjutkan loop pretraining utama | `python main.py pretrain --auto` |
| `sft` | Pelatihan SFT (chat, reasoning `<think>`, function call) | `python main.py sft --from run_001` |
| `eval` | Evaluasi otomatis kepatuhan format & perbandingan run | `python main.py eval --compare run_001 run_002` |
| `chat` | REPL chat interaktif terminal dengan eksekusi tool | `python main.py chat` |
| `export` | Ekspor checkpoint ke format resmi Hugging Face | `python main.py export --out export/model` |
| `runs` | Tampilkan ringkasan tabel semua run pelatihan | `python main.py runs` |
| `doctor` | Periksa kesiapan lingkungan, GPU, CUDA, dan dependensi | `python main.py doctor` |
| `replan` | Rencanakan ulang parameter proyek dengan mempertahankan override | `python main.py replan` |
| `config` | Tampilkan atau ubah konfigurasi settings/advisor | `python main.py config show` |
| `guide` | Tampilkan panduan langkah demi langkah di terminal | `python main.py guide` |

---

### 🧪 Menjalankan Unit Tests

Toolkit dilengkapi dengan unit test menyeluruh untuk memastikan kebenaran rumus arsitektur, template Jinja, mask-loss span, rotasi checkpoint, sistem internasionalisasi dwibahasa, dan integrasi pipeline end-to-end:

```bash
pytest -o pythonpath=. -v
```

---

### 📁 Struktur Direktori

```text
├── llmkit/                   # Modul inti toolkit
│   ├── advisor.py            # Aturan validasi advisor dwibahasa (A01-A18)
│   ├── bench.py              # Benchmark hardware & memory
│   ├── chat.py               # Interactive chat REPL dwibahasa
│   ├── chat_template.py      # Jinja template & loss masking span
│   ├── check.py              # Validator data mentah
│   ├── checkpoint.py         # Save/load checkpoint atomik (safetensors/pt)
│   ├── cli.py                # Command-line interface parser & handlers
│   ├── dataset.py            # Streaming memmap dataset loader
│   ├── data_spec.py          # Definisi kuota data & generator identity
│   ├── doctor.py             # Diagnostik hardware & dependensi dwibahasa
│   ├── evaluate.py           # Evaluasi pretrain & SFT dwibahasa
│   ├── export.py             # Ekspor ke format Hugging Face
│   ├── gpu_table.py          # Database spesifikasi GPU & deteksi VRAM
│   ├── guide.py              # Panduan urut langkah demi langkah dwibahasa
│   ├── i18n.py               # Sistem internasionalisasi dwibahasa (EN/ID)
│   ├── model.py              # Konfigurasi arsitektur Qwen3
│   ├── pilot.py              # Mini model pilot test
│   ├── planner.py            # Perhitungan parameter, token, FLOPs, biaya
│   ├── prep.py               # Tokenisasi teks ke shard biner
│   ├── settings.py           # Pengaturan konfigurasi & rules engine
│   ├── sft.py                # Supervised Fine-Tuning loop
│   ├── state.py              # Pelacak status proyek di disk
│   ├── tokenizer_train.py    # Pelatihan BPE tokenizer
│   ├── tools_sandbox.py      # Sandbox eksekusi function call
│   ├── train.py              # Pretraining engine dengan WSD schedule
│   └── ui.py                 # Formatting UI Rich terminal dwibahasa
├── tests/                    # Rangkaian unit test dan test fixture
├── main.py                   # Entry point CLI utama
├── plan.md                   # Dokumen spesifikasi teknis lengkap
├── .gitignore                # Filter file cache, checkpoints, skills, & dataset
└── README.md                 # Dokumentasi proyek dwibahasa (EN & ID)
```

---

### 📄 Lisensi

Proyek ini dilisensikan di bawah lisensi **Apache License 2.0** - lihat file [LICENSE](LICENSE) untuk detail selengkapnya.
Didistribusikan untuk riset mandiri, eksperimen pembuatan LLM, dan keperluan edukasi.
