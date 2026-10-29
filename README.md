# LLMKit - Toolkit Pembuatan Small Language Model (SLM) dari Nol

Toolkit end-to-end berbasis Python & PyTorch untuk merancang, melatih, mengevaluasi, dan mengekspor Small Language Model (SLM) berbasis arsitektur **Qwen3** dari tahap pretraining hingga Supervised Fine-Tuning (SFT) dengan mode penalaran (*reasoning*) dan pemanggilan fungsi (*tool calling*).

---

## 🚀 Fitur Utama

- **Arsitektur Qwen3 Modern**: Mendukung Grouped-Query Attention (GQA), RMSNorm, SwiGLU FFN, RoPE context extension ($\theta = 1,000,000$), dan tie word embeddings.
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

## 📋 Persyaratan Sistem

- **Sistem Operasi**: Windows 10/11, Linux (Ubuntu/Debian), atau macOS.
- **Python**: Versi 3.11 - 3.13.
- **GPU**: NVIDIA GPU dengan VRAM minimal 8 GB disarankan (RTX 3060/4060 ke atas untuk uji coba lokal, A100/H100 untuk pretraining skala penuh).
- **PyTorch**: 2.1+ dengan dukungan CUDA.

---

## 🛠️ Instalasi Dependensi

Pasang dependensi Python yang dibutuhkan:

```bash
pip install torch transformers tokenizers safetensors numpy rich tqdm jinja2
```

Lakukan verifikasi kesiapan sistem dan hardware Anda:

```bash
python main.py doctor
```

Jika semua dependensi dan GPU siap, sistem akan menampilkan status: `[SEMUA SISTEM SIAP]`.

---

## 📖 Panduan Penggunaan Lengkap (Step-by-Step)

### 1. Perencanaan Model (Kalkulator Cepat)
Simulasikan arsitektur, kebutuhan data, memori GPU, dan biaya sewa tanpa membuat folder proyek:
```bash
python main.py plan 0.6B --mode balanced --gpu H100
```
Untuk menguji dampak perubahan parameter (override delta):
```bash
python main.py plan 0.6B --set context_final=8192
```

---

### 2. Inisialisasi Proyek Baru
Mulai proyek baru dengan panduan interaktif terminal:
```bash
python main.py init
```
Perintah ini akan membuat file `project.json` dan struktur folder data:
- `data/pretrain/` (en_web, id_web, code, math, books_wiki, synthetic, dialog)
- `data/sft/` (chat, multiturn, think, tool_call, identity, code, refusal)

---

### 3. Cek & Validasi Integritas Data Mentah
Setelah menaruh file `.jsonl` pada folder `data/`, validasi format dan kecukupan volumenya:
```bash
python main.py check
```

Untuk membuat data identitas sintetis model dwibahasa secara otomatis:
```bash
python main.py make-identity
```

---

### 4. Pelatihan Tokenizer
Latih tokenizer Byte-level BPE khusus kosakata Indonesia dan Inggris:
```bash
python main.py tokenizer
```
Output tersimpan di folder `tokenizer/` lengkap dengan metrik fertilitas kata.

---

### 5. Konversi Data ke Shard Biner (Prep)
Lakukan tokenisasi data mentah ke format binary shard uint16:
```bash
python main.py prep
```
Output tersimpan di `tokenized/<kategori>/shard_*.bin`.

---

### 6. Benchmark Hardware GPU
Ukur performa throughput GPU riil dan temukan micro-batch size optimal tanpa OOM:
```bash
python main.py bench
```

---

### 7. Uji Coba Pilot (Model Mini 0.05B)
Jalankan model uji coba pada data riil untuk sweep learning rate dan verifikasi pipeline:
```bash
python main.py pilot
```

---

### 8. Pretraining Utama
Mulai pelatihan pretraining model dari nol:
```bash
# Jalankan pretraining dengan supervisor auto-resume jika terjadi crash/OOM
python main.py pretrain --auto

# Memulai run baru secara bersih
python main.py pretrain --new
```
Checkpoint otomatis disimpan secara berkala di `runs/run_XXX/ckpt/`.

---

### 9. Supervised Fine-Tuning (SFT)
Latih model pretrain dengan data instruksi percakapan, penalaran, dan tool calling:
```bash
# SFT dari checkpoint run pretrain terakhir
python main.py sft

# SFT dari run tertentu
python main.py sft --from run_001
```

---

### 10. Evaluasi & Perbandingan Run
Uji performa dan kepatuhan format model yang telah dilatih:
```bash
# Evaluasi run tertentu
python main.py eval --run run_001

# Bandingkan metrik dua run yang berbeda
python main.py eval --compare run_001 run_002
```

---

### 11. Chat Interaktif di Terminal
Uji coba langsung respons model di terminal dengan eksekusi sandbox otomatis:
```bash
python main.py chat
```

---

### 12. Ekspor ke Hugging Face & Format GGUF/Ollama
Ekspor bobot final model ke format standar Hugging Face:
```bash
python main.py export --run run_002 --out export/my_model
```

#### Konversi ke GGUF (llama.cpp) & Ollama
1. **Konversi ke file `.gguf`**:
   ```bash
   python <path-to-llama.cpp>/convert_hf_to_gguf.py export/my_model --outfile my_model.gguf --outtype bf16
   ```
2. **Kuantisasi ke Q8_0 (Opsional)**:
   ```bash
   llama-quantize my_model.gguf my_model-q8_0.gguf Q8_0
   ```
3. **Jalankan di Ollama**:
   Buat file `Modelfile`:
   ```dockerfile
   FROM ./my_model-q8_0.gguf
   PARAMETER stop "<|im_end|>"
   ```
   Lalu build dan jalankan:
   ```bash
   ollama create my_model -f Modelfile
   ollama run my_model
   ```

---

## 🗂️ Referensi Perintah CLI (`main.py`)

| Perintah | Deskripsi Singkat | Contoh Penggunaan |
|---|---|---|
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

## 🧪 Menjalankan Unit Tests

Toolkit dilengkapi dengan unit test menyeluruh untuk memastikan kebenaran rumus arsitektur, template Jinja, mask-loss span, rotasi checkpoint, dan integrasi pipeline end-to-end:

```bash
python -m unittest discover -s tests -v
```

---

## 📁 Struktur Direktori

```text
├── llmkit/                   # Modul inti toolkit
│   ├── advisor.py            # Aturan validasi advisor (A01-A18)
│   ├── bench.py              # Benchmark hardware & memory
│   ├── chat.py               # Interactive chat REPL
│   ├── chat_template.py      # Jinja template & loss masking span
│   ├── check.py              # Validator data mentah
│   ├── checkpoint.py         # Save/load checkpoint atomik (safetensors/pt)
│   ├── cli.py                # Command-line interface parser & handlers
│   ├── dataset.py            # Streaming memmap dataset loader
│   ├── data_spec.py          # Definisi kuota data & generator identity
│   ├── doctor.py             # Diagnostik hardware & dependensi
│   ├── evaluate.py           # Evaluasi pretrain & SFT
│   ├── export.py             # Ekspor ke format Hugging Face
│   ├── guide.py              # Panduan urut langkah demi langkah
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
│   └── ui.py                 # Formatting UI Rich terminal
├── tests/                    # Rangkaian unit test dan test fixture
├── main.py                   # Entry point CLI utama
├── plan.md                   # Dokumen spesifikasi teknis lengkap
├── .gitignore                # Filter file cache, checkpoints, & dataset
└── README.md                 # Dokumentasi proyek
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah lisensi **Apache License 2.0** - lihat file [LICENSE](LICENSE) untuk detail selengkapnya.
Didistribusikan untuk riset mandiri, eksperimen pembuatan LLM, dan keperluan edukasi.
