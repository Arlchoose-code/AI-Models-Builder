# PLAN: LLM Training Toolkit (CLI interaktif) — Spesifikasi untuk Implementasi

Dokumen ini adalah spesifikasi lengkap. Baca SEMUA sebelum menulis kode. Ikuti urutan fase di Bagian 15.

---

## 0. ATURAN UNTUK AI YANG MENGIMPLEMENTASIKAN (WAJIB DIBACA)

1. **Jangan mengarang.** Jangan mengarang nama fungsi/API/parameter library. Kalau tidak yakin sebuah API ada, tulis kode yang memeriksa dulu (`hasattr`, `try/except ImportError`) atau pakai cara yang sudah pasti ada. Jangan mengarang ID dataset; hanya pakai daftar di Bagian 14 dan tandai "verifikasi dulu".
2. **Tidak ada placeholder.** Dilarang menulis `# TODO`, `pass`, `...`, atau "implementasi menyusul" di kode yang dikirim. Kalau satu fitur belum waktunya (lihat fase), jangan dibuat file-nya sama sekali.
3. **Kerjakan per fase.** Setelah satu fase selesai: jalankan semua tes fase itu (Bagian 15), tampilkan hasilnya, lalu **berhenti dan tunggu konfirmasi pengguna**. Jangan lanjut ke fase berikutnya sendiri.
4. **Kode lengkap, bukan potongan.** Setiap file diberikan utuh, bisa langsung dijalankan.
5. **Semua angka dan rumus di dokumen ini adalah spesifikasi.** Kalau hasil kode berbeda dari "test vector", kodenya yang salah.
6. **Tes dulu dengan data kecil sintetis** sebelum menyatakan fitur selesai. Sediakan generator data dummy (`tests/make_dummy_data.py`).
7. **Gaya kode:** Python 3.13, type hints, `pathlib.Path` (bukan string path), semua file dibuka dengan `encoding="utf-8"`, fungsi kecil dan bernama jelas, docstring singkat. Tidak ada variabel global yang bermutasi.
8. **Bahasa:** semua teks yang tampil ke pengguna (pesan CLI, error, panduan) dalam **Bahasa Indonesia**. Nama variabel/fungsi/komentar kode dalam bahasa Inggris.
9. **Jangan pernah menghapus atau menimpa data milik pengguna** (`data/`) tanpa konfirmasi eksplisit. Semua penulisan file penting harus **atomic** (tulis ke `.tmp` lalu `os.replace`).
10. **Pesan error harus actionable:** sebutkan apa yang salah, file/baris mana, dan perintah apa yang harus dijalankan selanjutnya.
11. **Prioritas jika ada konflik:** Bagian 18 (Settings & Advisor) menimpa Bagian 6.5 (`seq_len`), 6.8 (aturan Stage C), 13.1 (daftar perintah), dan 13.2 (alur `init`). Selain itu, bagian bernomor lebih spesifik menang atas yang lebih umum.

---

## 1. TUJUAN

Satu program CLI (`python main.py <perintah>`) yang memandu pengguna melatih LLM dari nol (pretraining → SFT → eval → chat → export), dengan alur:

1. Pengguna menjalankan `init` **sekali**, menjawab pertanyaan di terminal.
2. Program mencetak **rencana lengkap** di terminal (arsitektur, jumlah token, compute, biaya, hyperparameter, komposisi data, checkpoint) dan **membuat folder data kosong beserta arahan** apa yang harus diisi, formatnya, dan contoh barisnya.
3. Pengguna mengisi data secara manual sesuai arahan.
4. Pengguna menjalankan perintah per tahap (`check`, `tokenizer`, `prep`, `bench`, `pilot`, `pretrain`, `sft`, `eval`, `chat`, `export`).
5. `status` selalu mendeteksi otomatis posisi proyek dari **kondisi file di disk** dan memberi tahu langkah selanjutnya.

Model target pengguna: dwibahasa **Inggris + Indonesia**, teks saja (tanpa vision), mendukung **tool calling**, dan **mode think / no-think** (hanya dua mode, TANPA level low/medium/high). Ukuran: 0.05B sampai ~1B (default contoh 0.6B).

Tidak boleh ada output berupa `plan.md` / `config.yaml` yang harus dibuka pengguna. Semua rencana **tampil di terminal**. Konfigurasi internal disimpan di `project.json` (JSON, bukan YAML).

---

## 2. LINGKUNGAN

- OS utama pengguna: **Windows** (perintah `py -3.13 main.py ...`). Juga harus jalan di Linux (Colab / server sewaan).
- Python 3.13. GPU lokal: RTX 4060 Laptop 8 GB. Cloud: Colab (T4/L4/A100), H100 sewaan.
- Dependensi: `torch`, `transformers>=4.51.0`, `tokenizers`, `safetensors`, `numpy`, `tqdm`. Opsional: `rich` (tabel berwarna; fallback ke teks biasa kalau tidak terpasang), `datasets` + `huggingface_hub` (hanya untuk perintah `fetch`, fase akhir).
- **Aturan Windows:**
  - Gunakan `if __name__ == "__main__":` di entry point. Multiprocessing memakai `spawn`; jangan pakai `fork`.
  - `torch.compile` **mati secara default** di Windows (flag `--compile` untuk menyalakan).
  - DataLoader `num_workers=0` di Windows (data dibaca via `np.memmap`, sudah cepat).
  - Di awal program: `sys.stdout.reconfigure(encoding="utf-8")` dan `sys.stderr.reconfigure(encoding="utf-8")`.
  - Path dengan spasi harus aman.
- Presisi: pakai `bf16` jika `torch.cuda.is_bf16_supported()`, selain itu `fp16` + `torch.amp.GradScaler`. Loss dihitung dalam fp32.

---

## 3. KEPUTUSAN YANG SUDAH FINAL (JANGAN DITANYAKAN ULANG)

| Hal | Keputusan |
|---|---|
| Arsitektur | Dense decoder-only. **Gunakan kelas `Qwen3ForCausalLM` + `Qwen3Config` dari `transformers`** (bobot diinisialisasi acak dari config). Alasan: tidak perlu menulis model sendiri (minim bug), mendukung RMSNorm, SwiGLU, GQA, RoPE, QK-norm, tied embeddings, gradient checkpointing, SDPA, dan hasilnya langsung bisa diexport ke format Hugging Face / GGUF. |
| Attention impl | `attn_implementation="sdpa"` |
| Tokenizer | Byte-level BPE dilatih sendiri dengan library `tokenizers`. Vocab **32.000** untuk model ≤0,2B, **64.000** untuk yang lebih besar. Angka dipecah per digit. Vocab ≤65.535 sehingga token disimpan `uint16`. |
| Format chat | ChatML ala Qwen3 (Bagian 8). Dua mode: `think` dan `no_think`. Default saat tidak ada tag = `no_think`. |
| Tool call | Format `<tool_call>{json}</tool_call>` ala Qwen3 (Bagian 8). |
| Tied embeddings | Ya (untuk semua ukuran ≤ 1B). |
| Optimizer | AdamW (betas 0.9/0.95, eps 1e-8, weight decay 0.1 hanya untuk matriks 2D, tanpa decay untuk norm dan embedding), grad clip 1.0. |
| LR schedule | WSD: warmup → stable → decay linear ke 10% puncak. |
| Packing pretrain | Dokumen digabung berurutan dengan token EOS (`<\|endoftext\|>`) di antaranya, dipotong blok `seq_len+1`. **Tanpa** document masking di MVP. |
| Context | Pretrain 2048/4096. Perpanjangan ke 32k = fase opsional (naikkan `rope_theta`). |
| Checkpoint | Format `torch.save` penuh (model + optimizer + dataloader state + RNG), atomic, rolling + milestone + best. Detail di Bagian 11. |
| Config internal | `project.json` (JSON). Tidak ada YAML. |

---

## 4. STRUKTUR PROYEK

Kode toolkit (satu paket Python, satu pintu masuk `main.py`):

```
llmkit/
  main.py                  # entry point: memanggil llmkit.cli.main()
  llmkit/
    __init__.py
    cli.py                 # argparse + routing perintah
    ui.py                  # print tabel/warna, input() dengan validasi, format angka/waktu
    state.py               # baca/tulis project.json, deteksi status dari disk
    gpu_table.py           # tabel GPU (TFLOPs efektif, VRAM, harga/jam) + deteksi GPU
    planner.py             # hitung parameter, preset arsitektur, token, compute, hyperparam, memori, komposisi
    data_spec.py           # definisi folder data, schema JSON, contoh baris, arahan
    check.py               # validator data
    chat_template.py       # render percakapan -> teks + mask loss
    tokenizer_train.py     # latih tokenizer + laporan fertility
    prep.py                # tokenisasi -> shard .bin, dedup, split val, filter leak
    dataset.py             # dataloader pretrain (weighted, resumable) + dataset SFT
    model.py               # build Qwen3ForCausalLM dari config, hitung parameter nyata
    checkpoint.py          # simpan/muat/rotasi checkpoint (atomic)
    train.py               # loop pretraining (+ auto-resume, spike rollback)
    bench.py               # ukur tok/s, cari micro-batch maksimal
    pilot.py               # run kecil validasi + sweep LR + tes resume
    sft.py                 # fine-tuning chat/think/tool
    evaluate.py            # eval pretrain & SFT
    tools_sandbox.py       # tool aman untuk eval: calculate, search (mock), get_time
    chat.py                # REPL chat dengan model
    export.py              # export HF folder + petunjuk GGUF/Ollama
    doctor.py              # cek environment
    guide.py               # teks help & guide
  tests/
    make_dummy_data.py
    test_planner.py
    test_chat_template.py
    test_checkpoint.py
    test_status.py
```

Folder proyek hasil `init` (dibuat di direktori kerja, nama sesuai jawaban pengguna):

```
<nama_proyek>/
  project.json             # jawaban init + rencana (dibaca semua perintah)
  data/
    pretrain/<sumber>/     # lihat Bagian 7 (pengguna mengisi .jsonl)
    sft/<kategori>/        # lihat Bagian 7
    eval_leak/             # (opsional) teks benchmark untuk deteksi kebocoran
  tokenizer/               # tokenizer.json + tokenizer_meta.json
  tokenized/               # shard .bin per sumber + _manifest.json + _val/
  runs/run_001/            # satu folder per run (config, log, checkpoint, hasil eval)
  export/
  logs/history.log         # catatan semua perintah yang pernah dijalankan
  bench.json               # hasil `bench`
```

Lokasi `runs/` (khususnya checkpoint) harus bisa diganti lewat `--out-dir` atau jawaban `init` (misal ke Google Drive saat di Colab).

---

## 5. PROJECT.JSON DAN DETEKSI STATUS OTOMATIS

### 5.1 `project.json`
Isi minimal:
```json
{
  "version": 1,
  "name": "bahasa-0.6B",
  "created": "2026-10-30T10:00:00",
  "answers": {
    "size_b": 0.6, "token_mode": "balanced", "tokens_override": null,
    "gpu": "H100", "features": {"tool_call": true, "think": true, "vision": false},
    "id_unique_tokens_b": 40.0, "long_context": false,
    "model_name": "Bahasa", "creator": "Pengguna", "runs_dir": "runs"
  },
  "plan": { "...": "hasil planner.make_plan(answers), lihat Bagian 6" },
  "history": []
}
```
Tulis atomic. Setiap perintah menambahkan satu baris ke `logs/history.log`: `ISO-waktu | perintah | argumen | hasil(ok/gagal) | ringkasan`.

### 5.2 `status`: deteksi dari file asli di disk (bukan dari catatan)
Urutan tahap dan aturan "selesai":

| Tahap | Dianggap selesai jika |
|---|---|
| init | `project.json` ada dan valid |
| data | `check --quick` lulus: tiap folder wajib punya file `.jsonl` dan estimasi token ≥ 90% target (lihat Bagian 9.1) |
| tokenizer | `tokenizer/tokenizer.json` dan `tokenizer_meta.json` ada, vocab sesuai rencana |
| prep | `tokenized/_manifest.json` ada dan **fingerprint data** (daftar path+ukuran+mtime semua file di `data/`) sama dengan yang tercatat di manifest. Jika beda → status `STALE` (bukan selesai). |
| bench | `bench.json` ada dan `config_hash` sama dengan hash config model saat ini |
| pilot | `pilot/result.json` ada |
| pretrain | ada `runs/*/run.json` dengan `kind="pretrain"` dan `status="finished"` |
| sft | ada run `kind="sft"` `status="finished"` |
| eval | ada hasil eval untuk run terbaru |
| export | folder `export/` berisi `model.safetensors` |

Tampilan status: `[x]` selesai, `[~]` sebagian/sedang berjalan, `[ ]` belum, `[!]` stale/bermasalah.

### 5.3 Deteksi kondisi run (`run.json`)
Loop training menulis `runs/<id>/run.json` yang diperbarui (atomic) minimal tiap 30 detik:
```json
{"kind":"pretrain","status":"running","pid":12345,"heartbeat":"ISO-waktu",
 "step":34000,"tokens_seen":35651584000,"tokens_target":59800000000,
 "last_loss":2.71,"last_val_loss":2.80,"tok_per_s":18400,
 "last_checkpoint":"ckpt/step_00034000","finished":false}
```
Nilai `status`: `running`, `paused` (keluar normal karena `--max-hours`/Ctrl+C), `finished`, `crashed`, `stopped_spike`.
Aturan deteksi di `status`:
- `status=="running"` tetapi `heartbeat` lebih tua dari 180 detik **atau** `pid` tidak hidup → laporkan **"Training berhenti tiba-tiba (sesi putus/crash?)"** dan sarankan `pretrain --resume`.
- Cek proses hidup: `psutil` jika ada, kalau tidak pakai `os.kill(pid, 0)` (Linux) / `ctypes` `OpenProcess` (Windows).

### 5.4 Blok "SELANJUTNYA"
`status` mencetak tepat satu rekomendasi: tahap pertama yang belum selesai, dengan perintah lengkapnya. Kasus khusus:
- pretrain `paused`/`crashed` → `pretrain --resume`.
- prep `STALE` → peringatan + `prep` ulang.
- Jika perintah tahap dijalankan tapi prasyarat belum ada → **tolak** dengan pesan jelas dan opsi (contoh: `sft` butuh checkpoint pretrain selesai; tawarkan "[1] tunggu selesai [2] SFT dari checkpoint terbaru (hasil kurang matang)").

Setiap perintah, setelah selesai, mencetak ringkasan hasil dan baris `Lanjut: python main.py <perintah berikutnya>`.


---

## 6. PLANNER (`planner.py`) — SEMUA PERHITUNGAN OTOMATIS

Fungsi inti: `make_plan(answers) -> dict`. Input: ukuran (B parameter), mode token atau jumlah token manual, GPU, fitur, batas data Indonesia unik. Output: seluruh rencana. Juga tersedia mode cepat tanpa proyek: `python main.py plan 0.5B --mode balanced --gpu H100` (hanya mencetak rencana).

### 6.1 Hitung jumlah parameter (RUMUS PERSIS)
Untuk config: `d` (hidden), `L` (layer), `Hq`, `Hkv`, `hd` (head_dim), `ffn`, `V` (vocab), tied embeddings:

```
attn_per_layer = d*(Hq*hd) + d*(Hkv*hd) + d*(Hkv*hd) + (Hq*hd)*d     # q,k,v,o tanpa bias
ffn_per_layer  = 3 * d * ffn                                          # gate, up, down (SwiGLU)
norm_per_layer = 2*d + 2*hd                                           # 2 RMSNorm + q_norm + k_norm
per_layer      = attn_per_layer + ffn_per_layer + norm_per_layer
total          = L*per_layer + V*d + d                                # embedding (tied) + final norm
total_untied   = total + V*d
non_embedding  = L*per_layer + d
```

**Test vector (harus tepat):**
- Qwen3-0.6B: `d=1024, L=28, Hq=16, Hkv=8, hd=128, ffn=3072, V=151936` → `total = 596,049,920`, `non_embedding = 440,467,456`.
- Preset 0.6B di bawah → `total = 598,097,920`.

Setelah model dibangun dengan `Qwen3ForCausalLM`, `sum(p.numel() for p in model.parameters())` **harus sama** dengan rumus (tied embeddings dihitung sekali). Tambahkan tes untuk ini; jika beda, rumus atau config salah.

### 6.2 Preset arsitektur (head_dim 64, `Hq = d/64`)

| Ukuran | d | L | Hq | Hkv | ffn | V | Total (tepat dari rumus) |
|---|---|---|---|---|---|---|---|
| 0.05B | 512 | 12 | 8 | 2 | 1408 | 32000 | 50,214,912 |
| 0.1B | 640 | 18 | 10 | 2 | 1728 | 32000 | 97,920,384 |
| 0.3B | 1024 | 20 | 16 | 4 | 2816 | 64000 | 291,024,384 |
| 0.6B | 1280 | 30 | 20 | 4 | 3456 | 64000 | 598,097,920 |
| 1B | 1792 | 28 | 28 | 4 | 4864 | 64000 | 1,052,482,816 |

Parameter tetap untuk semua preset: `hidden_act="silu"`, `rms_norm_eps=1e-6`, `tie_word_embeddings=True`, `attention_bias=False`, `rope_theta=500000`, `max_position_embeddings=` seq_len pretrain (diubah saat long-context), `use_sliding_window=False`, `initializer_range=0.02`.

**Ukuran di luar preset (misal 0.5B, 0.2B):** jalankan solver:
1. `V` = 32000 jika target ≤0.2B, selain itu 64000.
2. Cari kombinasi: `d` kelipatan 64 (rentang 256–4096), `hd=64`, `Hq=d/64`, `Hkv` = divisor dari `Hq` yang paling dekat ke `Hq/5` (minimal 1), `ffn` = pembulatan `2.7*d` ke kelipatan 64, `L` dari 8 sampai 48.
3. Pilih kombinasi dengan error `|total-target|/target` terkecil; jika seri, pilih yang rasio `d/L` paling dekat 44.
4. Error harus ≤ 3%. Jika tidak tercapai, cetak peringatan dan tampilkan 3 kandidat terbaik untuk dipilih pengguna.
5. Jika ukuran sama dengan preset (±5%), pakai preset.

### 6.3 Jumlah token (mode otomatis)

`tokens = rasio * total_params`, dibulatkan ke 0,1B terdekat.

| Mode | Rasio token/param | Arti |
|---|---|---|
| `chinchilla` | 20 | compute-optimal, untuk uji coba |
| `balanced` | 100 | **default**, bagus untuk biaya/kualitas |
| `overtrain` | 500 | kualitas tinggi untuk model kecil |
| `max` | 3000 | level "mantap banget" (0.6B ≈ 1.8T) |
| `custom` | – | pengguna memasukkan jumlah token (`--tokens 80B`) |

**Mode budget (terbalik):** `--budget-usd 500` atau `--hours 100` + GPU → hitung token maksimum:
`tokens = (jam * 3600 * tflops_efektif * 1e12) / (6 * params)`, dengan `jam = budget_usd / harga_per_jam` untuk mode dolar.
Cetak hasilnya beserta rasio token/param dan label mode terdekat.

**Test vector:** 0.6B preset, `balanced` → `tokens = round(100 * 598,097,920 / 1e9, 1) = 59.8B`.

### 6.4 Compute dan biaya
- `FLOPs = 6 * N * D` (N parameter, D token).
- `detik = FLOPs / (tflops_efektif * 1e12)`; `jam = detik/3600`; `biaya = jam * harga_per_jam`.
- Tabel GPU di `gpu_table.py` (**TFLOPs efektif** = yang realistis terpakai untuk model kecil, bisa diganti hasil `bench`). Semua nilai ini harus mudah diedit di satu tempat:

| GPU | VRAM GB | TFLOPs efektif | Harga $/jam | Presisi |
|---|---|---|---|---|
| H100 | 80 | 300 | 2.5 | bf16 |
| A100 | 40 | 120 | 1.5 | bf16 |
| L4 | 24 | 35 | 0.8 | bf16 |
| RTX 4090 | 24 | 80 | 0 | bf16 |
| RTX 4060 Laptop | 8 | 12 | 0 | bf16 |
| T4 | 16 | 15 | 0 | fp16 |
| custom | input | input | input | input |

- Deteksi otomatis: `torch.cuda.get_device_name(0)` dicocokkan dengan substring (case-insensitive); jika tidak cocok → tanya pengguna.
- Setelah `bench` dijalankan, **semua estimasi waktu dihitung ulang dari tok/s nyata**: `detik = tokens_tersisa / tok_per_s`.
- Tampilkan juga tabel perbandingan: untuk tiap GPU di tabel → jam, hari, biaya.

**Test vector:** 0.6B, 59.8B token, H100 → FLOPs = 6*598,097,920*59.8e9 ≈ 2.146e20 → ≈ 199 jam → ≈ $497.

### 6.5 Hyperparameter pretrain (otomatis)

- `seq_len`: 4096 jika VRAM ≥ 24 GB; 2048 jika lebih kecil (bisa di-override).
- `batch_tokens` (token per optimizer step) berdasarkan total token:

| Total token | batch_tokens |
|---|---|
| < 5B | 262,144 |
| 5B – 30B | 524,288 |
| 30B – 150B | 1,048,576 |
| 150B – 600B | 2,097,152 |
| > 600B | 4,194,304 |

- `total_steps = ceil(tokens / batch_tokens)`.
- `micro_batch` dan `grad_accum`: `batch_tokens = micro_batch * seq_len * grad_accum`. `micro_batch` ditentukan oleh `bench` (cari terbesar yang muat); sebelum ada `bench`, pakai estimasi memori (6.7).
- `peak_lr = 8e-4 * (0.125 / N_B) ** 0.25`, di-clamp ke [2e-4, 2e-3] (N_B = parameter dalam miliar). **Test vector:** N=0.598B → ≈ 5.41e-4. Angka ini titik awal; `pilot` menyapu 3 nilai (0.5×, 1×, 2×).
- `warmup_steps = min(2000, max(100, round(0.01 * total_steps)))`.
- Schedule WSD: warmup linear → stable di `peak_lr` sampai 80% langkah → decay linear ke `0.1 * peak_lr` pada 20% langkah terakhir.
- `weight_decay=0.1`, `betas=(0.9,0.95)`, `grad_clip=1.0`.
- `eval_every_tokens = clamp(total_tokens/50, 50M, 2B)`.

### 6.6 Storage
- `bytes_per_token = 2` (uint16, vocab ≤65.535).
- `tokenized_GB = tokens_unik_dibutuhkan * 2 / 1e9` (hitung dari token unik per sumber, bukan total training).
- Saran ruang kerja = 3× ukuran data mentah + tokenized.
- Ukuran checkpoint: `params * 2` (bobot bf16) dan `params * 12` (+optimizer) — tampilkan ke pengguna.

### 6.7 Estimasi memori (kasar; bench yang final)
```
mem_static  = params * 16 bytes            # fp32 master + grad + adam m,v
mem_act     = micro_batch * seq_len * d * L * c   # c ≈ 34 bytes tanpa grad-checkpoint, ≈ 4 dengan grad-checkpoint
mem_logits  = micro_batch * seq_len * V * 4 bytes (fp32 logits) * 2
total_est   = mem_static + mem_act + mem_logits + 1.5 GB overhead
```
Jika `mem_static` saja > 85% VRAM → cetak peringatan **"Model ini TIDAK muat untuk full training di GPU ini"** dan sarankan GPU lain/ukuran lebih kecil. (Contoh: 0.6B butuh ≈9.6 GB statis → tidak muat di 8 GB.)
Jika `total_est` > VRAM → aktifkan `grad_checkpointing=True` dan turunkan `micro_batch`.

### 6.8 Komposisi data pretrain (persen **dari total token**, otomatis)

Pembagian tahap berdasarkan total token:
- Tanpa long-context: Stage A = 80%, Stage B = 20%.
- Dengan long-context (diminta di `init`, hanya jika total ≥ 20B): A = 75%, B = 20%, C = 5%.

Bobot dalam tiap tahap (jumlah tiap kolom = 100):

| Folder sumber | Stage A (umum) | Stage B (kualitas tinggi) | Stage C (long ctx) |
|---|---|---|---|
| `en_web` | 36 | – | 15 |
| `code` | 25 | 30 | – |
| `id_web` | 12 | – | 10 |
| `math` | 8 | 25 | – |
| `books_wiki` | 8 | – | – |
| `synthetic` | 9 | – | – |
| `dialog` | 2 | – | – |
| `reasoning` | – | 15 | – |
| `en_hq` | – | 12 | – |
| `id_hq` | – | 8 | – |
| `instruct_text` | – | 10 | – |
| `long_docs` | – | – | 50 |
| `long_code` | – | – | 25 |

Penjelasan folder: `en_web` web Inggris berkualitas (FineWeb-Edu/DCLM), `id_web` web Indonesia, `code` kode (+issue/dokumentasi), `math` matematika/STEM, `books_wiki` buku + Wikipedia EN/ID, `synthetic` teks buatan model (rewrite ala textbook, terjemahan EN↔ID), `dialog` forum/percakapan, `reasoning` teks penalaran langkah demi langkah (boleh mengandung `<think>`), `en_hq`/`id_hq` subset teks paling bersih, `instruct_text` teks gaya instruksi/tool/format dalam bentuk teks mentah, `long_docs`/`long_code` dokumen dan repo panjang.

**Perhitungan token per folder:** `needed[folder] = Σ_stage (stage_tokens * weight/100)`.

**Cek ketersediaan (`available[folder]`)** dari hasil `check`/`prep`:
- `epochs = needed / available`.
- `epochs ≤ 1.0` → OK. `1–4` → OK dengan catatan "diulang N kali". `4–8` → **PERINGATAN**. `> 8` → **ERROR** (jangan lanjut).
- Khusus `id_web` dan `id_hq`: `available` dibatasi `answers.id_unique_tokens_b` bila diisi.
- **Auto-adjust (opsional, tanya pengguna):** jika ada folder dengan epochs > 4, tawarkan: batasi ke 4 epoch dan alihkan sisa bobotnya secara proporsional ke folder lain yang punya cadangan; cetak komposisi baru dan tampilkan peringatan bahwa porsi Indonesia turun.
- Folder opsional yang kosong (user sengaja tidak punya): bobotnya dibagi ulang proporsional ke folder lain dalam tahap yang sama, dengan pemberitahuan jelas.
- Hitung dan tampilkan **total porsi bahasa Indonesia** dan **porsi kode** hasil akhir.

Bahasa campuran: tidak ada rasio EN/ID yang dipaksa di pretrain selain yang tersirat dari bobot di atas; tampilkan saja hasilnya.

### 6.9 Rencana post-training (otomatis)

Jumlah sampel SFT total: `n_sft = clamp(round(400_000 * N_B), 20_000, 1_000_000)` (contoh: 0.1B → 40k, 0.6B → ≈239k, 1B → 400k). Dibulatkan ke ribuan.

| Kategori (folder `data/sft/...`) | % dari n_sft | Isi |
|---|---|---|
| `chat` | 30 | Tanya jawab umum, mode `no_think` |
| `think` | 20 | Penalaran: math 35%, kode 25%, logika 20%, sains 10%, perencanaan tool 10%; mode `think` |
| `tool_call` | 15 | Lihat aturan di bawah |
| `multiturn` | 15 | Percakapan 3–6 giliran |
| `code` | 10 | Menulis/menjelaskan/memperbaiki kode |
| `format` | 5 | Output JSON/tabel/daftar sesuai instruksi |
| `refusal` | 3 | Menolak sopan, "saya tidak tahu", tidak mengarang |
| `identity` | 2 | Siapa kamu, kamu bisa apa, siapa pembuatmu |

Aturan komposisi di dalam data:
- **Bahasa:** EN 55% / ID 45% (toleransi ±10 poin; di luar itu → peringatan).
- **Mode:** hanya dua. Kategori `think` seluruhnya mode `think`; kategori lain mode `no_think`. Sebagian `tool_call` (20% dari kategori itu) boleh mode `think`.
- **Di dalam `tool_call`:** 25% "tidak perlu tool" (jawab langsung, tool tersedia tapi tidak dipakai), 15% multi-call (beberapa tool call berurutan/paralel), 10% penanganan error tool (hasil tool berisi error → model mencoba ulang/menjelaskan), 50% satu tool call normal. Pakai minimal 8 jenis tool berbeda (search, calculate, get_time, dll).
- **Validitas:** 100% tool call harus JSON valid dan nama tool ada di daftar tools sampel itu.
- Hold-out: 1% sampel (min 200, max 5000), dipilih deterministik lewat hash `id`, jadi eval set SFT.

Hyperparameter SFT otomatis:
- `lr = 1e-4` jika N ≤ 0.15B; `5e-5` jika 0.15B < N ≤ 0.8B; `3e-5` jika N > 0.8B.
- `epochs = 3` (2 jika n_sft > 500k), `batch = 128 sequences` (pakai grad accumulation), `warmup 3%`, cosine ke 10%, `weight_decay=0`, `max_seq_len = seq_len`, sampel lebih panjang dibuang (dihitung & dilaporkan).
- Loss hanya pada token milik assistant (Bagian 8.3), dirata-rata per token.

### 6.10 Rencana checkpoint (otomatis)
- **Rolling:** setiap `ckpt_every_tokens = clamp(tokens/100, 100M, 1B)` **atau** 30 menit sejak checkpoint terakhir (mana yang duluan). Simpan 3 terakhir, hapus yang lebih lama.
- **Milestone permanen:** tiap 10% dari total token (`milestone_10`, `milestone_20`, dst.). Tidak pernah dihapus.
- **Best:** salin checkpoint dengan val loss terbaik ke `best/`.
- **Final:** `final/` saat selesai (bobot saja dalam safetensors, bf16).

---

## 7. ARAHAN DATA & CONTOH FORMAT (`data_spec.py`)

`init` membuat folder-folder kosong, lalu mencetak arahan per folder: **target jumlah token** (atau sampel), **format**, dan **contoh baris**. Setiap folder berisi `README.txt` kecil dengan teks yang sama (supaya arahan tetap ada nanti).

Ukuran file: bebas, boleh banyak file `.jsonl` (juga `.jsonl.gz` dan `.jsonl.zst` jika library ada). Satu baris = satu dokumen. Encoding UTF-8.

### 7.1 Pretrain: `data/pretrain/<sumber>/*.jsonl`
```json
{"text": "The water cycle describes how water moves through evaporation, condensation, and precipitation."}
{"text": "Komputer memproses data menggunakan instruksi yang dijalankan oleh prosesor."}
{"text": "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    ..."}
```
Field wajib: `text` (string, tidak kosong). Field opsional: `lang` (`en`/`id`), `source`, `id`. Field lain diabaikan.

Aturan konten per folder:
- `reasoning`: boleh mengandung `<think>...</think>` di dalam `text`.
- `id_web` dan `id_hq`: teks Indonesia. `en_web`, `en_hq`: Inggris.
- `books_wiki`, `synthetic`, `dialog`: campuran EN+ID.

### 7.2 SFT: `data/sft/<kategori>/*.jsonl` (satu format kanonik untuk semua kategori)

Skema satu baris:
```json
{
  "id": "unik-string",
  "lang": "id",
  "mode": "no_think",
  "tools": [ {"type":"function","function":{"name":"...","description":"...","parameters":{...}}} ],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "reasoning": "...", "tool_calls": [{"name":"...","arguments":{...}}]},
    {"role": "tool", "name": "...", "content": "..."}
  ]
}
```
Aturan (divalidasi oleh `check`):
- `lang`: `en` atau `id`. `mode`: `think` atau `no_think`. `tools`: opsional, daftar skema fungsi (JSON Schema).
- `system` opsional dan hanya boleh di posisi pertama. Jika tidak ada, renderer memakai system prompt default (Bagian 8.1).
- Urutan role: `user` → `assistant` → (`tool` → `assistant`)* → `user` ... Setelah `assistant` yang punya `tool_calls` WAJIB ada satu `tool` message per call (urut sama).
- `assistant.content` boleh kosong **hanya** jika ada `tool_calls`.
- `assistant.reasoning`: hanya untuk `mode=think` dan hanya di assistant message **setelah user message terakhir**. Pada mode `no_think` dilarang ada `reasoning`. Assistant message di riwayat (sebelum user terakhir) dilarang punya `reasoning`.
- `tool_calls[].arguments` adalah objek JSON (bukan string).
- Percakapan harus diakhiri `assistant` (bukan `tool`/`user`).
- Jika data sumber memakai `<think>...</think>` di dalam `content`, konversikan ke field `reasoning` (sediakan fungsi util `extract_think()` dan tawarkan di pesan error `check`).

**Contoh per kategori (satu baris JSONL; di dokumen ini diformat agar terbaca):**

`chat` (no_think, ID):
```json
{"id":"chat-0001","lang":"id","mode":"no_think","messages":[
 {"role":"user","content":"Apa fungsi CPU pada komputer?"},
 {"role":"assistant","content":"CPU (Central Processing Unit) menjalankan instruksi program dan memproses data. Ia ibarat otak komputer yang mengatur dan menghitung."}]}
```

`think` (mode think, math):
```json
{"id":"think-0001","lang":"en","mode":"think","messages":[
 {"role":"user","content":"A shirt costs $40 and is 25% off. What is the sale price?"},
 {"role":"assistant","reasoning":"25% of 40 is 10. So the discount is $10. 40 - 10 = 30.","content":"The sale price is $30."}]}
```

`tool_call` (satu panggilan, ID):
```json
{"id":"tool-0001","lang":"id","mode":"no_think",
 "tools":[{"type":"function","function":{"name":"calculate","description":"Hitung ekspresi matematika","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
 "messages":[
 {"role":"user","content":"Berapa 125 dikali 8?"},
 {"role":"assistant","content":"","tool_calls":[{"name":"calculate","arguments":{"expression":"125 * 8"}}]},
 {"role":"tool","name":"calculate","content":"1000"},
 {"role":"assistant","content":"125 × 8 = 1000."}]}
```

`tool_call` (tidak perlu tool; tool tersedia tapi dijawab langsung):
```json
{"id":"tool-0102","lang":"en","mode":"no_think",
 "tools":[{"type":"function","function":{"name":"search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
 "messages":[
 {"role":"user","content":"What is the capital of France?"},
 {"role":"assistant","content":"The capital of France is Paris."}]}
```

`tool_call` (search lalu jawab, mode think):
```json
{"id":"tool-0203","lang":"id","mode":"think",
 "tools":[{"type":"function","function":{"name":"search","description":"Cari di web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
 "messages":[
 {"role":"user","content":"Siapa pemenang Piala Dunia terakhir?"},
 {"role":"assistant","reasoning":"Informasi ini bisa berubah. Lebih aman mencari dulu.","content":"","tool_calls":[{"name":"search","arguments":{"query":"pemenang Piala Dunia terakhir"}}]},
 {"role":"tool","name":"search","content":"Argentina menjuarai Piala Dunia FIFA 2022 di Qatar."},
 {"role":"assistant","reasoning":"Hasil pencarian menyebut Argentina, 2022.","content":"Berdasarkan hasil pencarian, pemenang Piala Dunia terakhir adalah Argentina (Qatar 2022)."}]}
```

`tool_call` (error tool, lalu pulih):
```json
{"id":"tool-0310","lang":"en","mode":"no_think",
 "tools":[{"type":"function","function":{"name":"calculate","description":"Evaluate a math expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
 "messages":[
 {"role":"user","content":"What is 12 divided by 0.5 plus 3?"},
 {"role":"assistant","content":"","tool_calls":[{"name":"calculate","arguments":{"expression":"12 / 0.5 + 3"}}]},
 {"role":"tool","name":"calculate","content":"ERROR: invalid syntax"},
 {"role":"assistant","content":"","tool_calls":[{"name":"calculate","arguments":{"expression":"(12 / 0.5) + 3"}}]},
 {"role":"tool","name":"calculate","content":"27"},
 {"role":"assistant","content":"12 divided by 0.5 plus 3 equals 27."}]}
```

`multiturn` (3 giliran): `messages` berisi user/assistant bergantian minimal 3 pasang; tidak ada `reasoning` di riwayat.

`code`, `format`, `refusal`: format sama dengan `chat`.

`identity` (nama model & pembuat diisi dari jawaban `init`; contoh dengan model "Bahasa", pembuat "Pengguna"):
```json
{"id":"id-0001","lang":"id","mode":"no_think","messages":[
 {"role":"user","content":"Kamu siapa?"},
 {"role":"assistant","content":"Aku Bahasa, asisten AI dwibahasa (Indonesia dan Inggris) yang dibuat oleh Pengguna."}]}
```

### 7.3 `data/eval_leak/` (opsional)
File `.jsonl` berformat `{"text": "..."}` berisi teks soal benchmark (misal soal HellaSwag/MMLU/IndoMMLU) yang **tidak boleh** bocor ke data latih. `prep` membuang dokumen latih yang punya n-gram 13 kata yang sama.

### 7.4 Ketika pengguna harus "menyesuaikan format manual"
`check` (Bagian 9) menunjukkan baris yang salah. Sediakan juga util `python main.py convert <file> --from <jenis> --to <folder>` hanya untuk dua konversi umum (fase akhir): (a) kolom `text` dari Parquet/JSONL berkolom lain → `{"text":...}`; (b) format ShareGPT/OpenAI `messages` lama → skema SFT di atas. Jangan buat konverter lain.

---

## 8. CHAT TEMPLATE (`chat_template.py`) — SPESIFIKASI PERSIS

### 8.1 Special token (urutan ID tetap, diberikan ke tokenizer sebelum vocab lain)
```
<|endoftext|>  <|im_start|>  <|im_end|>  <think>  </think>
<tool_call>  </tool_call>  <tool_response>  </tool_response>  <tools>  </tools>
<|pad|>  <|reserved_0|> ... <|reserved_15|>
```
`eos_token=<|im_end|>` untuk chat; `<|endoftext|>` dipakai sebagai pemisah dokumen di pretrain dan `pad`-nya `<|pad|>`.

System prompt default (jika sampel tidak punya system):
`Kamu adalah {model_name}, asisten AI dwibahasa (Indonesia dan Inggris) buatan {creator}.`

### 8.2 Format render
Satu giliran: `<|im_start|>{role}\n{content}<|im_end|>\n`.

**System + tools** (jika `tools` ada, system menjadi):
```
<|im_start|>system
{system_text}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json fungsi 1, satu baris}
{json fungsi 2, satu baris}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```
JSON fungsi dibuat dengan `json.dumps(obj, ensure_ascii=False)` (default separator `", "` dan `": "`).

**Tag mode** ditempelkan di akhir isi user message TERAKHIR yang berasal dari manusia (bukan tool response): `" /think"` atau `" /no_think"`.
- Sampel `mode=think` → selalu diberi `/think`.
- Sampel `mode=no_think` → diberi `/no_think` pada 50% sampel (dipilih deterministik: `hash(id) % 2 == 0`), 50% sisanya tanpa tag.
- Tanpa tag = `no_think`.

**Assistant message** (selalu dimulai blok think; kosong bila no_think):
```
<|im_start|>assistant
<think>
{reasoning}
</think>

{content}<|im_end|>
```
- `no_think` / tanpa reasoning: bagian think menjadi persis `<think>\n\n</think>\n\n`.
- Jika ada `tool_calls`, setelah think (dan `content` jika ada, dipisah `\n`) tulis tiap call:
```
<tool_call>
{"name": "calculate", "arguments": {"expression": "125 * 8"}}
</tool_call>
```
  Beberapa call ditulis berurutan, dipisah satu `\n`, lalu diakhiri `<|im_end|>`. Jika `content` tidak kosong, tulis `content` dulu, `\n`, baru call pertama.
- Assistant message di riwayat (sebelum user terakhir) dirender **tanpa** reasoning (think kosong).

**Tool response** (semua `tool` message berurutan digabung menjadi satu giliran `user`):
```
<|im_start|>user
<tool_response>
{content tool 1}
</tool_response>
<tool_response>
{content tool 2}
</tool_response><|im_end|>
```

**Generation prompt (inferensi):** berhenti di `<|im_start|>assistant\n`. Model sendiri yang menulis `<think>...`. Opsi `--force-nothink` menambahkan `<think>\n\n</think>\n\n` setelahnya (jaring pengaman untuk model kecil).

**Contoh lengkap hasil render** (sampel `tool-0001` di atas, tag `/no_think` aktif):
```
<|im_start|>system
Kamu adalah Bahasa, asisten AI dwibahasa (Indonesia dan Inggris) buatan Pengguna.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "calculate", "description": "Hitung ekspresi matematika", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Berapa 125 dikali 8? /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "calculate", "arguments": {"expression": "125 * 8"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
1000
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

125 × 8 = 1000.<|im_end|>
```
Pisahkan antar giliran dengan `\n` (setelah setiap `<|im_end|>`).

### 8.3 Masking loss
`render(sample) -> (text, spans)` dengan `spans` = daftar `(char_start, char_end)` untuk bagian yang dilatih. Bagian yang dilatih: **mulai dari `<think>` pertama sampai dan termasuk `<|im_end|>`** pada setiap assistant message (setelah `<|im_start|>assistant\n`). Tidak dilatih: system, user, tool response, header role, dan `\n` setelah `<|im_end|>`.
Tokenisasi pakai `return_offsets_mapping=True`; label = `-100` untuk token yang offset-nya di luar spans.
**Tes wajib:** untuk 20 sampel, decode token ber-label ≠ -100: tiap segmen harus diawali `<think>` dan diakhiri `<|im_end|>`; tidak boleh mengandung `<|im_start|>`.

### 8.4 Jinja untuk export
Buat string `chat_template` Jinja yang menghasilkan teks identik dengan renderer Python (boleh diadaptasi dari template resmi Qwen3 di `Qwen/Qwen3-0.6B/tokenizer_config.json`, disesuaikan dengan aturan di atas). **Tes wajib:** untuk 50 sampel, hasil render Jinja == hasil render Python (string sama persis).

---

## 9. `check`: VALIDATOR DATA (`check.py`)

Perintah: `python main.py check [--quick] [--fix-hints]`.

### 9.1 Estimasi token sebelum tokenizer ada
Jangan men-tokenisasi semua data hanya untuk menghitung. Estimasi:
- Jika tokenizer sudah ada: ambil sampel 1 MB dari tiap file (awal, tengah, akhir), tokenisasi, hitung `bytes_per_token` sampel, ekstrapolasi ke ukuran file. Cache hasil di `.cache/data_index.json` dengan kunci `(path, size, mtime)` supaya tidak dihitung ulang.
- Jika belum: `tokens ≈ ukuran_bytes / bpt`, dengan `bpt`: `en_*`=4.3, `id_*`=3.9, `code`/`long_code`=3.6, `math`=3.5, lainnya=4.0. Label hasil sebagai **"estimasi"**.
- `--quick` hanya memeriksa keberadaan file, ukuran total, dan estimasi token (tanpa membaca isi).

### 9.2 Validasi pretrain
Baca tiap file baris demi baris (streaming; jangan memuat seluruh file ke RAM). Untuk tiap folder laporkan: jumlah file, jumlah dokumen, ukuran, estimasi token, % dari target, baris error pertama (maks 5 contoh per jenis error). Error yang dideteksi: JSON tidak valid, `text` hilang/bukan string/kosong, dokumen sangat pendek (<20 karakter, hanya warning), karakter kontrol berlebihan, UTF-8 rusak.
Periksa sampel 2000 dokumen per folder: duplikat persis (hash teks ternormalisasi) → persentase; rasio karakter non-Latin tak wajar pada folder `en_*`/`id_*` → warning kemungkinan salah folder.

### 9.3 Validasi SFT
Terapkan semua aturan Bagian 7.2. Laporkan per kategori: jumlah sampel vs target (`n_sft * persen`), proporsi `en`/`id`, proporsi mode `think`/`no_think`, jumlah tool call valid/tidak valid, jumlah sampel yang melebihi `max_seq_len` (jika tokenizer ada), sampel duplikat (id sama atau isi sama).
Validasi tool call: `arguments` harus cocok dengan `parameters` (cek `required` dan tipe dasar `string/number/integer/boolean/array/object`).

### 9.4 Contoh tampilan
```
=== CEK DATA PRETRAIN ===
folder        file   dokumen     estimasi token   target     status
en_web          12    9.1 jt     18.3B / 22.0B     83%       KURANG 3.7B
id_web           5    4.2 jt      9.4B /  9.0B    104%       OK
code             8    3.0 jt     11.0B / 15.0B     73%       KURANG
...
ERROR  data/pretrain/id_web/part3.jsonl baris 4421: field "text" kosong
WARN   2.1% dokumen duplikat di en_web (sampel 2000)

=== CEK DATA SFT ===
kategori      sampel   target    en/id      status
chat          81k      72k       57/43      OK
tool_call     31k      36k       52/48      KURANG 5k
  - 12 tool call tidak valid (lihat logs/check_sft_errors.txt)
```
Semua error lengkap ditulis ke `logs/check_errors.txt` (bukan dibanjiri di layar). Exit code 0 jika tidak ada ERROR, 1 jika ada.

---

## 10. PIPELINE DATA

### 10.1 `tokenizer` (`tokenizer_train.py`)
- Library `tokenizers`: `Tokenizer(BPE)`, `pre_tokenizer = Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])`, `decoder = ByteLevel()`, `BpeTrainer(vocab_size=V, special_tokens=<daftar Bagian 8.1>, initial_alphabet=ByteLevel.alphabet(), min_frequency=2)`.
- Data latih tokenizer: sampel dokumen dari `data/pretrain/*` dengan bobot: Inggris (en_*) 35%, Indonesia (id_*) 35%, kode 22%, matematika 8%. Total sekitar 1–2 GB teks (batas atas `--max-gb`, default 1.5). Pengambilan sampel acak deterministik (seed 42), streaming.
- Output: `tokenizer/tokenizer.json`, `tokenizer/tokenizer_meta.json`.
- **Laporan wajib** (dicetak dan disimpan di meta): ukuran vocab, ID setiap special token, `bytes_per_token` dan **`token_per_kata`** untuk sampel EN, ID (termasuk contoh kata berimbuhan: "mempertanggungjawabkan", "ketidakpastian", slang: "gak", "banget", "wkwk"), kode, dan math. Peringatan jika `token_per_kata` ID > 1.6× EN.
- Tes: encode→decode harus identik pada 1000 dokumen acak; angka "12345" harus menjadi 5 token terpisah; ID `<|im_end|>` konsisten.
- Jika vocab > 65.535 → cetak error (tidak boleh, karena uint16).

### 10.2 `prep` (`prep.py`)
Tokenisasi seluruh `data/pretrain/<sumber>/` ke shard biner.
1. Baca streaming, per dokumen: normalisasi unicode NFC, buang jika panjang < 20 karakter.
2. **Dedup persis:** hash (xxhash jika ada, jika tidak `hashlib.blake2b` 8 byte) teks ternormalisasi (lowercase + whitespace tunggal); simpan set hash di RAM per sumber (set `int`); dokumen duplikat dibuang. Laporkan jumlah yang dibuang.
3. **Filter kebocoran:** jika `data/eval_leak/` ada, bangun set hash n-gram 13 kata dari file itu; buang dokumen latih yang mengandung n-gram yang sama; laporkan.
4. **Split validasi:** dokumen dengan `hash(teks) % 1000 == 0` (≈0,1%) masuk validasi per sumber, maksimal ~5M token per sumber (sisanya kembali ke train). Simpan ke `tokenized/_val/<sumber>.bin`.
5. Tokenisasi dokumen + tambahkan `<|endoftext|>` setelah tiap dokumen. Tulis `uint16` ke shard `tokenized/<sumber>/shard_00000.bin`, ukuran ±100M token per shard (ukuran shard tepat tidak penting). Gunakan `multiprocessing.Pool` (spawn-safe) dengan jumlah proses `min(cpu_count()-1, 8)`.
6. Tulis `tokenized/_manifest.json`: per sumber → daftar shard, jumlah token, dokumen terbuang (dup/leak/pendek), plus `data_fingerprint` (Bagian 5.2) dan hash tokenizer.
7. Resumable: jika dijalankan ulang, shard yang sudah selesai (dan fingerprint input sumbernya sama) dilewati.
8. Setelah selesai: tampilkan token **aktual** per sumber, bandingkan dengan `needed` (Bagian 6.8), hitung ulang `epochs`, dan cetak peringatan/error yang sesuai.

### 10.3 Dataloader pretrain (`dataset.py`)
- Setiap sumber: `np.memmap(shard, dtype=np.uint16, mode="r")`.
- Pada tiap tahap (A/B/C), pemilihan sumber per **sequence** memakai `numpy.random.Generator` ber-seed (`seed + step`) dengan probabilitas = bobot tahap (Bagian 6.8). Tiap sumber punya **cursor** (indeks shard, offset) yang maju berurutan; saat semua shard habis → epoch+1 dan urutan shard diacak ulang (seed = `base_seed + epoch`).
- Satu sequence = `seq_len + 1` token berurutan (input = `[:-1]`, target = `[1:]`). Tidak ada padding.
- Satu batch = `micro_batch` sequence. Pindah tahap otomatis berdasarkan `tokens_seen` (batas tahap dari rencana).
- **State harus bisa disimpan dan dipulihkan persis:** `{stage, per_source: {cursor_shard, cursor_offset, epoch, order_seed}, rng_state, tokens_seen}` → masuk ke checkpoint. Setelah resume, batch berikutnya **identik** dengan yang akan didapat tanpa interupsi (tes wajib, Bagian 15).
- Prefetch sederhana dengan thread (bukan proses) agar aman di Windows.

---

## 11. TRAINING (`train.py`) DAN CHECKPOINT (`checkpoint.py`)

### 11.1 Loop pretraining
```
build model (Qwen3ForCausalLM dari config, dtype fp32 master, autocast bf16/fp16)
optimizer = AdamW(param groups: decay untuk tensor ndim>=2 selain embedding, no-decay untuk norm/embedding)
resume bila ada checkpoint (pilih `latest.json`)
loop per optimizer step:
    for i in range(grad_accum):
        x, y = loader.next_batch()
        with autocast: loss = model(x, labels=None) -> hitung cross_entropy(logits.float(), y) / grad_accum
        loss.backward()           # (fp16: scaler)
    clip_grad_norm_(1.0); lr = schedule(step); optimizer.step(); zero_grad()
    log tiap 10 step; eval val tiap eval_every_tokens; checkpoint sesuai aturan; update run.json tiap 30 detik
```
- Loss dihitung manual dari logits (fp32) agar bisa dipotong per-chunk bila VRAM kecil (opsi `loss_chunks`: bagi sequence jadi beberapa potong untuk `lm_head` + CE; aktif otomatis jika estimasi `mem_logits` > 25% VRAM).
- `gradient_checkpointing`: `model.gradient_checkpointing_enable()` bila rencana mengaktifkannya.
- Log `runs/<id>/metrics.jsonl` (satu JSON per baris): `step, tokens, loss, lr, grad_norm, tok_per_s, gpu_mem_gb, time`. Layar: `tqdm` satu baris berisi step, loss, lr, tok/s, ETA (dari tok/s rata-rata bergerak, bukan tebakan awal).
- Eval validasi: rata-rata loss per sumber di `tokenized/_val/` (maks 40 batch tiap sumber), cetak loss + perplexity, tulis ke `metrics.jsonl` dan `run.json`. Hitung juga **bits-per-byte** (`loss/ln2 * tokens/bytes`) agar bisa dibandingkan lintas tokenizer.
- Sampel teks: tiap eval cetak 3 kelanjutan dari prompt tetap (EN, ID, kode) dengan sampling `temperature=0.7, top_p=0.9`, simpan ke `runs/<id>/samples.txt`.
- **Ctrl+C / sinyal / `--max-hours`:** tangkap (`KeyboardInterrupt`, `SIGTERM` jika ada), simpan checkpoint, tulis `status="paused"`, keluar rapi dengan pesan cara `--resume`.
- **Deteksi loss spike + rollback otomatis:** pertahankan EMA loss (decay 0.98). Jika loss NaN/Inf **atau** `loss > 2.0 * ema` selama 5 step berturut-turut → muat checkpoint rolling terbaru yang sehat, **maju melewati 2000 sequence data** (lewati data pemicu), turunkan LR 20% (`lr_scale *= 0.8` untuk sisa training, tercatat di `run.json`), lanjut. Maksimum 3 rollback; sesudahnya berhenti dengan `status="stopped_spike"` dan pesan jelas.
- Seed: `seed=1337` (bisa diubah) untuk init model, dataloader, dropout (tidak ada dropout).

### 11.2 Perintah `pretrain`
```
python main.py pretrain                  # run baru (run_001, run_002, ...)
python main.py pretrain --resume         # lanjut dari checkpoint terakhir run terakhir
python main.py pretrain --resume run_001 --step 40000   # dari checkpoint tertentu
python main.py pretrain --new --lr 1e-3  # run baru dengan LR berbeda (config lain tetap)
python main.py pretrain --max-hours 11   # berhenti rapi sebelum sesi habis (Colab)
python main.py pretrain --auto           # supervisor: jika proses mati (exit code != 0), jalankan ulang otomatis dengan --resume, maks 20 kali, jeda 30 detik
python main.py pretrain --out-dir "G:/MyDrive/llm_runs"   # simpan run/checkpoint di Drive
```
Run baru menyalin config yang dipakai ke `runs/<id>/config.json` (supaya reproducible).
Jika `--resume` tetapi tidak ada checkpoint → pesan error + saran. Jika konfigurasi proyek berubah sejak checkpoint dibuat (hash config beda) → tolak dengan penjelasan.

### 11.3 Checkpoint (`checkpoint.py`)

Struktur di `runs/<id>/ckpt/`:
```
step_00034000/
   model.pt          # state_dict model (fp32)
   optim.pt          # state_dict optimizer (+ scaler jika fp16)
   train_state.json  # step, tokens_seen, lr_scale, rollback_count, loss_ema, config_hash
   data_state.json   # state dataloader (Bagian 10.3)
   rng_state.pt      # RNG torch (CPU+CUDA), numpy, python
   COMPLETE          # file kosong, ditulis TERAKHIR (penanda checkpoint utuh)
milestone_10/ milestone_20/ ...   # sama strukturnya; tidak pernah dihapus
best/                              # salinan ckpt dengan val loss terbaik
latest.json                        # {"path": "step_00034000"} diperbarui atomic
final/                             # model.safetensors (bf16) + config.json, saat selesai
```
Aturan:
- **Atomic:** tulis ke folder `<nama>.tmp`, `fsync`, buat `COMPLETE`, lalu `os.replace` ke nama akhir. Checkpoint tanpa `COMPLETE` dianggap rusak dan diabaikan/dibersihkan saat start.
- **Rotasi:** simpan 3 `step_*` terakhir; hapus yang lebih lama (kecuali milestone dan best).
- Frekuensi: Bagian 6.10 (token ATAU 30 menit, mana duluan; keduanya dicatat di `config.json` dan bisa diganti `--ckpt-every-tokens 500M`, `--ckpt-every-min 20`, `--keep 3`).
- Muat: `load_checkpoint(path)` memulihkan model, optimizer, scaler, step, tokens, data state, RNG. Setelah dimuat, cetak: `Resume dari step X (Y token, Z%)`.
- **Tes wajib:** latih 50 step → simpan di step 20 → muat → lanjut sampai 50 → loss dan bobot akhir sama dengan run tanpa interupsi (toleransi 1e-5 pada CPU dengan data dummy dan seed sama).
- Cek ruang disk sebelum menyimpan; jika kurang dari 2× ukuran checkpoint → peringatan keras dan lewati rotasi hapus yang berbahaya (jangan crash).

---

## 12. TAHAP LAIN

### 12.1 `bench` (`bench.py`)
- Bangun model sesuai config. Untuk `micro_batch` = 1, 2, 4, 8, ... (kelipatan dua) jalankan 3 step pemanasan + 20 step terukur dengan data acak (tanpa data asli), catat tok/s dan memori puncak. Berhenti saat OOM (tangkap `torch.cuda.OutOfMemoryError`, bersihkan cache). Coba juga dengan/ tanpa `gradient_checkpointing`.
- Pilih konfigurasi dengan tok/s tertinggi yang memori puncaknya ≤ 92% VRAM. Simpan `bench.json`: `{config_hash, gpu, micro_batch, grad_accum, grad_checkpointing, tok_per_s, peak_mem_gb, dtype, date}`.
- Hitung ulang dan cetak ETA pretraining: `tokens_total / tok_per_s`, plus biaya bila GPU berbayar.
- Hanya berjalan di GPU CUDA; di CPU cetak pesan bahwa bench tidak berlaku.

### 12.2 `pilot` (`pilot.py`)
Tujuan: memvalidasi pipeline murah sebelum run besar. Menjalankan model **preset kecil** (otomatis 0.05B bila model target ≥0.3B, selain itu 0.05B juga) pada data asli yang sudah di-`prep`:
1. **Sweep LR:** tiga run pendek (default 300M token tiap run, `--pilot-tokens`) dengan LR `0.5×, 1×, 2×` dari rumus untuk model kecil itu. Bandingkan val loss akhir; cetak tabel dan LR terbaik. Beri catatan: hasil ini hanya panduan; model besar biasanya cocok dengan LR sedikit lebih rendah.
2. **Cek kesehatan:** loss turun monoton secara umum, tidak ada NaN, tok/s wajar, memori stabil.
3. **Tes resume:** simpan, matikan proses, resume, pastikan loss tersambung.
4. **Sampel teks** EN/ID/kode dicetak.
5. **Tokenizer fertility** (dari `tokenizer_meta.json`) dicetak ulang.
Simpan `pilot/result.json` dengan LR terbaik; `plan` pretrain menawarkan memakainya (pengguna menjawab y/n).

### 12.3 `sft` (`sft.py`)
- Prasyarat: checkpoint pretrain `final/` atau pilihan `--from <run>[:step]`.
- Tokenisasi sampel lewat `chat_template.render` + mask (Bagian 8). Buang sampel > `max_seq_len`. Susun batch dengan pengelompokan panjang (length-grouped) dan padding; jangan pakai packing di MVP.
- Loss: cross-entropy pada token dengan label ≠ -100, **dirata-rata per token di seluruh batch** (jumlahkan loss lalu bagi total token berlabel dalam batch efektif, termasuk grad-accum).
- Hyperparameter: Bagian 6.9. Checkpoint dengan aturan sama (Bagian 11.3) tetapi frekuensi per `epoch/4` dan best berdasar val loss SFT.
- Run SFT disimpan sebagai `runs/<id>/` dengan `kind="sft"`.
- Sebelum training, cetak 3 sampel hasil render + mask (berwarna: token yang dilatih vs tidak) agar pengguna bisa memeriksa format.

### 12.4 `eval` (`evaluate.py`)
**Pretrain** (`eval --run run_001`): val loss/perplexity/BPB per sumber, tabel; sampel kelanjutan teks; opsional `--bench` menjalankan `lm_eval` (jika terpasang) pada tugas yang dipilih pengguna (misal hellaswag, arc_easy; untuk Indonesia, sebutkan tugas yang tersedia di instalasi pengguna — jangan mengarang nama tugas).

**SFT** (`eval --run run_002`), tes otomatis pada eval set SFT + set tes bawaan:
1. **Kepatuhan mode:** 50 prompt dengan `/no_think` → blok think harus kosong persis; 50 prompt dengan `/think` → think tidak kosong dan tertutup `</think>`. Laporkan persen patuh.
2. **Validitas tool call:** parse JSON di `<tool_call>`; nama ada di daftar tools; argumen sesuai skema. Laporkan persen valid.
3. **Ketepatan memutuskan tool:** pada sampel "perlu tool" vs "tidak perlu tool", akurasi keputusan.
4. **End-to-end:** jalankan tool nyata dari `tools_sandbox.py` (`calculate` dengan `ast` aman, `search` mock dari `tests/mock_kb.json`, `get_time`), kembalikan hasil ke model, bandingkan jawaban akhir dengan kunci (exact/contains).
5. **Bahasa balasan** sama dengan bahasa pertanyaan (deteksi heuristik sederhana: daftar stopword EN vs ID).
6. **Identitas:** 10 prompt identitas harus menyebut nama model/pembuat dari `project.json`.
7. **Multi-turn:** 20 dialog 3 giliran; periksa tidak ada pengulangan kalimat ≥ 20 token yang sama berturut-turut (deteksi degenerasi).
Cetak tabel skor + 5 contoh gagal. Simpan `runs/<id>/eval.json`. `eval --compare run_001 run_002` membandingkan dua run berdampingan.

### 12.5 `chat` (`chat.py`)
REPL: `python main.py chat --run run_002 [--step 40000] [--mode think|no_think|auto] [--tools] [--force-nothink] [--temp 0.6] [--top-p 0.9] [--rep-penalty 1.05] [--max-new 512]`.
- Perintah dalam chat: `/think`, `/nothink`, `/reset`, `/tools on|off`, `/system <teks>`, `/exit`.
- Dengan `--tools`: model boleh memanggil tool sandbox; program mengeksekusi tool, mengembalikan hasilnya, dan melanjutkan generasi otomatis (maks 5 iterasi), mencetak tiap tool call/hasilnya dengan warna berbeda.
- Blok `<think>` dicetak redup/terpisah (bukan dicampur dengan jawaban).
- Generasi memakai `model.generate` dengan `eos_token_id=<|im_end|>`.

### 12.6 `export` (`export.py`) — VERSI MINIMAL
Export **tidak** mengonversi ke GGUF di dalam toolkit. Tugasnya hanya menghasilkan folder Hugging Face yang bersih dan benar, karena itu input standar untuk semua alat lain (llama.cpp, Unsloth, vLLM).
- Simpan di `export/<nama>/`: `model.safetensors` (bf16), `config.json` (Qwen3), `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` (berisi `chat_template` Jinja Bagian 8.4, `eos_token="<|im_end|>"`, special token), `special_tokens_map.json`.
- Tes: `AutoModelForCausalLM.from_pretrained(folder)` + `AutoTokenizer.from_pretrained(folder)` harus memuat dan menghasilkan teks; `apply_chat_template` harus sama dengan renderer Python.
- Setelah selesai, cetak petunjuk (tidak dijalankan otomatis), dengan catatan **"cek dokumentasi llama.cpp/Unsloth/Ollama terbaru, perintahnya bisa berubah"**:
  - Jalur A (disarankan, tanpa Unsloth): `python <llama.cpp>/convert_hf_to_gguf.py export/<nama> --outfile model.gguf --outtype bf16`, lalu `llama-quantize model.gguf model-q8_0.gguf Q8_0`.
  - Jalur B (opsional): Unsloth `save_pretrained_gguf` bila model berhasil dimuat lewat `FastLanguageModel`. Tidak dijamin untuk model hasil pretrain sendiri; jalur A lebih pasti.
  - Ollama: `Modelfile` berisi `FROM ./model-q8_0.gguf` dan `PARAMETER stop "<|im_end|>"`, lalu `ollama create <nama> -f Modelfile`.
- Peringatan wajib dicetak: penyebab utama hasil jelek/ngelantur setelah dikonversi adalah **chat template atau token EOS yang tidak sama** dengan saat training. Karena itu tes Jinja == Python (Bagian 8.4) tidak boleh dilewati.

### 12.7 `doctor` (`doctor.py`)
Periksa dan laporkan (OK/WARN/ERROR + saran): versi Python, versi `torch` dan CUDA tersedia/tidak, nama GPU & VRAM, `bf16` didukung, versi `transformers` ≥ 4.51.0 (jika lebih rendah → pesan `KeyError: 'qwen3'` penyebabnya), `tokenizers`/`safetensors`/`numpy`/`tqdm` terpasang, `rich` opsional, ruang disk kosong di folder proyek vs kebutuhan (Bagian 6.6), RAM, jumlah CPU, OS (peringatan Windows: `torch.compile` mati, saran WSL2 untuk performa lebih baik), tes kecil: alokasi tensor di GPU + satu forward/backward kecil.

---

## 13. CLI, `init`, DAN UX (`cli.py`, `ui.py`, `guide.py`)

### 13.1 Daftar perintah
```
python main.py init                      # tanya jawab + rencana + buat folder + arahan data
python main.py plan 0.5B [--mode M] [--tokens 80B] [--gpu H100] [--budget-usd 500] [--hours 100]   # kalkulator cepat tanpa proyek
python main.py status                    # posisi proyek (auto-detect) + langkah berikutnya
python main.py check [--quick]           # validasi data
python main.py tokenizer                 # latih tokenizer
python main.py prep                      # tokenisasi -> shard .bin
python main.py bench                     # ukur kecepatan GPU
python main.py pilot                     # model kecil: sweep LR + tes resume
python main.py pretrain [opsi Bagian 11.2]
python main.py sft [--from run_001[:step]]
python main.py eval [--run ID] [--compare A B] [--bench]
python main.py chat [opsi Bagian 12.5]
python main.py export [--run ID]
python main.py runs                      # tabel semua run (id, jenis, token, loss, status)
python main.py doctor                    # cek environment
python main.py help [perintah]           # bantuan; tanpa argumen = semua perintah
python main.py guide                     # panduan urut dari nol sampai chat
python main.py replan                    # ubah jawaban init (ukuran, token, GPU) dan cetak rencana baru
python main.py config show [--why]      # lihat semua setting + alasan nilai auto (Bagian 18)
python main.py config set kunci=nilai    # kunci manual (contoh: context_final=8192)
python main.py config auto kunci         # kembalikan satu setting ke auto
python main.py plan --set kunci=nilai    # simulasi dampak setting tanpa menyimpan
```
Opsi global: `--project <folder>` (default: folder kerja saat ini jika ada `project.json`), `--yes` (setuju semua konfirmasi, untuk skrip), `--dry-run` (tampilkan apa yang akan dilakukan tanpa menjalankan).

### 13.2 Alur `init` (pertanyaan, urut persis, dengan nilai default dan validasi)
```
1. Nama proyek? [bahasa-0.6B]
2. Ukuran model (miliar parameter)? 0.05 / 0.1 / 0.3 / 0.6 / 1 / angka lain [0.6]
3. Mode token? 1=chinchilla(20x) 2=balanced(100x) 3=overtrain(500x) 4=max(3000x) 5=jumlah manual 6=berdasarkan budget [2]
     (jika 5: tanya jumlah token, terima "60B", "2T", "500M"; jika 6: tanya $ atau jam GPU)
4. GPU? (deteksi otomatis; tampilkan hasil deteksi dan minta konfirmasi) [hasil deteksi]
5. Fitur: tool calling? [Y/n]   think/no-think? [Y/n]   (vision: tidak tersedia di versi ini)
6. Perkiraan data bahasa Indonesia UNIK yang kamu punya (miliar token, kosongkan jika belum tahu)? []
7. Perlu long-context 32k di tahap akhir? [y/N]
8. Nama model (untuk data identitas)? [Bahasa]   Pembuat? [Pengguna]
9. Folder untuk run/checkpoint? [runs]   (bisa path Google Drive)
```
Validasi input: angka harus valid, pilihan harus ada; input salah → pesan dan tanya ulang (maks 3x lalu pakai default). Jika `--yes` dan semua opsi diberikan lewat flag/JSON (`--answers jawaban.json`), jalan tanpa pertanyaan (untuk tes).

### 13.3 Output `init` (semuanya di terminal)
Cetak berurutan: (1) ringkasan jawaban; (2) tabel arsitektur + jumlah parameter nyata; (3) token, compute, waktu, biaya, tabel perbandingan GPU; (4) hyperparameter dan checkpoint; (5) estimasi memori + peringatan muat/tidak; (6) komposisi data pretrain per tahap dalam persen **dan** token absolut, plus total porsi Indonesia/kode; (7) rencana SFT per kategori (persen + jumlah sampel); (8) kebutuhan storage; (9) **arahan data**: daftar folder, target tiap folder, format + contoh baris (Bagian 7), dan di akhir "Langkah kamu sekarang: isi folder di atas, lalu jalankan `python main.py check`".

Contoh tampilan (angka harus dihasilkan oleh kode, bukan di-hardcode):
```
=== RENCANA: bahasa-0.6B ===
Arsitektur : 30 layer | hidden 1280 | 20 Q-head / 4 KV-head | FFN 3456 | vocab 64k
Parameter  : 598,097,920 (0.60B) | non-embedding 516,177,920
Token      : 59.8B (rasio 100x, mode balanced)
Compute    : 2.15e20 FLOPs | H100: ~199 jam (8.3 hari) ~ $497
Batch      : 1,048,576 token/step (micro 16 x accum 16 x seq 4096) | 57,030 step
LR         : puncak 5.4e-4 | warmup 570 step | WSD (stable 80%, decay 20%)
Checkpoint : tiap 598M token / 30 menit | simpan 3 terakhir | milestone tiap 10%
Memori     : statis 9.6 GB + aktivasi ~... -> muat di H100 (80 GB), TIDAK muat di 4060 (8 GB)

Pretrain (token): Stage A 47.8B | Stage B 12.0B
  en_web 17.2B (36% A) | code 12.0B+3.6B | id_web 5.7B | math 3.8B+3.0B | ...
  Porsi Indonesia total: ~xx% | kode: ~xx%
SFT: 239,000 sampel | chat 71.7k | think 47.8k | tool_call 35.9k | multiturn 35.9k | code 23.9k | format 11.9k | refusal 7.2k | identity 4.8k
Storage: tokenized ~120 GB | ruang kerja disarankan ~400 GB
```
(`non-embedding` = `total - V*d` = 516,177,920 untuk preset 0.6B; periksa dengan tes.)

### 13.4 `help` dan `guide`
- `help`: daftar semua perintah, satu baris penjelasan tiap perintah. `help <perintah>`: penjelasan, opsi, contoh pemakaian, kesalahan umum dan solusinya.
- `guide`: panduan berurutan 1 sampai 10 (dari `init` sampai `export`), tiap langkah berisi: apa yang dilakukan, perintahnya, hasil yang diharapkan, dan "kalau gagal".
- Isi `help`/`guide` disimpan sebagai string terstruktur di `guide.py` (bukan file terpisah).

### 13.5 Aturan UX
- Warna (jika `rich` ada): hijau = OK, kuning = peringatan, merah = error, biru = info. Tanpa `rich`, pakai awalan teks `[OK]`, `[WARN]`, `[ERROR]`.
- Angka besar diformat: `59.8B`, `1.8T`, `598M`; waktu: `199 jam (8.3 hari)`.
- Setiap perintah panjang menampilkan progress (tqdm) dan estimasi sisa waktu.
- Konfirmasi (`y/N`) sebelum tindakan yang menghabiskan >1 jam atau menimpa file.
- Jika dijalankan di luar folder proyek: pesan "Tidak menemukan project.json. Jalankan `python main.py init` atau gunakan `--project`."

---

## 14. KANDIDAT DATASET (hanya untuk dicetak sebagai saran; JANGAN diunduh otomatis di MVP)

Cetak daftar ini di `guide`/`init` dengan catatan **"verifikasi dulu bahwa dataset masih ada, dan cek lisensinya sebelum dipakai"**. Jangan menambah ID lain yang tidak yakin.

Pretrain:
- `en_web`/`en_hq`: `HuggingFaceFW/fineweb-edu`, `HuggingFaceTB/smollm-corpus`
- `id_web`/`id_hq`: `HuggingFaceFW/fineweb-2` (konfigurasi bahasa Indonesia `ind_Latn`), Wikipedia Indonesia (`wikimedia/wikipedia`, konfigurasi bahasa `id`)
- `code`: `bigcode/the-stack-v2` (perlu persetujuan akses), subset Python-edu dari `HuggingFaceTB/smollm-corpus`
- `math`: `HuggingFaceTB/finemath`, `open-web-math/open-web-math`
- `books_wiki`: Wikipedia EN/ID (`wikimedia/wikipedia`)
- `synthetic`: `Cosmopedia` (bagian dari smollm-corpus) + terjemahan EN→ID buatan sendiri

SFT:
- `chat`/`multiturn`: `HuggingFaceTB/smoltalk` (Inggris; Indonesia perlu diterjemahkan atau dibuat sintetis)
- `tool_call`: `NousResearch/hermes-function-calling-v1`, `Salesforce/xlam-function-calling-60k`, `Team-ACE/ToolACE`, `glaiveai/glaive-function-calling-v2` (semua perlu dikonversi ke skema Bagian 7.2)
- `think`: `open-thoughts/OpenThoughts-114k`, `open-r1/OpenR1-Math-220k` (konversi `<think>` ke field `reasoning`)
- Sampel Indonesia untuk semua kategori SFT: dibuat dengan terjemahan atau generasi sintetis oleh model besar; pengguna yang menyiapkan.

---

## 15. FASE IMPLEMENTASI & TES PENERIMAAN

Kerjakan berurutan. Setelah tiap fase: jalankan tes, tampilkan output, **berhenti dan tunggu konfirmasi**.

### Fase 1 — Perencana, init, status, bantuan (belum butuh GPU)
File: `main.py, cli.py, ui.py, state.py, gpu_table.py, planner.py, settings.py, advisor.py (Bagian 18), data_spec.py, check.py (bagian pretrain+SFT), guide.py, doctor.py, tests/make_dummy_data.py, tests/test_planner.py, tests/test_status.py`.
Tes penerimaan:
1. `total_params` Qwen3-0.6B = 596,049,920 dan non-embedding = 440,467,456; preset 0.6B = 598,097,920; semua preset di tabel 6.2 persis sama.
2. `plan 0.6B --mode balanced --gpu H100`: token 59.8B; FLOPs ≈ 2.146e20; waktu ≈ 199 jam; `peak_lr` ≈ 5.41e-4; `batch_tokens`=1,048,576; steps 57,030.
3. `plan 0.5B`: solver menghasilkan config dengan error ≤3% dan cetak kandidat.
4. `plan 0.6B --budget-usd 500 --gpu H100` memberi token ≈ 60B.
5. Komposisi A/B: tiap kolom persis 100; `needed` per folder dihitung benar; skenario `id_unique_tokens_b=2` dan `max` memicu ERROR epochs.
6. `init --answers jawaban.json --yes` membuat folder + `project.json` + folder data dengan `README.txt`.
7. `status` pada proyek kosong, pada data dummy, dan simulasi `run.json` dengan heartbeat basi (harus mendeteksi "berhenti tiba-tiba").
8. `check` pada data dummy: mendeteksi baris rusak, duplikat, folder kurang, ketidakseimbangan bahasa.
9. `help`, `guide`, `doctor` berjalan tanpa error di Windows dan Linux.

### Fase 2 — Tokenizer, prep, model, bench
File: `chat_template.py, tokenizer_train.py, prep.py, dataset.py, model.py, bench.py, tests/test_chat_template.py`.
Tes: tokenizer round-trip, digit terpisah, ID special token stabil; `prep` pada data dummy (dedup, leak filter, val split, manifest, resumable); dataloader deterministik; jumlah parameter model nyata = rumus; render chat + mask (Bagian 8.3) pada 20 sampel; `bench` jalan di GPU/CPU-fallback.

### Fase 3 — Pretraining
File: `checkpoint.py, train.py` (+ pembaruan `status`).
Tes: latih model 0.05B pada data dummy 200 step: loss turun; tes resume identik (Bagian 11.3); rotasi checkpoint (hanya 3 `step_*` + milestone + best); `--max-hours` menyimpan lalu `paused`; simulasi crash (kill proses) lalu `--resume` dan `--auto`; spike rollback dengan data yang sengaja dirusak; `status` menampilkan progres dan rekomendasi benar di tiap kondisi.

### Fase 4 — SFT, chat
File: `sft.py, chat.py, tools_sandbox.py` (+ `make-identity` helper yang membuat `data/sft/identity/auto.jsonl` dari template EN/ID dengan nama model/pembuat).
Tes: SFT pada data dummy; mask benar; hasil `chat` bisa memanggil tool sandbox end-to-end (walau model dummy jelek, mekanismenya harus jalan: parse, eksekusi, kembalikan hasil).

### Fase 5 — Eval, pilot, export, auto
File: `evaluate.py, pilot.py, export.py` (+ `convert`, `fetch` opsional bila diminta).
Tes: eval menghasilkan semua metrik; Jinja == Python render; export bisa dimuat ulang via `from_pretrained`; `pilot` jalan di data dummy.

---

## 16. DI LUAR SCOPE (JANGAN DIBUAT KECUALI DIMINTA)
Tahap vision (flag `vision` ada di settings tetapi tahapnya belum tersedia, lihat Bagian 18), optimizer Muon, RL/GRPO/DPO, multi-GPU/FSDP/DeepSpeed, MoE, near-dedup MinHash (sarankan `datatrove` atau `text-dedup` sebagai alat eksternal), konversi GGUF otomatis, document masking, `fetch` otomatis, level reasoning low/medium/high, GUI/web. Jangan menambah dependensi besar di luar daftar Bagian 2.

## 17. CATATAN KEJUJURAN UNTUK PENGGUNA (cetak di `init`)
- Angka (rasio token, LR, TFLOPs efektif, harga GPU) adalah **titik awal**, bukan kebenaran. `bench` dan `pilot` yang menentukan angka nyata.
- Model 0.6B dari nol dengan ≤100B token hampir pasti kalah dari Qwen3-0.6B (dilatih 36T token). Jalur lebih hemat untuk hasil produk: continued pretraining dari model yang sudah ada. Toolkit ini valid untuk belajar dan untuk model buatan sendiri.
- Kualitas data lebih menentukan daripada jumlah token.

---

## 18. SETTINGS & ADVISOR (`settings.py`, `advisor.py`) — MENIMPA BAGIAN 6.5, 6.8, 13.1, 13.2

### 18.1 Konsep
- Setiap setting punya `mode`: `auto` (dihitung program) atau `manual` (dikunci pengguna). Default semua `auto`.
- Disimpan di `project.json` → `"settings"`:
```json
"settings": {
  "context_final": {"value": 8192, "mode": "auto", "reason": "tool call aktif: hasil search butuh konteks panjang"},
  "vision": {"value": false, "mode": "auto", "reason": "model <0.6B / VRAM <24 GB"}
}
```
- Setting `manual` **tidak pernah ditimpa** oleh `replan`/`init` ulang. Setting `auto` dihitung ulang setiap `replan`, `config set`, atau perubahan GPU/ukuran/token.
- Urutan perhitungan (karena saling bergantung): `goal` → `profile` → `languages` → `features` → `vocab_size` → `arch_style` → `context_pretrain` → `context_final` → `code_focus` → `think_max_tokens` → `precision` → `grad_checkpointing` → `micro_batch` → `ckpt_every` → sisanya. Tiap fungsi aturan murni (input: answers + setting sebelumnya, output: nilai + alasan) sehingga mudah dites.
- Perintah (lihat 13.1): `config show` (tabel: kunci, nilai, mode auto/manual, alasan jika `--why`), `config set k=v` (validasi nilai, lalu jalankan advisor), `config auto k`, `plan --set k=v` (hitung ulang rencana dengan override sementara, cetak selisih biaya/waktu/memori dibanding rencana tersimpan, TIDAK menyimpan).

### 18.2 Daftar setting, nilai yang valid, dan aturan auto

| Kunci | Nilai valid | Aturan auto |
|---|---|---|
| `goal` | `belajar`, `produk`, `maksimal`, `custom` | Ditanya di `init` (default `produk`). Preset di 18.4. |
| `profile` | `laptop`, `colab`, `cloud` | `colab` jika `"google.colab" in sys.modules` atau env `COLAB_GPU` ada; `laptop` jika nama GPU mengandung "Laptop"/"Mobile" atau VRAM ≤ 12 GB; selain itu `cloud`. Efek profil di 18.3. |
| `languages` | `en+id`, `id`, `en` | `en+id`. |
| `vision` | `true`, `false` | `false`. Disarankan `true` hanya jika ukuran ≥ 0.6B **dan** VRAM ≥ 24 GB **dan** teks sudah selesai. Lihat 18.5. |
| `think` | `true`, `false` | `false` jika ukuran < 0.1B; selain itu `true`. |
| `tool_call` | `true`, `false` | `true`. |
| `think_max_tokens` | 512 / 1024 / 2048 / 4096 | 512 jika N ≤ 0.15B; 1024 jika N ≤ 0.8B; 2048 jika lebih besar. Dipakai: `check` SFT (peringatan/pembuangan sampel `reasoning` lebih panjang dari batas), eval, dan `--max-new` default di chat. |
| `context_pretrain` | 1024 / 2048 / 4096 / 8192 | N ≤ 0.15B → 2048; N > 0.15B dan VRAM ≥ 24 GB → 4096; selain itu 2048. (Menggantikan aturan `seq_len` di 6.5.) |
| `context_final` | 2048 / 4096 / 8192 / 16384 / 32768 | `tool_call` aktif → minimal 8192. `long_context` diminta di `init` **dan** total token ≥ 20B **dan** VRAM ≥ 24 GB → 32768. Selain itu 4096. Tidak boleh < `context_pretrain`. |
| `code_focus` | `ringan`, `sedang`, `berat` | `sedang`. Pengaruh: bobot `code` Stage A = 10% / 25% / 35%; bobot lain di Stage A dikecilkan proporsional. Kategori SFT `code` = 5% / 10% / 20% (sisa diambil dari `chat`). |
| `vocab_size` | 32000 / 64000 | 32000 jika N ≤ 0.2B, atau `languages != en+id` dan N ≤ 0.6B; selain itu 64000. Wajib ≤ 65.535. |
| `arch_style` | `preset`, `dalam_ramping`, `seimbang` | `preset` (pakai tabel 6.2 jika ukuran cocok). `dalam_ramping`/`seimbang` memaksa solver 6.2 dengan target `d/L` = 36 / 44. |
| `precision` | `bf16`, `fp16` | `bf16` jika `torch.cuda.is_bf16_supported()`, selain itu `fp16` (+GradScaler). |
| `grad_checkpointing` | `true`, `false` | `true` jika estimasi memori (6.7) tanpa checkpointing > 85% VRAM; selain itu `false`. |
| `micro_batch` | integer | Dari `bench`; sebelum `bench`, dari estimasi memori. |
| `ckpt_every` | `{tokens, menit, keep}` | Ikut profil (18.3). |
| `dedup` / `leak_filter` | `true`, `false` | `true` keduanya. |
| `optimizer` | `adamw` | `adamw` (satu-satunya yang tersedia di versi ini). |

**Aturan turunan wajib diterapkan ke planner:**
1. **Stage C (menimpa 6.8):** aktif jika `context_final > context_pretrain` (bukan hanya long-context). Bobot Stage C = 5% dari total token. `seq_len` Stage C = `context_final`. `rope_theta` Stage C = `500000 * (context_final / context_pretrain)` (4k→8k: 1.000.000; 4k→32k: 4.000.000; model diubah config-nya saat pindah tahap). Komposisi Stage C: jika `context_final ≥ 32768` pakai kolom Stage C di tabel 6.8 (`long_docs`, `long_code`, dst.); jika lebih kecil, pakai bobot yang sama dengan Stage A (`long_docs` tidak diperlukan).
2. **SFT `max_seq_len`** = `min(context_final, 4096)` jika VRAM < 24 GB, selain itu `context_final`.
3. **`think=false`:** kategori SFT `think` = 0%; tag `/think`/`/no_think` tidak ditambahkan; template tetap menulis blok think kosong (konsisten). Persentase 20% dialihkan: `chat` +10, `code` +5, `multiturn` +5. Folder `reasoning` dan Stage B `reasoning` dialihkan ke `math`.
4. **`tool_call=false`:** kategori `tool_call` = 0%, tidak ada blok `# Tools`; 15% dialihkan: `chat` +8, `format` +4, `multiturn` +3.
5. **`languages=id` atau `en`:** hapus folder bahasa lain dari rencana (`en_*` atau `id_*`), bobotnya dibagi proporsional ke folder lain dalam tahap yang sama; target bahasa SFT menjadi 100% satu bahasa.
6. **`vision=true`:** hanya menyimpan flag dan memunculkan saran 18.5; **tidak mengubah** pipeline teks.

### 18.3 Efek profil
| | laptop | colab | cloud |
|---|---|---|---|
| Ukuran maksimal yang disarankan | yang `mem_static` ≤ 60% VRAM | sesuai GPU sesi | bebas |
| `context_pretrain` | 2048 | 2048 (T4/L4), 4096 (A100) | 4096 |
| `grad_checkpointing` | cenderung `true` | ikut estimasi | ikut estimasi |
| `ckpt_every` | 15 menit / keep 3 | 10 menit / keep 3 | token-based (6.10) / keep 3 |
| Lokasi run | lokal | Drive (tanya path) | lokal/volume |
| `pretrain` default | – | `--max-hours 11 --auto` | `--auto` |
| `torch.compile` | mati di Windows | mati | hidup di Linux |
| Pesan khusus | saran WSL2, cooling, steker | saran simpan ke Drive, sesi bisa putus | saran matikan instance setelah selesai |

### 18.4 Preset `goal`
| goal | token_mode | think | tool_call | context_final | long_context | catatan |
|---|---|---|---|---|---|---|
| `belajar` | `chinchilla` | `false` jika N<0.3B | `true` | 2048–4096 | tidak | murah dan cepat, ukuran kecil dianjurkan |
| `produk` | `balanced` | auto | `true` | 8192 | tidak | default |
| `maksimal` | `overtrain` (atau `max` jika data unik cukup) | `true` | `true` | 8192 (32768 jika syarat terpenuhi) | ya | cek data unik dulu |
| `custom` | pengguna menentukan semua | | | | | |
Memilih `goal` hanya mengisi nilai `auto`; setting manual tetap menang.

### 18.5 Vision (flag, bukan tahap)
- `vision=true` menampilkan di `status` satu baris: `[ ] vision (belum tersedia di versi ini)`. Tidak ada perintah vision.
- Saran yang dicetak: "Vision dikerjakan setelah model teks selesai dan stabil: encoder gambar (misalnya SigLIP) + projector + fine-tune pada data gambar-instruksi (Indonesia perlu diterjemahkan/dibuat sintetis). Pada model <0.6B hasilnya biasanya lemah. Encoder bisa lebih besar dari LLM-nya sendiri."
- Aturan konflik: `vision=true` dengan ukuran < 0.3B → `WARN`.

### 18.6 Advisor: aturan saran dan peringatan
Dijalankan: akhir `init`, `replan`, `config set`, `plan --set`, dan di awal `pretrain`/`sft` (hanya tingkat `ERROR` yang memblokir; bisa dilewati dengan `--force`). Setiap temuan: `{id, level: info|warn|error, pesan, saran, setting_terkait}`. Tampil dikelompokkan per level, ERROR dulu.

| ID | Kondisi | Level | Pesan/saran (Bahasa Indonesia) |
|---|---|---|---|
| A01 | `mem_static` > 85% VRAM | error | "Model tidak muat untuk full training di GPU ini. Pilih GPU lain atau ukuran ≤ X B." |
| A02 | estimasi total memori > VRAM walau grad checkpoint aktif dan micro_batch=1 | error | "Turunkan context_pretrain atau ukuran model." |
| A03 | `context_final` ≥ 16384 dan VRAM < 24 GB | warn | "Konteks panjang di GPU ini berisiko OOM; saran 4096–8192." |
| A04 | `tool_call=true` dan `context_final` < 4096 | warn | "Hasil search/tool sering panjang; saran context_final 8192." |
| A05 | `think=true` dan N < 0.1B | warn | "Penalaran model sekecil ini biasanya buruk; saran think=false." |
| A06 | `epochs` suatu folder > 4 (6.8) | warn (error jika > 8) | "Data <folder> diulang N kali. Tambah data unik/sintetis atau turunkan bobotnya." |
| A07 | porsi Indonesia total < 10% dan `languages=en+id` | warn | "Porsi bahasa Indonesia kecil; kemampuan Indonesia akan lemah." |
| A08 | rasio token/param > 1000 dan data unik total < 30% dari token latih | warn | "Terlalu banyak pengulangan data; pertimbangkan mode lebih rendah." |
| A09 | biaya estimasi > budget (jika `--budget-usd` diberikan) | error | "Biaya $X melebihi budget $Y. Kurangi token atau ukuran." |
| A10 | `tool_call=true` tetapi folder `sft/tool_call` kosong saat `sft` | error | "Isi data tool call dulu, atau matikan tool_call." |
| A11 | `vocab_size` > 65535 | error | "Vocab harus ≤ 65.535 (uint16)." |
| A12 | `profile=colab` dan checkpoint tidak di Drive | warn | "Sesi Colab bisa putus; gunakan --out-dir ke Google Drive." |
| A13 | `profile=laptop` dan estimasi waktu > 14 hari | warn | "Terlalu lama untuk laptop; pertimbangkan sewa GPU atau kurangi token." |
| A14 | `vision=true` | info | Teks saran 18.5. |
| A15 | ukuran ≥ 0.3B, token < 20× parameter (di bawah chinchilla) | warn | "Model akan undertrained; naikkan token." |
| A16 | N=0.6B dan token ≤ 100B | info | "Ekspektasi realistis: bahasa lancar, tool call dasar, pengetahuan terbatas; kemungkinan masih di bawah Qwen3-0.6B." |
| A17 | `think_max_tokens` + panjang prompt tipikal > `context_final` | warn | "Batas think terlalu besar untuk konteks ini." |
| A18 | setting manual berbeda dari saran auto | info | "Kamu menimpa auto: <kunci>=<manual> (auto menyarankan <nilai>)." |

### 18.7 Tampilan contoh
```
$ python main.py config show --why
kunci               nilai    mode    alasan
context_pretrain    4096     auto    N=0.6B dan VRAM 80 GB
context_final       8192     auto    tool_call aktif: hasil search butuh konteks panjang
vision              false    auto    ukuran/VRAM belum memadai; kerjakan setelah teks selesai
think               true     auto    ukuran >= 0.1B
code_focus          sedang   manual  (kamu atur)
...

SARAN
[WARN] A04 tool_call aktif tapi context_final=4096. Saran: set context_final=8192.   (config set context_final=8192)
[INFO] A16 Ekspektasi realistis untuk 0.6B dengan 59.8B token: ...
```
`plan --set context_final=32768` mencetak selisih, contoh: `Memori: +18 GB | Waktu: +2 jam | Peringatan baru: A03`.

### 18.8 Penambahan pada `init` (menimpa 13.2)
Setelah pertanyaan 4 (GPU), tambahkan: **4b.** Tujuan? `1=belajar 2=produk 3=maksimal 4=custom` [2]. **4c.** Profil lingkungan? (deteksi otomatis, minta konfirmasi) [hasil deteksi]. Pertanyaan 5 (fitur) diganti: tool calling [Y/n], think/no-think [Y/n], vision [y/N]. Setelah rencana tercetak, tampilkan blok **SARAN** (advisor) dan tanya: "Ubah setting? (n/ketik `config set ...`)". Semua nilai lain tetap `auto`.

### 18.9 Tambahan Fase 1
File tambahan Fase 1: `settings.py`, `advisor.py`, `tests/test_settings.py`. Tes penerimaan:
1. Tiap aturan auto di 18.2 diuji pada ≥ 3 kombinasi (ukuran, VRAM, fitur).
2. `config set` manual tidak berubah setelah `replan`; `config auto` mengembalikan nilai auto.
3. Setiap aturan advisor A01–A18 punya satu tes yang memicunya dan satu yang tidak.
4. `think=false` dan `tool_call=false` menghasilkan persentase SFT yang jumlahnya tetap 100.
5. `languages=id` menghasilkan komposisi pretrain tanpa folder `en_*` dengan jumlah kolom 100.
6. Stage C aktif saat `context_final=8192` dan `context_pretrain=4096`; `rope_theta` Stage C = 1,000,000.
7. `plan --set` tidak mengubah `project.json`.
