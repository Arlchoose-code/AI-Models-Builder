"""Pintu masuk CLI (argparse + routing perintah) dan alur interaktif init / plan / config (Bagian 13 & 18)."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from llmkit.advisor import run_advisor
from llmkit.bench import run_benchmark
from llmkit.chat import run_chat_repl
from llmkit.check import check_dataset
from llmkit.data_spec import (
    PRETRAIN_FOLDER_DESCRIPTIONS,
    SFT_CATEGORY_DESCRIPTIONS,
    convert_file,
    create_project_data_scaffold,
    make_identity_file,
)
from llmkit.doctor import run_doctor
from llmkit.export import export_hf_model
from llmkit.gpu_table import detect_gpu, get_gpu_spec
from llmkit.guide import (
    DATASET_CANDIDATES,
    HONESTY_NOTES,
    print_command_help,
    print_full_guide,
)
from llmkit.pilot import run_pilot
from llmkit.planner import make_plan
from llmkit.prep import run_prep
from llmkit.settings import SETTING_KEYS, compute_auto_settings, detect_system_profile
from llmkit.sft import run_sft
from llmkit.state import (
    atomic_write_json,
    detect_stage_status,
    load_project,
    log_history,
    save_project,
)
from llmkit.tokenizer_train import train_tokenizer
from llmkit.train import run_pretrain
from llmkit.ui import (
    confirm,
    format_exact_number,
    format_number,
    format_time,
    print_error,
    print_header,
    print_info,
    print_ok,
    print_table,
    print_warn,
    prompt_input,
)


def parse_token_string(s: str) -> float:
    """Ubah string token ('60B', '2T', '500M') menjadi angka float token."""
    s = s.strip().upper()
    if s.endswith("T"):
        return float(s[:-1]) * 1e12
    elif s.endswith("B"):
        return float(s[:-1]) * 1e9
    elif s.endswith("M"):
        return float(s[:-1]) * 1e6
    elif s.endswith("K"):
        return float(s[:-1]) * 1e3
    return float(s)


def display_plan_summary(plan: dict[str, Any], project_name: str = "Model") -> None:
    """Tampilkan rencana lengkap di terminal sesuai spesifikasi Bagian 13.3."""
    arch = plan["architecture"]
    tok = plan["tokens"]
    comp = plan["compute"]
    hyp = plan["hyperparameters"]
    mem = plan["memory"]
    pre = plan["pretrain_data"]
    sft = plan["sft_data"]
    stor = plan["storage"]

    print_header(f"RENCANA LENGKAP: {project_name}")
    print(f"Arsitektur : {arch['L']} layer | hidden {arch['d']} | {arch['Hq']} Q-head / {arch['Hkv']} KV-head | FFN {arch['ffn']} | vocab {format_number(arch['V'])}")
    print(f"Parameter  : {format_exact_number(arch['total_params'])} ({arch['total_params'] / 1e9:.2f}B) | non-embedding {format_exact_number(arch['non_embedding'])}")
    print(f"Token      : {format_number(tok['target'])} (rasio {tok['ratio']:.0f}x, mode {tok['mode']})")

    gpu_label = comp["gpu"]
    print(f"Compute    : {comp['flops']:.2e} FLOPs | {gpu_label}: ~{comp['hours']:.0f} jam ({comp['hours'] / 24.0:.1f} hari) ~ ${comp['cost']:.0f}")

    print(f"Batch      : {format_exact_number(hyp['batch_tokens'])} token/step (micro {hyp['micro_batch']} x accum {hyp['grad_accum']} x seq {hyp['context_pretrain']}) | {format_exact_number(hyp['total_steps'])} step")
    print(f"LR         : puncak {hyp['peak_lr']:.2e} | warmup {hyp['warmup_steps']} step | WSD (stable 80%, decay 20%)")
    print(f"Checkpoint : tiap {format_number(hyp['ckpt_every_tokens'])} token / 30 menit | simpan 3 terakhir | milestone tiap 10%")

    mem_note = f"statis {mem['mem_static_gb']:.1f} GB + aktivasi ~{mem['mem_act_gb']:.1f} GB (total ~{mem['total_gb']:.1f} GB)"
    print(f"Memori     : {mem_note}")

    print("\nKomposisi Pretrain:")
    for st_name, st_info in pre["stages"].items():
        if st_info:
            print(f"  Stage {st_name} ({st_info['pct']:.0f}% = {format_number(st_info['tokens'])}):")
            weights_str = ", ".join(f"{f}: {w:.0f}%" for f, w in st_info["weights"].items() if w > 0)
            print(f"    {weights_str}")
    print(f"  Porsi Bahasa Indonesia total: ~{pre['pct_id']:.1f}% | Porsi Kode total: ~{pre['pct_code']:.1f}%")

    print(f"\nRencana SFT: {sft['n_sft']:,} sampel total")
    sft_cats_str = " | ".join(f"{c}: {info['samples']:,}" for c, info in sft["categories"].items())
    print(f"  {sft_cats_str}")

    print(f"\nKebutuhan Storage:")
    print(f"  Tokenized biner: ~{stor['tokenized_gb']:.1f} GB | Ruang kerja disarankan: ~{stor['suggested_workspace_gb']:.0f} GB")
    print(f"  Checkpoint bobot: ~{stor['ckpt_weight_gb']:.1f} GB | Checkpoint penuh (+optim): ~{stor['ckpt_full_gb']:.1f} GB")

    # Tabel perbandingan GPU
    print("\nTabel Perbandingan GPU:")
    comp_headers = ["GPU", "VRAM", "Waktu", "Hari", "Biaya", "Presisi"]
    comp_rows = []
    for g in comp["comparisons"]:
        comp_rows.append([
            g["gpu"],
            f"{g['vram_gb']:.0f} GB",
            f"{g['hours']:.0f} jam",
            f"{g['days']:.1f} hari",
            f"${g['cost']:.0f}",
            g["precision"],
        ])
    print_table(comp_headers, comp_rows)


def handle_init(args: argparse.Namespace) -> None:
    """Alur perintah init (Bagian 13.2 & 18.8)."""
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    proj_file = project_dir / "project.json"
    if proj_file.exists() and not args.yes:
        if not confirm("project.json sudah ada di direktori ini. Timpa dan inisialisasi ulang?", default=False):
            print("Inisialisasi dibatalkan.")
            return

    answers: dict[str, Any] = {}

    # Jika file jawaban disediakan (--answers)
    if args.answers:
        with open(args.answers, "r", encoding="utf-8") as f:
            answers = json.load(f)
    elif args.yes:
        # Default otomatis
        answers = {
            "name": "bahasa-0.6B",
            "size_b": 0.6,
            "token_mode": "balanced",
            "gpu": "H100",
            "goal": "produk",
            "profile": "cloud",
            "features": {"tool_call": True, "think": True, "vision": False},
            "id_unique_tokens_b": None,
            "long_context": False,
            "model_name": "Bahasa",
            "creator": "Pengguna",
            "runs_dir": "runs"
        }
    else:
        print_header("INISIALISASI PROYEK LLM BARU")

        # 1. Nama proyek
        p_name = prompt_input("1. Nama proyek?", default="bahasa-0.6B")
        answers["name"] = p_name

        # 2. Ukuran model
        size_str = prompt_input("2. Ukuran model (miliar parameter)? (0.05 / 0.1 / 0.3 / 0.6 / 1)", default="0.6")
        answers["size_b"] = float(size_str)

        # 3. Mode token
        print("3. Mode token:")
        print("   1=chinchilla (20x)   2=balanced (100x, default)   3=overtrain (500x)")
        print("   4=max (3000x)        5=jumlah token manual        6=berdasarkan budget/jam")
        t_choice = prompt_input("   Pilihan mode", default="2")
        tokens_override = None
        budget_usd = None
        hours = None
        mode_label = "balanced"

        if t_choice == "1":
            mode_label = "chinchilla"
        elif t_choice == "2":
            mode_label = "balanced"
        elif t_choice == "3":
            mode_label = "overtrain"
        elif t_choice == "4":
            mode_label = "max"
        elif t_choice == "5":
            mode_label = "custom"
            raw_tok = prompt_input("   Masukkan jumlah token (contoh: 60B, 2T, 500M)", default="60B")
            tokens_override = parse_token_string(raw_tok)
        elif t_choice == "6":
            b_type = prompt_input("   Berdasarkan budget dollar ($) atau jam GPU (h)? (ketik '$' atau 'h')", default="$")
            if b_type == "$":
                budget_usd = float(prompt_input("   Maksimal budget dalam USD?", default="500"))
            else:
                hours = float(prompt_input("   Maksimal jam sewa GPU?", default="100"))

        answers["token_mode"] = mode_label
        answers["tokens_override"] = tokens_override
        answers["budget_usd"] = budget_usd
        answers["hours"] = hours

        # 4. GPU
        detected_dev, detected_spec = detect_gpu()
        def_gpu = detected_spec.name if detected_spec else "H100"
        gpu_choice = prompt_input(f"4. GPU? (terdeteksi: {detected_dev or 'Tidak ada GPU CUDA'})", default=def_gpu)
        answers["gpu"] = gpu_choice

        # 4b. Tujuan (Goal)
        print("4b. Tujuan proyek: 1=belajar 2=produk 3=maksimal 4=custom")
        g_choice = prompt_input("    Pilihan tujuan", default="2")
        g_map = {"1": "belajar", "2": "produk", "3": "maksimal", "4": "custom"}
        answers["goal"] = g_map.get(g_choice, "produk")

        # 4c. Profil lingkungan
        detected_prof = detect_system_profile(get_gpu_spec(gpu_choice))
        prof_choice = prompt_input(f"4c. Profil lingkungan? (laptop/colab/cloud)", default=detected_prof)
        answers["profile"] = prof_choice

        # 5. Fitur
        tc_ok = confirm("5a. Aktifkan tool calling?", default=True)
        th_ok = confirm("5b. Aktifkan penalaran think/no-think?", default=True)
        vis_ok = confirm("5c. Flag vision (tahap lanjutan)?", default=False)
        answers["features"] = {"tool_call": tc_ok, "think": th_ok, "vision": vis_ok}

        # 6. Data unik Indonesia
        id_tok = prompt_input("6. Perkiraan data bahasa Indonesia UNIK yang kamu miliki (miliar token, kosongkan jika belum tahu)?", default="")
        answers["id_unique_tokens_b"] = float(id_tok) if id_tok else None

        # 7. Long-context
        long_ok = confirm("7. Perlu long-context 32k di tahap akhir?", default=False)
        answers["long_context"] = long_ok

        # 8. Identitas model & pembuat
        m_name = prompt_input("8a. Nama model?", default="Bahasa")
        c_name = prompt_input("8b. Nama pembuat / organisasi?", default="Pengguna")
        answers["model_name"] = m_name
        answers["creator"] = c_name

        # 9. Folder runs / checkpoint
        r_dir = prompt_input("9. Folder untuk run dan checkpoint?", default="runs")
        answers["runs_dir"] = r_dir

    # Hitung rencana awal
    plan = make_plan(answers)
    display_plan_summary(plan, project_name=answers.get("name", "Model"))

    # Buat folder scaffolding & README panduan data
    create_project_data_scaffold(
        project_dir,
        plan,
        model_name=answers.get("model_name", "Bahasa"),
        creator=answers.get("creator", "Pengguna")
    )

    # Catat ke project.json
    project_payload = {
        "version": 1,
        "name": answers.get("name", "bahasa-0.6B"),
        "created": datetime.datetime.now().isoformat(),
        "answers": answers,
        "plan": plan,
        "settings": plan.get("settings", {}),
        "history": []
    }
    save_project(project_dir, project_payload)

    # Tampilkan temuan Advisor (Bagian 18.6)
    findings = run_advisor(answers, plan)
    if findings:
        print("\n" + "=" * 50)
        print("SARAN & PERINGATAN ADVISOR:")
        for f in findings:
            if f.level == "error":
                print_error(f"[{f.id}] {f.message} -> Saran: {f.suggestion}")
            elif f.level == "warn":
                print_warn(f"[{f.id}] {f.message} -> Saran: {f.suggestion}")
            else:
                print_info(f"[{f.id}] {f.message}")
        print("=" * 50)

    print("\n" + HONESTY_NOTES)
    print_ok(f"Inisialisasi selesai di {project_dir}!")
    print(f"Langkah kamu sekarang: isi folder di data/pretrain/ dan data/sft/, lalu jalankan: python main.py check")
    log_history(project_dir, "init", "", True, "Inisialisasi proyek baru berhasil")


def handle_plan_calculator(args: argparse.Namespace) -> None:
    """Mode kalkulator cepat tanpa membuat proyek, atau simulasi --set (Bagian 6, 13.1, & 18.1)."""
    project_dir = Path(args.project).resolve()
    has_proj = (project_dir / "project.json").exists()

    size_str = str(args.size).strip().upper().rstrip("B")
    size_b = float(size_str)
    tok_mode = args.mode or "balanced"
    tokens_override = parse_token_string(args.tokens) if args.tokens else None
    budget_usd = float(args.budget_usd) if args.budget_usd else None
    hours = float(args.hours) if args.hours else None
    gpu_name = args.gpu or "H100"

    sim_answers = {
        "size_b": size_b,
        "token_mode": tok_mode,
        "tokens_override": tokens_override,
        "budget_usd": budget_usd,
        "hours": hours,
        "gpu": gpu_name,
        "features": {"tool_call": True, "think": True, "vision": False},
        "languages": "en+id",
        "long_context": False,
    }

    override_settings = {}
    if args.set:
        for item in args.set:
            if "=" in item:
                k, v = item.split("=", 1)
                k = k.strip()
                v = v.strip()
                parsed_val: Any = v
                if v.lower() == "true":
                    parsed_val = True
                elif v.lower() == "false":
                    parsed_val = False
                elif v.isdigit():
                    parsed_val = int(v)
                override_settings[k] = {"value": parsed_val, "mode": "manual"}

    if args.set:
        if has_proj:
            proj = load_project(project_dir)
            base_answers = proj.get("answers", sim_answers)
            base_plan = proj.get("plan", make_plan(base_answers))
            base_settings = proj.get("settings", base_plan.get("settings", {}))
        else:
            base_answers = sim_answers
            base_plan = make_plan(sim_answers)
            base_settings = base_plan.get("settings", {})

        merged_settings = dict(base_settings)
        merged_settings.update(override_settings)
        sim_plan = make_plan(base_answers, current_settings=merged_settings)

        mem_diff = sim_plan["memory"]["total_gb"] - base_plan["memory"]["total_gb"]
        time_diff = sim_plan["compute"]["hours"] - base_plan["compute"]["hours"]
        cost_diff = sim_plan["compute"]["cost"] - base_plan["compute"]["cost"]

        base_findings = {f.id for f in run_advisor(base_answers, base_plan)}
        sim_findings = run_advisor(base_answers, sim_plan)
        new_warnings = [f.id for f in sim_findings if f.id not in base_findings]

        print_header("HASIL SIMULASI OVERRIDE (--set)")
        diff_str = f"Memori: {mem_diff:+.1f} GB | Waktu: {time_diff:+.1f} jam | Biaya: {cost_diff:+.0f}$ | Peringatan baru: {', '.join(new_warnings) or 'Tidak ada'}"
        print(diff_str)
        print("(Simulasi ini bersifat sementara dan TIDAK disimpan ke project.json)\n")

        display_plan_summary(sim_plan, project_name=f"Simulasi Override ({base_answers.get('name', 'Model')})")
        return

    plan = make_plan(sim_answers)
    display_plan_summary(plan, project_name=f"Simulasi {size_b}B")

    findings = run_advisor(sim_answers, plan)
    if findings:
        print("\nTemuan Advisor:")
        for f in findings:
            print(f"[{f.level.upper()}] {f.id}: {f.message} (Saran: {f.suggestion})")


def handle_replan(args: argparse.Namespace) -> None:
    """Alur perintah replan (Bagian 13.1 & 18.1): perbarui jawaban init dan hitung ulang rencana."""
    project_dir = Path(args.project).resolve()
    proj = load_project(project_dir)
    answers = proj.get("answers", {})
    curr_settings = proj.get("settings", {})

    # Setting manual tidak pernah ditimpa oleh replan
    manual_settings = {k: v for k, v in curr_settings.items() if v.get("mode") == "manual"}

    answers_file = getattr(args, "answers", None)
    if answers_file:
        with open(answers_file, "r", encoding="utf-8") as f:
            updated_answers = json.load(f)
    elif getattr(args, "yes", False):
        updated_answers = dict(answers)
    else:
        print_header("REPLAN: PERBARUI PARAMETER PROYEK")
        size_str = prompt_input("Ukuran model (miliar parameter)?", default=str(answers.get("size_b", 0.6)))
        answers["size_b"] = float(size_str)

        print("Mode token: 1=chinchilla 2=balanced 3=overtrain 4=max 5=manual")
        t_choice = prompt_input("Pilihan mode", default="2")
        if t_choice == "1":
            answers["token_mode"] = "chinchilla"
            answers["tokens_override"] = None
        elif t_choice == "2":
            answers["token_mode"] = "balanced"
            answers["tokens_override"] = None
        elif t_choice == "3":
            answers["token_mode"] = "overtrain"
            answers["tokens_override"] = None
        elif t_choice == "4":
            answers["token_mode"] = "max"
            answers["tokens_override"] = None
        elif t_choice == "5":
            answers["token_mode"] = "custom"
            answers["tokens_override"] = parse_token_string(prompt_input("Jumlah token?", default="60B"))

        answers["gpu"] = prompt_input("GPU?", default=answers.get("gpu", "H100"))
        answers["goal"] = prompt_input("Tujuan (belajar/produk/maksimal/custom)?", default=answers.get("goal", "produk"))
        updated_answers = answers

    new_plan = make_plan(updated_answers, current_settings=manual_settings)
    display_plan_summary(new_plan, project_name=updated_answers.get("name", "Model"))

    proj["answers"] = updated_answers
    proj["plan"] = new_plan
    proj["settings"] = new_plan["settings"]
    save_project(project_dir, proj)

    findings = run_advisor(updated_answers, new_plan)
    if findings:
        print("\n" + "=" * 50)
        print("SARAN & PERINGATAN ADVISOR:")
        for f in findings:
            if f.level == "error":
                print_error(f"[{f.id}] {f.message} -> Saran: {f.suggestion}")
            elif f.level == "warn":
                print_warn(f"[{f.id}] {f.message} -> Saran: {f.suggestion}")
            else:
                print_info(f"[{f.id}] {f.message}")
        print("=" * 50)

    print_ok(f"Rencana berhasil diperbarui dan disimpan ke project.json")
    log_history(project_dir, "replan", "", True, "Replan proyek berhasil")



def handle_status(args: argparse.Namespace) -> None:
    """Perintah status auto-detect (Bagian 5.2)."""
    project_dir = Path(args.project).resolve()
    status_info = detect_stage_status(project_dir)

    print_header(f"STATUS PROYEK: {project_dir.name}")
    headers = ["Tahap", "Status", "Rincian"]
    rows = []
    for stage_name, s_data in status_info.get("stages", {}).items():
        rows.append([stage_name, s_data.get("status", "[ ]"), s_data.get("label", "")])
    print_table(headers, rows)

    if status_info.get("warning"):
        print_warn(f"\nPerhatian: {status_info['warning']}")

    print(f"\nSELANJUTNYA: Tahap '{status_info['next_step']}'")
    print(f"Perintah   : {status_info['next_command']}\n")


def handle_config(args: argparse.Namespace) -> None:
    """Perintah config (show, set, auto) sesuai Bagian 18."""
    project_dir = Path(args.project).resolve()
    proj = load_project(project_dir)
    answers = proj.get("answers", {})
    curr_settings = proj.get("settings", {})
    plan = proj.get("plan", {})

    if args.config_action == "show":
        headers = ["Kunci", "Nilai", "Mode"]
        if args.why:
            headers.append("Alasan")
        rows = []
        for k, v in curr_settings.items():
            row = [k, str(v.get("value")), v.get("mode", "auto")]
            if args.why:
                row.append(str(v.get("reason", "")))
            rows.append(row)
        print_table(headers, rows, title="PENGATURAN PROYEK")

        findings = run_advisor(answers, plan)
        if findings:
            print("\nSARAN:")
            for f in findings:
                print(f"[{f.level.upper()}] {f.id} {f.message} Saran: {f.suggestion}")

    elif args.config_action == "set":
        pair = args.key_val
        if "=" not in pair:
            print_error("Format salah. Gunakan: python main.py config set kunci=nilai")
            return
        key, raw_val = pair.split("=", 1)
        key = key.strip()
        val: Any = raw_val.strip()
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.isdigit():
            val = int(val)

        curr_settings[key] = {"value": val, "mode": "manual", "reason": "(kamu atur)"}
        new_plan = make_plan(answers, current_settings=curr_settings)
        proj["plan"] = new_plan
        proj["settings"] = new_plan["settings"]
        save_project(project_dir, proj)
        print_ok(f"Setting '{key}' berhasil dikunci manual ke: {val}")

    elif args.config_action == "auto":
        key = args.key.strip()
        if key in curr_settings:
            curr_settings[key]["mode"] = "auto"
            new_plan = make_plan(answers, current_settings=curr_settings)
            proj["plan"] = new_plan
            proj["settings"] = new_plan["settings"]
            save_project(project_dir, proj)
            auto_val = new_plan["settings"][key]["value"]
            print_ok(f"Setting '{key}' dikembalikan ke mode auto (nilai saat ini: {auto_val})")


def handle_runs(args: argparse.Namespace) -> None:
    """Tampilkan riwayat semua run (Bagian 13.1)."""
    project_dir = Path(args.project).resolve()
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        print("Belum ada folder runs/ yang terbentuk.")
        return

    headers = ["Run ID", "Jenis", "Status", "Step", "Tokens", "Loss Terakhir"]
    rows = []
    for r_dir in sorted(runs_dir.glob("run_*")):
        r_json = r_dir / "run.json"
        if r_json.exists():
            try:
                with open(r_json, "r", encoding="utf-8") as f:
                    d = json.load(f)
                rows.append([
                    r_dir.name,
                    d.get("kind", "-"),
                    d.get("status", "-"),
                    f"{d.get('step', 0):,}",
                    format_number(d.get("tokens_seen", 0)),
                    str(d.get("last_loss", "-")),
                ])
            except Exception:
                pass
    print_table(headers, rows, title="DAFTAR RUN TRAINING")


def build_parser() -> argparse.ArgumentParser:
    """Bangun parser CLI argument."""
    parser = argparse.ArgumentParser(description="LLM Training Toolkit (CLI interaktif)", add_help=False)
    parser.add_argument("--project", default=".", help="Direktori proyek (default: direktori saat ini)")
    parser.add_argument("--yes", "-y", action="store_true", help="Setujui semua konfirmasi otomatis")
    parser.add_argument("--dry-run", action="store_true", help="Simulasikan tindakan tanpa menyimpan ke disk")

    subparsers = parser.add_subparsers(dest="subcommand")

    # init
    p_init = subparsers.add_parser("init", help="Inisialisasi proyek baru")
    p_init.add_argument("--answers", help="Path ke file jawaban JSON untuk otomatisasi")

    # plan
    p_plan = subparsers.add_parser("plan", help="Kalkulator rencana cepat")
    p_plan.add_argument("size", nargs="?", default="0.6B", help="Ukuran model (contoh: 0.6B, 0.5B)")
    p_plan.add_argument("--mode", default="balanced", help="Mode token (chinchilla, balanced, overtrain, max)")
    p_plan.add_argument("--tokens", help="Jumlah token manual (contoh: 80B, 2T)")
    p_plan.add_argument("--gpu", default="H100", help="Nama GPU (H100, A100, L4, RTX 4090, dll)")
    p_plan.add_argument("--budget-usd", help="Batas budget dalam USD")
    p_plan.add_argument("--hours", help="Batas jam sewa GPU")
    p_plan.add_argument("--set", action="append", help="Uji coba override setting (kunci=nilai)")

    # status
    subparsers.add_parser("status", help="Periksa status proyek saat ini")

    # check
    p_check = subparsers.add_parser("check", help="Validasi dataset")
    p_check.add_argument("--quick", action="store_true", help="Pemeriksaan cepat tanpa membaca seluruh baris")
    p_check.add_argument("--fix-hints", action="store_true", help="Tampilkan detail petunjuk perbaikan error")

    # tokenizer
    p_tok = subparsers.add_parser("tokenizer", help="Latih tokenizer Byte-level BPE")
    p_tok.add_argument("--vocab-size", type=int, default=64000, help="Ukuran vocab (default 64000)")
    p_tok.add_argument("--max-gb", type=float, default=1.5, help="Batas maksimal ukuran data teks latih (GB)")

    # prep
    p_prep = subparsers.add_parser("prep", help="Tokenisasi teks ke shard biner uint16")

    # bench
    subparsers.add_parser("bench", help="Benchmark kecepatan hardware GPU")

    # pilot
    subparsers.add_parser("pilot", help="Jalankan model uji coba 0.05B untuk sweep LR")

    # pretrain
    p_train = subparsers.add_parser("pretrain", help="Mulai atau lanjutkan pretraining")
    p_train.add_argument("--resume", action="store_true", help="Lanjutkan training dari checkpoint terakhir")
    p_train.add_argument("--new", action="store_true", help="Mulai run baru meskipun ada checkpoint sebelumnya")
    p_train.add_argument("--run", help="ID run tertentu untuk di-resume (contoh: run_001)")
    p_train.add_argument("--step", type=int, help="Step spesifik untuk di-resume")
    p_train.add_argument("--lr", type=float, help="Ubah peak LR")
    p_train.add_argument("--max-hours", type=float, help="Jeda otomatis setelah N jam")
    p_train.add_argument("--auto", action="store_true", help="Supervisor auto-restart jika crash")
    p_train.add_argument("--out-dir", help="Lokasi penyimpanan runs (misal ke Google Drive)")
    p_train.add_argument("--compile", action="store_true", help="Aktifkan torch.compile")

    # sft
    p_sft = subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    p_sft.add_argument("--from", dest="from_run", help="ID run atau path checkpoint awal (contoh: run_001 atau run_001:step_00010000)")
    p_sft.add_argument("--from-ckpt", help="Path checkpoint pretrain awal")
    p_sft.add_argument("--epochs", type=int, help="Jumlah epoch SFT")
    p_sft.add_argument("--lr", type=float, help="Learning rate SFT")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluasi model")
    p_eval.add_argument("--run", help="ID run yang akan dievaluasi")
    p_eval.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"), help="Bandingkan dua run berdampingan")
    p_eval.add_argument("--bench", action="store_true", help="Jalankan benchmark lm_eval jika terpasang")

    # chat
    p_chat = subparsers.add_parser("chat", help="REPL chat terminal interaktif")
    p_chat.add_argument("--run", help="ID run yang digunakan")
    p_chat.add_argument("--step", type=int, help="Step checkpoint tertentu")
    p_chat.add_argument("--mode", default="auto", choices=["think", "no_think", "auto"], help="Mode chat")
    p_chat.add_argument("--tools", action="store_true", help="Aktifkan dukungan tool sandbox")
    p_chat.add_argument("--force-nothink", action="store_true", help="Paksa blok think kosong")

    # export
    p_exp = subparsers.add_parser("export", help="Ekspor ke format Hugging Face")
    p_exp.add_argument("--run", help="ID run yang diekspor")
    p_exp.add_argument("--out-name", help="Nama folder ekspor")

    # replan
    p_replan = subparsers.add_parser("replan", help="Ubah jawaban init dan perbarui rencana")
    p_replan.add_argument("--answers", help="Path ke file jawaban JSON untuk otomatisasi")
    p_replan.add_argument("-y", "--yes", action="store_true", help="Pakai konfigurasi saat ini tanpa prompt interaktif")

    # make-identity
    subparsers.add_parser("make-identity", help="Buat atau perbarui file data identitas SFT")

    # config
    p_cfg = subparsers.add_parser("config", help="Kelola settings proyek")
    cfg_subs = p_cfg.add_subparsers(dest="config_action")
    p_c_show = cfg_subs.add_parser("show", help="Lihat semua settings")
    p_c_show.add_argument("--why", action="store_true", help="Tampilkan alasan nilai auto")
    p_c_set = cfg_subs.add_parser("set", help="Kunci setting ke manual")
    p_c_set.add_argument("key_val", help="Pasangan kunci=nilai (contoh: context_final=8192)")
    p_c_auto = cfg_subs.add_parser("auto", help="Kembalikan setting ke mode auto")
    p_c_auto.add_argument("key", help="Nama setting")

    # convert
    p_conv = subparsers.add_parser("convert", help="Konversi format data teks")
    p_conv.add_argument("source", help="File sumber (JSONL / Parquet)")
    p_conv.add_argument("--from-type", required=True, choices=["text_column", "sharegpt"], help="Jenis format asal")
    p_conv.add_argument("--to", required=True, help="File tujuan output .jsonl")
    p_conv.add_argument("--column", default="text", help="Nama kolom teks (jika dari text_column)")

    # helper lain
    subparsers.add_parser("runs", help="Tampilkan riwayat semua run training")
    subparsers.add_parser("doctor", help="Periksa lingkungan sistem")
    subparsers.add_parser("guide", help="Panduan lengkap berurutan")

    p_help = subparsers.add_parser("help", help="Bantuan perintah")
    p_help.add_argument("target_command", nargs="?", help="Perintah spesifik")

    return parser


def main() -> None:
    """Fungsi utama dispatcher CLI."""
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    project_dir = Path(args.project).resolve()

    if not args.subcommand or args.subcommand == "help":
        target = getattr(args, "target_command", None)
        print_command_help(target)
        return

    if args.subcommand == "guide":
        print_full_guide()
        return

    if args.subcommand == "doctor":
        run_doctor(project_dir)
        return

    if args.subcommand == "plan":
        handle_plan_calculator(args)
        return

    if args.subcommand == "init":
        handle_init(args)
        return

    # Perintah di bawah ini mensyaratkan project.json harus ada
    if not (project_dir / "project.json").exists():
        print_error("Tidak menemukan project.json di direktori kerja.")
        print("Silakan jalankan: python main.py init  atau gunakan opsi --project <folder>.")
        sys.exit(1)

    if args.subcommand == "status":
        handle_status(args)
    elif args.subcommand == "replan":
        handle_replan(args)
    elif args.subcommand == "make-identity":
        make_identity_file(project_dir)
    elif args.subcommand == "check":
        ok, res = check_dataset(project_dir, quick=args.quick, fix_hints=args.fix_hints)
        print(f"\nStatus Pemeriksaan Data: {'[OK] Lulus' if ok else '[ERROR] Ditemukan masalah'}")
        if not ok:
            print(f"Lihat rincian lengkap error di: {project_dir / 'logs' / 'check_errors.txt'}")
            sys.exit(1)
    elif args.subcommand == "tokenizer":
        train_tokenizer(project_dir, vocab_size=args.vocab_size, max_gb=args.max_gb)
    elif args.subcommand == "prep":
        run_prep(project_dir)
    elif args.subcommand == "bench":
        run_benchmark(project_dir)
    elif args.subcommand == "pilot":
        run_pilot(project_dir)
    elif args.subcommand == "pretrain":
        if args.auto:
            max_restarts = 20
            restart_count = 0
            while restart_count <= max_restarts:
                try:
                    res = run_pretrain(
                        project_dir,
                        resume=True if restart_count > 0 else args.resume,
                        new_run=args.new if restart_count == 0 else False,
                        run_id=args.run,
                        specific_step=args.step,
                        max_hours=args.max_hours,
                        override_lr=args.lr,
                        custom_out_dir=args.out_dir,
                        compile_model=args.compile
                    )
                    if res.get("status") in ("finished", "paused", "stopped_spike"):
                        break
                except Exception as e:
                    restart_count += 1
                    if restart_count > max_restarts:
                        print_error(f"Pretrain crash berulang melebihi batas {max_restarts} kali: {e}")
                        sys.exit(1)
                    print_warn(f"Pretrain crash: {e}. Supervisor auto-restart ({restart_count}/{max_restarts}) dalam 30 detik...")
                    time.sleep(30 if not args.dry_run else 0.1)
        else:
            run_pretrain(
                project_dir,
                resume=args.resume,
                new_run=args.new,
                run_id=args.run,
                specific_step=args.step,
                max_hours=args.max_hours,
                override_lr=args.lr,
                custom_out_dir=args.out_dir,
                compile_model=args.compile
            )
    elif args.subcommand == "sft":
        from_ckpt = getattr(args, "from_run", None) or args.from_ckpt
        run_sft(
            project_dir,
            from_checkpoint=from_ckpt,
            epochs_override=args.epochs,
            lr_override=args.lr
        )
    elif args.subcommand == "eval":
        from llmkit.evaluate import compare_runs, evaluate_pretrain_model, evaluate_sft_model
        if args.compare:
            compare_runs(project_dir, args.compare[0], args.compare[1])
            return

        run_id = args.run
        if not run_id:
            existing_runs = sorted(list((project_dir / "runs").glob("run_*")), key=lambda p: p.name)
            run_id = existing_runs[-1].name if existing_runs else "run_001"

        run_json = project_dir / "runs" / run_id / "run.json"
        kind = "sft"
        if run_json.exists():
            try:
                with open(run_json, "r", encoding="utf-8") as f:
                    kind = json.load(f).get("kind", "sft")
            except Exception:
                pass

        if kind == "pretrain":
            evaluate_pretrain_model(project_dir, run_id=run_id, bench=args.bench)
        else:
            evaluate_sft_model(project_dir, run_id=run_id)
    elif args.subcommand == "chat":
        run_chat_repl(
            project_dir,
            run_id=args.run,
            specific_step=args.step,
            initial_mode=args.mode,
            use_tools=args.tools,
            force_nothink=args.force_nothink
        )
    elif args.subcommand == "export":
        export_hf_model(project_dir, run_id=args.run, out_name=args.out_name)
    elif args.subcommand == "config":
        handle_config(args)
    elif args.subcommand == "runs":
        handle_runs(args)
    elif args.subcommand == "convert":
        src = Path(args.source)
        dst = Path(args.to)
        num = convert_file(src, dst, from_format=args.from_type, text_column=args.column)
        print_ok(f"Berhasil mengonversi {num:,} baris ke: {dst}")
