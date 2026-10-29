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
    get_honesty_notes,
    print_command_help,
    print_full_guide,
)
from llmkit.i18n import (
    get_lang,
    load_saved_lang,
    prompt_language_selection,
    set_lang,
    tr,
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
    """Tampilkan rencana lengkap di terminal sesuai spesifikasi Bagian 13.3 (dwibahasa)."""
    arch = plan["architecture"]
    tok = plan["tokens"]
    comp = plan["compute"]
    hyp = plan["hyperparameters"]
    mem = plan["memory"]
    pre = plan["pretrain_data"]
    sft = plan["sft_data"]
    stor = plan["storage"]

    print_header(tr(f"RENCANA LENGKAP: {project_name}", f"COMPLETE PLAN: {project_name}"))
    print(tr(
        f"Arsitektur : {arch['L']} layer | hidden {arch['d']} | {arch['Hq']} Q-head / {arch['Hkv']} KV-head | FFN {arch['ffn']} | vocab {format_number(arch['V'])}",
        f"Architecture: {arch['L']} layers | hidden {arch['d']} | {arch['Hq']} Q-head / {arch['Hkv']} KV-head | FFN {arch['ffn']} | vocab {format_number(arch['V'])}"
    ))
    print(tr(
        f"Parameter  : {format_exact_number(arch['total_params'])} ({arch['total_params'] / 1e9:.2f}B) | non-embedding {format_exact_number(arch['non_embedding'])}",
        f"Parameters  : {format_exact_number(arch['total_params'])} ({arch['total_params'] / 1e9:.2f}B) | non-embedding {format_exact_number(arch['non_embedding'])}"
    ))
    print(tr(
        f"Token      : {format_number(tok['target'])} (rasio {tok['ratio']:.0f}x, mode {tok['mode']})",
        f"Tokens      : {format_number(tok['target'])} (ratio {tok['ratio']:.0f}x, mode {tok['mode']})"
    ))

    gpu_label = comp["gpu"]
    print(tr(
        f"Compute    : {comp['flops']:.2e} FLOPs | {gpu_label}: ~{comp['hours']:.0f} jam ({comp['hours'] / 24.0:.1f} hari) ~ ${comp['cost']:.0f}",
        f"Compute     : {comp['flops']:.2e} FLOPs | {gpu_label}: ~{comp['hours']:.0f} hours ({comp['hours'] / 24.0:.1f} days) ~ ${comp['cost']:.0f}"
    ))

    print(tr(
        f"Batch      : {format_exact_number(hyp['batch_tokens'])} token/step (micro {hyp['micro_batch']} x accum {hyp['grad_accum']} x seq {hyp['context_pretrain']}) | {format_exact_number(hyp['total_steps'])} step",
        f"Batch       : {format_exact_number(hyp['batch_tokens'])} tokens/step (micro {hyp['micro_batch']} x accum {hyp['grad_accum']} x seq {hyp['context_pretrain']}) | {format_exact_number(hyp['total_steps'])} steps"
    ))
    print(tr(
        f"LR         : puncak {hyp['peak_lr']:.2e} | warmup {hyp['warmup_steps']} step | WSD (stable 80%, decay 20%)",
        f"LR          : peak {hyp['peak_lr']:.2e} | warmup {hyp['warmup_steps']} steps | WSD (stable 80%, decay 20%)"
    ))
    print(tr(
        f"Checkpoint : tiap {format_number(hyp['ckpt_every_tokens'])} token / 30 menit | simpan 3 terakhir | milestone tiap 10%",
        f"Checkpoint  : every {format_number(hyp['ckpt_every_tokens'])} tokens / 30 mins | keep last 3 | milestone every 10%"
    ))

    mem_note = tr(
        f"statis {mem['mem_static_gb']:.1f} GB + aktivasi ~{mem['mem_act_gb']:.1f} GB (total ~{mem['total_gb']:.1f} GB)",
        f"static {mem['mem_static_gb']:.1f} GB + activation ~{mem['mem_act_gb']:.1f} GB (total ~{mem['total_gb']:.1f} GB)"
    )
    print(tr(f"Memori     : {mem_note}", f"Memory      : {mem_note}"))

    print("\n" + tr("Komposisi Pretrain:", "Pretrain Composition:"))
    for st_name, st_info in pre["stages"].items():
        if st_info:
            print(f"  Stage {st_name} ({st_info['pct']:.0f}% = {format_number(st_info['tokens'])}):")
            weights_str = ", ".join(f"{f}: {w:.0f}%" for f, w in st_info["weights"].items() if w > 0)
            print(f"    {weights_str}")
    print(tr(
        f"  Porsi Bahasa Indonesia total: ~{pre['pct_id']:.1f}% | Porsi Kode total: ~{pre['pct_code']:.1f}%",
        f"  Total Indonesian portion: ~{pre['pct_id']:.1f}% | Total Code portion: ~{pre['pct_code']:.1f}%"
    ))

    print("\n" + tr(f"Rencana SFT: {sft['n_sft']:,} sampel total", f"SFT Plan: {sft['n_sft']:,} total samples"))
    sft_cats_str = " | ".join(f"{c}: {info['samples']:,}" for c, info in sft["categories"].items())
    print(f"  {sft_cats_str}")

    print("\n" + tr("Kebutuhan Storage:", "Storage Requirements:"))
    print(tr(
        f"  Tokenized biner: ~{stor['tokenized_gb']:.1f} GB | Ruang kerja disarankan: ~{stor['suggested_workspace_gb']:.0f} GB",
        f"  Binary tokenized: ~{stor['tokenized_gb']:.1f} GB | Suggested workspace: ~{stor['suggested_workspace_gb']:.0f} GB"
    ))
    print(tr(
        f"  Checkpoint bobot: ~{stor['ckpt_weight_gb']:.1f} GB | Checkpoint penuh (+optim): ~{stor['ckpt_full_gb']:.1f} GB",
        f"  Weight checkpoint: ~{stor['ckpt_weight_gb']:.1f} GB | Full checkpoint (+optim): ~{stor['ckpt_full_gb']:.1f} GB"
    ))

    # Tabel perbandingan GPU
    print("\n" + tr("Tabel Perbandingan GPU:", "GPU Comparison Table:"))
    comp_headers = [tr("GPU", "GPU"), tr("VRAM", "VRAM"), tr("Waktu", "Time"), tr("Hari", "Days"), tr("Biaya", "Cost"), tr("Presisi", "Precision")]
    comp_rows = []
    for g in comp["comparisons"]:
        comp_rows.append([
            g["gpu"],
            f"{g['vram_gb']:.0f} GB",
            tr(f"{g['hours']:.0f} jam", f"{g['hours']:.0f} hrs"),
            tr(f"{g['days']:.1f} hari", f"{g['days']:.1f} days"),
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
        if not confirm(tr("project.json sudah ada di direktori ini. Timpa dan inisialisasi ulang?", "project.json already exists in this directory. Overwrite and reinitialize?"), default=False):
            print(tr("Inisialisasi dibatalkan.", "Initialization cancelled."))
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
        print_header(tr("INISIALISASI PROYEK LLM BARU", "NEW LLM PROJECT INITIALIZATION"))

        # 0. Bahasa antarmuka proyek
        if not getattr(args, "lang", None):
            print(tr("Pilihan Bahasa Antarmuka / Interface Language:", "Interface Language / Pilihan Bahasa Antarmuka:"))
            print("  1. English (en)")
            print("  2. Bahasa Indonesia (id)")
            init_lang_choice = prompt_input(
                tr("Pilih bahasa antarmuka [1/2]", "Select interface language [1/2]"),
                default="1" if get_lang() == "en" else "2"
            )
            if str(init_lang_choice).strip().lower() in ("1", "en", "english"):
                set_lang("en", persist_global=True, project_dir=project_dir)
            elif str(init_lang_choice).strip().lower() in ("2", "id", "indonesia", "indo"):
                set_lang("id", persist_global=True, project_dir=project_dir)

        # 1. Nama proyek
        p_name = prompt_input(tr("1. Nama proyek?", "1. Project name?"), default="bahasa-0.6B")
        answers["name"] = p_name

        # 2. Ukuran model
        size_str = prompt_input(tr("2. Ukuran model (miliar parameter)? (0.05 / 0.1 / 0.3 / 0.6 / 1)", "2. Model size (billion parameters)? (0.05 / 0.1 / 0.3 / 0.6 / 1)"), default="0.6")
        answers["size_b"] = float(size_str)

        # 3. Mode token
        print(tr("3. Mode token:", "3. Token mode:"))
        print(tr("   1=chinchilla (20x)   2=balanced (100x, default)   3=overtrain (500x)", "   1=chinchilla (20x)   2=balanced (100x, default)   3=overtrain (500x)"))
        print(tr("   4=max (3000x)        5=jumlah token manual        6=berdasarkan budget/jam", "   4=max (3000x)        5=manual token count         6=based on budget/hours"))
        t_choice = prompt_input(tr("   Pilihan mode", "   Mode choice"), default="2")
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
            raw_tok = prompt_input(tr("   Masukkan jumlah token (contoh: 60B, 2T, 500M)", "   Enter token count (e.g. 60B, 2T, 500M)"), default="60B")
            tokens_override = parse_token_string(raw_tok)
        elif t_choice == "6":
            b_type = prompt_input(tr("   Berdasarkan budget dollar ($) atau jam GPU (h)? (ketik '$' atau 'h')", "   Based on dollar budget ($) or GPU hours (h)? (type '$' or 'h')"), default="$")
            if b_type == "$":
                budget_usd = float(prompt_input(tr("   Maksimal budget dalam USD?", "   Maximum budget in USD?"), default="500"))
            else:
                hours = float(prompt_input(tr("   Maksimal jam sewa GPU?", "   Maximum GPU rental hours?"), default="100"))

        answers["token_mode"] = mode_label
        answers["tokens_override"] = tokens_override
        answers["budget_usd"] = budget_usd
        answers["hours"] = hours

        # 4. GPU
        detected_dev, detected_spec = detect_gpu()
        def_gpu = detected_spec.name if detected_spec else "H100"
        gpu_choice = prompt_input(tr(f"4. GPU? (terdeteksi: {detected_dev or 'Tidak ada GPU CUDA'})", f"4. GPU? (detected: {detected_dev or 'No CUDA GPU'})"), default=def_gpu)
        answers["gpu"] = gpu_choice

        # 4b. Tujuan (Goal)
        print(tr("4b. Tujuan proyek: 1=belajar 2=produk 3=maksimal 4=custom", "4b. Project goal: 1=learning 2=product 3=maximum 4=custom"))
        g_choice = prompt_input(tr("    Pilihan tujuan", "    Goal choice"), default="2")
        g_map = {"1": "belajar", "2": "produk", "3": "maksimal", "4": "custom"}
        answers["goal"] = g_map.get(g_choice, "produk")

        # 4c. Profil lingkungan
        detected_prof = detect_system_profile(get_gpu_spec(gpu_choice))
        prof_choice = prompt_input(tr("4c. Profil lingkungan? (laptop/colab/cloud)", "4c. Environment profile? (laptop/colab/cloud)"), default=detected_prof)
        answers["profile"] = prof_choice

        # 5. Fitur
        tc_ok = confirm(tr("5a. Aktifkan tool calling?", "5a. Enable tool calling?"), default=True)
        th_ok = confirm(tr("5b. Aktifkan penalaran think/no-think?", "5b. Enable think/no-think reasoning?"), default=True)
        vis_ok = confirm(tr("5c. Flag vision (tahap lanjutan)?", "5c. Vision flag (advanced stage)?"), default=False)
        answers["features"] = {"tool_call": tc_ok, "think": th_ok, "vision": vis_ok}

        # 6. Data unik Indonesia
        id_tok = prompt_input(tr("6. Perkiraan data bahasa Indonesia UNIK yang kamu miliki (miliar token, kosongkan jika belum tahu)?", "6. Estimated UNIQUE Indonesian text data you have (billion tokens, leave blank if unsure)?"), default="")
        answers["id_unique_tokens_b"] = float(id_tok) if id_tok else None

        # 7. Long-context
        long_ok = confirm(tr("7. Perlu long-context 32k di tahap akhir?", "7. Need long-context 32k in final stage?"), default=False)
        answers["long_context"] = long_ok

        # 8. Identitas model & pembuat
        m_name = prompt_input(tr("8a. Nama model?", "8a. Model name?"), default="Bahasa")
        c_name = prompt_input(tr("8b. Nama pembuat / organisasi?", "8b. Creator / organization name?"), default="Pengguna" if get_lang() == "id" else "User")
        answers["model_name"] = m_name
        answers["creator"] = c_name

        # 9. Folder runs / checkpoint
        r_dir = prompt_input(tr("9. Folder untuk run dan checkpoint?", "9. Directory for runs and checkpoints?"), default="runs")
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
        "lang": get_lang(),
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
        print(tr("SARAN & PERINGATAN ADVISOR:", "ADVISOR SUGGESTIONS & WARNINGS:"))
        for f in findings:
            if f.level == "error":
                print_error(f"[{f.id}] {f.message} -> {tr('Saran', 'Suggestion')}: {f.suggestion}")
            elif f.level == "warn":
                print_warn(f"[{f.id}] {f.message} -> {tr('Saran', 'Suggestion')}: {f.suggestion}")
            else:
                print_info(f"[{f.id}] {f.message}")
        print("=" * 50)

    print("\n" + get_honesty_notes())
    print_ok(tr(f"Inisialisasi selesai di {project_dir}!", f"Initialization complete at {project_dir}!"))
    print(tr("Langkah kamu sekarang: isi folder di data/pretrain/ dan data/sft/, lalu jalankan: python main.py check", "Your next step: populate folders in data/pretrain/ and data/sft/, then run: python main.py check"))
    log_history(project_dir, "init", "", True, tr("Inisialisasi proyek baru berhasil", "New project initialization successful"))


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

        print_header(tr("HASIL SIMULASI OVERRIDE (--set)", "OVERRIDE SIMULATION RESULTS (--set)"))
        diff_str = tr(
            f"Memori: {mem_diff:+.1f} GB | Waktu: {time_diff:+.1f} jam | Biaya: {cost_diff:+.0f}$ | Peringatan baru: {', '.join(new_warnings) or 'Tidak ada'}",
            f"Memory: {mem_diff:+.1f} GB | Time: {time_diff:+.1f} hrs | Cost: {cost_diff:+.0f}$ | New warnings: {', '.join(new_warnings) or 'None'}"
        )
        print(diff_str)
        print(tr("(Simulasi ini bersifat sementara dan TIDAK disimpan ke project.json)\n", "(This simulation is temporary and NOT saved to project.json)\n"))

        display_plan_summary(sim_plan, project_name=tr(f"Simulasi Override ({base_answers.get('name', 'Model')})", f"Override Simulation ({base_answers.get('name', 'Model')})"))
        return

    plan = make_plan(sim_answers)
    display_plan_summary(plan, project_name=tr(f"Simulasi {size_b}B", f"Simulation {size_b}B"))

    findings = run_advisor(sim_answers, plan)
    if findings:
        print(tr("\nTemuan Advisor:", "\nAdvisor Findings:"))
        for f in findings:
            print(f"[{f.level.upper()}] {f.id}: {f.message} ({tr('Saran', 'Suggestion')}: {f.suggestion})")


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
        print_header(tr("REPLAN: PERBARUI PARAMETER PROYEK", "REPLAN: UPDATE PROJECT PARAMETERS"))
        size_str = prompt_input(tr("Ukuran model (miliar parameter)?", "Model size (billion parameters)?"), default=str(answers.get("size_b", 0.6)))
        answers["size_b"] = float(size_str)

        print(tr("Mode token: 1=chinchilla 2=balanced 3=overtrain 4=max 5=manual", "Token mode: 1=chinchilla 2=balanced 3=overtrain 4=max 5=manual"))
        t_choice = prompt_input(tr("Pilihan mode", "Mode choice"), default="2")
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
            answers["tokens_override"] = parse_token_string(prompt_input(tr("Jumlah token?", "Token count?"), default="60B"))

        answers["gpu"] = prompt_input(tr("GPU?", "GPU?"), default=answers.get("gpu", "H100"))
        answers["goal"] = prompt_input(tr("Tujuan (belajar/produk/maksimal/custom)?", "Goal (learning/product/maximum/custom)?"), default=answers.get("goal", "produk"))
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
        print(tr("SARAN & PERINGATAN ADVISOR:", "ADVISOR SUGGESTIONS & WARNINGS:"))
        for f in findings:
            if f.level == "error":
                print_error(f"[{f.id}] {f.message} -> {tr('Saran', 'Suggestion')}: {f.suggestion}")
            elif f.level == "warn":
                print_warn(f"[{f.id}] {f.message} -> {tr('Saran', 'Suggestion')}: {f.suggestion}")
            else:
                print_info(f"[{f.id}] {f.message}")
        print("=" * 50)

    print_ok(tr("Rencana berhasil diperbarui dan disimpan ke project.json", "Plan successfully updated and saved to project.json"))
    log_history(project_dir, "replan", "", True, tr("Replan proyek berhasil", "Project replan successful"))



def handle_status(args: argparse.Namespace) -> None:
    """Perintah status auto-detect (Bagian 5.2)."""
    project_dir = Path(args.project).resolve()
    status_info = detect_stage_status(project_dir)

    print_header(tr(f"STATUS PROYEK: {project_dir.name}", f"PROJECT STATUS: {project_dir.name}"))
    headers = [tr("Tahap", "Stage"), tr("Status", "Status"), tr("Rincian", "Details")]
    rows = []
    for stage_name, s_data in status_info.get("stages", {}).items():
        rows.append([stage_name, s_data.get("status", "[ ]"), s_data.get("label", "")])
    print_table(headers, rows)

    if status_info.get("warning"):
        print_warn(tr(f"\nPerhatian: {status_info['warning']}", f"\nWarning: {status_info['warning']}"))

    print(tr(f"\nSELANJUTNYA: Tahap '{status_info['next_step']}'", f"\nNEXT: Stage '{status_info['next_step']}'"))
    print(tr(f"Perintah   : {status_info['next_command']}\n", f"Command    : {status_info['next_command']}\n"))


def handle_config(args: argparse.Namespace) -> None:
    """Perintah config (show, set, auto) sesuai Bagian 18."""
    project_dir = Path(args.project).resolve()
    proj = load_project(project_dir)
    answers = proj.get("answers", {})
    curr_settings = proj.get("settings", {})
    plan = proj.get("plan", {})

    if args.config_action == "show":
        headers = [tr("Kunci", "Key"), tr("Nilai", "Value"), tr("Mode", "Mode")]
        if args.why:
            headers.append(tr("Alasan", "Reason"))
        rows = []
        for k, v in curr_settings.items():
            row = [k, str(v.get("value")), v.get("mode", "auto")]
            if args.why:
                row.append(str(v.get("reason", "")))
            rows.append(row)
        print_table(headers, rows, title=tr("PENGATURAN PROYEK", "PROJECT SETTINGS"))

        findings = run_advisor(answers, plan)
        if findings:
            print(tr("\nSARAN:", "\nSUGGESTIONS:"))
            for f in findings:
                print(f"[{f.level.upper()}] {f.id} {f.message} {tr('Saran', 'Suggestion')}: {f.suggestion}")

    elif args.config_action == "set":
        pair = args.key_val
        if "=" not in pair:
            print_error(tr("Format salah. Gunakan: python main.py config set kunci=nilai", "Invalid format. Use: python main.py config set key=val"))
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

        curr_settings[key] = {"value": val, "mode": "manual", "reason": tr("(kamu atur)", "(user set)")}
        new_plan = make_plan(answers, current_settings=curr_settings)
        proj["plan"] = new_plan
        proj["settings"] = new_plan["settings"]
        save_project(project_dir, proj)
        print_ok(tr(f"Setting '{key}' berhasil dikunci manual ke: {val}", f"Setting '{key}' manually locked to: {val}"))

    elif args.config_action == "auto":
        key = args.key.strip()
        if key in curr_settings:
            curr_settings[key]["mode"] = "auto"
            new_plan = make_plan(answers, current_settings=curr_settings)
            proj["plan"] = new_plan
            proj["settings"] = new_plan["settings"]
            save_project(project_dir, proj)
            auto_val = new_plan["settings"][key]["value"]
            print_ok(tr(f"Setting '{key}' dikembalikan ke mode auto (nilai saat ini: {auto_val})", f"Setting '{key}' reverted to auto mode (current value: {auto_val})"))


def handle_runs(args: argparse.Namespace) -> None:
    """Tampilkan riwayat semua run (Bagian 13.1)."""
    project_dir = Path(args.project).resolve()
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        print(tr("Belum ada folder runs/ yang terbentuk.", "No runs/ directory created yet."))
        return

    headers = [tr("Run ID", "Run ID"), tr("Jenis", "Type"), tr("Status", "Status"), tr("Step", "Step"), tr("Tokens", "Tokens"), tr("Loss Terakhir", "Last Loss")]
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
    print_table(headers, rows, title=tr("DAFTAR RUN TRAINING", "TRAINING RUNS LIST"))


def build_parser() -> argparse.ArgumentParser:
    """Bangun parser CLI argument."""
    parser = argparse.ArgumentParser(
        description=tr("LLM Training Toolkit (CLI interaktif)", "LLM Training Toolkit (Interactive CLI)"),
        add_help=False
    )
    parser.add_argument("--project", default=".", help=tr("Direktori proyek (default: direktori saat ini)", "Project directory (default: current directory)"))
    parser.add_argument("--yes", "-y", action="store_true", help=tr("Setujui semua konfirmasi otomatis", "Automatically confirm all prompts"))
    parser.add_argument("--dry-run", action="store_true", help=tr("Simulasikan tindakan tanpa menyimpan ke disk", "Simulate actions without saving to disk"))
    parser.add_argument("--lang", choices=["en", "id"], help=tr("Bahasa antarmuka CLI (id / en)", "CLI interface language (id / en)"))

    subparsers = parser.add_subparsers(dest="subcommand")

    # lang
    p_lang = subparsers.add_parser("lang", help=tr("Ubah atau lihat bahasa antarmuka CLI", "Change or view CLI interface language"))
    p_lang.add_argument("code", nargs="?", choices=["en", "id"], help=tr("Kode bahasa (en / id)", "Language code (en / id)"))

    # init
    p_init = subparsers.add_parser("init", help=tr("Inisialisasi proyek baru", "Initialize a new project"))
    p_init.add_argument("--answers", help=tr("Path ke file jawaban JSON untuk otomatisasi", "Path to JSON answers file for automation"))

    # plan
    p_plan = subparsers.add_parser("plan", help=tr("Kalkulator rencana cepat", "Quick plan calculator"))
    p_plan.add_argument("size", nargs="?", default="0.6B", help=tr("Ukuran model (contoh: 0.6B, 0.5B)", "Model size (e.g. 0.6B, 0.5B)"))
    p_plan.add_argument("--mode", default="balanced", help=tr("Mode token (chinchilla, balanced, overtrain, max)", "Token mode (chinchilla, balanced, overtrain, max)"))
    p_plan.add_argument("--tokens", help=tr("Jumlah token manual (contoh: 80B, 2T)", "Manual token count (e.g. 80B, 2T)"))
    p_plan.add_argument("--gpu", default="H100", help=tr("Nama GPU (H100, A100, L4, RTX 4090, dll)", "GPU name (H100, A100, L4, RTX 4090, etc.)"))
    p_plan.add_argument("--budget-usd", help=tr("Batas budget dalam USD", "Budget limit in USD"))
    p_plan.add_argument("--hours", help=tr("Batas jam sewa GPU", "GPU rental hours limit"))
    p_plan.add_argument("--set", action="append", help=tr("Uji coba override setting (kunci=nilai)", "Test override setting (key=value)"))

    # status
    subparsers.add_parser("status", help=tr("Periksa status proyek saat ini", "Check current project status"))

    # check
    p_check = subparsers.add_parser("check", help=tr("Validasi dataset", "Validate dataset"))
    p_check.add_argument("--quick", action="store_true", help=tr("Pemeriksaan cepat tanpa membaca seluruh baris", "Quick check without scanning all lines"))
    p_check.add_argument("--fix-hints", action="store_true", help=tr("Tampilkan detail petunjuk perbaikan error", "Show detailed error fix hints"))

    # tokenizer
    p_tok = subparsers.add_parser("tokenizer", help=tr("Latih tokenizer Byte-level BPE", "Train Byte-level BPE tokenizer"))
    p_tok.add_argument("--vocab-size", type=int, default=64000, help=tr("Ukuran vocab (default 64000)", "Vocab size (default 64000)"))
    p_tok.add_argument("--max-gb", type=float, default=1.5, help=tr("Batas maksimal ukuran data teks latih (GB)", "Max size of training text data (GB)"))

    # prep
    p_prep = subparsers.add_parser("prep", help=tr("Tokenisasi teks ke shard biner uint16", "Tokenize text into binary uint16 shards"))

    # bench
    subparsers.add_parser("bench", help=tr("Benchmark kecepatan hardware GPU", "Benchmark GPU hardware speed"))

    # pilot
    subparsers.add_parser("pilot", help=tr("Jalankan model uji coba 0.05B untuk sweep LR", "Run 0.05B pilot model for LR sweep"))

    # pretrain
    p_train = subparsers.add_parser("pretrain", help=tr("Mulai atau lanjutkan pretraining", "Start or resume pretraining"))
    p_train.add_argument("--resume", action="store_true", help=tr("Lanjutkan training dari checkpoint terakhir", "Resume training from last checkpoint"))
    p_train.add_argument("--new", action="store_true", help=tr("Mulai run baru meskipun ada checkpoint sebelumnya", "Start new run even if previous checkpoints exist"))
    p_train.add_argument("--run", help=tr("ID run tertentu untuk di-resume (contoh: run_001)", "Specific run ID to resume (e.g. run_001)"))
    p_train.add_argument("--step", type=int, help=tr("Step spesifik untuk di-resume", "Specific step to resume"))
    p_train.add_argument("--lr", type=float, help=tr("Ubah peak LR", "Override peak LR"))
    p_train.add_argument("--max-hours", type=float, help=tr("Jeda otomatis setelah N jam", "Pause automatically after N hours"))
    p_train.add_argument("--auto", action="store_true", help=tr("Supervisor auto-restart jika crash", "Auto-restart supervisor on crash"))
    p_train.add_argument("--out-dir", help=tr("Lokasi penyimpanan runs (misal ke Google Drive)", "Runs storage directory (e.g. to Google Drive)"))
    p_train.add_argument("--compile", action="store_true", help=tr("Aktifkan torch.compile", "Enable torch.compile"))

    # sft
    p_sft = subparsers.add_parser("sft", help=tr("Supervised Fine-Tuning", "Supervised Fine-Tuning"))
    p_sft.add_argument("--from", dest="from_run", help=tr("ID run atau path checkpoint awal (contoh: run_001 atau run_001:step_00010000)", "Run ID or initial checkpoint path (e.g. run_001 or run_001:step_00010000)"))
    p_sft.add_argument("--from-ckpt", help=tr("Path checkpoint pretrain awal", "Initial pretrain checkpoint path"))
    p_sft.add_argument("--epochs", type=int, help=tr("Jumlah epoch SFT", "Number of SFT epochs"))
    p_sft.add_argument("--lr", type=float, help=tr("Learning rate SFT", "SFT learning rate"))

    # eval
    p_eval = subparsers.add_parser("eval", help=tr("Evaluasi model", "Evaluate model"))
    p_eval.add_argument("--run", help=tr("ID run yang akan dievaluasi", "Run ID to evaluate"))
    p_eval.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"), help=tr("Bandingkan dua run berdampingan", "Compare two runs side-by-side"))
    p_eval.add_argument("--bench", action="store_true", help=tr("Jalankan benchmark lm_eval jika terpasang", "Run lm_eval benchmark if installed"))

    # chat
    p_chat = subparsers.add_parser("chat", help=tr("REPL chat terminal interaktif", "Interactive terminal chat REPL"))
    p_chat.add_argument("--run", help=tr("ID run yang digunakan", "Run ID to use"))
    p_chat.add_argument("--step", type=int, help=tr("Step checkpoint tertentu", "Specific checkpoint step"))
    p_chat.add_argument("--mode", default="auto", choices=["think", "no_think", "auto"], help=tr("Mode chat", "Chat mode"))
    p_chat.add_argument("--tools", action="store_true", help=tr("Aktifkan dukungan tool sandbox", "Enable tool sandbox support"))
    p_chat.add_argument("--force-nothink", action="store_true", help=tr("Paksa blok think kosong", "Force empty think block"))

    # export
    p_exp = subparsers.add_parser("export", help=tr("Ekspor ke format Hugging Face", "Export to Hugging Face format"))
    p_exp.add_argument("--run", help=tr("ID run yang diekspor", "Run ID to export"))
    p_exp.add_argument("--out-name", help=tr("Nama folder ekspor", "Export folder name"))

    # replan
    p_replan = subparsers.add_parser("replan", help=tr("Ubah jawaban init dan perbarui rencana", "Update init answers and recalculate plan"))
    p_replan.add_argument("--answers", help=tr("Path ke file jawaban JSON untuk otomatisasi", "Path to JSON answers file for automation"))
    p_replan.add_argument("-y", "--yes", action="store_true", help=tr("Pakai konfigurasi saat ini tanpa prompt interaktif", "Use current configuration without interactive prompt"))

    # make-identity
    subparsers.add_parser("make-identity", help=tr("Buat atau perbarui file data identitas SFT", "Create or update SFT identity data file"))

    # config
    p_cfg = subparsers.add_parser("config", help=tr("Kelola settings proyek", "Manage project settings"))
    cfg_subs = p_cfg.add_subparsers(dest="config_action")
    p_c_show = cfg_subs.add_parser("show", help=tr("Lihat semua settings", "View all settings"))
    p_c_show.add_argument("--why", action="store_true", help=tr("Tampilkan alasan nilai auto", "Show rationale for auto values"))
    p_c_set = cfg_subs.add_parser("set", help=tr("Kunci setting ke manual", "Lock setting to manual"))
    p_c_set.add_argument("key_val", help=tr("Pasangan kunci=nilai (contoh: context_final=8192)", "Key=value pair (e.g. context_final=8192)"))
    p_c_auto = cfg_subs.add_parser("auto", help=tr("Kembalikan setting ke mode auto", "Revert setting to auto mode"))
    p_c_auto.add_argument("key", help=tr("Nama setting", "Setting name"))

    # convert
    p_conv = subparsers.add_parser("convert", help=tr("Konversi format data teks", "Convert text data format"))
    p_conv.add_argument("source", help=tr("File sumber (JSONL / Parquet)", "Source file (JSONL / Parquet)"))
    p_conv.add_argument("--from-type", required=True, choices=["text_column", "sharegpt"], help=tr("Jenis format asal", "Source format type"))
    p_conv.add_argument("--to", required=True, help=tr("File tujuan output .jsonl", "Target output .jsonl file"))
    p_conv.add_argument("--column", default="text", help=tr("Nama kolom teks (jika dari text_column)", "Text column name (if from text_column)"))

    # helper lain
    subparsers.add_parser("runs", help=tr("Tampilkan riwayat semua run training", "Display training run history"))
    subparsers.add_parser("doctor", help=tr("Periksa lingkungan sistem", "Check system environment"))
    subparsers.add_parser("guide", help=tr("Panduan lengkap berurutan", "Complete sequential guide"))

    p_help = subparsers.add_parser("help", help=tr("Bantuan perintah", "Command help"))
    p_help.add_argument("target_command", nargs="?", help=tr("Perintah spesifik", "Specific command"))

    return parser


def main() -> None:
    """Fungsi utama dispatcher CLI."""
    # Pre-parse language argument or resolve language first
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--lang", choices=["en", "id"])
    temp_parser.add_argument("--project", default=".")
    temp_parser.add_argument("--yes", "-y", action="store_true")
    temp_args, _ = temp_parser.parse_known_args()

    project_dir = Path(temp_args.project).resolve()

    chosen_lang: Optional[str] = None
    persist_choice = False
    if temp_args.lang:
        chosen_lang = temp_args.lang
        persist_choice = True
    elif "lang" in sys.argv:
        idx = sys.argv.index("lang")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in ("en", "id"):
            chosen_lang = sys.argv[idx + 1]
            persist_choice = True

    if chosen_lang:
        set_lang(chosen_lang, persist_global=persist_choice, project_dir=project_dir)
    elif "LLMKIT_LANG" in os.environ:
        set_lang(os.environ["LLMKIT_LANG"], persist_global=False, project_dir=project_dir)
    else:
        saved = load_saved_lang(project_dir)
        if saved:
            set_lang(saved, persist_global=False, project_dir=project_dir)
        else:
            is_automated = (
                getattr(temp_args, "yes", False)
                or "pytest" in sys.modules
                or "unittest" in sys.modules
                or not sys.stdin.isatty()
            )
            if not is_automated:
                chosen = prompt_language_selection()
                set_lang(chosen, persist_global=True, project_dir=project_dir)
            else:
                set_lang("id", persist_global=False, project_dir=project_dir)

    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # If subcommand is lang without code or with code
    if args.subcommand == "lang":
        if getattr(args, "code", None):
            set_lang(args.code, persist_global=True, project_dir=project_dir)
            print_ok(tr(f"Bahasa antarmuka berhasil diatur ke: {args.code}", f"Interface language successfully set to: {args.code}"))
        else:
            if sys.stdin.isatty() and not getattr(args, "yes", False):
                code = prompt_language_selection()
                set_lang(code, persist_global=True, project_dir=project_dir)
                print_ok(tr(f"Bahasa antarmuka berhasil diatur ke: {code}", f"Interface language successfully set to: {code}"))
            else:
                curr = get_lang()
                print(tr(f"Bahasa antarmuka saat ini: {curr} ({'Bahasa Indonesia' if curr == 'id' else 'English'})", f"Current interface language: {curr} ({'Bahasa Indonesia' if curr == 'id' else 'English'})"))
        return

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
        print_error(tr("Tidak menemukan project.json di direktori kerja.", "Could not find project.json in working directory."))
        print(tr("Silakan jalankan: python main.py init  atau gunakan opsi --project <folder>.", "Please run: python main.py init or use the --project <folder> option."))
        sys.exit(1)

    if args.subcommand == "status":
        handle_status(args)
    elif args.subcommand == "replan":
        handle_replan(args)
    elif args.subcommand == "make-identity":
        make_identity_file(project_dir)
    elif args.subcommand == "check":
        ok, res = check_dataset(project_dir, quick=args.quick, fix_hints=args.fix_hints)
        print(tr(f"\nStatus Pemeriksaan Data: {'[OK] Lulus' if ok else '[ERROR] Ditemukan masalah'}", f"\nData Check Status: {'[OK] Passed' if ok else '[ERROR] Issues found'}"))
        if not ok:
            print(tr(f"Lihat rincian lengkap error di: {project_dir / 'logs' / 'check_errors.txt'}", f"See complete error details at: {project_dir / 'logs' / 'check_errors.txt'}"))
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
                        print_error(tr(f"Pretrain crash berulang melebihi batas {max_restarts} kali: {e}", f"Pretrain crashed repeatedly exceeding limit of {max_restarts} times: {e}"))
                        sys.exit(1)
                    print_warn(tr(f"Pretrain crash: {e}. Supervisor auto-restart ({restart_count}/{max_restarts}) dalam 30 detik...", f"Pretrain crash: {e}. Supervisor auto-restart ({restart_count}/{max_restarts}) in 30 seconds..."))
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
        print_ok(tr(f"Berhasil mengonversi {num:,} baris ke: {dst}", f"Successfully converted {num:,} rows to: {dst}"))
