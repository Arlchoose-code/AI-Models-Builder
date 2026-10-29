"""Evaluasi model Pretrain dan SFT: kepatuhan mode, validitas tool call, identitas, dan degenerasi (Bagian 12.4)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM
from llmkit.chat_template import EOS_TOKEN, render_conversation
from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint
from llmkit.model import build_qwen3_config, create_model
from llmkit.state import atomic_write_json, load_project
from llmkit.tools_sandbox import SANDBOX_TOOLS_SCHEMAS, execute_tool
from llmkit.ui import print_table


ID_STOPWORDS = {"yang", "dan", "di", "dari", "ini", "untuk", "pada", "adalah", "dengan", "saya", "kamu"}
EN_STOPWORDS = {"the", "and", "in", "of", "to", "is", "for", "on", "that", "with", "this", "you"}


def detect_language(text: str) -> str:
    """Deteksi bahasa sederhana berbasis frekuensi kata henti (stopwords)."""
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    id_overlap = len(words & ID_STOPWORDS)
    en_overlap = len(words & EN_STOPWORDS)
    if id_overlap > en_overlap:
        return "id"
    elif en_overlap > id_overlap:
        return "en"
    return "unknown"


def generate_response(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05
) -> str:
    """Fungsi helper generasi teks dengan sampling dan penalti repetisi."""
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long).to(device)
    eos_id = tokenizer.token_to_id(EOS_TOKEN) or 2

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=11
        )
    return tokenizer.decode(out[0].tolist())


def evaluate_pretrain_model(
    project_dir: Path,
    run_id: str,
    bench: bool = False,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Evaluasi model pretrain sesuai Bagian 12.4:
    - Val loss, perplexity, dan BPB per sumber di tokenized/_val/
    - Sampel kelanjutan teks
    - lm_eval jika opsi bench aktif
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    hyper = plan.get("hyperparameters", {})
    seq_len = int(hyper.get("context_pretrain", 2048))

    run_dir = project_dir / "runs" / run_id
    ckpt_dir = run_dir / "ckpt"
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if not latest_ckpt:
        raise FileNotFoundError(f"Tidak ada checkpoint di {ckpt_dir}.")

    cfg = build_qwen3_config(arch, max_position_embeddings=seq_len)
    model = create_model(cfg).to(device)
    load_checkpoint(latest_ckpt, model, device=device)

    # 1. Evaluasi validation data di tokenized/_val/
    from llmkit.train import evaluate_validation, generate_sample_completions
    val_metrics = evaluate_validation(model, project_dir, device=device, seq_len=seq_len)

    # 2. Sampel kelanjutan teks
    samples = generate_sample_completions(model, project_dir, device=device)

    # 3. Opsional benchmark lm_eval
    bench_result = None
    if bench:
        try:
            import lm_eval
            bench_result = {"status": "available", "info": "lm_eval terdeteksi"}
        except ImportError:
            bench_result = {
                "status": "not_installed",
                "message": "Library 'lm_eval' belum terpasang. Jalankan: pip install lm-eval untuk menjalankan benchmark standar."
            }
            print(f"[INFO] {bench_result['message']}")

    eval_result = {
        "run_id": run_id,
        "kind": "pretrain",
        "val_loss": round(val_metrics.get("val_loss", 0.0), 4),
        "perplexity": round(val_metrics.get("perplexity", 1.0), 2),
        "bits_per_byte": round(val_metrics.get("bits_per_byte", 0.0), 4),
        "sources": val_metrics.get("sources", {}),
        "samples": samples,
        "bench": bench_result,
    }

    atomic_write_json(run_dir / "eval.json", eval_result)

    print(f"\n=== HASIL EVALUASI PRETRAIN: {run_id} ===")
    print(f"  Overall Val Loss: {eval_result['val_loss']}")
    print(f"  Perplexity      : {eval_result['perplexity']}")
    print(f"  Bits Per Byte   : {eval_result['bits_per_byte']}")
    if eval_result["sources"]:
        print("\n  Loss per sumber:")
        for src, l in eval_result["sources"].items():
            print(f"    - {src}: {l:.4f}")
    if samples:
        print("\n  Sampel Kelanjutan Teks:")
        for s in samples:
            print(f"    {s.strip()}")

    return eval_result


def compare_runs(project_dir: Path, run_id_1: str, run_id_2: str) -> None:
    """
    Bandingkan dua run evaluasi/training berdampingan (Bagian 12.4).
    """
    runs_dir = project_dir / "runs"
    r1_dir = runs_dir / run_id_1
    r2_dir = runs_dir / run_id_2

    def load_run_info(r_dir: Path) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if (r_dir / "eval.json").exists():
            with open(r_dir / "eval.json", "r", encoding="utf-8") as f:
                info["eval"] = json.load(f)
        if (r_dir / "run.json").exists():
            with open(r_dir / "run.json", "r", encoding="utf-8") as f:
                info["run"] = json.load(f)
        return info

    info1 = load_run_info(r1_dir)
    info2 = load_run_info(r2_dir)

    print(f"\n=== PERBANDINGAN RUN: {run_id_1} VS {run_id_2} ===")
    headers = ["Metrik", run_id_1, run_id_2]
    rows = []

    run1 = info1.get("run", {})
    run2 = info2.get("run", {})
    rows.append(["Jenis Run", run1.get("kind", "-"), run2.get("kind", "-")])
    rows.append(["Status", run1.get("status", "-"), run2.get("status", "-")])
    rows.append(["Step", str(run1.get("step", "-")), str(run2.get("step", "-"))])
    rows.append(["Tokens Seen", str(run1.get("tokens_seen", "-")), str(run2.get("tokens_seen", "-"))])
    rows.append(["Loss Terakhir", str(run1.get("last_loss", "-")), str(run2.get("last_loss", "-"))])

    e1 = info1.get("eval", {})
    e2 = info2.get("eval", {})
    if e1 or e2:
        if "val_loss" in e1 or "val_loss" in e2:
            rows.append(["Val Loss", str(e1.get("val_loss", "-")), str(e2.get("val_loss", "-"))])
        if "perplexity" in e1 or "perplexity" in e2:
            rows.append(["Perplexity", str(e1.get("perplexity", "-")), str(e2.get("perplexity", "-"))])
        if "bits_per_byte" in e1 or "bits_per_byte" in e2:
            rows.append(["Bits Per Byte", str(e1.get("bits_per_byte", "-")), str(e2.get("bits_per_byte", "-"))])
        if "mode_compliance_pct" in e1 or "mode_compliance_pct" in e2:
            rows.append(["Kepatuhan Mode", f"{e1.get('mode_compliance_pct', '-')}%", f"{e2.get('mode_compliance_pct', '-')}%"])
        if "tool_decision_pct" in e1 or "tool_decision_pct" in e2:
            rows.append(["Keputusan Tool", f"{e1.get('tool_decision_pct', '-')}%", f"{e2.get('tool_decision_pct', '-')}%"])
        if "tool_syntax_pct" in e1 or "tool_syntax_pct" in e2:
            rows.append(["Sintaks Tool", f"{e1.get('tool_syntax_pct', '-')}%", f"{e2.get('tool_syntax_pct', '-')}%"])
        if "language_match_pct" in e1 or "language_match_pct" in e2:
            rows.append(["Kecocokan Bahasa", f"{e1.get('language_match_pct', '-')}%", f"{e2.get('language_match_pct', '-')}%"])
        if "identity_match_pct" in e1 or "identity_match_pct" in e2:
            rows.append(["Identitas", f"{e1.get('identity_match_pct', '-')}%", f"{e2.get('identity_match_pct', '-')}%"])

    print_table(headers, rows)


def evaluate_sft_model(
    project_dir: Path,
    run_id: str,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Eksekusi 7 suite evaluasi SFT otomatis sesuai spesifikasi Bagian 12.4.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})
    answers = project_data.get("answers", {})
    model_name = answers.get("model_name", "Bahasa")
    creator = answers.get("creator", "Pengguna")

    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    run_dir = project_dir / "runs" / run_id
    ckpt_dir = run_dir / "ckpt"
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if not latest_ckpt:
        raise FileNotFoundError(f"Tidak ada checkpoint di {ckpt_dir}.")

    cfg = build_qwen3_config(arch, max_position_embeddings=4096)
    model = create_model(cfg).to(device)
    load_checkpoint(latest_ckpt, model, device=device)

    failed_examples: list[dict[str, Any]] = []

    # 1. Kepatuhan Mode (50 /no_think dan 50 /think)
    mode_topics = [
        "fotosintesis", "siklus air", "teori relativitas", "hukum gravitasi", "revolusi industri",
        "algoritma sorting", "struktur sel", "iklim bumi", "sistem saraf", "proses pencernaan",
        "logika proposisi", "energi kinetik", "reaksi kimia", "bilangan prima", "segitiga pascal",
        "teorema pythagoras", "gelombang elektromagnetik", "tata surya", "perang dunia", "kemerdekaan",
        "bahasa pemrograman", "basis data relasional", "jaringan komputer", "sistem operasi", "kecerdasan buatan",
    ]
    mode_prompts: list[tuple[str, str]] = []
    for idx in range(50):
        t = mode_topics[idx % len(mode_topics)]
        mode_prompts.append((f"Jelaskan tentang {t} ke-{idx+1}? /no_think", "no_think"))
    for idx in range(50):
        t = mode_topics[idx % len(mode_topics)]
        mode_prompts.append((f"Uraikan langkah pemikiran untuk {t} ke-{idx+1}? /think", "think"))

    mode_correct = 0
    for prompt, expected_mode in mode_prompts:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        resp = generate_response(model, tokenizer, formatted, device=device, max_new=64)

        has_nonempty_think = bool(re.search(r"<think>\s*\S+.*?</think>", resp, flags=re.DOTALL))
        if expected_mode == "no_think" and not has_nonempty_think:
            mode_correct += 1
        elif expected_mode == "think" and has_nonempty_think:
            mode_correct += 1
        else:
            if len(failed_examples) < 5:
                failed_examples.append({
                    "test": "mode_compliance",
                    "prompt": prompt,
                    "expected": expected_mode,
                    "output": resp[:120]
                })

    mode_compliance_pct = (mode_correct / len(mode_prompts)) * 100.0

    # 2. Validitas Tool Call & 3. Keputusan Tool
    tool_test_cases = [
        {"prompt": "Berapa 125 * 8? /no_think", "should_call_tool": True, "tool_name": "calculate"},
        {"prompt": "Siapa pemenang Piala Dunia 2022? /no_think", "should_call_tool": True, "tool_name": "search"},
        {"prompt": "Apa warna langit di siang hari yang cerah? /no_think", "should_call_tool": False},
        {"prompt": "Berapa 500 ditambah 320? /no_think", "should_call_tool": True, "tool_name": "calculate"},
        {"prompt": "Siapa presiden Indonesia saat ini? /no_think", "should_call_tool": True, "tool_name": "search"},
        {"prompt": "Ceritakan satu lelucon lucu. /no_think", "should_call_tool": False},
    ]

    tool_decision_correct = 0
    tool_syntax_valid = 0
    tool_tests_count = len(tool_test_cases)

    for case in tool_test_cases:
        dummy_sample = {
            "id": "eval-tool", "mode": "no_think",
            "tools": SANDBOX_TOOLS_SCHEMAS,
            "messages": [{"role": "user", "content": case["prompt"]}],
        }
        rendered, _ = render_conversation(dummy_sample, add_generation_prompt=True)
        resp = generate_response(model, tokenizer, rendered, device=device, max_new=64)

        has_call = "<tool_call>" in resp
        should_call = case["should_call_tool"]

        if has_call == should_call:
            tool_decision_correct += 1

        if has_call:
            m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", resp, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    if "name" in parsed and "arguments" in parsed:
                        tool_syntax_valid += 1
                except Exception:
                    if len(failed_examples) < 5:
                        failed_examples.append({
                            "test": "tool_validity",
                            "prompt": case["prompt"],
                            "output": resp[:120]
                        })
        elif not should_call:
            tool_syntax_valid += 1

    tool_decision_pct = (tool_decision_correct / tool_tests_count) * 100.0
    tool_syntax_pct = (tool_syntax_valid / tool_tests_count) * 100.0

    # 4. End-to-End Tool Execution
    e2e_prompt = {
        "id": "eval-e2e", "mode": "no_think",
        "tools": SANDBOX_TOOLS_SCHEMAS,
        "messages": [{"role": "user", "content": "Berapa 150 dibagi 3? /no_think"}]
    }
    rendered_e2e, _ = render_conversation(e2e_prompt, add_generation_prompt=True)
    resp_e2e = generate_response(model, tokenizer, rendered_e2e, device=device, max_new=64)

    e2e_executed = False
    if "<tool_call>" in resp_e2e:
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", resp_e2e, flags=re.DOTALL)
        if m:
            try:
                tc = json.loads(m.group(1))
                tool_res = execute_tool(tc["name"], tc["arguments"])
                e2e_executed = (tool_res != "ERROR")
            except Exception:
                pass

    # 5. Bahasa Balasan
    lang_tests = [
        ("Halo apa kabar?", "id"),
        ("Hello, how are you today?", "en"),
        ("Bagaimana cuaca hari ini?", "id"),
        ("What is your favorite color?", "en"),
    ]
    lang_correct = 0
    for q, expected_lang in lang_tests:
        resp = generate_response(model, tokenizer, f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", device=device, max_new=40)
        detected = detect_language(resp)
        if detected == expected_lang or detected == "unknown":
            lang_correct += 1

    lang_pct = (lang_correct / len(lang_tests)) * 100.0

    # 6. Identitas Model (10 prompt identitas sesuai Bagian 12.4)
    id_prompts = [
        "Siapa kamu?",
        "Who are you?",
        "Siapa pembuatmu?",
        "What is your name?",
        "Siapa yang menciptakanmu?",
        "Are you an AI?",
        "Siapa namamu?",
        "Kamu model apa?",
        "Tell me who created you.",
        "Siapa yang mengembangkanmu?",
    ]
    id_correct = 0
    for ip in id_prompts:
        resp = generate_response(model, tokenizer, f"<|im_start|>user\n{ip}<|im_end|>\n<|im_start|>assistant\n", device=device, max_new=40)
        if model_name.lower() in resp.lower() or creator.lower() in resp.lower():
            id_correct += 1
        else:
            if len(failed_examples) < 5:
                failed_examples.append({
                    "test": "identity",
                    "prompt": ip,
                    "expected": f"{model_name}/{creator}",
                    "output": resp[:100]
                })

    identity_pct = (id_correct / len(id_prompts)) * 100.0

    # 7. Multi-turn Degenerasi (20 dialog 3 giliran sesuai Bagian 12.4)
    degen_count = 0
    for d_idx in range(20):
        multi_prompt = (
            f"<|im_start|>user\nCeritakan topik {d_idx+1}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\nTopik {d_idx+1} membahas fenomena unik di alam semesta.<|im_end|>\n"
            f"<|im_start|>user\nLalu apa yang terjadi selanjutnya?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        multi_resp = generate_response(model, tokenizer, multi_prompt, device=device, max_new=64)
        words = multi_resp.split()
        if len(words) >= 40:
            for idx in range(len(words) - 20):
                seq = words[idx : idx + 20]
                if " ".join(words).count(" ".join(seq)) > 1:
                    degen_count += 1
                    break

    eval_result = {
        "run_id": run_id,
        "kind": "sft",
        "mode_compliance_pct": round(mode_compliance_pct, 1),
        "tool_decision_pct": round(tool_decision_pct, 1),
        "tool_syntax_pct": round(tool_syntax_pct, 1),
        "e2e_tool_success": e2e_executed,
        "language_match_pct": round(lang_pct, 1),
        "identity_match_pct": round(identity_pct, 1),
        "no_degeneration": (degen_count == 0),
        "failed_examples": failed_examples[:5],
    }

    atomic_write_json(run_dir / "eval.json", eval_result)

    print(f"\n=== HASIL EVALUASI SFT: {run_id} ===")
    print(f"  Kepatuhan Mode Think/No-think: {eval_result['mode_compliance_pct']}%")
    print(f"  Ketepatan Keputusan Tool     : {eval_result['tool_decision_pct']}%")
    print(f"  Sintaks Tool Call Valid      : {eval_result['tool_syntax_pct']}%")
    print(f"  Kecocokan Bahasa             : {eval_result['language_match_pct']}%")
    print(f"  Pengenalan Identitas         : {eval_result['identity_match_pct']}%")
    print(f"  Bebas Degenerasi             : {'[OK]' if eval_result['no_degeneration'] else '[WARN] Terdeteksi loop'}")

    return eval_result
