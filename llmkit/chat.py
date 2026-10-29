"""REPL Chat interaktif dengan dukungan mode think/no-think dan loop eksekusi tool otomatis (Bagian 12.5)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional
import torch
from tokenizers import Tokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM
from llmkit.chat_template import EOS_TOKEN, get_default_system, render_conversation
from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint
from llmkit.i18n import tr
from llmkit.model import build_qwen3_config, create_model
from llmkit.state import load_project
from llmkit.tools_sandbox import SANDBOX_TOOLS_SCHEMAS, execute_tool
from llmkit.ui import print_info, print_warn


def run_chat_repl(
    project_dir: Path,
    run_id: Optional[str] = None,
    specific_step: Optional[int] = None,
    initial_mode: str = "auto",
    use_tools: bool = False,
    force_nothink: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    max_new_tokens: int = 512
) -> None:
    """
    REPL terminal untuk berinteraksi langsung dengan model hasil training.
    """
    project_data = load_project(project_dir)
    plan = project_data.get("plan", {})
    arch = plan.get("architecture", {})

    tok_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        print(tr("[ERROR] Tokenizer belum dilatih. Jalankan 'tokenizer' terlebih dahulu.", "[ERROR] Tokenizer not trained yet. Run 'tokenizer' first."))
        return

    tokenizer = Tokenizer.from_file(str(tok_path))
    eos_id = tokenizer.token_to_id(EOS_TOKEN) or 2

    runs_dir = project_dir / "runs"
    if run_id:
        target_run_dir = runs_dir / run_id
    else:
        existing_runs = sorted(list(runs_dir.glob("run_*")), key=lambda p: p.name)
        if not existing_runs:
            print(tr("[ERROR] Belum ada run training yang ditemukan.", "[ERROR] No training runs found yet."))
            return
        target_run_dir = existing_runs[-1]

    ckpt_dir = target_run_dir / "ckpt"
    if specific_step is not None:
        ckpt_path = ckpt_dir / f"step_{specific_step:08d}"
    else:
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    if not ckpt_path or not ckpt_path.exists():
        print(tr(f"[ERROR] Checkpoint tidak ditemukan di {ckpt_dir}.", f"[ERROR] Checkpoint not found in {ckpt_dir}."))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_qwen3_config(arch, max_position_embeddings=4096)
    model = create_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    print_info(tr(
        f"Chat REPL aktif | Model: {target_run_dir.name} | Checkpoint: {ckpt_path.name}",
        f"Chat REPL active | Model: {target_run_dir.name} | Checkpoint: {ckpt_path.name}"
    ))
    print(tr(
        "Ketik pesanmu. Perintah khusus: /think, /nothink, /reset, /tools on|off, /system <teks>, /exit",
        "Type your message. Special commands: /think, /nothink, /reset, /tools on|off, /system <text>, /exit"
    ))

    answers = project_data.get("answers", {})
    model_name = answers.get("model_name", "Bahasa")
    creator = answers.get("creator", "Pengguna")

    current_mode = initial_mode
    tools_active = use_tools
    custom_system: Optional[str] = get_default_system(model_name, creator)
    history_messages: list[dict[str, Any]] = []

    while True:
        try:
            mode_badge = f"[{current_mode}]"
            tool_badge = "[+tools]" if tools_active else ""
            user_input = input(f"\nUser {mode_badge}{tool_badge}> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + tr("Keluar dari chat.", "Exiting chat."))
            break

        if not user_input:
            continue

        # Perintah CLI internal dalam chat
        if user_input == "/exit":
            print(tr("Sampai jumpa!", "Goodbye!"))
            break
        elif user_input == "/reset":
            history_messages.clear()
            print_info(tr("Riwayat percakapan dibersihkan.", "Conversation history cleared."))
            continue
        elif user_input == "/think":
            current_mode = "think"
            print_info(tr("Mode dialihkan ke: think", "Mode switched to: think"))
            continue
        elif user_input == "/nothink":
            current_mode = "no_think"
            print_info(tr("Mode dialihkan ke: no_think", "Mode switched to: no_think"))
            continue
        elif user_input.startswith("/tools "):
            arg = user_input.split()[1].lower()
            tools_active = (arg == "on")
            st_str = tr('aktif' if tools_active else 'mati', 'active' if tools_active else 'inactive')
            print_info(tr(f"Fitur tools: {st_str}", f"Tools feature: {st_str}"))
            continue
        elif user_input.startswith("/system "):
            custom_system = user_input[8:].strip()
            print_info(tr(f"System prompt diperbarui: {custom_system}", f"System prompt updated: {custom_system}"))
            continue

        # Susun riwayat giliran
        history_messages.append({"role": "user", "content": user_input})

        # Loop Tool Execution (maksimal 5 iterasi)
        iterations = 0
        while iterations < 5:
            iterations += 1

            sample_dict = {
                "id": f"chat-{len(history_messages)}",
                "mode": "think" if current_mode == "think" else "no_think",
                "tools": SANDBOX_TOOLS_SCHEMAS if tools_active else [],
                "messages": history_messages,
            }

            prompt_text, _ = render_conversation(
                sample_dict,
                default_system=custom_system,
                add_generation_prompt=True,
                force_nothink=force_nothink
            )

            # Generasi respons
            input_ids = torch.tensor([tokenizer.encode(prompt_text).ids], dtype=torch.long).to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_id,
                    pad_token_id=11
                )

            gen_ids = out[0][input_ids.shape[1]:].tolist()
            reply = tokenizer.decode(gen_ids).strip()

            # Bersihkan penutup <|im_end|> jika ada
            if reply.endswith("<|im_end|>"):
                reply = reply[:-10].strip()

            # Pisahkan pemikiran <think> dari jawaban
            think_match = re.search(r"<think>(.*?)</think>", reply, flags=re.DOTALL)
            content_part = reply
            if think_match:
                reasoning_txt = think_match.group(1).strip()
                if reasoning_txt:
                    pikir_tag = tr("[Pikir]", "[Think]")
                    print(f"\n[bold cyan]{pikir_tag}[/bold cyan]\n{reasoning_txt}")
                content_part = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

            # Cek apakah ada panggilan tool <tool_call>
            tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content_part, flags=re.DOTALL)
            if tool_match and tools_active:
                raw_call = tool_match.group(1).strip()
                try:
                    tc_obj = json.loads(raw_call)
                    fn_name = tc_obj.get("name", "")
                    fn_args = tc_obj.get("arguments", {})
                    print(f"\n[Tool Call] {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

                    tool_output = execute_tool(fn_name, fn_args)
                    print(f"[Tool Output] {tool_output}")

                    # Tambahkan assistant message dengan tool call ke riwayat
                    history_messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tc_obj]
                    })
                    # Tambahkan balasan tool
                    history_messages.append({
                        "role": "tool",
                        "name": fn_name,
                        "content": tool_output
                    })
                    # Lanjutkan generasi otomatis
                    continue

                except Exception as e:
                    print_warn(f"Gagal mem-parse tool call: {e}")

            # Respons selesai (tanpa tool call baru)
            print(f"\nAssistant> {content_part}")
            history_messages.append({"role": "assistant", "content": content_part})
            break
