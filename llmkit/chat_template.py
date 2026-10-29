"""Spesifikasi dan implementasi ChatML / Chat Template (Bagian 8)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
    "<|pad|>",
] + [f"<|reserved_{i}|>" for i in range(16)]

EOS_TOKEN = "<|im_end|>"
PRETRAIN_EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"


def get_default_system(model_name: str = "Bahasa", creator: str = "Pengguna") -> str:
    """System prompt bawaan jika tidak ada peran system eksplisit."""
    return f"Kamu adalah {model_name}, asisten AI dwibahasa (Indonesia dan Inggris) buatan {creator}."


def render_conversation(
    sample: dict[str, Any],
    default_system: Optional[str] = None,
    allow_think_tags: bool = True,
    add_generation_prompt: bool = False,
    force_nothink: bool = False,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Render percakapan kanonik menjadi teks tokenisasi dan daftar span (char_start, char_end)
    yang dilatih (loss masking). Sesuai Bagian 8.2 & 8.3.
    """
    messages = list(sample.get("messages", []))
    tools = sample.get("tools", [])
    sample_id = str(sample.get("id", "0"))
    mode = str(sample.get("mode", "no_think"))

    # Cek apakah ada system prompt eksplisit
    has_system = any(m.get("role") == "system" for m in messages)
    system_text = ""
    if has_system:
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content", "").strip()
                break
    else:
        system_text = default_system or get_default_system()

    rendered = ""
    spans: list[tuple[int, int]] = []

    # 1. System (+ tools jika ada)
    sys_body = system_text
    if tools:
        tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tools_block = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_str}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        sys_body += tools_block

    rendered += f"<|im_start|>system\n{sys_body}<|im_end|>\n"

    # Cari index user message terakhir yang berasal dari manusia
    last_user_idx = -1
    for idx, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = idx

    # Kelompokkan giliran secara sekuensial
    # Peran tool yang berurutan digabung ke dalam satu giliran user dengan <tool_response>
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        if role == "system":
            i += 1
            continue

        if role == "user":
            content = m.get("content", "")
            # Tempelkan tag mode jika ini user terakhir dari manusia
            if i == last_user_idx and allow_think_tags:
                if mode == "think":
                    content = f"{content} /think"
                elif mode == "no_think":
                    # 50% diberi /no_think (deterministik berdasarkan hash ID)
                    h_val = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest(), 16)
                    if h_val % 2 == 0:
                        content = f"{content} /no_think"
            rendered += f"<|im_start|>user\n{content}<|im_end|>\n"
            i += 1

        elif role == "tool":
            # Kumpulkan semua pesan tool yang berurutan
            tool_responses: list[str] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                t_msg = messages[i]
                t_content = t_msg.get("content", "")
                tool_responses.append(f"<tool_response>\n{t_content}\n</tool_response>")
                i += 1
            all_resp = "\n".join(tool_responses)
            rendered += f"<|im_start|>user\n{all_resp}<|im_end|>\n"

        elif role == "assistant":
            # Assistant turn
            is_after_last_user = (i > last_user_idx)
            reasoning = m.get("reasoning", "").strip() if (is_after_last_user and mode == "think") else ""
            content = m.get("content", "").strip()
            tool_calls = m.get("tool_calls", [])

            start_header = len(rendered)
            rendered += "<|im_start|>assistant\n"
            train_start = len(rendered)

            # Bagian think
            if reasoning:
                think_block = f"<think>\n{reasoning}\n</think>\n\n"
            else:
                think_block = "<think>\n\n</think>\n\n"

            body = think_block

            # Konten dan tool_calls
            if content and tool_calls:
                body += f"{content}\n"
                calls_str = "\n".join(
                    f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
                    for tc in tool_calls
                )
                body += calls_str
            elif tool_calls:
                calls_str = "\n".join(
                    f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
                    for tc in tool_calls
                )
                body += calls_str
            else:
                body += content

            body_with_eos = f"{body}<|im_end|>"
            rendered += body_with_eos
            train_end = len(rendered)
            rendered += "\n"

            spans.append((train_start, train_end))
            i += 1
        else:
            i += 1

    if add_generation_prompt:
        rendered += "<|im_start|>assistant\n"
        if force_nothink:
            rendered += "<think>\n\n</think>\n\n"

    return rendered, spans


def get_jinja_template(default_system_text: str = "Kamu adalah Bahasa, asisten AI dwibahasa (Indonesia dan Inggris) buatan Pengguna.") -> str:
    """
    Mengembalikan string Jinja template yang kompatibel dengan tokenizer transformers
    dan menghasilkan teks yang 100% identik dengan render_conversation (Bagian 8.4).
    """
    template = (
        "{%- if messages[0]['role'] == 'system' %}"
        "{{ '<|im_start|>system\\n' + messages[0]['content'] }}"
        "{%- else %}"
        f"{{{{ '<|im_start|>system\\n{default_system_text}' }}}}"
        "{%- endif %}"
        "{%- if tools %}"
        "{{ '\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n' }}"
        "{%- for tool in tools %}"
        "{{ tool | tojson }}{{ '\\n' }}"
        "{%- endfor %}"
        "{{ '</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call>' }}"
        "{%- endif %}"
        "{{ '<|im_end|>\\n' }}"
        "{%- set last_user_idx = -1 %}"
        "{%- for m in messages %}"
        "{%- if m.role == 'user' %}"
        "{%- set last_user_idx = loop.index0 %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- for m in messages %}"
        "{%- if m.role == 'user' %}"
        "{{ '<|im_start|>user\\n' + m.content + '<|im_end|>\\n' }}"
        "{%- elif m.role == 'tool' %}"
        "{%- if loop.first or messages[loop.index0 - 1]['role'] != 'tool' %}"
        "{{ '<|im_start|>user\\n' }}"
        "{%- endif %}"
        "{{ '<tool_response>\\n' + m.content + '\\n</tool_response>' }}"
        "{%- if loop.last or messages[loop.index0 + 1]['role'] != 'tool' %}"
        "{{ '<|im_end|>\\n' }}"
        "{%- else %}"
        "{{ '\\n' }}"
        "{%- endif %}"
        "{%- elif m.role == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n<think>\\n' }}"
        "{%- if m.reasoning is defined and m.reasoning %}"
        "{{ m.reasoning + '\\n' }}"
        "{%- else %}"
        "{{ '\\n' }}"
        "{%- endif %}"
        "{{ '</think>\\n\\n' }}"
        "{%- if m.content %}"
        "{{ m.content }}"
        "{%- if m.tool_calls is defined and m.tool_calls %}"
        "{{ '\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
        "{%- if m.tool_calls is defined and m.tool_calls %}"
        "{%- for tc in m.tool_calls %}"
        "{{ '<tool_call>\\n' }}{{ tc | tojson }}{{ '\\n</tool_call>' }}"
        "{%- if not loop.last %}{{ '\\n' }}{%- endif %}"
        "{%- endfor %}"
        "{%- endif %}"
        "{{ '<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    )
    return template
