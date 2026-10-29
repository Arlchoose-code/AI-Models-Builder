"""Sandbox eksekusi fungsi/tool aman (calculate via AST, mock search, get_time) (Bagian 12.4)."""

from __future__ import annotations

import ast
import datetime
import json
import operator
from pathlib import Path
from typing import Any, Optional


# AST Safe Calculator
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            left = _eval_ast_node(node.left)
            right = _eval_ast_node(node.right)
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Operator tidak diizinkan: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            operand = _eval_ast_node(node.operand)
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Operator unari tidak diizinkan: {op_type}")
    raise ValueError(f"Ekspresi tidak valid: {ast.dump(node)}")


def safe_calculate(expression: str) -> str:
    """Hitung ekspresi matematika secara aman menggunakan parsing AST."""
    try:
        clean_expr = expression.strip().replace("^", "**")
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_ast_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


# Mock Knowledge Base untuk Search
DEFAULT_MOCK_KB: dict[str, str] = {
    "piala dunia terakhir": "Argentina menjuarai Piala Dunia FIFA 2022 di Qatar setelah mengalahkan Prancis melalui adu penalti.",
    "world cup": "Argentina won the 2022 FIFA World Cup in Qatar against France.",
    "capital of france": "The capital of France is Paris.",
    "ibu kota perancis": "Ibu kota Perancis adalah Paris.",
    "ibu kota indonesia": "Ibu kota Indonesia saat ini berpindah secara bertahap dari Jakarta ke Nusantara di Kalimantan Timur.",
    "presiden indonesia": "Presiden Republik Indonesia ke-8 adalah Prabowo Subianto.",
    "indonesia independence": "Indonesia memproklamasikan kemerdekaan pada 17 Agustus 1945.",
}


def safe_search(query: str, kb_file: Optional[Path] = None) -> str:
    """Pencarian berbasis mock knowledge base."""
    kb = dict(DEFAULT_MOCK_KB)
    if kb_file is None:
        cand_kb = Path("tests/mock_kb.json")
        if cand_kb.exists():
            kb_file = cand_kb
    if kb_file and kb_file.exists():
        try:
            with open(kb_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                kb.update(loaded)
        except Exception:
            pass

    q_lower = query.lower().strip()
    for k, v in kb.items():
        if k in q_lower or any(word in q_lower for word in k.split()):
            return v
    return f"Tidak ditemukan informasi yang cocok untuk query: '{query}'."


def get_current_time() -> str:
    """Dapatkan waktu saat ini dalam format terbaca."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# Definisi Skema Tool Standar
SANDBOX_TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Hitung ekspresi matematika sederhana secara aman (+, -, *, /, **, %)",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Ekspresi matematika, contoh: '125 * 8'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Cari informasi fakta dan peristiwa terkini di basis data pengetahuan",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci pencarian"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Dapatkan tanggal dan waktu saat ini",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def execute_tool(name: str, arguments: dict[str, Any], kb_file: Optional[Path] = None) -> str:
    """Eksekutor tunggal fungsi sandbox berdasarkan nama fungsi dan argumen."""
    if name == "calculate":
        expr = str(arguments.get("expression", ""))
        return safe_calculate(expr)
    elif name == "search":
        q = str(arguments.get("query", ""))
        return safe_search(q, kb_file=kb_file)
    elif name == "get_time":
        return get_current_time()
    else:
        return f"ERROR: Fungsi '{name}' tidak dikenal dalam sandbox."
