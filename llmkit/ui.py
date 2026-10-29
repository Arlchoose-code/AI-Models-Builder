"""Komponen UI, pemformatan teks, warna, tabel, dan input interaktif."""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional, Sequence

# Windows UTF-8 stdout/stderr safety
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
    console = Console()
except ImportError:
    _HAS_RICH = False
    console = None


def format_number(n: float | int) -> str:
    """Format angka besar menjadi string ramah dibaca (k, M, B, T)."""
    abs_n = abs(float(n))
    sign = "-" if n < 0 else ""
    if abs_n >= 1e12:
        return f"{sign}{abs_n / 1e12:.1f}T".replace(".0T", "T")
    if abs_n >= 1e9:
        return f"{sign}{abs_n / 1e9:.1f}B".replace(".0B", "B")
    if abs_n >= 1e6:
        return f"{sign}{abs_n / 1e6:.1f}M".replace(".0M", "M")
    if abs_n >= 1e3:
        return f"{sign}{abs_n / 1e3:.1f}k".replace(".0k", "k")
    if isinstance(n, int) or abs_n == int(abs_n):
        return f"{sign}{int(abs_n)}"
    return f"{sign}{abs_n:.2f}"


def format_exact_number(n: int) -> str:
    """Format angka persis dengan pemisah koma: 598,097,920."""
    return f"{n:,}"


def format_time(seconds: float) -> str:
    """Format waktu: '199 jam (8.3 hari)', '12.5 menit', '45 detik'."""
    if seconds < 0:
        return "0 detik"
    if seconds < 60:
        return f"{seconds:.1f} detik".replace(".0 detik", " detik")
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} menit".replace(".0 menit", " menit")
    hours = minutes / 60.0
    days = hours / 24.0
    if days >= 1.0:
        return f"{hours:.0f} jam ({days:.1f} hari)"
    return f"{hours:.1f} jam".replace(".0 jam", " jam")


def print_ok(msg: str) -> None:
    """Cetak pesan sukses / OK."""
    if _HAS_RICH and console is not None:
        console.print(f"[bold green][OK][/bold green] {msg}")
    else:
        print(f"[OK] {msg}")


def print_warn(msg: str) -> None:
    """Cetak pesan peringatan."""
    if _HAS_RICH and console is not None:
        console.print(f"[bold yellow][WARN][/bold yellow] {msg}")
    else:
        print(f"[WARN] {msg}")


def print_error(msg: str) -> None:
    """Cetak pesan error."""
    if _HAS_RICH and console is not None:
        console.print(f"[bold red][ERROR][/bold red] {msg}")
    else:
        print(f"[ERROR] {msg}")


def print_info(msg: str) -> None:
    """Cetak pesan info."""
    if _HAS_RICH and console is not None:
        console.print(f"[bold cyan][INFO][/bold cyan] {msg}")
    else:
        print(f"[INFO] {msg}")


def print_header(title: str) -> None:
    """Cetak header seksi/panel."""
    line = f"=== {title} ==="
    if _HAS_RICH and console is not None:
        console.print(f"\n[bold magenta]{line}[/bold magenta]")
    else:
        print(f"\n{line}")


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: Optional[str] = None
) -> None:
    """Cetak tabel dengan rich atau fallback teks ASCII."""
    if _HAS_RICH and console is not None:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        for h in headers:
            table.add_column(str(h))
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    else:
        if title:
            print(f"\n--- {title} ---")
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(cell)))
        header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)
        for row in rows:
            print(" | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row)))


def prompt_input(
    question: str,
    default: Optional[str] = None,
    validator: Optional[Callable[[str], tuple[bool, Optional[str], Any]]] = None,
    max_retries: int = 3
) -> Any:
    """
    Tanya pengguna dengan validasi.
    validator mengembalikan (is_valid, error_msg, parsed_value).
    """
    default_str = f" [{default}]" if default is not None else ""
    full_prompt = f"{question}{default_str}: "

    retries = 0
    while retries < max_retries:
        try:
            raw = input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDibatalkan oleh pengguna.")
            sys.exit(1)

        if not raw and default is not None:
            raw = str(default)

        if validator is None:
            return raw

        valid, err_msg, parsed = validator(raw)
        if valid:
            return parsed

        retries += 1
        print_warn(f"{err_msg or 'Input tidak valid.'} (Percobaan {retries}/{max_retries})")

    if default is not None and validator is not None:
        valid, _, parsed = validator(str(default))
        if valid:
            print_info(f"Menggunakan nilai default: {default}")
            return parsed

    print_error("Terlalu banyak kesalahan input.")
    sys.exit(1)


def confirm(question: str, default: bool = False, assume_yes: bool = False) -> bool:
    """Konfirmasi Ya/Tidak (y/N atau Y/n)."""
    if assume_yes:
        return True
    tag = "[Y/n]" if default else "[y/N]"
    full_prompt = f"{question} {tag}: "
    try:
        raw = input(full_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nDibatalkan.")
        sys.exit(1)

    if not raw:
        return default
    return raw in ("y", "ya", "yes", "true", "1")
