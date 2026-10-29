"""Entry point utama untuk LLM Training Toolkit CLI."""

import multiprocessing
import sys

# Konfigurasi encoding UTF-8 untuk stdout dan stderr di Windows
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Pastikan multiprocessing aman pada platform Windows
if sys.platform == "win32":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from llmkit.cli import main

if __name__ == "__main__":
    main()
