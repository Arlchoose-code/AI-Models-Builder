"""Sistem Internasionalisasi (i18n) dwibahasa (EN / ID) untuk LLMKit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

LANG_EN = "en"
LANG_ID = "id"
SUPPORTED_LANGS = [LANG_EN, LANG_ID]

_CURRENT_LANG: str = LANG_ID


def get_config_dir() -> Path:
    """Direktori konfigurasi global pengguna."""
    cfg_dir = Path.home() / ".llmkit"
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cfg_dir


def get_global_config_path() -> Path:
    """Path file konfigurasi global."""
    return get_config_dir() / "config.json"


def load_saved_lang(project_dir: Optional[Path] = None) -> Optional[str]:
    """
    Muat bahasa tersimpan dari environment, project.json, atau global config.
    """
    # 1. Environment variable
    env_lang = os.environ.get("LLMKIT_LANG", "").strip().lower()
    if env_lang in SUPPORTED_LANGS:
        return env_lang

    # 2. project.json jika ada
    if project_dir is not None:
        p_json = Path(project_dir) / "project.json"
        if p_json.exists():
            try:
                with open(p_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val = str(data.get("lang", "")).strip().lower()
                    if val in SUPPORTED_LANGS:
                        return val
            except Exception:
                pass

    # 3. Global config
    cfg_path = get_global_config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                val = str(data.get("lang", "")).strip().lower()
                if val in SUPPORTED_LANGS:
                    return val
        except Exception:
            pass

    return None


def save_lang(lang: str, persist_global: bool = True, project_dir: Optional[Path] = None) -> None:
    """Simpan preferensi bahasa ke global config dan/atau project.json."""
    if lang not in SUPPORTED_LANGS:
        return

    if persist_global:
        cfg_path = get_global_config_path()
        cfg_data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                cfg_data = {}
        cfg_data["lang"] = lang
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
        except Exception:
            pass

    if project_dir is not None:
        p_json = Path(project_dir) / "project.json"
        if p_json.exists():
            try:
                with open(p_json, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                p_data["lang"] = lang
                with open(p_json, "w", encoding="utf-8") as f:
                    json.dump(p_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass


def get_lang() -> str:
    """Dapatkan kode bahasa aktif saat ini ('en' atau 'id')."""
    return _CURRENT_LANG


def set_lang(lang: str, persist_global: bool = False, project_dir: Optional[Path] = None) -> None:
    """Set bahasa aktif saat ini."""
    global _CURRENT_LANG
    lang = lang.strip().lower()
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Bahasa '{lang}' tidak didukung. Pilih salah satu dari: {SUPPORTED_LANGS}")
    _CURRENT_LANG = lang
    if persist_global or project_dir is not None:
        save_lang(lang, persist_global=persist_global, project_dir=project_dir)


def tr(id_str: str, en_str: str, lang: Optional[str] = None, **kwargs) -> str:
    """
    Pilih teks bahasa Indonesia atau Inggris berdasarkan bahasa aktif.
    """
    active = (lang or _CURRENT_LANG).strip().lower()
    text = en_str if active == LANG_EN else id_str
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def prompt_language_selection() -> str:
    """Tampilkan pilihan bahasa di terminal secara interaktif pada awal eksekusi."""
    print("\n" + "=" * 55)
    print("Select Language / Pilih Bahasa:")
    print("  1. English (en)")
    print("  2. Bahasa Indonesia (id)")
    print("=" * 55)
    try:
        choice = input("Choice / Pilihan [1/2] (default: 1): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return LANG_EN

    if choice in ("2", "id", "indonesia", "indo"):
        return LANG_ID
    return LANG_EN
