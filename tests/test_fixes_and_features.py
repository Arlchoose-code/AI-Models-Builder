"""Unit tests untuk fitur baru dan bug fixes (tests/test_fixes_and_features.py)."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import Qwen3Config

from llmkit.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from llmkit.data_spec import make_identity_file, create_project_data_scaffold
from llmkit.evaluate import compare_runs, evaluate_pretrain_model
from llmkit.model import build_qwen3_config, create_adamw_optimizer, create_model
from llmkit.planner import make_plan
from llmkit.sft import run_sft
from llmkit.state import load_project, save_project
from llmkit.train import evaluate_validation, run_pretrain
from llmkit.cli import handle_replan
from llmkit.i18n import get_lang, load_saved_lang, save_lang, set_lang, tr
from tests.make_dummy_data import create_dummy_pretrain_data, create_dummy_sft_data


class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestFixesAndFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.answers = {
            "name": "test-model",
            "size_b": 0.05,
            "token_mode": "balanced",
            "gpu": "RTX 4060 Laptop",
            "goal": "belajar",
            "profile": "laptop",
            "features": {"tool_call": True, "think": True, "vision": False},
            "languages": "en+id",
            "long_context": False,
            "model_name": "TestAI",
            "creator": "AuthorName",
            "runs_dir": "runs"
        }
        self.plan = make_plan(self.answers)
        self.proj_payload = {
            "version": 1,
            "name": "test-model",
            "answers": self.answers,
            "plan": self.plan,
            "settings": self.plan["settings"],
            "history": []
        }
        save_project(self.temp_dir, self.proj_payload)

        # Mock tokenizer untuk tes yang membutuhkannya
        tok_dir = self.temp_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok = Tokenizer(BPE())
        tok.save(str(tok_dir / "tokenizer.json"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_make_identity_file(self):
        """Uji generator identity synthetic (Section 15 Fase 4)."""
        create_project_data_scaffold(self.temp_dir, self.plan, "TestAI", "AuthorName")
        out_path = make_identity_file(self.temp_dir, "TestAI", "AuthorName")
        self.assertTrue(out_path.exists())
        self.assertEqual(out_path.name, "auto.jsonl")

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(lines), 4)
        for sample in lines:
            self.assertIn("messages", sample)
            msgs = sample["messages"]
            self.assertTrue(any("TestAI" in m["content"] for m in msgs))
            self.assertTrue(any("AuthorName" in m["content"] for m in msgs))

    def test_checkpoint_final_safetensors_loading(self):
        """Uji find_latest_checkpoint dan load_checkpoint pada folder final/ dengan model.safetensors."""
        arch = self.plan["architecture"]
        cfg = build_qwen3_config(arch, max_position_embeddings=128)
        model = create_model(cfg, seed=42)
        final_dir = self.temp_dir / "runs" / "run_001" / "ckpt" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        # Simpan melalui save_pretrained untuk menghasilkan model.safetensors
        model.save_pretrained(str(final_dir))
        with open(final_dir / "train_state.json", "w") as f:
            json.dump({"step": 100}, f)

        latest = find_latest_checkpoint(self.temp_dir / "runs" / "run_001" / "ckpt")
        self.assertEqual(latest, final_dir)

        model_loaded = create_model(cfg, seed=123)
        loaded_res = load_checkpoint(latest, model_loaded)
        self.assertEqual(loaded_res["step"], 100)
        # Verifikasi bobot cocok
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_sft_from_missing_checkpoint_error(self):
        """Uji bahwa run_sft menolak jika checkpoint sumber pretrain tidak ditemukan."""
        create_project_data_scaffold(self.temp_dir, self.plan, "TestAI", "AuthorName")
        create_dummy_sft_data(self.temp_dir / "data", with_errors=False)

        # Tidak ada pretrain run_001 sama sekali
        with self.assertRaises(FileNotFoundError) as ctx:
            run_sft(self.temp_dir, from_checkpoint="run_001")
        self.assertIn("SFT membutuhkan checkpoint pretrain selesai", str(ctx.exception))

    def test_evaluate_validation_numpy_memmap(self):
        """Uji bahwa evaluate_validation di train.py memuat shard numpy memmap dengan benar tanpa NameError."""
        val_dir = self.temp_dir / "tokenized" / "_val"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_shard = val_dir / "wiki.bin"
        tokens = np.random.randint(0, 100, size=512, dtype=np.uint16)
        with open(val_shard, "wb") as f:
            f.write(tokens.tobytes())

        cfg = Qwen3Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        model = create_model(cfg, seed=42)
        device = "cpu"
        res = evaluate_validation(model, self.temp_dir, device=device, seq_len=64)
        self.assertIn("val_loss", res)
        self.assertIn("perplexity", res)
        self.assertIn("sources", res)
        self.assertIn("wiki", res["sources"])
        self.assertGreater(res["val_loss"], 0.0)

    def test_evaluate_pretrain_model_and_compare_runs(self):
        """Uji evaluate_pretrain_model dan compare_runs."""
        val_dir = self.temp_dir / "tokenized" / "_val"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_shard = val_dir / "general.bin"
        tokens = np.random.randint(0, 100, size=512, dtype=np.uint16)
        with open(val_shard, "wb") as f:
            f.write(tokens.tobytes())

        arch = self.plan["architecture"]
        cfg = build_qwen3_config(arch, max_position_embeddings=128)
        model = create_model(cfg, seed=42)
        run_dir = self.temp_dir / "runs" / "run_001"
        ckpt_dir = run_dir / "ckpt" / "step_00000010"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(run_dir / "ckpt", 10, model, create_adamw_optimizer(model), {"step": 10}, {})

        # Pretrain eval
        eval_res = evaluate_pretrain_model(self.temp_dir, "run_001")
        self.assertIn("val_loss", eval_res)
        self.assertIn("perplexity", eval_res)
        self.assertEqual(eval_res["kind"], "pretrain")

        # Buat run_002 tiruan untuk uji compare_runs
        run_dir2 = self.temp_dir / "runs" / "run_002"
        run_dir2.mkdir(parents=True, exist_ok=True)
        with open(run_dir2 / "eval.json", "w") as f:
            json.dump(eval_res, f)

        # Compare runs run_001 vs run_002
        compare_runs(self.temp_dir, "run_001", "run_002")

    def test_replan_command(self):
        """Uji handle_replan merencanakan ulang dan mempertahankan override setting."""
        # Beri satu manual override
        proj = load_project(self.temp_dir)
        proj["settings"]["context_final"] = {"value": 8192, "mode": "manual"}
        save_project(self.temp_dir, proj)

        ans_file = self.temp_dir / "new_answers.json"
        new_ans = dict(self.answers)
        new_ans["token_mode"] = "chinchilla"
        new_ans["gpu"] = "A100-80GB"
        with open(ans_file, "w", encoding="utf-8") as f:
            json.dump(new_ans, f)

        args = DummyArgs(
            answers=str(ans_file),
            yes=True,
            project=str(self.temp_dir)
        )
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(self.temp_dir))
            handle_replan(args)
        finally:
            os.chdir(orig_cwd)

        updated_proj = load_project(self.temp_dir)
        self.assertEqual(updated_proj["answers"]["token_mode"], "chinchilla")
        self.assertEqual(updated_proj["answers"]["gpu"], "A100-80GB")
        # Manual override context_final harus tetap dipertahankan
        self.assertEqual(updated_proj["settings"]["context_final"]["value"], 8192)
        self.assertEqual(updated_proj["settings"]["context_final"]["mode"], "manual")

    def test_i18n_translation_and_switching(self):
        """Uji fungsionalitas penerjemahan dan pergantian bahasa runtime."""
        orig_lang = get_lang()
        try:
            set_lang("id")
            self.assertEqual(get_lang(), "id")
            self.assertEqual(tr("halo", "hello"), "halo")
            self.assertEqual(tr("Token: {n}", "Tokens: {n}", n=100), "Token: 100")

            set_lang("en")
            self.assertEqual(get_lang(), "en")
            self.assertEqual(tr("halo", "hello"), "hello")
            self.assertEqual(tr("Token: {n}", "Tokens: {n}", n=100), "Tokens: 100")

            with self.assertRaises(ValueError):
                set_lang("fr")
        finally:
            set_lang(orig_lang)

    def test_i18n_persistence(self):
        """Uji penyimpanan preferensi bahasa ke project.json."""
        orig_lang = get_lang()
        try:
            # Simpan preferensi bahasa en ke project temp
            save_lang("en", persist_global=False, project_dir=self.temp_dir)
            loaded = load_saved_lang(self.temp_dir)
            self.assertEqual(loaded, "en")

            # Verifikasi bahwa file project.json memuat field lang: en
            with open(self.temp_dir / "project.json", "r", encoding="utf-8") as f:
                proj_data = json.load(f)
            self.assertEqual(proj_data.get("lang"), "en")
        finally:
            set_lang(orig_lang)

    def test_detect_stage_status_localization(self):
        """Uji lokalisasi label status panggung (detect_stage_status) dalam bahasa EN dan ID."""
        orig_lang = get_lang()
        try:
            from llmkit.state import detect_stage_status
            set_lang("id")
            res_id = detect_stage_status(self.temp_dir)
            self.assertEqual(res_id["stages"]["init"]["label"], "selesai")
            self.assertEqual(res_id["stages"]["data"]["label"], "belum diisi")

            set_lang("en")
            res_en = detect_stage_status(self.temp_dir)
            self.assertEqual(res_en["stages"]["init"]["label"], "finished")
            self.assertEqual(res_en["stages"]["data"]["label"], "not populated yet")
        finally:
            set_lang(orig_lang)

    def test_init_interactive_language_selection(self):
        """Uji bahwa prompt interaktif handle_init dapat memilih bahasa English dan tersimpan."""
        orig_lang = get_lang()
        try:
            from unittest.mock import patch
            from llmkit.cli import handle_init
            clean_proj = self.temp_dir / "clean_proj"
            clean_proj.mkdir()

            # Input stream: 1 (en), project_name, size, token_mode, gpu, goal, profile, tc, th, vis, id_tok, long_ctx, model_name, creator, runs_dir
            inputs = ["1", "my-en-model", "0.05", "2", "RTX 4060 Laptop", "1", "laptop", "y", "y", "n", "", "n", "EnModel", "Dev", "runs"]
            with patch("builtins.input", side_effect=inputs):
                args = DummyArgs(project=str(clean_proj), yes=False, answers=None, lang=None)
                handle_init(args)

            with open(clean_proj / "project.json", "r", encoding="utf-8") as f:
                proj = json.load(f)
            self.assertEqual(proj.get("lang"), "en")
            self.assertEqual(proj.get("name"), "my-en-model")
        finally:
            set_lang(orig_lang)


if __name__ == "__main__":
    unittest.main()
