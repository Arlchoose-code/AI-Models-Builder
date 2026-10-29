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


if __name__ == "__main__":
    unittest.main()
