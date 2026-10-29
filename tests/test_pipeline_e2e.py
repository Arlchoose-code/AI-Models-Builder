"""End-to-end Integration Test untuk seluruh pipeline LLM Training Toolkit (tests/test_pipeline_e2e.py)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmkit.chat_template import render_conversation
from llmkit.check import check_dataset
from llmkit.data_spec import create_project_data_scaffold
from llmkit.dataset import PretrainDataLoader
from llmkit.export import export_hf_model
from llmkit.model import (
    build_qwen3_config,
    count_parameters,
    create_model,
)
from llmkit.pilot import run_pilot
from llmkit.planner import calculate_params, make_plan
from llmkit.prep import run_prep
from llmkit.sft import run_sft
from llmkit.state import load_project, save_project
from llmkit.tokenizer_train import train_tokenizer
from llmkit.tools_sandbox import execute_tool
from llmkit.train import run_pretrain
from tests.make_dummy_data import create_dummy_pretrain_data, create_dummy_sft_data


class TestPipelineE2E(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_flow(self):
        """Uji alur lengkap dari inisialisasi, prep, training, eval, hingga ekspor."""
        project_dir = self.temp_dir

        # 1. Init
        answers = {
            "name": "test-e2e",
            "size_b": 0.05,
            "token_mode": "balanced",
            "gpu": "RTX 4060 Laptop",
            "goal": "belajar",
            "profile": "laptop",
            "features": {"tool_call": True, "think": True, "vision": False},
            "languages": "en+id",
            "long_context": False,
            "model_name": "BahasaTest",
            "creator": "Tester",
            "runs_dir": "runs"
        }
        plan = make_plan(answers)
        proj_payload = {
            "version": 1,
            "name": "test-e2e",
            "answers": answers,
            "plan": plan,
            "settings": plan["settings"],
            "history": []
        }
        save_project(project_dir, proj_payload)
        create_project_data_scaffold(project_dir, plan, "BahasaTest", "Tester")
        self.assertTrue((project_dir / "project.json").exists())

        # 2. Data Dummy Bersih
        create_dummy_pretrain_data(project_dir / "data", with_errors=False)
        create_dummy_sft_data(project_dir / "data", with_errors=False)

        # 3. Check data
        ok, res = check_dataset(project_dir, quick=False)
        self.assertTrue(ok)
        self.assertEqual(len(res["errors"]), 0)

        # 4. Tokenizer Train
        tok, meta = train_tokenizer(project_dir, vocab_size=1000, max_gb=0.01)
        self.assertTrue((project_dir / "tokenizer" / "tokenizer.json").exists())
        self.assertLessEqual(meta["vocab_size"], 1000)
        self.assertGreater(meta["vocab_size"], 256)

        # 5. Prep Shards
        prep_res = run_prep(project_dir, tokens_per_shard=10_000)
        self.assertTrue((project_dir / "tokenized" / "_manifest.json").exists())
        self.assertGreater(prep_res["total_tokens"], 0)

        # 6. Model Parameters Verification (Qwen3ForCausalLM vs calculate_params)
        p_calc = calculate_params(d=512, L=12, Hq=8, Hkv=2, hd=64, ffn=1408, V=1000)
        cfg = build_qwen3_config({"d": 512, "L": 12, "Hq": 8, "Hkv": 2, "hd": 64, "ffn": 1408, "V": 1000})
        m = create_model(cfg)
        tot_m, non_emb_m = count_parameters(m)
        self.assertEqual(tot_m, p_calc["total"])
        self.assertEqual(non_emb_m, p_calc["non_embedding"])

        # 7. Sandbox Tools Execution
        calc_out = execute_tool("calculate", {"expression": "25 * 4 + 10"})
        self.assertEqual(calc_out, "110")
        search_out = execute_tool("search", {"query": "capital of france"})
        self.assertIn("Paris", search_out)

        # 8. Pretrain Run (10 steps)
        # Override plan hyperparameters untuk tes cepat
        plan["hyperparameters"]["total_steps"] = 10
        plan["tokens"]["target"] = 10000
        plan["hyperparameters"]["micro_batch"] = 2
        plan["hyperparameters"]["grad_accum"] = 1
        plan["hyperparameters"]["context_pretrain"] = 128
        proj_payload["plan"] = plan
        save_project(project_dir, proj_payload)

        train_res = run_pretrain(project_dir, seed=42)
        self.assertEqual(train_res["status"], "finished")
        self.assertEqual(train_res["step"], 10)
        self.assertTrue((project_dir / "runs" / "run_001" / "ckpt" / "step_00000010" / "COMPLETE").exists())

        # 9. SFT Run (1 epoch)
        sft_res = run_sft(project_dir, epochs_override=1, seed=42)
        self.assertEqual(sft_res["status"], "finished")
        self.assertTrue((project_dir / "runs" / "run_002" / "ckpt" / "step_00000001" / "COMPLETE").exists())

        # 10. Export Model
        export_dir = export_hf_model(project_dir, run_id="run_002", out_name="my_export")
        self.assertTrue((export_dir / "model.safetensors").exists())
        self.assertTrue((export_dir / "config.json").exists())
        self.assertTrue((export_dir / "tokenizer_config.json").exists())

        # 11. Muat kembali via AutoModel & AutoTokenizer dari HuggingFace
        loaded_model = AutoModelForCausalLM.from_pretrained(str(export_dir))
        self.assertIsNotNone(loaded_model)
        loaded_tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
        self.assertIsNotNone(loaded_tokenizer)


if __name__ == "__main__":
    unittest.main()
