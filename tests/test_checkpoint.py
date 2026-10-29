"""Unit tests untuk simpan/muat checkpoint, rotasi, dan determinisme resume (tests/test_checkpoint.py)."""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from llmkit.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from llmkit.model import create_adamw_optimizer, create_model


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ckpt_dir = self.temp_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Model kecil untuk uji cepat
        self.cfg = Qwen3Config(
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

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_rotation_and_completeness(self):
        """Uji rotasi hanya menyimpan 3 step_* terakhir dan mengabaikan folder tanpa COMPLETE."""
        model = create_model(self.cfg, seed=42)
        optimizer = create_adamw_optimizer(model)

        # Simpan step 10, 20, 30, 40
        for s in [10, 20, 30, 40]:
            save_checkpoint(
                self.ckpt_dir, s, model, optimizer,
                train_state={"step": s}, data_state={}, keep_recent=3
            )

        step_folders = sorted([d.name for d in self.ckpt_dir.glob("step_*")])
        # Hanya 3 terakhir yang tersisa (20, 30, 40)
        self.assertEqual(step_folders, ["step_00000020", "step_00000030", "step_00000040"])

        # Uji checkpoint tanpa COMPLETE
        corrupt = self.ckpt_dir / "step_00000050"
        corrupt.mkdir()
        (corrupt / "model.pt").touch()
        # latest checkpoint tidak boleh memilih folder corrupt
        latest = find_latest_checkpoint(self.ckpt_dir)
        self.assertEqual(latest.name, "step_00000040")

    def test_resume_determinism(self):
        """Tes wajib 11.3: Latih 50 step vs simpan di step 20 -> muat -> lanjut 50 step (hasil identik)."""
        torch.manual_seed(1337)
        np.random.seed(1337)

        # 1. Run tanpa interupsi (50 step)
        model_a = create_model(self.cfg, seed=1337)
        optim_a = create_adamw_optimizer(model_a, lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Dummy data tetap
        torch.manual_seed(42)
        fixed_inputs = [torch.randint(0, 100, (2, 32)) for _ in range(50)]
        fixed_targets = [torch.randint(0, 100, (2, 32)) for _ in range(50)]

        for s in range(50):
            x, y = fixed_inputs[s], fixed_targets[s]
            out = model_a(x)
            loss = loss_fn(out.logits.view(-1, 100), y.view(-1))
            loss.backward()
            optim_a.step()
            optim_a.zero_grad()

        # 2. Run dengan interupsi di step 20
        model_b = create_model(self.cfg, seed=1337)
        optim_b = create_adamw_optimizer(model_b, lr=1e-3)

        for s in range(20):
            x, y = fixed_inputs[s], fixed_targets[s]
            out = model_b(x)
            loss = loss_fn(out.logits.view(-1, 100), y.view(-1))
            loss.backward()
            optim_b.step()
            optim_b.zero_grad()

        # Simpan di step 20
        t_state = {"step": 20}
        d_state = {"cursor": 20}
        saved_p = save_checkpoint(self.ckpt_dir, 20, model_b, optim_b, t_state, d_state)

        # Inisialisasi model_c baru dan muat checkpoint step 20
        model_c = create_model(self.cfg, seed=999)
        optim_c = create_adamw_optimizer(model_c, lr=1e-3)
        load_checkpoint(saved_p, model_c, optim_c)

        # Lanjutkan 21..50 pada model_c
        for s in range(20, 50):
            x, y = fixed_inputs[s], fixed_targets[s]
            out = model_c(x)
            loss = loss_fn(out.logits.view(-1, 100), y.view(-1))
            loss.backward()
            optim_c.step()
            optim_c.zero_grad()

        # Verifikasi bobot model_a dan model_c identik (toleransi 1e-5)
        for (n_a, p_a), (n_c, p_c) in zip(model_a.named_parameters(), model_c.named_parameters()):
            diff = (p_a - p_c).abs().max().item()
            self.assertLess(diff, 1e-5, f"Perbedaan bobot ditemukan pada layer {n_a}: {diff}")


if __name__ == "__main__":
    unittest.main()
