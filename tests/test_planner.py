"""Unit tests untuk kalkulator planner, formula parameter, solver, dan komposisi data (tests/test_planner.py)."""

import math
import unittest
from llmkit.gpu_table import get_gpu_spec
from llmkit.planner import (
    PRESETS,
    calculate_params,
    calculate_tokens,
    compute_flops_and_cost,
    compute_pretrain_composition,
    get_batch_tokens,
    make_plan,
    solve_architecture,
)


class TestPlanner(unittest.TestCase):
    def test_qwen3_exact_test_vectors(self):
        """Tes 1: Uji rumus parameter Qwen3-0.6B resmi dan semua preset Tabel 6.2."""
        # Qwen3-0.6B resmi: d=1024, L=28, Hq=16, Hkv=8, hd=128, ffn=3072, V=151936
        q_res = calculate_params(d=1024, L=28, Hq=16, Hkv=8, hd=128, ffn=3072, V=151936)
        self.assertEqual(q_res["total"], 596_049_920)
        self.assertEqual(q_res["non_embedding"], 440_467_456)

        # Preset 0.05B
        p005 = calculate_params(d=512, L=12, Hq=8, Hkv=2, hd=64, ffn=1408, V=32000)
        self.assertEqual(p005["total"], 50_214_912)

        # Preset 0.1B
        p01 = calculate_params(d=640, L=18, Hq=10, Hkv=2, hd=64, ffn=1728, V=32000)
        self.assertEqual(p01["total"], 97_920_384)

        # Preset 0.3B
        p03 = calculate_params(d=1024, L=20, Hq=16, Hkv=4, hd=64, ffn=2816, V=64000)
        self.assertEqual(p03["total"], 291_024_384)

        # Preset 0.6B
        p06 = calculate_params(d=1280, L=30, Hq=20, Hkv=4, hd=64, ffn=3456, V=64000)
        self.assertEqual(p06["total"], 598_097_920)
        self.assertEqual(p06["non_embedding"], 516_177_920)

        # Preset 1B
        p1 = calculate_params(d=1792, L=28, Hq=28, Hkv=4, hd=64, ffn=4864, V=64000)
        self.assertEqual(p1["total"], 1_052_482_816)

    def test_plan_06b_balanced_h100(self):
        """Tes 2: plan 0.6B --mode balanced --gpu H100."""
        answers = {
            "size_b": 0.6,
            "token_mode": "balanced",
            "gpu": "H100",
            "features": {"tool_call": True, "think": True, "vision": False},
            "languages": "en+id",
            "long_context": False,
        }
        plan = make_plan(answers)

        # Token 59.8B
        tokens = plan["tokens"]["target"]
        self.assertEqual(tokens, 59.8e9)

        # FLOPs ≈ 2.146e20
        flops = plan["compute"]["flops"]
        self.assertAlmostEqual(flops / 1e20, 2.146, places=2)

        # Waktu ≈ 199 jam
        hours = plan["compute"]["hours"]
        self.assertAlmostEqual(hours, 199, delta=2)

        # Peak LR ≈ 5.41e-4
        lr = plan["hyperparameters"]["peak_lr"]
        self.assertAlmostEqual(lr * 1e4, 5.41, delta=0.1)

        # Batch tokens = 1,048,576
        batch_tokens = plan["hyperparameters"]["batch_tokens"]
        self.assertEqual(batch_tokens, 1_048_576)

        # Total steps = 57,030
        steps = plan["hyperparameters"]["total_steps"]
        self.assertEqual(steps, 57_030)

    def test_plan_05b_solver(self):
        """Tes 3: plan 0.5B dengan solver."""
        solved = solve_architecture(target_params=int(0.5e9), vocab_size=64000)
        self.assertTrue(solved["is_within_3pct"])
        self.assertLessEqual(solved["best"]["rel_error"], 0.03)
        self.assertEqual(len(solved["top3"]), 3)

    def test_plan_budget_mode(self):
        """Tes 4: plan 0.6B --budget-usd 500 --gpu H100 memberi token ≈ 60B."""
        spec_h100 = get_gpu_spec("H100")
        self.assertIsNotNone(spec_h100)
        params = 598_097_920
        tokens, label, ratio = calculate_tokens(
            total_params=params,
            budget_usd=500.0,
            gpu_spec=spec_h100
        )
        self.assertAlmostEqual(tokens / 1e9, 60.0, delta=2.0)

    def test_composition_weights_and_epoch_error(self):
        """Tes 5: Komposisi A/B jumlah kolom = 100, skenario id_unique_tokens_b=2 dan max memicu epochs > 8."""
        comp = compute_pretrain_composition(
            total_tokens=59.8e9,
            context_pretrain=2048,
            context_final=4096,
            languages="en+id"
        )
        # Tiap kolom harus 100
        sum_A = sum(comp["stages"]["A"]["weights"].values())
        sum_B = sum(comp["stages"]["B"]["weights"].values())
        self.assertAlmostEqual(sum_A, 100.0, places=2)
        self.assertAlmostEqual(sum_B, 100.0, places=2)

        # Skenario id_unique_tokens_b=2 dan max (3000x params)
        max_tokens, _, _ = calculate_tokens(598_097_920, mode="max")
        comp_max = compute_pretrain_composition(
            total_tokens=max_tokens,
            context_pretrain=2048,
            context_final=4096,
            languages="en+id"
        )
        needed_id = comp_max["needed"]["id_web"]
        available_id = 2.0 * 1e9
        epochs = needed_id / available_id
        self.assertGreater(epochs, 8.0)


if __name__ == "__main__":
    unittest.main()
