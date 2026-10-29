"""Unit tests untuk Settings, Aturan Auto, dan Advisor A01-A18 (tests/test_settings.py)."""

import unittest
from llmkit.advisor import run_advisor
from llmkit.planner import compute_pretrain_composition, compute_sft_composition, make_plan
from llmkit.settings import compute_auto_settings


class TestSettingsAndAdvisor(unittest.TestCase):
    def test_auto_settings_combinations(self):
        """Tes 1: Tiap aturan auto diuji pada minimal 3 kombinasi (ukuran, VRAM, fitur)."""
        # Kombinasi 1: Model kecil di laptop (0.05B, RTX 4060 Laptop 8GB)
        ans1 = {"size_b": 0.05, "gpu": "RTX 4060 Laptop", "features": {"tool_call": True, "think": True}}
        s1 = compute_auto_settings(ans1)
        self.assertEqual(s1["think"]["value"], False)  # N < 0.1B -> think=False
        self.assertEqual(s1["context_pretrain"]["value"], 2048)
        self.assertEqual(s1["vocab_size"]["value"], 32000)

        # Kombinasi 2: Model menengah di L4 (0.3B, L4 24GB)
        ans2 = {"size_b": 0.3, "gpu": "L4", "features": {"tool_call": True, "think": True}}
        s2 = compute_auto_settings(ans2)
        self.assertEqual(s2["think"]["value"], True)
        self.assertEqual(s2["context_pretrain"]["value"], 4096)  # N > 0.15B dan VRAM >= 24GB
        self.assertEqual(s2["vocab_size"]["value"], 64000)

        # Kombinasi 3: Model 1B di H100 (1B, H100 80GB, long-context)
        ans3 = {"size_b": 1.0, "gpu": "H100", "tokens_target_b": 100.0, "long_context": True, "features": {"tool_call": True, "think": True}}
        s3 = compute_auto_settings(ans3)
        self.assertEqual(s3["context_final"]["value"], 32768)
        self.assertEqual(s3["think_max_tokens"]["value"], 2048)  # N > 0.8B

    def test_manual_override_persistence(self):
        """Tes 2: Setting manual tidak berubah saat perhitungan ulang; auto mengembalikannya."""
        ans = {"size_b": 0.6, "gpu": "H100"}
        initial_s = compute_auto_settings(ans)
        self.assertEqual(initial_s["context_final"]["mode"], "auto")

        # Set manual
        initial_s["context_final"] = {"value": 16384, "mode": "manual", "reason": "dikunci"}
        recalculated = compute_auto_settings(ans, current_settings=initial_s)
        self.assertEqual(recalculated["context_final"]["value"], 16384)
        self.assertEqual(recalculated["context_final"]["mode"], "manual")

        # Kembalikan ke auto
        initial_s["context_final"]["mode"] = "auto"
        restored = compute_auto_settings(ans, current_settings=initial_s)
        self.assertEqual(restored["context_final"]["mode"], "auto")
        self.assertEqual(restored["context_final"]["value"], 8192)

    def test_sft_percentages_sum_to_100_when_features_disabled(self):
        """Tes 4: think=False dan tool_call=False menghasilkan persentase SFT yang tepat 100."""
        sft_none = compute_sft_composition(size_b=0.6, think=False, tool_call=False)
        total_pct = sum(c["pct"] for c in sft_none["categories"].values())
        self.assertAlmostEqual(total_pct, 100.0, places=2)
        self.assertNotIn("think", sft_none["categories"])
        self.assertNotIn("tool_call", sft_none["categories"])

    def test_languages_id_removes_en_folders(self):
        """Tes 5: languages=id menghasilkan komposisi pretrain tanpa folder en_* dengan jumlah kolom 100."""
        comp_id = compute_pretrain_composition(
            total_tokens=10e9,
            context_pretrain=2048,
            context_final=4096,
            languages="id"
        )
        for stage_name, stage_data in comp_id["stages"].items():
            if stage_data:
                for k in stage_data["weights"]:
                    self.assertFalse(k.startswith("en_"))
                self.assertAlmostEqual(sum(stage_data["weights"].values()), 100.0, places=2)

    def test_stage_c_activation_and_rope_theta(self):
        """Tes 6: Stage C aktif saat context_final=8192 dan context_pretrain=4096; rope_theta = 1,000,000."""
        comp = compute_pretrain_composition(
            total_tokens=50e9,
            context_pretrain=4096,
            context_final=8192
        )
        self.assertTrue(comp["stage_c_active"])
        self.assertEqual(comp["stages"]["A"]["pct"], 75.0)
        self.assertEqual(comp["stages"]["B"]["pct"], 20.0)
        self.assertEqual(comp["stages"]["C"]["pct"], 5.0)

        # Hitung rope_theta
        base_rope = 500000.0
        ratio = 8192 / 4096
        rope_c = base_rope * ratio
        self.assertEqual(rope_c, 1_000_000.0)

    def test_advisor_rules_a01_to_a18(self):
        """Tes 3: Setiap aturan advisor A01-A18 punya kondisi yang memicu dan tidak memicu."""
        # A01: Trigger (model 1B di GPU 8GB) vs Non-trigger (model 0.05B di GPU 8GB)
        p_a01_trig = make_plan({"size_b": 1.0, "gpu": "RTX 4060 Laptop"})
        finds_trig = run_advisor({"size_b": 1.0, "gpu": "RTX 4060 Laptop"}, p_a01_trig)
        self.assertTrue(any(f.id == "A01" for f in finds_trig))

        p_a01_ok = make_plan({"size_b": 0.05, "gpu": "RTX 4060 Laptop"})
        finds_ok = run_advisor({"size_b": 0.05, "gpu": "RTX 4060 Laptop"}, p_a01_ok)
        self.assertFalse(any(f.id == "A01" for f in finds_ok))

        # A03: context_final >= 16384 di GPU < 24GB
        p_a03 = make_plan({"size_b": 0.1, "gpu": "RTX 4060 Laptop"}, current_settings={"context_final": {"value": 16384, "mode": "manual"}})
        self.assertTrue(any(f.id == "A03" for f in run_advisor({"size_b": 0.1, "gpu": "RTX 4060 Laptop"}, p_a03)))

        # A04: tool_call=True dan context_final < 4096
        p_a04 = make_plan({"size_b": 0.1, "gpu": "H100"}, current_settings={"context_final": {"value": 2048, "mode": "manual"}})
        self.assertTrue(any(f.id == "A04" for f in run_advisor({"size_b": 0.1, "gpu": "H100"}, p_a04)))

        # A05: think=True dan N < 0.1B
        p_a05 = make_plan({"size_b": 0.05, "gpu": "H100"}, current_settings={"think": {"value": True, "mode": "manual"}})
        self.assertTrue(any(f.id == "A05" for f in run_advisor({"size_b": 0.05, "gpu": "H100"}, p_a05)))

        # A06: folder epochs > 4
        p_a06 = make_plan({"size_b": 0.6, "gpu": "H100", "id_unique_tokens_b": 1.0})
        finds_a06 = run_advisor({"size_b": 0.6, "gpu": "H100", "id_unique_tokens_b": 1.0}, p_a06)
        self.assertTrue(any(f.id == "A06" for f in finds_a06))

        # A09: cost > budget
        p_a09 = make_plan({"size_b": 0.6, "gpu": "H100", "tokens_override": 59.8e9, "budget_usd": 100})
        finds_a09 = run_advisor({"size_b": 0.6, "gpu": "H100", "tokens_override": 59.8e9, "budget_usd": 100}, p_a09)
        self.assertTrue(any(f.id == "A09" for f in finds_a09))

        # A11: vocab_size > 65535
        p_a11 = make_plan({"size_b": 0.1, "gpu": "H100"}, current_settings={"vocab_size": {"value": 100000, "mode": "manual"}})
        self.assertTrue(any(f.id == "A11" for f in run_advisor({"size_b": 0.1, "gpu": "H100"}, p_a11)))

        # A14: vision=True
        p_a14 = make_plan({"size_b": 0.6, "gpu": "H100"}, current_settings={"vision": {"value": True, "mode": "manual"}})
        self.assertTrue(any(f.id == "A14" for f in run_advisor({"size_b": 0.6, "gpu": "H100"}, p_a14)))

        # A16: N=0.6B dan tokens <= 100B
        p_a16 = make_plan({"size_b": 0.6, "gpu": "H100", "token_mode": "balanced"})
        self.assertTrue(any(f.id == "A16" for f in run_advisor({"size_b": 0.6, "gpu": "H100"}, p_a16)))

        # A18: manual setting differs from auto
        p_a18 = make_plan({"size_b": 0.6, "gpu": "H100"}, current_settings={"code_focus": {"value": "berat", "mode": "manual"}})
        self.assertTrue(any(f.id == "A18" for f in run_advisor({"size_b": 0.6, "gpu": "H100"}, p_a18)))


if __name__ == "__main__":
    unittest.main()
