"""Unit tests untuk deteksi status disk, heartbeat basi, dan validator check (tests/test_status.py)."""

import datetime
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from llmkit.check import check_dataset
from llmkit.data_spec import create_project_data_scaffold
from llmkit.planner import make_plan
from llmkit.state import (
    atomic_write_json,
    detect_stage_status,
    save_project,
)
from tests.make_dummy_data import create_dummy_pretrain_data, create_dummy_sft_data


class TestStatusAndCheck(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_status_empty_project(self):
        """Uji status pada folder kosong (belum ada project.json)."""
        res = detect_stage_status(self.temp_dir)
        self.assertEqual(res["next_step"], "init")
        self.assertEqual(res["next_command"], "python main.py init")

    def test_status_with_initialized_project(self):
        """Uji status pada proyek yang baru diinisialisasi."""
        answers = {"size_b": 0.05, "gpu": "RTX 4060 Laptop", "model_name": "Bahasa", "creator": "Pengguna"}
        plan = make_plan(answers)
        proj_data = {"version": 1, "name": "test-model", "answers": answers, "plan": plan, "settings": plan["settings"]}
        save_project(self.temp_dir, proj_data)
        create_project_data_scaffold(self.temp_dir, plan, "Bahasa", "Pengguna")

        res = detect_stage_status(self.temp_dir)
        self.assertEqual(res["stages"]["init"]["status"], "[x]")
        self.assertEqual(res["stages"]["data"]["status"], "[ ]")
        self.assertEqual(res["next_step"], "data")

    def test_status_stale_heartbeat_detection(self):
        """Tes 7: Deteksi run pretrain berhenti tiba-tiba jika heartbeat > 180 detik atau PID mati."""
        answers = {"size_b": 0.05, "gpu": "RTX 4060 Laptop", "runs_dir": "runs"}
        plan = make_plan(answers)
        save_project(self.temp_dir, {"version": 1, "answers": answers, "plan": plan, "settings": plan["settings"]})

        # Simulasi run.json dengan status running tetapi heartbeat 300 detik yang lalu
        run_dir = self.temp_dir / "runs" / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.datetime.now() - datetime.timedelta(seconds=300)).isoformat()
        run_data = {
            "kind": "pretrain",
            "status": "running",
            "pid": 99999999,  # PID yang dipastikan tidak hidup
            "heartbeat": old_time,
            "step": 500,
            "tokens_seen": 1000000,
            "finished": False
        }
        atomic_write_json(run_dir / "run.json", run_data)

        res = detect_stage_status(self.temp_dir)
        self.assertEqual(res["stages"]["pretrain"]["status"], "[!]")
        self.assertEqual(res["next_command"], "python main.py pretrain --resume")
        self.assertIn("Training berhenti tiba-tiba", res["warning"])

    def test_check_dataset_with_corrupt_data(self):
        """Tes 8: check mendeteksi baris rusak, text kosong, dan format salah pada data dummy."""
        answers = {"size_b": 0.05, "gpu": "RTX 4060 Laptop"}
        plan = make_plan(answers)
        save_project(self.temp_dir, {"version": 1, "answers": answers, "plan": plan, "settings": plan["settings"]})

        # Buat data dengan baris rusak
        create_dummy_pretrain_data(self.temp_dir / "data", with_errors=True)
        create_dummy_sft_data(self.temp_dir / "data", with_errors=True)

        ok, summary = check_dataset(self.temp_dir, quick=False, fix_hints=True)
        self.assertFalse(ok)
        self.assertGreater(len(summary["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
