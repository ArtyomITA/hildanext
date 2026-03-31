# WSD 8k config generation + training-loop integration tests.
# Entrypoints: unittest test methods.
# Verifies create_stage0_config, _apply_stage0_to_cfg, and logging fields.
from __future__ import annotations
import unittest
import tempfile
import json
import math
from pathlib import Path
from hildanext.config import (
    AppConfig, default_config, load_config, save_config, to_dict,
    TrainConfig, ExperimentConfig, WSDConfig, Stage0Config,
)
from hildanext.wsd_stage0 import create_stage0_config, _apply_stage0_to_cfg
from hildanext.training import _compute_lr, _t_bucket_key, _T_BUCKET_NAMES
from reporting import emit_payload


class WSDConfig8kTests(unittest.TestCase):
    """Tests for the 8k WSD Stage0 configuration."""

    def _make_base_cfg(self) -> AppConfig:
        return default_config(Path("E:/DIFFUSION/HildaNext/hildanext"))

    # ---- create_stage0_config ----

    def test_create_stage0_config_steps(self):
        """create_stage0_config must produce 8000 total steps."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="E:/DIFFUSION/HildaNext/dolma_v1_6_sample_1767050862")
        self.assertEqual(out.stage0.steps_total_stage0, 8000)
        emit_payload(
            "test_create_stage0_config_steps",
            "create_stage0_config must set steps_total_stage0=8000.",
            {"steps": out.stage0.steps_total_stage0},
        )

    def test_create_stage0_config_seq_len(self):
        """seq_len must be 1024."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        self.assertEqual(out.data.seq_len, 1024)
        self.assertEqual(out.stage0.seq_len, 1024)

    def test_create_stage0_config_experiment_flags(self):
        """Experiment flags must match S0-A/C/D spec."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        exp = out.experiment
        self.assertEqual(exp.attention_mode, "bidirectional_only_stable")
        self.assertEqual(exp.time_param, "continuous_time")
        self.assertEqual(exp.loss_weighting, "inv_t")
        self.assertEqual(exp.shift_mode, "preserve_left_shift")
        self.assertAlmostEqual(exp.t_min, 0.001)
        self.assertAlmostEqual(exp.t_max, 1.0)
        self.assertEqual(exp.experiment_id, "s0_ct_bidir_8k")

    def test_create_stage0_config_wsd_fractions(self):
        """WSD fractions: warmup=10%, stable=70%, decay=20%."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        self.assertAlmostEqual(out.stage0.warmup_frac, 0.10)
        self.assertAlmostEqual(out.stage0.stable_frac, 0.70)
        self.assertAlmostEqual(out.stage0.decay_frac, 0.20)

    def test_create_stage0_config_grad_accum(self):
        """grad_accum=16, batch=1."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        self.assertEqual(out.train.batch_size, 1)
        self.assertEqual(out.train.accum_steps, 16)
        self.assertEqual(out.stage0.grad_accum_steps, 16)

    def test_create_stage0_config_saves_loadable_json(self):
        """Written JSON must be loadable via load_config."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            create_stage0_config(cfg, out_path, dolma_path="dummy")
            loaded = load_config(out_path)
        self.assertEqual(loaded.stage0.steps_total_stage0, 8000)
        self.assertEqual(loaded.experiment.attention_mode, "bidirectional_only_stable")

    # ---- _apply_stage0_to_cfg ----

    def test_apply_stage0_wsd_steps(self):
        """_apply_stage0_to_cfg computes warmup/stable/decay from fracs and total."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            cfg = create_stage0_config(cfg, out_path, dolma_path="dummy")
        run_cfg = _apply_stage0_to_cfg(cfg, run_id="test_run")
        w = run_cfg.wsd.warmup_steps
        s = run_cfg.wsd.stable_steps
        d = run_cfg.wsd.decay_steps
        total = w + s + d
        emit_payload(
            "test_apply_stage0_wsd_steps",
            "Warmup+stable+decay must sum to steps_total_stage0.",
            {"warmup": w, "stable": s, "decay": d, "total": total, "expected": 8000},
        )
        self.assertEqual(total, 8000)
        self.assertEqual(w, 800)   # 10% of 8000
        self.assertEqual(s, 5600)  # 70% of 8000
        self.assertEqual(d, 1600)  # 20% of 8000

    def test_apply_stage0_grad_clip(self):
        """_apply_stage0_to_cfg must set grad_clip=1.0."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            cfg = create_stage0_config(cfg, out_path, dolma_path="dummy")
        run_cfg = _apply_stage0_to_cfg(cfg)
        self.assertAlmostEqual(run_cfg.train.grad_clip, 1.0)

    def test_apply_stage0_lr_min_ratio(self):
        """_apply_stage0_to_cfg must set lr_min_ratio=0.1."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            cfg = create_stage0_config(cfg, out_path, dolma_path="dummy")
        run_cfg = _apply_stage0_to_cfg(cfg)
        self.assertAlmostEqual(run_cfg.train.lr_min_ratio, 0.1)

    def test_apply_stage0_seq_len_propagated(self):
        """seq_len from stage0 must propagate to data.seq_len."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            cfg = create_stage0_config(cfg, out_path, dolma_path="dummy")
        run_cfg = _apply_stage0_to_cfg(cfg)
        self.assertEqual(run_cfg.data.seq_len, 1024)

    def test_apply_stage0_ladder_blocks(self):
        """Ladder blocks must include seq_len as max element."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            cfg = create_stage0_config(cfg, out_path, dolma_path="dummy")
        run_cfg = _apply_stage0_to_cfg(cfg)
        self.assertIn(1024, run_cfg.wsd.ladder_blocks)
        self.assertEqual(run_cfg.wsd.ladder_blocks[0], 1)  # starts at 1
        self.assertEqual(run_cfg.wsd.ladder_blocks[-1], 1024)  # ends at seq_len

    # ---- TrainConfig new fields ----

    def test_train_config_has_grad_clip(self):
        """TrainConfig dataclass must have grad_clip field."""
        tc = TrainConfig()
        self.assertTrue(hasattr(tc, "grad_clip"))
        self.assertAlmostEqual(tc.grad_clip, 1.0)

    def test_train_config_has_lr_min_ratio(self):
        """TrainConfig dataclass must have lr_min_ratio field."""
        tc = TrainConfig()
        self.assertTrue(hasattr(tc, "lr_min_ratio"))
        self.assertAlmostEqual(tc.lr_min_ratio, 0.1)

    # ---- t-bucket helpers ----

    def test_t_bucket_key_ranges(self):
        """_t_bucket_key must partition [0,1] into 4 ranges."""
        self.assertEqual(_t_bucket_key(0.05), "0.0-0.1")
        self.assertEqual(_t_bucket_key(0.1), "0.1-0.3")
        self.assertEqual(_t_bucket_key(0.2), "0.1-0.3")
        self.assertEqual(_t_bucket_key(0.3), "0.3-0.6")
        self.assertEqual(_t_bucket_key(0.5), "0.3-0.6")
        self.assertEqual(_t_bucket_key(0.6), "0.6-1.0")
        self.assertEqual(_t_bucket_key(0.9), "0.6-1.0")

    def test_t_bucket_names_consistent(self):
        """_T_BUCKET_NAMES must list all 4 bucket keys."""
        self.assertEqual(len(_T_BUCKET_NAMES), 4)
        self.assertIn("0.0-0.1", _T_BUCKET_NAMES)
        self.assertIn("0.6-1.0", _T_BUCKET_NAMES)

    # ---- cooldown & eval settings ----

    def test_cooldown_settings(self):
        """Cooldown must be every 100 steps, 120 seconds."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        self.assertEqual(out.train.cooldown_every_steps, 100)
        self.assertEqual(out.train.cooldown_seconds, 120)

    def test_eval_and_save_every(self):
        """eval_every and save_every must be 500."""
        cfg = self._make_base_cfg()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_cfg.json"
            out = create_stage0_config(cfg, out_path, dolma_path="dummy")
        self.assertEqual(out.stage0.eval_every_steps, 500)
        self.assertEqual(out.stage0.save_every_steps, 500)

    # ---- roundtrip JSON serialization ----

    def test_config_roundtrip_preserves_experiment(self):
        """Save → load must preserve all experiment flags."""
        cfg = self._make_base_cfg()
        cfg.experiment.attention_mode = "bidirectional_only_stable"
        cfg.experiment.time_param = "continuous_time"
        cfg.experiment.loss_weighting = "inv_t"
        cfg.experiment.t_min = 0.001
        cfg.experiment.t_max = 1.0
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.json"
            save_config(cfg, p)
            loaded = load_config(p)
        self.assertEqual(loaded.experiment.attention_mode, "bidirectional_only_stable")
        self.assertEqual(loaded.experiment.time_param, "continuous_time")
        self.assertEqual(loaded.experiment.loss_weighting, "inv_t")
        self.assertAlmostEqual(loaded.experiment.t_min, 0.001)
        self.assertAlmostEqual(loaded.experiment.t_max, 1.0)
        emit_payload(
            "test_config_roundtrip_preserves_experiment",
            "JSON roundtrip must preserve all ExperimentConfig fields.",
            {"ok": True},
        )


if __name__ == "__main__":
    unittest.main()
