# LR schedule tests for linear warmup + cosine decay.
# Entrypoints: unittest test methods.
# Verifies _compute_lr at boundary and mid-points.
from __future__ import annotations
import unittest
import math
from hildanext.training import _compute_lr
from reporting import emit_payload


class LRScheduleTests(unittest.TestCase):

    # ---- boundary values ----

    def test_lr_at_step_0_is_zero(self):
        """At step 0, linear warmup → LR = 0."""
        lr = _compute_lr(step=0, warmup_steps=800, total_steps=8000, base_lr=3e-5, min_ratio=0.1)
        emit_payload(
            "test_lr_at_step_0_is_zero",
            "Linear warmup starts at 0.",
            {"lr": lr},
        )
        self.assertAlmostEqual(lr, 0.0, places=10)

    def test_lr_at_warmup_end_equals_base(self):
        """At the end of warmup, LR = base_lr."""
        lr = _compute_lr(step=800, warmup_steps=800, total_steps=8000, base_lr=3e-5, min_ratio=0.1)
        emit_payload(
            "test_lr_at_warmup_end_equals_base",
            "At step=warmup_steps, LR should equal base_lr (start of cosine decay).",
            {"lr": lr, "base_lr": 3e-5},
        )
        # At step=warmup_steps, progress=0, cosine=1 → lr=base_lr*(0.1+0.9*1)=base_lr
        self.assertAlmostEqual(lr, 3e-5, places=10)

    def test_lr_at_total_steps_equals_min(self):
        """At the last step, LR = base_lr * min_ratio."""
        lr = _compute_lr(step=8000, warmup_steps=800, total_steps=8000, base_lr=3e-5, min_ratio=0.1)
        expected = 3e-5 * 0.1  # 3e-6
        emit_payload(
            "test_lr_at_total_steps_equals_min",
            "At step=total_steps, cosine decay reaches min_ratio.",
            {"lr": lr, "expected": expected},
        )
        self.assertAlmostEqual(lr, expected, places=10)

    # ---- warmup linearity ----

    def test_warmup_is_linear(self):
        """During warmup, LR should increase linearly."""
        base_lr = 3e-5
        warmup = 800
        lrs = [_compute_lr(s, warmup, 8000, base_lr, 0.1) for s in range(0, warmup + 1, 100)]
        # Check linearity: lr(step) = base_lr * step / warmup
        results = []
        for i, s in enumerate(range(0, warmup + 1, 100)):
            expected = base_lr * s / warmup
            results.append({"step": s, "lr": lrs[i], "expected": expected})
            self.assertAlmostEqual(lrs[i], expected, places=10, msg=f"step={s}")
        emit_payload(
            "test_warmup_is_linear",
            "LR increases linearly during warmup.",
            {"samples": results},
        )

    def test_warmup_midpoint(self):
        """At half-warmup, LR should be half of base_lr."""
        lr = _compute_lr(step=400, warmup_steps=800, total_steps=8000, base_lr=3e-5, min_ratio=0.1)
        expected = 3e-5 * 400 / 800
        self.assertAlmostEqual(lr, expected, places=10)

    # ---- cosine decay shape ----

    def test_cosine_decay_monotonically_decreasing(self):
        """After warmup, LR must monotonically decrease."""
        lrs = [_compute_lr(s, 800, 8000, 3e-5, 0.1) for s in range(800, 8001, 50)]
        for i in range(1, len(lrs)):
            self.assertLessEqual(lrs[i], lrs[i - 1] + 1e-12,
                                 f"LR not decreasing at step {800 + i * 50}")

    def test_cosine_midpoint(self):
        """At the cosine midpoint, LR should be about (1+min_ratio)/2 * base_lr."""
        mid_step = 800 + (8000 - 800) // 2  # 4400
        lr = _compute_lr(mid_step, 800, 8000, 3e-5, 0.1)
        # progress = 0.5, cos(pi*0.5) = 0 → lr = base*(0.1 + 0.9*0.5) = base*0.55
        expected = 3e-5 * (0.1 + 0.9 * 0.5)
        emit_payload(
            "test_cosine_midpoint",
            "At cosine midpoint, LR ≈ base*(min_ratio + (1-min_ratio)*0.5).",
            {"step": mid_step, "lr": lr, "expected": expected},
        )
        self.assertAlmostEqual(lr, expected, places=9)

    # ---- min_ratio variations ----

    def test_min_ratio_0_reaches_zero(self):
        """With min_ratio=0.0, final LR should be 0."""
        lr = _compute_lr(step=8000, warmup_steps=800, total_steps=8000, base_lr=3e-5, min_ratio=0.0)
        self.assertAlmostEqual(lr, 0.0, places=10)

    def test_min_ratio_1_keeps_constant(self):
        """With min_ratio=1.0, LR stays constant = base_lr after warmup."""
        for s in range(800, 8001, 500):
            lr = _compute_lr(s, 800, 8000, 3e-5, 1.0)
            self.assertAlmostEqual(lr, 3e-5, places=10, msg=f"step={s}")

    # ---- edge cases ----

    def test_warmup_0_starts_at_base(self):
        """With warmup=0, step 0 should already use cosine from base_lr."""
        lr = _compute_lr(step=0, warmup_steps=0, total_steps=100, base_lr=1e-4, min_ratio=0.1)
        # progress=0, cosine=1 → lr = base*(0.1 + 0.9*1) = base
        self.assertAlmostEqual(lr, 1e-4, places=10)

    def test_lr_never_negative(self):
        """LR must never be negative."""
        for s in range(0, 10001, 100):
            lr = _compute_lr(s, 800, 8000, 3e-5, 0.1)
            self.assertGreaterEqual(lr, 0.0, f"negative LR at step {s}")


if __name__ == "__main__":
    unittest.main()
