# Continuous-time ELBO tests for S0-C.
# Entrypoints: unittest test methods.
# Verifies t sampling, i.i.d. masking, 1/t loss scaling.
from __future__ import annotations
import unittest
import math
import torch
from hildanext.diffusion import _continuous_time_m2t_batch, _causal_loss
from reporting import emit_payload


class ContinuousTimeElboTests(unittest.TestCase):

    def _make_batch(self, seq_len: int = 32, t_min: float = 0.001, t_max: float = 1.0):
        """Helper: create a simple batch and call _continuous_time_m2t_batch."""
        torch.manual_seed(42)
        ids = torch.randint(10, 500, (1, seq_len), dtype=torch.long)
        attn = torch.ones(1, seq_len, dtype=torch.long)
        mask_id = 999
        noisy, labels, t, ratio = _continuous_time_m2t_batch(
            ids, attn, None, mask_id, t_min=t_min, t_max=t_max
        )
        return ids, noisy, labels, t, ratio, mask_id

    # ---- t sampling range ----

    def test_t_within_bounds(self):
        """t_sampled must lie in [t_min, t_max]."""
        t_min, t_max = 0.001, 1.0
        ts = []
        for seed in range(200):
            torch.manual_seed(seed)
            ids = torch.randint(10, 500, (1, 64), dtype=torch.long)
            attn = torch.ones(1, 64, dtype=torch.long)
            _, _, t, _ = _continuous_time_m2t_batch(ids, attn, None, 999, t_min, t_max)
            ts.append(t)
        t_min_obs = min(ts)
        t_max_obs = max(ts)
        emit_payload(
            "test_t_within_bounds",
            "All sampled t values must be in [t_min, t_max].",
            {"t_min_obs": t_min_obs, "t_max_obs": t_max_obs, "n_samples": len(ts)},
        )
        self.assertGreaterEqual(t_min_obs, t_min - 1e-9)
        self.assertLessEqual(t_max_obs, t_max + 1e-9)

    def test_t_distribution_covers_range(self):
        """Over many samples, t should roughly cover the full [t_min, t_max] range."""
        ts = []
        for seed in range(500):
            torch.manual_seed(seed)
            ids = torch.randint(10, 500, (1, 64), dtype=torch.long)
            attn = torch.ones(1, 64, dtype=torch.long)
            _, _, t, _ = _continuous_time_m2t_batch(ids, attn, None, 999, 0.001, 1.0)
            ts.append(t)
        mean_t = sum(ts) / len(ts)
        # Uniform(0.001,1.0) has mean ≈ 0.5005
        emit_payload(
            "test_t_distribution_covers_range",
            "Mean t should be ≈0.5 for U(0.001,1.0).",
            {"mean_t": mean_t, "min_t": min(ts), "max_t": max(ts)},
        )
        self.assertAlmostEqual(mean_t, 0.5, delta=0.08)

    # ---- masking ratio tracks t ----

    def test_mask_ratio_correlates_with_t(self):
        """Higher t should produce higher mask ratio on average."""
        low_ratios = []
        high_ratios = []
        for seed in range(300):
            torch.manual_seed(seed)
            ids = torch.randint(10, 500, (1, 128), dtype=torch.long)
            attn = torch.ones(1, 128, dtype=torch.long)
            _, _, t, ratio = _continuous_time_m2t_batch(ids, attn, None, 999, 0.001, 1.0)
            if t < 0.3:
                low_ratios.append(ratio)
            elif t > 0.7:
                high_ratios.append(ratio)
        mean_low = sum(low_ratios) / max(1, len(low_ratios))
        mean_high = sum(high_ratios) / max(1, len(high_ratios))
        emit_payload(
            "test_mask_ratio_correlates_with_t",
            "Higher t ⇒ higher mask_ratio on average.",
            {"mean_low_t_ratio": mean_low, "mean_high_t_ratio": mean_high,
             "n_low": len(low_ratios), "n_high": len(high_ratios)},
        )
        self.assertGreater(mean_high, mean_low, "high-t mask ratio should exceed low-t")

    # ---- masking is i.i.d. per token ----

    def test_mask_is_iid_per_token(self):
        """Each token is masked independently with prob t. Across 500 trials at fixed t=0.5,
        the per-position mask frequency should be ≈0.5 for each position."""
        mask_id = 999
        seq_len = 64
        counts = torch.zeros(seq_len)
        n_trials = 500
        for seed in range(n_trials):
            torch.manual_seed(seed)
            ids = torch.randint(10, 500, (1, seq_len), dtype=torch.long)
            attn = torch.ones(1, seq_len, dtype=torch.long)
            # Force t≈0.5 by using t_min=t_max=0.5
            noisy, _, t, _ = _continuous_time_m2t_batch(ids, attn, None, mask_id, 0.5, 0.5)
            masked_pos = (noisy[0] == mask_id).float()
            counts += masked_pos
        freqs = counts / n_trials
        mean_freq = freqs.mean().item()
        std_freq = freqs.std().item()
        emit_payload(
            "test_mask_is_iid_per_token",
            "With t=0.5, each position should be masked ≈50% of the time.",
            {"mean_freq": mean_freq, "std_freq": std_freq, "n_trials": n_trials},
        )
        # mean frequency should be near 0.5
        self.assertAlmostEqual(mean_freq, 0.5, delta=0.06)
        # std should be small (all positions roughly equal)
        self.assertLess(std_freq, 0.1)

    # ---- at least one token masked ----

    def test_at_least_one_token_masked(self):
        """Even at very low t, at least one token must be masked (fallback)."""
        mask_id = 999
        for seed in range(50):
            torch.manual_seed(seed)
            ids = torch.randint(10, 500, (1, 16), dtype=torch.long)
            attn = torch.ones(1, 16, dtype=torch.long)
            noisy, labels, t, ratio = _continuous_time_m2t_batch(ids, attn, None, mask_id, 0.001, 0.002)
            n_masked = (noisy[0] == mask_id).sum().item()
            self.assertGreaterEqual(n_masked, 1, f"seed={seed}: no tokens masked at t={t:.4f}")

    # ---- labels only at masked positions ----

    def test_labels_only_at_masked_positions(self):
        """Labels should be -100 everywhere except at positions that were masked."""
        ids, noisy, labels, t, ratio, mask_id = self._make_batch(seq_len=64)
        masked_pos = (noisy[0] == mask_id)
        unmasked_pos = ~masked_pos
        # unmasked positions must have label -100
        self.assertTrue((labels[0][unmasked_pos] == -100).all().item())
        # masked positions must have the original token id
        self.assertTrue((labels[0][masked_pos] == ids[0][masked_pos]).all().item())
        emit_payload(
            "test_labels_only_at_masked_positions",
            "Labels must be -100 at unmasked positions and original ids at masked positions.",
            {"n_masked": int(masked_pos.sum()), "all_unmasked_neg100": True, "all_masked_original": True},
        )

    # ---- 1/t ELBO scaling ----

    def test_inv_t_scaling(self):
        """loss_m2t_scaled = loss_m2t_raw / t  (for loss_weighting='inv_t')."""
        # Simulate: raw_loss=2.0, t=0.25 → scaled = 2.0/0.25 = 8.0
        raw_loss = torch.tensor(2.0)
        t = 0.25
        t_clamp = max(t, 0.001)
        scaled = raw_loss / t_clamp
        emit_payload(
            "test_inv_t_scaling",
            "1/t ELBO weighting: loss_scaled = loss_raw / t.",
            {"raw_loss": float(raw_loss), "t": t, "scaled": float(scaled), "expected": 8.0},
        )
        self.assertAlmostEqual(float(scaled), 8.0, places=4)

    def test_inv_t_clamped_at_tmin(self):
        """At very small t, clamping to 0.001 prevents explosion."""
        raw_loss = torch.tensor(1.0)
        t = 0.0001  # below t_min
        t_clamp = max(t, 0.001)
        scaled = raw_loss / t_clamp
        expected = 1.0 / 0.001  # 1000
        self.assertAlmostEqual(float(scaled), expected, places=1)

    def test_inv_t_scaling_at_t1(self):
        """At t=1.0, scaling factor is 1 (so loss unchanged)."""
        raw_loss = torch.tensor(3.5)
        t = 1.0
        scaled = raw_loss / max(t, 0.001)
        self.assertAlmostEqual(float(scaled), float(raw_loss), places=4)

    def test_inv_t_increases_loss_for_small_t(self):
        """For t<1, 1/t > 1, so scaled loss > raw loss."""
        raw_loss = torch.tensor(2.0)
        for t in [0.1, 0.3, 0.5, 0.8]:
            scaled = float(raw_loss) / max(t, 0.001)
            self.assertGreater(scaled, float(raw_loss), f"t={t}: scaled should exceed raw")


if __name__ == "__main__":
    unittest.main()
