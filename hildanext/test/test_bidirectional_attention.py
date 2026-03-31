# Bidirectional attention tests for S0-A (WSD phase-dependent attention mode).
# Entrypoints: unittest test methods.
# Verifies that causal/non-causal mask is applied per WSD phase.
from __future__ import annotations
import unittest
import torch
from hildanext.masks import batch_doc_attention_mask
from hildanext.diffusion import wsd_block
from hildanext.config import WSDConfig, ExperimentConfig
from reporting import emit_payload


def _determine_bidirectional(phase: str, attn_mode: str) -> bool:
    """Replicate the phase→bidirectional logic from training._run."""
    if attn_mode == "bidirectional_always":
        return True
    if attn_mode == "bidirectional_only_stable":
        return phase == "stable"
    return False


class BidirectionalAttentionTests(unittest.TestCase):

    # ---- mask builder level ----

    def test_causal_mask_is_lower_triangular(self):
        """causal=True must produce a lower-triangular mask (no future→past)."""
        docs = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        m = batch_doc_attention_mask(docs, causal=True)
        upper_strict = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        leaked = m[0][upper_strict].any().item()
        emit_payload(
            "test_causal_mask_is_lower_triangular",
            "Causal mask must be lower-triangular: no attention from earlier pos to later pos above diagonal.",
            {"mask": m[0].int().tolist(), "leaked_above_diag": leaked},
        )
        self.assertFalse(leaked, "causal mask leaked above diagonal")

    def test_bidirectional_mask_is_symmetric(self):
        """causal=False must yield a symmetric mask (all within-doc positions see each other)."""
        docs = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        m = batch_doc_attention_mask(docs, causal=False)
        sym = torch.equal(m[0], m[0].T)
        all_true = m[0].all().item()
        emit_payload(
            "test_bidirectional_mask_is_symmetric",
            "Non-causal mask for a single doc must be symmetric and fully attending.",
            {"mask": m[0].int().tolist(), "symmetric": sym, "all_true": all_true},
        )
        self.assertTrue(sym, "bidirectional mask is not symmetric")
        self.assertTrue(all_true, "bidirectional mask has False entries for same doc")

    def test_bidirectional_still_blocks_cross_doc(self):
        """Even with causal=False, cross-doc attention must be blocked."""
        docs = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        m = batch_doc_attention_mask(docs, causal=False)
        cross_01 = m[0, 0, 2].item()
        cross_10 = m[0, 2, 0].item()
        emit_payload(
            "test_bidirectional_still_blocks_cross_doc",
            "Bidirectional mode must still block cross-document attention.",
            {"mask": m[0].int().tolist(), "cross_01": cross_01, "cross_10": cross_10},
        )
        self.assertFalse(cross_01)
        self.assertFalse(cross_10)

    # ---- WSD phase → bidirectional mapping ----

    def test_warmup_is_causal_in_bidir_only_stable(self):
        """attention_mode=bidirectional_only_stable ⇒ warmup phase ⇒ causal (bidirectional=False)."""
        bidir = _determine_bidirectional("warmup", "bidirectional_only_stable")
        emit_payload(
            "test_warmup_is_causal_in_bidir_only_stable",
            "Warmup phase in bidirectional_only_stable mode must be causal.",
            {"phase": "warmup", "bidirectional": bidir},
        )
        self.assertFalse(bidir)

    def test_stable_is_bidirectional_in_bidir_only_stable(self):
        """attention_mode=bidirectional_only_stable ⇒ stable phase ⇒ bidirectional."""
        bidir = _determine_bidirectional("stable", "bidirectional_only_stable")
        emit_payload(
            "test_stable_is_bidirectional_in_bidir_only_stable",
            "Stable phase in bidirectional_only_stable mode must be bidirectional.",
            {"phase": "stable", "bidirectional": bidir},
        )
        self.assertTrue(bidir)

    def test_decay_is_causal_in_bidir_only_stable(self):
        """attention_mode=bidirectional_only_stable ⇒ decay phase ⇒ causal."""
        bidir = _determine_bidirectional("decay", "bidirectional_only_stable")
        emit_payload(
            "test_decay_is_causal_in_bidir_only_stable",
            "Decay phase in bidirectional_only_stable mode must be causal.",
            {"phase": "decay", "bidirectional": bidir},
        )
        self.assertFalse(bidir)

    def test_bidirectional_always_all_phases(self):
        """attention_mode=bidirectional_always ⇒ all phases bidirectional."""
        for ph in ("warmup", "stable", "decay"):
            bidir = _determine_bidirectional(ph, "bidirectional_always")
            self.assertTrue(bidir, f"bidirectional_always must be True for phase={ph}")

    def test_causal_always_all_phases(self):
        """attention_mode=causal_always ⇒ all phases causal."""
        for ph in ("warmup", "stable", "decay"):
            bidir = _determine_bidirectional(ph, "causal_always")
            self.assertFalse(bidir, f"causal_always must be False for phase={ph}")

    # ---- wsd_block integration: verify phase at 8k-step boundaries ----

    def test_wsd_phase_at_8k_boundaries(self):
        """With 8k config (warmup=800,stable=5600,decay=1600), verify phases and bidir at key steps."""
        wsd_cfg = WSDConfig(
            warmup_steps=800,
            stable_steps=5600,
            decay_steps=1600,
            start_block_size=1,
            max_block_size=1024,
            end_block_size=32,
            ladder_blocks=[1, 4, 32, 64, 128, 256, 512, 1024],
            decay_blocks=[1024, 512, 256, 128, 64, 32],
            enforce_divisibility=True,
        )
        test_points = [
            (0, "warmup", False),       # first step → warmup
            (400, "warmup", False),      # mid warmup
            (799, "warmup", False),      # last warmup step
            (800, "stable", True),       # first stable step
            (4000, "stable", True),      # mid stable
            (6399, "stable", True),      # last stable step
            (6400, "decay", False),      # first decay step
            (7999, "decay", False),      # last step
        ]
        results = []
        for step, expected_phase, expected_bidir in test_points:
            ws = wsd_block(step, wsd_cfg, seq_len=1024)
            bidir = _determine_bidirectional(ws.phase, "bidirectional_only_stable")
            results.append({
                "step": step,
                "expected_phase": expected_phase,
                "got_phase": ws.phase,
                "expected_bidir": expected_bidir,
                "got_bidir": bidir,
                "block_size": ws.block_size,
            })
            self.assertEqual(ws.phase, expected_phase, f"step={step} expected phase={expected_phase} got={ws.phase}")
            self.assertEqual(bidir, expected_bidir, f"step={step} expected bidir={expected_bidir} got={bidir}")
        emit_payload(
            "test_wsd_phase_at_8k_boundaries",
            "Verifies WSD phases and bidirectional flag at 8k-config boundaries.",
            {"results": results},
        )

    # ---- ExperimentConfig defaults ----

    def test_experiment_config_defaults(self):
        """ExperimentConfig defaults must match the S0-A/C/D spec."""
        exp = ExperimentConfig()
        emit_payload(
            "test_experiment_config_defaults",
            "Checks that ExperimentConfig defaults align with S0-A/C/D specification.",
            {
                "attention_mode": exp.attention_mode,
                "time_param": exp.time_param,
                "loss_weighting": exp.loss_weighting,
                "shift_mode": exp.shift_mode,
                "t_min": exp.t_min,
                "t_max": exp.t_max,
            },
        )
        self.assertEqual(exp.attention_mode, "bidirectional_only_stable")
        self.assertEqual(exp.time_param, "continuous_time")
        self.assertEqual(exp.loss_weighting, "inv_t")
        self.assertEqual(exp.shift_mode, "preserve_left_shift")
        self.assertAlmostEqual(exp.t_min, 0.001)
        self.assertAlmostEqual(exp.t_max, 1.0)


if __name__ == "__main__":
    unittest.main()
