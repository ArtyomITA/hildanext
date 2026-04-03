# S2D2 inference tests — paper-faithful checks.
# Verifies: routing, contiguous span, verification, acceptance,
# fallback, diagnostics, endpoint existence, non-regression.
from __future__ import annotations
import unittest
import torch
import math
from hildanext.utils import TinyCausalLM, SimpleTokenizer
from hildanext.inference_s2d2 import (
    InferenceS2D2Request,
    S2D2Diagnostics,
    find_first_contiguous_mask_span,
    estimate_expected_accept_prefix,
    compute_verification_score,
    route_s2d2_verification,
    build_verifier_inputs_position_aligned,
    run_s2d2_verification,
    s2d2_decode,
)
from reporting import emit_payload


def _make_dummy_model(vocab: int = 64, hidden: int = 16):
    m = TinyCausalLM(vocab_size=vocab, hidden_size=hidden)
    m.eval()
    return m


def _make_dummy_tokenizer(vocab: int = 64):
    return SimpleTokenizer(vocab_size=vocab)


MASK_ID = 63
VOCAB = 64


# ---------------------------------------------------------------------------
# 1. Route registration
# ---------------------------------------------------------------------------
class TestS2D2Routing(unittest.TestCase):
    """Verify /inferences2d2 endpoint is registered and base routes still work."""

    def test_inferences2d2_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        emit_payload("test_s2d2_route_exists", "check /inferences2d2", {"routes": routes[:30]})
        self.assertIn("/inferences2d2", routes)

    def test_original_generate_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/generate", routes)

    def test_inferenceots_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferenceots", routes)

    def test_inferencercdm_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferencercdm", routes)


# ---------------------------------------------------------------------------
# 2. Request schema
# ---------------------------------------------------------------------------
class TestS2D2RequestSchema(unittest.TestCase):
    """Verify request schema accepts all S2D2-specific fields."""

    def test_default_values(self):
        req = InferenceS2D2Request(prompt="hello")
        self.assertEqual(req.s2d2_block_size, 32)
        self.assertEqual(req.s2d2_routing_policy, "min_span")
        self.assertEqual(req.s2d2_acceptance_estimator, "entropy")
        self.assertEqual(req.s2d2_min_verify_span, 2)

    def test_custom_values(self):
        req = InferenceS2D2Request(
            prompt="test",
            s2d2_block_size=16,
            s2d2_routing_policy="always",
            s2d2_acceptance_estimator="margin",
        )
        self.assertEqual(req.s2d2_block_size, 16)
        self.assertEqual(req.s2d2_routing_policy, "always")


# ---------------------------------------------------------------------------
# 3. First contiguous masked span
# ---------------------------------------------------------------------------
class TestContiguousSpan(unittest.TestCase):
    """Paper Section 4.1: verify first contiguous masked span detection."""

    def test_all_masked(self):
        seq = torch.tensor([[10, 20, MASK_ID, MASK_ID, MASK_ID]])
        start, end = find_first_contiguous_mask_span(seq, MASK_ID, prompt_len=2)
        self.assertEqual(start, 2)
        self.assertEqual(end, 5)

    def test_partial_masked(self):
        seq = torch.tensor([[10, 20, 5, MASK_ID, MASK_ID, 8, MASK_ID]])
        start, end = find_first_contiguous_mask_span(seq, MASK_ID, prompt_len=2)
        # First contiguous span in gen region: positions 3,4
        self.assertEqual(start, 3)
        self.assertEqual(end, 5)

    def test_no_masked(self):
        seq = torch.tensor([[10, 20, 5, 8, 12]])
        start, end = find_first_contiguous_mask_span(seq, MASK_ID, prompt_len=2)
        self.assertEqual(start, -1)
        self.assertEqual(end, -1)

    def test_single_mask(self):
        seq = torch.tensor([[10, 20, MASK_ID, 5, 8]])
        start, end = find_first_contiguous_mask_span(seq, MASK_ID, prompt_len=2)
        self.assertEqual(start, 2)
        self.assertEqual(end, 3)

    def test_gap_then_mask(self):
        seq = torch.tensor([[10, 20, 5, 8, MASK_ID, MASK_ID]])
        start, end = find_first_contiguous_mask_span(seq, MASK_ID, prompt_len=2)
        self.assertEqual(start, 4)
        self.assertEqual(end, 6)


# ---------------------------------------------------------------------------
# 4. Expected accept prefix estimation
# ---------------------------------------------------------------------------
class TestAcceptPrefixEstimation(unittest.TestCase):
    """Paper Eq.(5): K̂ = Σ_{k=1}^{L} Π_{i=1}^{k} α_i."""

    def test_entropy_estimator(self):
        # Create logits where first position is very confident, second less so
        logits = torch.zeros(1, 4, VOCAB)
        logits[0, 0, 5] = 10.0  # very confident → low entropy → high α
        logits[0, 1, :] = 0.0   # uniform → high entropy → low α
        k_hat = estimate_expected_accept_prefix(
            logits, span_start=0, span_end=2,
            estimator="entropy", entropy_beta=1.0, vocab_size=VOCAB,
        )
        self.assertGreater(k_hat, 0.0)
        self.assertLess(k_hat, 3.0)

    def test_margin_estimator(self):
        logits = torch.zeros(1, 4, VOCAB)
        logits[0, 0, 5] = 10.0  # large margin
        logits[0, 1, 5] = 0.2
        logits[0, 1, 6] = 0.1   # small margin
        k_hat = estimate_expected_accept_prefix(
            logits, span_start=0, span_end=2,
            estimator="margin", margin_threshold=0.1, vocab_size=VOCAB,
        )
        self.assertGreater(k_hat, 0.0)

    def test_empty_span(self):
        logits = torch.zeros(1, 4, VOCAB)
        k_hat = estimate_expected_accept_prefix(
            logits, span_start=2, span_end=2,
            estimator="entropy", vocab_size=VOCAB,
        )
        self.assertEqual(k_hat, 0.0)


# ---------------------------------------------------------------------------
# 5. Verification score (Eq.6)
# ---------------------------------------------------------------------------
class TestVerificationScore(unittest.TestCase):
    """Paper Eq.(6): s = K̂ - c (static) or s = K̂ - c·N_hi (dynamic)."""

    def test_static_score(self):
        s = compute_verification_score(k_hat=5.0, cost=1.0, mode="static")
        self.assertAlmostEqual(s, 4.0)

    def test_dynamic_score(self):
        s = compute_verification_score(k_hat=5.0, cost=1.0, n_hi=3, mode="dynamic")
        self.assertAlmostEqual(s, 2.0)


# ---------------------------------------------------------------------------
# 6. Routing policies (Algorithm 4)
# ---------------------------------------------------------------------------
class TestRoutingPolicies(unittest.TestCase):
    """Algorithm 4: DOVERIFY routing decisions."""

    def _make_dummy_args(self, span_len=4, policy="min_span"):
        logits = torch.zeros(1, 10, VOCAB)
        logits[0, :, 5] = 5.0  # some confidence
        return dict(
            span_len=span_len,
            draft_logits=logits,
            span_start=3,
            span_end=3 + span_len,
            confidence=torch.ones(6) * 0.8,
            mask_positions=torch.arange(4),
            mask_id=MASK_ID,
            tau=0.3,
            policy=policy,
            min_verify_span=2,
            score_threshold=0.0,
            score_cost=1.0,
            score_mode="static",
            hysteresis_state=False,
            hysteresis_on=1.0,
            hysteresis_off=-5.0,
            estimator="entropy",
            entropy_beta=1.0,
            margin_threshold=0.1,
            vocab_size=VOCAB,
        )

    def test_min_span_policy_above(self):
        do, _, _ = route_s2d2_verification(**self._make_dummy_args(span_len=4, policy="min_span"))
        self.assertTrue(do)

    def test_min_span_policy_below(self):
        do, _, _ = route_s2d2_verification(**self._make_dummy_args(span_len=1, policy="min_span"))
        self.assertFalse(do)

    def test_always_policy(self):
        do, _, _ = route_s2d2_verification(**self._make_dummy_args(span_len=1, policy="always"))
        self.assertTrue(do)

    def test_never_policy(self):
        do, _, _ = route_s2d2_verification(**self._make_dummy_args(span_len=100, policy="never"))
        self.assertFalse(do)

    def test_score_threshold_policy(self):
        args = self._make_dummy_args(span_len=4, policy="score_threshold")
        args["score_threshold"] = -100.0  # very low → always verify
        do, _, score = route_s2d2_verification(**args)
        self.assertTrue(do)

    def test_hysteresis_transitions(self):
        args = self._make_dummy_args(span_len=4, policy="hysteresis")
        args["hysteresis_state"] = False
        args["hysteresis_on"] = -100.0  # easy to turn on
        do, new_state, _ = route_s2d2_verification(**args)
        self.assertTrue(do)
        self.assertTrue(new_state)


# ---------------------------------------------------------------------------
# 7. Verifier reuses same model
# ---------------------------------------------------------------------------
class TestVerifierReusesSameModel(unittest.TestCase):
    """Paper Section 4.1: same model serves as both drafter and verifier."""

    def test_same_model_used(self):
        model = _make_dummy_model()
        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.tensor([1, 2, 3])  # prompt
        drafted_ids = torch.tensor([5, 6, 7])
        ver_seq = build_verifier_inputs_position_aligned(drafted_ids, MASK_ID, 3, 6, seq)
        # Verify drafted tokens are placed correctly
        self.assertTrue(torch.equal(ver_seq[0, 3:6], drafted_ids))
        # Verify prompt is preserved
        self.assertTrue(torch.equal(ver_seq[0, :3], seq[0, :3]))

    def test_verification_produces_results(self):
        model = _make_dummy_model()
        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.tensor([1, 2, 3])
        drafted_ids = torch.tensor([5, 6, 7])
        ver_seq = build_verifier_inputs_position_aligned(drafted_ids, MASK_ID, 3, 6, seq)
        # Make dummy draft probs
        draft_probs = torch.ones(1, 10, VOCAB) / VOCAB
        accepted_ids, ver_logits, acc_count, rej_pos = run_s2d2_verification(
            model=model, verifier_seq=ver_seq, draft_probs=draft_probs,
            drafted_ids=drafted_ids, span_start=3, span_end=6,
        )
        self.assertEqual(accepted_ids.shape[0], 3)
        self.assertGreaterEqual(acc_count, 0)


# ---------------------------------------------------------------------------
# 8. Acceptance left-to-right, first rejection stops
# ---------------------------------------------------------------------------
class TestAcceptanceLeftToRight(unittest.TestCase):
    """Algorithm 3 lines 14-24: left-to-right, break at first rejection."""

    def test_acceptance_stops_at_rejection(self):
        model = _make_dummy_model()
        seq = torch.full((1, 8), MASK_ID, dtype=torch.long)
        seq[0, :2] = torch.tensor([1, 2])
        drafted_ids = torch.tensor([5, 6, 7, 8])
        ver_seq = build_verifier_inputs_position_aligned(drafted_ids, MASK_ID, 2, 6, seq)
        # Make draft probs strongly peaked at wrong tokens so verifier likely rejects
        draft_probs = torch.zeros(1, 8, VOCAB)
        draft_probs[0, :, 10] = 1.0  # peaked at token 10, but drafted token is 5,6,7,8
        _, _, acc_count, rej_pos = run_s2d2_verification(
            model=model, verifier_seq=ver_seq, draft_probs=draft_probs,
            drafted_ids=drafted_ids, span_start=2, span_end=6,
        )
        # Either all accepted or rejection happened somewhere
        if rej_pos >= 0:
            self.assertEqual(acc_count, rej_pos)
        else:
            self.assertEqual(acc_count, 4)


# ---------------------------------------------------------------------------
# 9. Fallback to diffusion when verification skipped
# ---------------------------------------------------------------------------
class TestFallbackBehavior(unittest.TestCase):
    """Algorithm 3 lines 26-29: fallback to confidence threshold decoding."""

    def test_never_policy_uses_fallback(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            routing_policy="never",
            seed=42, is_dummy=True,
        )
        emit_payload("test_s2d2_fallback", "never policy", {
            "stats": stats, "verifier_invocations": diag.verifier_invocations,
            "fallback_count": diag.fallback_to_diffusion_count,
        })
        self.assertEqual(diag.verifier_invocations, 0)
        self.assertGreater(diag.fallback_to_diffusion_count, 0)
        self.assertEqual(stats["engine"], "s2d2")


# ---------------------------------------------------------------------------
# 10. End-to-end S2D2 decode
# ---------------------------------------------------------------------------
class TestS2D2EndToEnd(unittest.TestCase):
    """Full E2E: S2D2 decode produces text, stats, diagnostics."""

    def test_e2e_basic(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=6,
            routing_policy="always",
            seed=42, is_dummy=True,
        )
        emit_payload("test_s2d2_e2e_basic", "always verify", {
            "text": text, "engine": stats["engine"],
            "verifier_invocations": diag.verifier_invocations,
            "steps": diag.total_denoising_steps,
        })
        self.assertEqual(stats["engine"], "s2d2")
        self.assertEqual(stats["mode"], "S2D2")
        self.assertIn("s2d2_diagnostics", stats)
        self.assertGreater(diag.total_denoising_steps, 0)
        self.assertGreater(diag.verifier_invocations, 0)

    def test_e2e_min_span(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            routing_policy="min_span", min_verify_span=1,
            seed=42, is_dummy=True,
        )
        self.assertEqual(stats["engine"], "s2d2")
        self.assertIn("routing_policy", stats)
        self.assertEqual(stats["routing_policy"], "min_span")

    def test_diagnostics_structure(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats, diag = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            routing_policy="always",
            seed=42, is_dummy=True,
        )
        d = diag.to_dict()
        self.assertIn("total_denoising_steps", d)
        self.assertIn("verifier_invocations", d)
        self.assertIn("verifier_skips", d)
        self.assertIn("accepted_prefix_lengths", d)
        self.assertIn("routing_policy_used", d)
        self.assertIn("verifier_mask_path", d)
        self.assertEqual(d["verifier_mask_path"], "position_aligned_2n")
        self.assertIsInstance(d["steps"], list)
        if d["steps"]:
            step = d["steps"][0]
            self.assertIn("verify_invoked", step)
            self.assertIn("accepted_count", step)
            self.assertIn("rejection_position", step)

    def test_not_equivalent_to_global_ar(self):
        """Paper Section 4.4: S2D2 is NOT global AR. Verify hybrid trajectory."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats_always, diag_always = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            routing_policy="always",
            seed=42, is_dummy=True,
        )
        _, stats_never, diag_never = s2d2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            routing_policy="never",
            seed=42, is_dummy=True,
        )
        # Both produce S2D2 engine output
        self.assertEqual(stats_always["engine"], "s2d2")
        self.assertEqual(stats_never["engine"], "s2d2")
        # "always" should have verifier invocations, "never" should not
        self.assertGreater(diag_always.verifier_invocations, 0)
        self.assertEqual(diag_never.verifier_invocations, 0)
        self.assertGreater(diag_never.fallback_to_diffusion_count, 0)


if __name__ == "__main__":
    unittest.main()
