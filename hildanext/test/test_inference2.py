# Inferenza2 hybrid engine tests.
# Verifies: routing, hybrid beam state, RCD math reuse, OTS search,
# independent RCD state per beam, hybrid scoring, fallback, determinism.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
import math
from hildanext.utils import TinyCausalLM, SimpleTokenizer
from hildanext.inference2 import (
    Inference2Request,
    HybridBeamState,
    Inference2Diagnostics,
    clone_hybrid_beam,
    _hybrid_forward,
    update_hybrid_rcd_state,
    expand_hybrid_candidates,
    score_hybrid_candidate,
    score_hybrid_candidate_fallback,
    prune_hybrid_beams,
    initialize_hybrid_warm_start,
    inferenza2_decode,
)
from hildanext.inference_rcd import (
    compute_normalized_entropy,
    compute_rcd_residuals_from_probs,
    build_rcd_inputs_embeds,
    _get_embedding_layer,
)
from hildanext.inference_ots import (
    _add_gumbel_noise,
    _transfer_tokens,
    ots_decode,
)
from hildanext.inference_rcd import rcd_decode
from reporting import emit_payload


def _make_dummy_model(vocab: int = 64, hidden: int = 16):
    m = TinyCausalLM(vocab_size=vocab, hidden_size=hidden)
    m.eval()
    return m


def _make_dummy_tokenizer(vocab: int = 64):
    return SimpleTokenizer(vocab_size=vocab)


MASK_ID = 63
VOCAB = 64


# ===================================================================
# 1. Non-regression: old engines still work
# ===================================================================
class TestNonRegression(unittest.TestCase):
    """Verify inference_rcd.py and inference_ots.py still work unchanged."""

    def test_rcd_still_works(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = rcd_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, seed=42, is_dummy=True,
        )
        emit_payload("test_rcd_nonregression", "RCD still works", {"engine": stats["engine"]})
        self.assertEqual(stats["engine"], "rcd")
        self.assertTrue(len(text) > 0)

    def test_ots_still_works(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = ots_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, beam_size=2,
            gumbel_temperature=0.5, search_interval=2,
            seed=42, is_dummy=True,
        )
        emit_payload("test_ots_nonregression", "OTS still works", {"engine": stats["engine"]})
        self.assertEqual(stats["engine"], "ots")
        self.assertTrue(len(text) > 0)


# ===================================================================
# 2. Registration: inferenza2 is discoverable
# ===================================================================
class TestInferenza2Routing(unittest.TestCase):
    """New /api/inferenza2 endpoint is present."""

    def test_inferenza2_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        emit_payload("test_inferenza2_route", "check /inferenza2",
                     {"routes_sample": routes[:30]})
        self.assertIn("/inferenza2", routes)

    def test_old_endpoints_still_exist(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferencercdm", routes)
        self.assertIn("/inferenceots", routes)
        self.assertIn("/generate", routes)


# ===================================================================
# 3. Hybrid beam state: independent RCD state
# ===================================================================
class TestHybridBeamState(unittest.TestCase):
    """After expansion, child beams have independent RCD states."""

    def test_clone_has_independent_rcd_state(self):
        seq = torch.full((1, 15), MASK_ID, dtype=torch.long)
        seq[0, :5] = torch.arange(5)
        alpha = torch.rand(1, 15)
        delta = torch.randn(1, 15, 16)

        parent = HybridBeamState(
            seq=seq, prompt_len=5,
            alpha_prev=alpha, delta_prev=delta,
        )
        child = clone_hybrid_beam(parent)

        # Mutate child RCD state
        child.alpha_prev[0, 0] = 99.0
        child.delta_prev[0, 0, 0] = -99.0

        # Parent must be unchanged
        self.assertNotAlmostEqual(float(parent.alpha_prev[0, 0].item()), 99.0)
        self.assertNotAlmostEqual(float(parent.delta_prev[0, 0, 0].item()), -99.0)
        emit_payload("test_hybrid_beam_independent_rcd", "clone independence",
                     {"parent_alpha0": float(parent.alpha_prev[0, 0].item()),
                      "child_alpha0": float(child.alpha_prev[0, 0].item())})

    def test_expansion_produces_children_with_rcd(self):
        model = _make_dummy_model()
        embed_layer = _get_embedding_layer(model)
        embed_weight = embed_layer.weight.detach()

        seq = torch.full((1, 15), MASK_ID, dtype=torch.long)
        seq[0, :5] = torch.arange(5)
        alpha = torch.rand(1, 15)
        delta = torch.randn(1, 15, 16)

        beam = HybridBeamState(
            seq=seq, prompt_len=5,
            alpha_prev=alpha.clone(), delta_prev=delta.clone(),
        )

        children = expand_hybrid_candidates(
            model, beam, beam_size=3, mask_id=MASK_ID,
            tokens_per_step=2, gumbel_temp=0.5,
            embed_layer=embed_layer, embed_weight=embed_weight,
            effective_v=VOCAB, t_res=1.0, use_rcd=True,
        )

        self.assertEqual(len(children), 3)
        for child, revealed, x0_full in children:
            self.assertIsNotNone(child.alpha_prev)
            self.assertIsNotNone(child.delta_prev)
            self.assertEqual(child.alpha_prev.shape, (1, 15))
            self.assertEqual(child.delta_prev.shape, (1, 15, 16))

        emit_payload("test_expansion_rcd_children", "children have RCD state",
                     {"num_children": len(children)})


# ===================================================================
# 4. RCD math correctness — reused from inference_rcd
# ===================================================================
class TestRCDMathReuse(unittest.TestCase):
    """Residuals come from probs × embedding codebook, not hidden states."""

    def test_residual_from_probs_times_codebook(self):
        V, D = 32, 8
        probs = F.softmax(torch.randn(1, 5, V), dim=-1)
        E = torch.randn(V, D)
        delta = compute_rcd_residuals_from_probs(probs, E)
        expected = torch.matmul(probs, E)
        self.assertTrue(torch.allclose(delta, expected, atol=1e-6))
        emit_payload("test_rcd_math_residual", "probs @ E", {"shape": list(delta.shape)})

    def test_normalized_entropy_bounds(self):
        V = 32
        uniform = torch.ones(1, 5, V) / V
        alpha = compute_normalized_entropy(uniform, V)
        self.assertTrue(torch.allclose(alpha, torch.ones_like(alpha), atol=1e-3))

        onehot = torch.zeros(1, 5, V)
        onehot[:, :, 0] = 1.0
        alpha_oh = compute_normalized_entropy(onehot, V)
        self.assertTrue((alpha_oh < 0.01).all())
        emit_payload("test_rcd_entropy_bounds", "entropy bounds",
                     {"uniform_alpha": float(alpha.mean()), "onehot_alpha": float(alpha_oh.mean())})


# ===================================================================
# 5. OTS search: expansion + pruning
# ===================================================================
class TestOTSSearchInHybrid(unittest.TestCase):
    """Multiple candidates are expanded and pruned in hybrid mode."""

    def test_hybrid_full_decode_has_checkpoints(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=6,
            seed=42, is_dummy=True,
            rcd_enabled=True, ots_enabled=True,
            beam_size=3, gumbel_temperature=0.5,
            search_interval=3,
        )
        self.assertEqual(stats["engine"], "inferenza2")
        self.assertTrue(diag.rcd_enabled)
        self.assertTrue(diag.ots_enabled)
        self.assertGreater(diag.total_beams_explored, 3)
        self.assertGreater(diag.total_search_checkpoints, 0)
        emit_payload("test_hybrid_search", "hybrid OTS checkpoints",
                     {"checkpoints": diag.total_search_checkpoints,
                      "beams_explored": diag.total_beams_explored})


# ===================================================================
# 6. Hybrid forward path: uses RCD inputs_embeds
# ===================================================================
class TestHybridForwardPath(unittest.TestCase):
    """Hybrid beam forward uses RCD-style inputs_embeds."""

    def test_hybrid_forward_with_rcd(self):
        model = _make_dummy_model()
        embed_layer = _get_embedding_layer(model)

        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.arange(3)
        alpha = torch.rand(1, 10)
        delta = torch.randn(1, 10, 16)

        beam = HybridBeamState(
            seq=seq, prompt_len=3,
            alpha_prev=alpha, delta_prev=delta,
        )

        logits = _hybrid_forward(model, beam, MASK_ID, embed_layer, use_rcd=True)
        self.assertEqual(logits.shape, (1, 10, VOCAB))
        emit_payload("test_hybrid_forward_rcd", "RCD forward works",
                     {"shape": list(logits.shape)})

    def test_hybrid_forward_without_rcd(self):
        model = _make_dummy_model()
        embed_layer = _get_embedding_layer(model)

        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.arange(3)

        beam = HybridBeamState(seq=seq, prompt_len=3)

        logits = _hybrid_forward(model, beam, MASK_ID, embed_layer, use_rcd=False)
        self.assertEqual(logits.shape, (1, 10, VOCAB))


# ===================================================================
# 7. Hybrid scoring: dedicated helper
# ===================================================================
class TestHybridScoring(unittest.TestCase):
    """Main path uses diffusion-native scoring (not confidence-only)."""

    def test_diffusion_native_score_returns_float(self):
        model = _make_dummy_model()
        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.arange(3)
        seq[0, 3] = 5  # one revealed position

        beam = HybridBeamState(seq=seq, prompt_len=3)
        revealed = torch.zeros(1, 10, dtype=torch.bool)
        revealed[0, 3] = True

        x0_full = seq.clone()
        x0_full[0, 3:] = torch.randint(0, VOCAB - 1, (7,))

        score = score_hybrid_candidate(model, beam, revealed, x0_full, MASK_ID)
        self.assertIsInstance(score, float)
        emit_payload("test_hybrid_scoring", "diffusion score",
                     {"score": score, "type": type(score).__name__})

    def test_fallback_score_when_enabled(self):
        seq = torch.full((1, 10), MASK_ID, dtype=torch.long)
        seq[0, :3] = torch.arange(3)
        seq[0, 3] = 5

        beam = HybridBeamState(seq=seq, prompt_len=3)
        revealed = torch.zeros(1, 10, dtype=torch.bool)
        revealed[0, 3] = True
        conf = torch.rand(1, 10)

        score = score_hybrid_candidate_fallback(beam, conf, revealed)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)


# ===================================================================
# 8. Graceful fallback
# ===================================================================
class TestGracefulFallback(unittest.TestCase):
    """Invalid pruning mode raises if fallback disabled; works if enabled."""

    def test_invalid_pruning_mode_raises(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        with self.assertRaises(ValueError):
            inferenza2_decode(
                model=model, tokenizer=tok, device=torch.device("cpu"),
                mask_id=MASK_ID, vocab_size=VOCAB, prompt="test",
                max_new_tokens=4, max_steps=2, seed=42, is_dummy=True,
                pruning_mode="invalid_mode", allow_fallback_score=False,
            )

    def test_invalid_pruning_mode_falls_back(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="test",
            max_new_tokens=4, max_steps=2, seed=42, is_dummy=True,
            pruning_mode="invalid_mode", allow_fallback_score=True,
        )
        self.assertEqual(diag.pruning_mode_used, "fallback_confidence")
        self.assertIn("fallback_confidence_scoring", diag.approximations_used)
        emit_payload("test_graceful_fallback", "fallback activated",
                     {"pruning_mode": diag.pruning_mode_used})


# ===================================================================
# 9. Determinism
# ===================================================================
class TestDeterminism(unittest.TestCase):
    """Fixed seed produces stable outputs."""

    def test_same_seed_same_output(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text1, stats1, _ = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, seed=123, is_dummy=True,
            beam_size=2, search_interval=2, gumbel_temperature=0.3,
        )
        text2, stats2, _ = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, seed=123, is_dummy=True,
            beam_size=2, search_interval=2, gumbel_temperature=0.3,
        )
        self.assertEqual(text1, text2)
        emit_payload("test_determinism", "same seed same output",
                     {"match": text1 == text2})


# ===================================================================
# 10. Ablation modes
# ===================================================================
class TestAblationModes(unittest.TestCase):
    """Hybrid can run as rcd_only or ots_only via ablation flags."""

    def test_rcd_only_mode(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, seed=42, is_dummy=True,
            rcd_enabled=True, ots_enabled=False, beam_size=3,
        )
        self.assertEqual(diag.hybrid_mode_active, "rcd_only")
        self.assertTrue(diag.rcd_enabled)
        self.assertFalse(diag.ots_enabled)
        self.assertEqual(diag.total_search_checkpoints, 0)
        emit_payload("test_ablation_rcd_only", "rcd_only mode",
                     {"mode": diag.hybrid_mode_active})

    def test_ots_only_mode(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=6, seed=42, is_dummy=True,
            rcd_enabled=False, ots_enabled=True, beam_size=3,
            search_interval=3, gumbel_temperature=0.5,
        )
        self.assertEqual(diag.hybrid_mode_active, "ots_only")
        self.assertFalse(diag.rcd_enabled)
        self.assertTrue(diag.ots_enabled)
        emit_payload("test_ablation_ots_only", "ots_only mode",
                     {"mode": diag.hybrid_mode_active})

    def test_full_hybrid_mode(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = inferenza2_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=6, seed=42, is_dummy=True,
            rcd_enabled=True, ots_enabled=True, beam_size=3,
            search_interval=3, gumbel_temperature=0.5,
        )
        self.assertEqual(diag.hybrid_mode_active, "rcd_plus_ots")
        self.assertTrue(diag.rcd_enabled)
        self.assertTrue(diag.ots_enabled)
        self.assertIn("rcd_no_trained_weights", diag.approximations_used)
        self.assertIn("ots_scoring_uses_plain_forward_not_rcd_augmented", diag.approximations_used)
        emit_payload("test_ablation_full_hybrid", "full hybrid mode",
                     {"mode": diag.hybrid_mode_active,
                      "approximations": diag.approximations_used})


if __name__ == "__main__":
    unittest.main()
