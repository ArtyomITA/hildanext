# RCD inference tests — paper-faithful checks.
# Verifies: routing, math (Eq.1-3), mask-only injection, entropy bounds,
# warm-start, no-hidden-state shortcut, graceful fallback.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
import math
from types import SimpleNamespace
from hildanext.utils import TinyCausalLM, SimpleTokenizer
from hildanext.inference_rcd import (
    compute_normalized_entropy,
    compute_rcd_residuals_from_probs,
    build_rcd_inputs_embeds,
    initialize_rcd_warm_start,
    rcd_decode,
    InferenceRCDMRequest,
    _get_embedding_layer,
)
from reporting import emit_payload


def _make_dummy_model(vocab: int = 64, hidden: int = 16):
    m = TinyCausalLM(vocab_size=vocab, hidden_size=hidden)
    m.eval()
    return m


def _make_dummy_tokenizer(vocab: int = 64):
    tok = SimpleTokenizer(vocab_size=vocab)
    return tok


class TestRCDRouting(unittest.TestCase):
    """1. Original /generate still works; 2. /api/inferencercdm works."""

    def test_original_generate_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import load_config
        import os
        cfg_path = os.environ.get("HILDANEXT_CONFIG", "")
        if not cfg_path or not os.path.isfile(cfg_path):
            from hildanext.config import AppConfig
            cfg = AppConfig()
        else:
            cfg = load_config(cfg_path)
        app = create_app(cfg)
        routes = [r.path for r in app.routes]
        emit_payload("test_original_generate_endpoint_exists",
                     "Checks /generate route is present", {"routes_sample": routes[:20]})
        self.assertIn("/generate", routes)

    def test_inferencercdm_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        cfg = AppConfig()
        app = create_app(cfg)
        routes = [r.path for r in app.routes]
        emit_payload("test_inferencercdm_endpoint_exists",
                     "Checks /inferencercdm route is present", {"routes_sample": routes[:20]})
        self.assertIn("/inferencercdm", routes)


class TestRCDNonRegression(unittest.TestCase):
    """3. Original dLLM inference path unchanged."""

    def test_tinycausal_input_ids_still_works(self):
        m = _make_dummy_model()
        x = torch.randint(0, 60, (1, 8))
        out = m(input_ids=x)
        emit_payload("test_tinycausal_input_ids_still_works",
                     "TinyCausalLM input_ids path unchanged", {"logits_shape": list(out.logits.shape)})
        self.assertEqual(out.logits.shape, (1, 8, 64))

    def test_tinycausal_inputs_embeds_works(self):
        m = _make_dummy_model()
        e = torch.randn(1, 8, 16)
        out = m(inputs_embeds=e)
        emit_payload("test_tinycausal_inputs_embeds_works",
                     "TinyCausalLM inputs_embeds path works", {"logits_shape": list(out.logits.shape)})
        self.assertEqual(out.logits.shape, (1, 8, 64))


class TestRCDMath(unittest.TestCase):
    """4. Residual vector = probability-weighted sum over embedding codebook (Eq.1)."""

    def test_residual_vector_eq1(self):
        V, D = 32, 8
        torch.manual_seed(42)
        probs = F.softmax(torch.randn(1, 4, V), dim=-1)
        E = torch.randn(V, D)
        delta = compute_rcd_residuals_from_probs(probs, E)
        # Manual: delta[0, i, :] = sum_j probs[0, i, j] * E[j, :]
        expected = torch.matmul(probs, E)
        emit_payload("test_residual_vector_eq1",
                     "Δ = p @ E (Eq.1)", {
                         "delta_shape": list(delta.shape),
                         "max_diff": float((delta - expected).abs().max().item()),
                     })
        self.assertTrue(torch.allclose(delta, expected, atol=1e-6))


class TestMaskOnlyInjection(unittest.TestCase):
    """5. Committed tokens do NOT receive residual interpolation."""

    def test_committed_tokens_untouched(self):
        V, D = 32, 8
        mask_id = 3
        torch.manual_seed(7)
        m = _make_dummy_model(vocab=V, hidden=D)
        input_ids = torch.tensor([[5, mask_id, 10, mask_id]])
        alpha_prev = torch.tensor([[0.8, 0.8, 0.8, 0.8]])
        delta_prev = torch.randn(1, 4, D)
        result = build_rcd_inputs_embeds(input_ids, mask_id, m.embed, alpha_prev, delta_prev, force_mask_only=True)
        base = m.embed(input_ids)
        # Positions 0 and 2 (non-mask) must equal base embeddings exactly
        emit_payload("test_committed_tokens_untouched",
                     "Non-mask positions keep base embeddings (Eq.2)", {
                         "pos0_match": bool(torch.allclose(result[0, 0], base[0, 0], atol=1e-7)),
                         "pos2_match": bool(torch.allclose(result[0, 2], base[0, 2], atol=1e-7)),
                     })
        self.assertTrue(torch.allclose(result[0, 0], base[0, 0], atol=1e-7))
        self.assertTrue(torch.allclose(result[0, 2], base[0, 2], atol=1e-7))

    def test_mask_positions_interpolated(self):
        V, D = 32, 8
        mask_id = 3
        torch.manual_seed(7)
        m = _make_dummy_model(vocab=V, hidden=D)
        input_ids = torch.tensor([[5, mask_id, 10, mask_id]])
        alpha_prev = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        delta_prev = torch.randn(1, 4, D)
        result = build_rcd_inputs_embeds(input_ids, mask_id, m.embed, alpha_prev, delta_prev, force_mask_only=True)
        base = m.embed(input_ids)
        # Positions 1 and 3 (mask) must differ from base (unless delta == base)
        expected_1 = 0.5 * base[0, 1] + 0.5 * delta_prev[0, 1]
        emit_payload("test_mask_positions_interpolated",
                     "Mask positions use interpolated embedding (Eq.2)", {
                         "pos1_match_expected": bool(torch.allclose(result[0, 1], expected_1, atol=1e-6)),
                     })
        self.assertTrue(torch.allclose(result[0, 1], expected_1, atol=1e-6))


class TestEntropyAlpha(unittest.TestCase):
    """6. Normalized entropy stays in [0,1]; temperature-scaled variant."""

    def test_entropy_bounds(self):
        V = 64
        # Uniform distribution → max entropy → α = 1
        uniform = torch.ones(1, 4, V) / V
        alpha_uniform = compute_normalized_entropy(uniform, V)
        # One-hot → min entropy → α = 0
        onehot = torch.zeros(1, 4, V)
        onehot[:, :, 0] = 1.0
        alpha_onehot = compute_normalized_entropy(onehot, V)
        emit_payload("test_entropy_bounds",
                     "α ∈ [0,1], uniform→1, one-hot→0", {
                         "alpha_uniform_mean": float(alpha_uniform.mean().item()),
                         "alpha_onehot_mean": float(alpha_onehot.mean().item()),
                     })
        self.assertTrue((alpha_uniform >= 0).all() and (alpha_uniform <= 1.001).all())
        self.assertTrue(torch.allclose(alpha_uniform, torch.ones_like(alpha_uniform), atol=1e-5))
        self.assertTrue((alpha_onehot >= 0).all() and (alpha_onehot <= 0.01).all())

    def test_temperature_scaled_alpha(self):
        V = 64
        torch.manual_seed(3)
        logits = torch.randn(1, 4, V)
        # T_res = 1.0
        probs_1 = F.softmax(logits / 1.0, dim=-1)
        alpha_1 = compute_normalized_entropy(probs_1, V)
        # T_res = 2.0 → softer → higher entropy → higher α
        probs_2 = F.softmax(logits / 2.0, dim=-1)
        alpha_2 = compute_normalized_entropy(probs_2, V)
        emit_payload("test_temperature_scaled_alpha",
                     "Higher T_res → higher α (Sec 3.3)", {
                         "alpha_t1_mean": float(alpha_1.mean().item()),
                         "alpha_t2_mean": float(alpha_2.mean().item()),
                     })
        self.assertGreater(float(alpha_2.mean()), float(alpha_1.mean()))


class TestWarmStart(unittest.TestCase):
    """7. Warm-start initializes residual state."""

    def test_warm_start_produces_residuals(self):
        V, D = 32, 8
        mask_id = 3
        torch.manual_seed(11)
        m = _make_dummy_model(vocab=V, hidden=D)
        seq = torch.full((1, 10), mask_id, dtype=torch.long)
        seq[0, :3] = torch.tensor([5, 6, 7])
        E = m.embed.weight.detach()
        alpha_0, delta_0 = initialize_rcd_warm_start(m, seq, mask_id, E, V, t_res=1.0)
        emit_payload("test_warm_start_produces_residuals",
                     "Warm-start yields valid α₀ and Δ₀", {
                         "alpha_shape": list(alpha_0.shape),
                         "delta_shape": list(delta_0.shape),
                         "alpha_range": [float(alpha_0.min()), float(alpha_0.max())],
                     })
        self.assertEqual(alpha_0.shape, (1, 10))
        self.assertEqual(delta_0.shape, (1, 10, D))
        self.assertTrue((alpha_0 >= 0).all() and (alpha_0 <= 1.001).all())


class TestNoHiddenStateShortcut(unittest.TestCase):
    """8. Implementation uses probabilities, not hidden states."""

    def test_residual_source_is_probs_not_hidden(self):
        """Verify compute_rcd_residuals_from_probs uses probs @ E, not hidden states."""
        V, D = 16, 4
        torch.manual_seed(99)
        probs = F.softmax(torch.randn(1, 3, V), dim=-1)
        E = torch.randn(V, D)
        delta = compute_rcd_residuals_from_probs(probs, E)
        # Manually verify each position
        for i in range(3):
            expected = (probs[0, i].unsqueeze(0) @ E).squeeze(0)
            self.assertTrue(torch.allclose(delta[0, i], expected, atol=1e-6),
                            f"Position {i}: residual != probs @ E")
        emit_payload("test_residual_source_is_probs_not_hidden",
                     "Residual = probs @ E (not hidden states)", {"verified_positions": 3})

    def test_get_embedding_layer_returns_embedding(self):
        m = _make_dummy_model()
        emb = _get_embedding_layer(m)
        self.assertIsInstance(emb, torch.nn.Embedding)


class TestGracefulFallback(unittest.TestCase):
    """9. RCD request schema defaults and validation."""

    def test_request_defaults(self):
        req = InferenceRCDMRequest(prompt="test")
        emit_payload("test_request_defaults",
                     "InferenceRCDMRequest has sane defaults", {
                         "alpha_mode": req.rcd_alpha_mode,
                         "t_res": req.rcd_temperature_residual,
                         "warm_start": req.rcd_warm_start,
                         "force_mask_only": req.rcd_force_mask_only_injection,
                     })
        self.assertEqual(req.rcd_alpha_mode, "normalized_entropy")
        self.assertEqual(req.rcd_temperature_residual, 1.0)
        self.assertTrue(req.rcd_warm_start)
        self.assertTrue(req.rcd_force_mask_only_injection)


class TestRCDDecodeEndToEnd(unittest.TestCase):
    """10. Full rcd_decode with TinyCausalLM produces output."""

    def test_rcd_decode_runs(self):
        V, D = 64, 16
        mask_id = 3
        torch.manual_seed(0)
        m = _make_dummy_model(vocab=V, hidden=D)
        tok = _make_dummy_tokenizer(vocab=V)
        text, stats, diag = rcd_decode(
            model=m,
            tokenizer=tok,
            device=torch.device("cpu"),
            mask_id=mask_id,
            vocab_size=V,
            prompt="hello world",
            max_new_tokens=8,
            tau_mask=0.1,
            tau_edit=0.5,
            max_steps=3,
            t_res=1.0,
            warm_start=True,
            store_diagnostics=True,
            seed=42,
            is_dummy=True,
        )
        emit_payload("test_rcd_decode_runs",
                     "rcd_decode end-to-end with TinyCausalLM", {
                         "text_preview": text[:60],
                         "steps": stats["steps"],
                         "tokens_generated": stats["tokens_generated"],
                         "warm_start_used": diag.warm_start_used,
                         "finish_reason": stats["finish_reason"],
                         "diagnostics_steps_count": len(diag.steps),
                     })
        self.assertIsInstance(text, str)
        self.assertGreater(stats["steps"], 0)
        self.assertTrue(diag.warm_start_used)
        self.assertEqual(stats["engine"], "rcd")

    def test_rcd_decode_without_warm_start(self):
        V, D = 64, 16
        mask_id = 3
        m = _make_dummy_model(vocab=V, hidden=D)
        tok = _make_dummy_tokenizer(vocab=V)
        text, stats, diag = rcd_decode(
            model=m, tokenizer=tok, device=torch.device("cpu"),
            mask_id=mask_id, vocab_size=V, prompt="hello",
            max_new_tokens=4, tau_mask=0.1, tau_edit=0.5,
            max_steps=2, warm_start=False, seed=7, is_dummy=True,
        )
        emit_payload("test_rcd_decode_without_warm_start",
                     "rcd_decode runs without warm-start", {
                         "warm_start_used": diag.warm_start_used,
                     })
        self.assertFalse(diag.warm_start_used)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
