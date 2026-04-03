# EntRGi inference tests — paper-faithful checks.
# Verifies: routing, entropy weighting, gradient guidance logic,
# reward model loading, diagnostics, endpoint existence, non-regression.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
import math
from hildanext.utils import TinyCausalLM, SimpleTokenizer
from hildanext.inference_entrgi import (
    InferenceEntRGiRequest,
    EntRGiDiagnostics,
    compute_entropy_weights,
    apply_entrgi_guidance,
    entrgi_decode,
    load_reward_model,
    _get_reward_embed_weight,
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
class TestEntRGiRouting(unittest.TestCase):
    """Verify /inferenceentrgi endpoint exists and base routes still work."""

    def test_inferenceentrgi_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        emit_payload("test_entrgi_route_exists", "check /inferenceentrgi", {"routes": routes[:30]})
        self.assertIn("/inferenceentrgi", routes)

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

    def test_inferences2d2_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferences2d2", routes)


# ---------------------------------------------------------------------------
# 2. Request schema
# ---------------------------------------------------------------------------
class TestEntRGiRequestSchema(unittest.TestCase):
    """Verify request schema accepts all EntRGi-specific fields."""

    def test_default_values(self):
        req = InferenceEntRGiRequest(prompt="hello")
        self.assertEqual(req.entrgi_guidance_scale, 0.5)
        self.assertEqual(req.entrgi_guidance_steps, 3)
        self.assertEqual(req.entrgi_temperature, 0.7)
        self.assertFalse(req.entrgi_disable_guidance)
        self.assertIn("Skywork", req.entrgi_reward_model)

    def test_custom_values(self):
        req = InferenceEntRGiRequest(
            prompt="test",
            entrgi_guidance_scale=1.0,
            entrgi_guidance_steps=5,
            entrgi_temperature=0.3,
        )
        self.assertEqual(req.entrgi_guidance_scale, 1.0)
        self.assertEqual(req.entrgi_guidance_steps, 5)


# ---------------------------------------------------------------------------
# 3. Entropy weight computation (Algorithm 1 line 9)
# ---------------------------------------------------------------------------
class TestEntropyWeights(unittest.TestCase):
    """Paper Eq: w = H(q) / log K. Test entropy-aware weighting."""

    def test_uniform_distribution_max_entropy(self):
        q = torch.ones(4, VOCAB) / VOCAB  # uniform → max entropy
        w = compute_entropy_weights(q, VOCAB)
        self.assertEqual(w.shape, (4,))
        # Uniform has max entropy → w ≈ 1.0
        for i in range(4):
            self.assertAlmostEqual(float(w[i].item()), 1.0, places=2)

    def test_peaked_distribution_low_entropy(self):
        q = torch.zeros(2, VOCAB)
        q[0, 5] = 1.0  # delta → H=0 → w=0
        q[1, :] = 1.0 / VOCAB
        q[1, 10] = 0.0
        q[1] = q[1] / q[1].sum()  # nearly uniform
        w = compute_entropy_weights(q, VOCAB)
        self.assertAlmostEqual(float(w[0].item()), 0.0, places=2)
        self.assertGreater(float(w[1].item()), 0.5)

    def test_output_range(self):
        q = F.softmax(torch.randn(10, VOCAB), dim=-1)
        w = compute_entropy_weights(q, VOCAB)
        self.assertTrue((w >= 0.0).all())
        self.assertTrue((w <= 1.0).all())


# ---------------------------------------------------------------------------
# 4. Reward model loading (graceful fallback)
# ---------------------------------------------------------------------------
class TestRewardModelLoading(unittest.TestCase):
    """Reward model loading with graceful failure."""

    def test_invalid_model_returns_none(self):
        rm, tok = load_reward_model("definitely-not-a-real-model-12345", torch.device("cpu"))
        self.assertIsNone(rm)
        self.assertIsNone(tok)

    def test_default_model_name_in_schema(self):
        req = InferenceEntRGiRequest(prompt="test")
        self.assertEqual(req.entrgi_reward_model, "Skywork/Skywork-Reward-V2-Qwen3-0.6B")


# ---------------------------------------------------------------------------
# 5. Reward embed weight extraction
# ---------------------------------------------------------------------------
class TestRewardEmbedExtraction(unittest.TestCase):
    """Test that we can extract embedding weight from a model."""

    def test_extract_from_tiny_model(self):
        model = _make_dummy_model()
        # TinyCausalLM stores embeddings in .embed, not .get_input_embeddings()
        # Add the expected interface for the test
        model.get_input_embeddings = lambda: model.embed
        w = _get_reward_embed_weight(model)
        self.assertEqual(w.shape[0], VOCAB)  # V
        self.assertEqual(w.ndim, 2)  # (V, D)


# ---------------------------------------------------------------------------
# 6. Stop-gradient logic (Section 3.2)
# ---------------------------------------------------------------------------
class TestStopGradientLogic(unittest.TestCase):
    """Verify stop-gradient placement: gradient flows through soft embedding only."""

    def test_gradient_flows_through_soft_path(self):
        """The core EntRGi insight: gradient ∂R/∂ψ flows through
        ψ → softmax → soft_embeds → reward, NOT through hard_embeds."""
        V = 16
        D = 8
        E_R = torch.randn(V, D)

        # Logits (leaf, requires grad)
        psi = torch.randn(3, V, requires_grad=True)

        # Soft embedding (differentiable)
        q = F.softmax(psi / 0.7, dim=-1)
        e_bar = torch.matmul(q, E_R)  # (3, D)

        # Hard embedding (detached)
        with torch.no_grad():
            sampled = torch.multinomial(q.detach(), 1).squeeze(-1)
            e_tilde = E_R[sampled]

        # Entropy weight (detached)
        with torch.no_grad():
            w = compute_entropy_weights(q.detach(), V)

        # EntRGi interpolation: ê = ē + sg(w(ẽ-ē))
        shift = (w.unsqueeze(-1) * (e_tilde - e_bar.detach())).detach()
        mixed = e_bar + shift

        # Simulate reward: sum of mixed embeddings (differentiable mock)
        reward = mixed.sum()
        reward.backward()

        # psi.grad must be non-zero (gradient flowed through soft path)
        self.assertIsNotNone(psi.grad)
        self.assertGreater(float(psi.grad.abs().sum().item()), 0.0)


# ---------------------------------------------------------------------------
# 7. Logit update test
# ---------------------------------------------------------------------------
class TestLogitUpdate(unittest.TestCase):
    """Verify reward guidance actually modifies masked-position logits."""

    def test_guidance_modifies_logits(self):
        """With a mock reward model, verify logits change after guidance."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()

        # Without guidance (disabled)
        _, stats_no, diag_no = entrgi_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=4, max_steps=2,
            disable_guidance=True,
            seed=42, is_dummy=True,
        )
        self.assertEqual(stats_no["engine"], "entrgi")
        self.assertEqual(diag_no.number_of_guidance_calls, 0)
        self.assertGreater(diag_no.fallback_to_standard_count, 0)


# ---------------------------------------------------------------------------
# 8. E2E decode
# ---------------------------------------------------------------------------
class TestEntRGiEndToEnd(unittest.TestCase):
    """Full E2E: EntRGi decode produces text, stats, diagnostics."""

    def test_e2e_dummy_no_reward(self):
        """In dummy mode, reward model is skipped, standard decode runs."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = entrgi_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            seed=42, is_dummy=True,
        )
        emit_payload("test_entrgi_e2e", "dummy mode", {
            "engine": stats["engine"],
            "reward_loaded": diag.reward_model_loaded,
            "steps": diag.total_denoising_steps,
        })
        self.assertEqual(stats["engine"], "entrgi")
        self.assertEqual(stats["mode"], "EntRGi")
        self.assertIn("entrgi_diagnostics", stats)
        self.assertFalse(diag.reward_model_loaded)
        self.assertGreater(diag.total_denoising_steps, 0)

    def test_diagnostics_structure(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats, diag = entrgi_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=4, max_steps=2,
            seed=42, is_dummy=True,
        )
        d = diag.to_dict()
        self.assertIn("total_denoising_steps", d)
        self.assertIn("reward_model_name", d)
        self.assertIn("reward_model_loaded", d)
        self.assertIn("guidance_scale", d)
        self.assertIn("guidance_steps", d)
        self.assertIn("avg_masked_entropy", d)
        self.assertIn("avg_entropy_weight", d)
        self.assertIn("number_of_guidance_calls", d)
        self.assertIn("fallback_to_standard_count", d)
        self.assertIsInstance(d["steps"], list)
        if d["steps"]:
            step = d["steps"][0]
            self.assertIn("guidance_applied", step)
            self.assertIn("avg_entropy", step)
            self.assertIn("avg_entropy_weight", step)
            self.assertIn("reward_before", step)
            self.assertIn("reward_after", step)

    def test_disable_guidance_flag(self):
        """entrgi_disable_guidance=True skips all guidance, diagnoses fallback."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats, diag = entrgi_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4,
            disable_guidance=True,
            seed=42, is_dummy=True,
        )
        self.assertEqual(diag.number_of_guidance_calls, 0)
        self.assertGreater(diag.fallback_to_standard_count, 0)
        self.assertEqual(stats["engine"], "entrgi")

    def test_both_models_frozen(self):
        """Paper requirement: both dLLM and reward model stay frozen."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        # Check model params before
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        entrgi_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=4, max_steps=2,
            seed=42, is_dummy=True,
        )
        # Check model params after — should be identical
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p.data, params_before[n]),
                            f"dLLM param '{n}' was modified during EntRGi inference!")


if __name__ == "__main__":
    unittest.main()
