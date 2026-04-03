from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from hildanext.inference_entrgi import (
    InferenceEntRGiRequest,
    build_reward_id_map,
    compute_entropy_weights,
    entrgi_decode,
    load_reward_model,
    select_lowest_entropy_mask_positions,
    _get_reward_embed_weight,
)
from hildanext.utils import SimpleTokenizer, TinyCausalLM


MASK_ID = 63
VOCAB = 64


def _make_dummy_model(vocab: int = VOCAB, hidden: int = 16):
    model = TinyCausalLM(vocab_size=vocab, hidden_size=hidden)
    model.eval()
    return model


def _make_dummy_tokenizer(vocab: int = VOCAB):
    return SimpleTokenizer(vocab_size=vocab)


class _AlignedTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def convert_ids_to_tokens(self, idx: int) -> str:
        return f"tok{idx}"

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(token.replace("tok", ""))

    def get_vocab(self):
        return {f"tok{i}": i for i in range(self.vocab_size)}


class _ShiftedTokenizer(_AlignedTokenizer):
    def get_vocab(self):
        return {f"tok{i}": (i + 1 if i + 1 < self.vocab_size else 0) for i in range(self.vocab_size)}

    def convert_tokens_to_ids(self, token: str) -> int:
        idx = int(token.replace("tok", ""))
        return idx + 1 if idx + 1 < self.vocab_size else 0


class TestEntRGiRequestSchema(unittest.TestCase):
    def test_default_values(self):
        req = InferenceEntRGiRequest(prompt="hello")
        self.assertEqual(req.entrgi_guidance_scale, 0.5)
        self.assertEqual(req.entrgi_guidance_steps, 3)
        self.assertEqual(req.entrgi_temperature, 0.7)
        self.assertFalse(req.entrgi_disable_guidance)
        self.assertIn("Skywork", req.entrgi_reward_model)


class TestEntropyWeights(unittest.TestCase):
    def test_uniform_distribution_has_weight_one(self):
        q = torch.ones(4, VOCAB) / VOCAB
        w = compute_entropy_weights(q, VOCAB)
        self.assertTrue(torch.allclose(w, torch.ones_like(w), atol=1e-2))

    def test_delta_distribution_has_weight_zero(self):
        q = torch.zeros(2, VOCAB)
        q[:, 5] = 1.0
        w = compute_entropy_weights(q, VOCAB)
        self.assertTrue(torch.allclose(w, torch.zeros_like(w), atol=1e-2))


class TestLowestEntropySelection(unittest.TestCase):
    def test_budget_uses_remaining_steps(self):
        q = torch.full((5, 8), 1 / 8.0)
        chosen, budget, _, _ = select_lowest_entropy_mask_positions(q, remaining_steps=2)
        self.assertEqual(budget, 3)
        self.assertEqual(int(chosen.numel()), 3)

    def test_prefers_low_entropy_positions(self):
        q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=torch.float32,
        )
        chosen, budget, _, _ = select_lowest_entropy_mask_positions(q, remaining_steps=3)
        self.assertEqual(budget, 1)
        self.assertEqual(int(chosen[0].item()), 0)


class TestRewardAlignment(unittest.TestCase):
    def test_identity_alignment(self):
        tok = _AlignedTokenizer(VOCAB)
        mapping, mode = build_reward_id_map(tok, tok, VOCAB, VOCAB)
        self.assertEqual(mode, "identity")
        self.assertTrue(torch.equal(mapping, torch.arange(VOCAB)))

    def test_misaligned_tokenizer_fails(self):
        base = _AlignedTokenizer(VOCAB)
        reward = _ShiftedTokenizer(VOCAB)
        mapping, mode = build_reward_id_map(base, reward, VOCAB, VOCAB)
        self.assertIsNone(mapping)
        self.assertIn(mode, {"token_roundtrip_mismatch", "reward_token_missing"})


class TestRewardLoading(unittest.TestCase):
    def test_invalid_model_returns_none(self):
        rm, tok = load_reward_model("definitely-not-a-real-model-12345", torch.device("cpu"))
        self.assertIsNone(rm)
        self.assertIsNone(tok)

    def test_extract_embed_weight(self):
        model = _make_dummy_model()
        model.get_input_embeddings = lambda: model.embed
        weight = _get_reward_embed_weight(model)
        self.assertEqual(tuple(weight.shape), (VOCAB, 16))


class TestEntRGiEndToEnd(unittest.TestCase):
    def test_dummy_decode_runs_without_reward_model(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = entrgi_decode(
            model=model,
            tokenizer=tok,
            device=torch.device("cpu"),
            mask_id=MASK_ID,
            vocab_size=VOCAB,
            prompt="hello",
            max_new_tokens=8,
            max_steps=4,
            seed=42,
            is_dummy=True,
        )
        self.assertEqual(stats["engine"], "entrgi")
        self.assertEqual(stats["mode"], "EntRGi")
        self.assertIn("entrgi_diagnostics", stats)
        self.assertFalse(diag.reward_model_loaded)
        self.assertEqual(diag.tokenizer_alignment_mode, "guidance_disabled")
        self.assertGreater(diag.total_denoising_steps, 0)
        self.assertIsInstance(text, str)

    def test_diagnostics_payload_updated(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats, diag = entrgi_decode(
            model=model,
            tokenizer=tok,
            device=torch.device("cpu"),
            mask_id=MASK_ID,
            vocab_size=VOCAB,
            prompt="hello",
            max_new_tokens=4,
            max_steps=2,
            seed=7,
            is_dummy=True,
        )
        payload = diag.to_dict()
        self.assertIn("reward_tokenizer_aligned", payload)
        self.assertIn("tokenizer_alignment_mode", payload)
        self.assertIn("selection_policy_used", payload)
        self.assertIn("avg_selected_entropy", payload)
        self.assertIn("selection_policy_used", stats)
        self.assertIn("avg_selected_entropy", stats)
        if payload["steps"]:
            step = payload["steps"][0]
            self.assertIn("selection_budget", step)
            self.assertIn("selected_count", step)
            self.assertIn("avg_selected_entropy", step)

    def test_disable_guidance_keeps_entropy_ranked_decode(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        _, stats, diag = entrgi_decode(
            model=model,
            tokenizer=tok,
            device=torch.device("cpu"),
            mask_id=MASK_ID,
            vocab_size=VOCAB,
            prompt="hello",
            max_new_tokens=6,
            max_steps=3,
            disable_guidance=True,
            seed=123,
            is_dummy=True,
        )
        self.assertEqual(diag.number_of_guidance_calls, 0)
        self.assertGreater(diag.fallback_to_standard_count, 0)
        self.assertEqual(stats["selection_policy_used"], "lowest_entropy_budget")

    def test_models_stay_frozen(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        entrgi_decode(
            model=model,
            tokenizer=tok,
            device=torch.device("cpu"),
            mask_id=MASK_ID,
            vocab_size=VOCAB,
            prompt="hello",
            max_new_tokens=4,
            max_steps=2,
            seed=9,
            is_dummy=True,
        )
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, params_before[name]), name)


# ---------------------------------------------------------------------------
# Route registration & non-regression
# ---------------------------------------------------------------------------
class TestEntRGiRouting(unittest.TestCase):
    """Verify /inferenceentrgi endpoint exists and base routes still work."""

    def test_inferenceentrgi_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferenceentrgi", routes)

    def test_original_generate_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/generate", routes)

    def test_inferences2d2_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferences2d2", routes)

    def test_inferenceots_still_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        self.assertIn("/inferenceots", routes)


# ---------------------------------------------------------------------------
# Stop-gradient logic (Section 3.2 of paper)
# ---------------------------------------------------------------------------
class TestStopGradientLogic(unittest.TestCase):
    """Verify gradient flows through soft embedding only (paper Sec 3.2)."""

    def test_gradient_flows_through_soft_path(self):
        V = 16
        D = 8
        E_R = torch.randn(V, D)

        psi = torch.randn(3, V, requires_grad=True)
        q = F.softmax(psi / 0.7, dim=-1)
        e_bar = torch.matmul(q, E_R)

        with torch.no_grad():
            sampled = torch.multinomial(q.detach(), 1).squeeze(-1)
            e_tilde = E_R[sampled]
            w = compute_entropy_weights(q.detach(), V)

        shift = (w.unsqueeze(-1) * (e_tilde - e_bar.detach())).detach()
        mixed = e_bar + shift

        reward = mixed.sum()
        reward.backward()

        self.assertIsNotNone(psi.grad)
        self.assertGreater(float(psi.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
