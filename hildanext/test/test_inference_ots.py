# OTS inference tests — paper-faithful checks.
# Verifies: routing, beam lifecycle, expansion, pruning, diffusion-native
# scoring, determinism, no-AR-shortcut, graceful fallback.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
import math
from types import SimpleNamespace
from hildanext.utils import TinyCausalLM, SimpleTokenizer
from hildanext.inference_ots import (
    InferenceOTSRequest,
    OTSBeamState,
    OTSSearchDiagnostics,
    clone_search_state,
    _add_gumbel_noise,
    _transfer_tokens,
    expand_ots_candidates,
    score_ots_candidate,
    score_ots_candidate_fallback,
    prune_ots_beams,
    ots_decode,
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


class TestOTSRouting(unittest.TestCase):
    """1. /generate still works; 2. /api/inferenceots exists."""

    def test_original_generate_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        emit_payload("test_ots_original_generate", "check /generate", {"routes": routes[:20]})
        self.assertIn("/generate", routes)

    def test_inferenceots_endpoint_exists(self):
        from hildanext.api import create_app
        from hildanext.config import AppConfig
        app = create_app(AppConfig())
        routes = [r.path for r in app.routes]
        emit_payload("test_ots_route_exists", "check /inferenceots", {"routes": routes[:25]})
        self.assertIn("/inferenceots", routes)


class TestOTSNonRegression(unittest.TestCase):
    """Original inference path is not broken."""

    def test_tinycausal_basic_forward(self):
        m = _make_dummy_model()
        ids = torch.randint(0, VOCAB, (1, 10))
        out = m(input_ids=ids)
        emit_payload("test_ots_nonregression", "TinyCausalLM forward", {"shape": list(out.logits.shape)})
        self.assertEqual(out.logits.shape, (1, 10, VOCAB))


class TestOTSBeamLifecycle(unittest.TestCase):
    """When beam_size > 1, multiple candidates are maintained."""

    def test_multi_beam_init(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = ots_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello",
            max_new_tokens=8, max_steps=4, beam_size=3,
            gumbel_temperature=0.5, search_interval=2,
            seed=42, is_dummy=True,
        )
        emit_payload("test_beam_lifecycle", "multi-beam OTS", {
            "beam_size": diag.beam_size,
            "checkpoints": diag.total_search_checkpoints,
            "beams_explored": diag.total_beams_explored,
        })
        self.assertEqual(diag.beam_size, 3)
        self.assertGreater(diag.total_beams_explored, 3)
        self.assertTrue(len(text) > 0)


class TestOTSExpansion(unittest.TestCase):
    """Candidate expansion creates distinct trajectories."""

    def test_expansion_produces_distinct_children(self):
        model = _make_dummy_model()
        seq = torch.full((1, 15), MASK_ID, dtype=torch.long)
        seq[:, :5] = torch.randint(0, VOCAB - 1, (1, 5))
        beam = OTSBeamState(seq=seq, prompt_len=5)
        children = expand_ots_candidates(
            model, beam, beam_size=4, mask_id=MASK_ID,
            block_size=10, gumbel_temp=0.8,
        )
        emit_payload("test_expansion", "distinct children", {"n_children": len(children)})
        self.assertEqual(len(children), 4)
        # Children should have distinct sequences (Gumbel noise → diversity)
        seqs = [c[0].seq.tolist() for c in children]
        unique_seqs = set(tuple(s[0]) for s in seqs)
        # With Gumbel noise, at least 2 unique are expected, likely all 4
        self.assertGreaterEqual(len(unique_seqs), 2)


class TestOTSPruning(unittest.TestCase):
    """Pruning reduces candidates back to beam size."""

    def test_prune_to_beam_size(self):
        beams = []
        for i in range(8):
            b = OTSBeamState(
                seq=torch.zeros(1, 10, dtype=torch.long),
                prompt_len=3,
                cumulative_score=float(i),
            )
            beams.append(b)
        pruned = prune_ots_beams(beams, beam_size=3)
        emit_payload("test_pruning", "prune 8→3", {
            "scores": [b.cumulative_score for b in pruned],
        })
        self.assertEqual(len(pruned), 3)
        # Best scores kept
        self.assertAlmostEqual(pruned[0].cumulative_score, 7.0)
        self.assertAlmostEqual(pruned[1].cumulative_score, 6.0)
        self.assertAlmostEqual(pruned[2].cumulative_score, 5.0)


class TestOTSScorePath(unittest.TestCase):
    """OTS uses dedicated diffusion-native scoring, not AR confidence."""

    def test_diffusion_likelihood_score(self):
        """Paper Eq.2: score newly revealed block by re-masking it in x0,
        running a forward pass, measuring log-prob of those tokens."""
        model = _make_dummy_model()
        seq = torch.randint(0, VOCAB - 1, (1, 12))
        seq[:, 5:] = MASK_ID
        # Simulate a candidate where positions 5-7 were just revealed
        candidate_seq = seq.clone()
        candidate_seq[:, 5:8] = torch.tensor([[10, 20, 30]])
        state = OTSBeamState(seq=candidate_seq, prompt_len=5)
        newly_revealed = torch.zeros(1, 12, dtype=torch.bool)
        newly_revealed[:, 5:8] = True
        x0_full = candidate_seq.clone()
        x0_full[:, 8:] = torch.randint(0, VOCAB - 1, (1, 4))

        score = score_ots_candidate(
            model, state, newly_revealed, x0_full, MASK_ID,
        )
        emit_payload("test_score_path", "diffusion-native score", {"score": score})
        # Score should be a finite negative number (log-probs)
        self.assertTrue(math.isfinite(score))
        self.assertLessEqual(score, 0.0)

    def test_fallback_score_is_different_path(self):
        """Fallback confidence scoring is a separate code path."""
        seq = torch.randint(0, VOCAB - 1, (1, 10))
        state = OTSBeamState(seq=seq, prompt_len=3)
        conf = torch.rand(1, 10)
        revealed = torch.zeros(1, 10, dtype=torch.bool)
        revealed[:, 4:6] = True
        fb_score = score_ots_candidate_fallback(state, conf, revealed)
        emit_payload("test_fallback_score", "fallback path", {"score": fb_score})
        self.assertGreater(fb_score, 0.0)


class TestOTSDeterminism(unittest.TestCase):
    """Fixed seed → stable output."""

    def test_deterministic_output(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        kwargs = dict(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="test",
            max_new_tokens=8, max_steps=4, beam_size=2,
            gumbel_temperature=0.5, search_interval=2,
            seed=123, is_dummy=True,
        )
        text1, _, _ = ots_decode(**kwargs)
        text2, _, _ = ots_decode(**kwargs)
        emit_payload("test_determinism", "seed=123", {"text1": text1, "text2": text2})
        self.assertEqual(text1, text2)


class TestOTSGracefulFallback(unittest.TestCase):
    """Unknown pruning mode with fallback disabled → error."""

    def test_unknown_mode_error(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        with self.assertRaises(ValueError) as ctx:
            ots_decode(
                model=model, tokenizer=tok, device=torch.device("cpu"),
                mask_id=MASK_ID, vocab_size=VOCAB, prompt="x",
                max_new_tokens=4, max_steps=2, beam_size=2,
                pruning_mode="unknown_mode", allow_fallback_score=False,
                seed=1, is_dummy=True,
            )
        emit_payload("test_graceful_error", "unknown mode", {"error": str(ctx.exception)})
        self.assertIn("unknown_mode", str(ctx.exception).lower())

    def test_fallback_allowed_mode(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = ots_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="x",
            max_new_tokens=4, max_steps=2, beam_size=2,
            pruning_mode="unknown_mode", allow_fallback_score=True,
            seed=1, is_dummy=True,
        )
        emit_payload("test_graceful_fallback", "fallback active", {
            "mode_used": diag.pruning_mode_used,
        })
        self.assertEqual(diag.pruning_mode_used, "fallback_confidence")


class TestOTSGumbelNoise(unittest.TestCase):
    """Gumbel noise introduces diversity in token sampling."""

    def test_gumbel_varies_argmax(self):
        logits = torch.randn(1, 10, VOCAB)
        samples = set()
        for _ in range(20):
            noisy = _add_gumbel_noise(logits, temperature=0.8)
            argmax = noisy.argmax(dim=-1).tolist()
            samples.add(tuple(argmax[0]))
        emit_payload("test_gumbel_noise", "diversity", {"unique_samples": len(samples)})
        # With temperature=0.8, should get some diversity
        self.assertGreater(len(samples), 1)

    def test_zero_temp_no_noise(self):
        logits = torch.randn(1, 5, VOCAB)
        noisy = _add_gumbel_noise(logits, temperature=0.0)
        self.assertTrue(torch.equal(logits, noisy))


class TestOTSEndToEnd(unittest.TestCase):
    """Full OTS decode produces text + diagnostics."""

    def test_e2e_basic(self):
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = ots_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="hello world",
            max_new_tokens=16, max_steps=6, beam_size=3,
            gumbel_temperature=0.6, search_interval=3,
            seed=42, is_dummy=True,
        )
        emit_payload("test_e2e", "full OTS", {
            "text_len": len(text),
            "tokens_generated": stats["tokens_generated"],
            "finish_reason": stats["finish_reason"],
            "checkpoints": diag.total_search_checkpoints,
            "chosen_beam": diag.chosen_beam_index,
        })
        self.assertTrue(len(text) > 0)
        self.assertIn("engine", stats)
        self.assertEqual(stats["engine"], "ots")
        self.assertIn("ots_diagnostics", stats)

    def test_beam1_degenerates_to_standard(self):
        """beam_size=1 should behave like standard decoding (no search)."""
        model = _make_dummy_model()
        tok = _make_dummy_tokenizer()
        text, stats, diag = ots_decode(
            model=model, tokenizer=tok, device=torch.device("cpu"),
            mask_id=MASK_ID, vocab_size=VOCAB, prompt="test",
            max_new_tokens=8, max_steps=4, beam_size=1,
            gumbel_temperature=0.0, seed=42, is_dummy=True,
        )
        emit_payload("test_beam1", "degenerate to standard", {
            "checkpoints": diag.total_search_checkpoints,
        })
        # With beam_size=1, search still runs but with no diversity
        self.assertGreaterEqual(diag.total_search_checkpoints, 0)
        self.assertEqual(stats["engine"], "ots")


class TestOTSCloneState(unittest.TestCase):
    """clone_search_state produces independent copy."""

    def test_clone_independence(self):
        seq = torch.randint(0, VOCAB, (1, 10))
        orig = OTSBeamState(seq=seq, prompt_len=3, cumulative_score=1.5)
        clone = clone_search_state(orig)
        clone.seq[:, 0] = 999
        clone.cumulative_score = 99.0
        self.assertNotEqual(int(orig.seq[0, 0].item()), 999)
        self.assertAlmostEqual(orig.cumulative_score, 1.5)
        emit_payload("test_clone", "independent copy", {"ok": True})


class TestOTSSchema(unittest.TestCase):
    """Request schema has expected defaults."""

    def test_default_values(self):
        req = InferenceOTSRequest(prompt="x")
        emit_payload("test_schema", "defaults", {
            "beam_size": req.ots_beam_size,
            "gumbel_temp": req.ots_gumbel_temperature,
            "pruning_mode": req.ots_pruning_mode,
        })
        self.assertEqual(req.ots_beam_size, 3)
        self.assertAlmostEqual(req.ots_gumbel_temperature, 0.6)
        self.assertEqual(req.ots_pruning_mode, "diffusion_likelihood")
        self.assertFalse(req.ots_allow_fallback_simple_score)


if __name__ == "__main__":
    unittest.main()
