# Left-shift label alignment tests for S0-D.
# Entrypoints: unittest test methods.
# Verifies logits[i] predicts token i+1 and position 0 is never predicted.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
from hildanext.diffusion import _causal_loss
from reporting import emit_payload


class LeftShiftLabelTests(unittest.TestCase):

    # ---- _causal_loss left-shift behaviour ----

    def test_causal_loss_uses_left_shift(self):
        """_causal_loss must use logits[:,:-1] vs labels[:,1:] (standard left-shift)."""
        torch.manual_seed(0)
        B, S, V = 1, 6, 32
        logits = torch.randn(B, S, V)
        labels = torch.tensor([[10, 5, 8, -100, 12, 3]], dtype=torch.long)

        got = _causal_loss(logits, labels)

        # Reference: manual left-shift
        l = logits[:, :-1, :].contiguous()     # positions 0..4 → predict 1..5
        y = labels[:, 1:].contiguous()          # targets at 1..5
        expected = F.cross_entropy(l.view(-1, V), y.view(-1), ignore_index=-100)

        emit_payload(
            "test_causal_loss_uses_left_shift",
            "Verifies _causal_loss implements standard left-shift: logits[i] predicts token[i+1].",
            {
                "got": float(got.item()),
                "expected": float(expected.item()),
                "abs_diff": float(abs(got.item() - expected.item())),
            },
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_position_0_never_predicted(self):
        """Position 0 is never a prediction target (no label at index 0 after shift)."""
        B, S, V = 1, 8, 64
        logits = torch.randn(B, S, V)
        # Put a real label ONLY at position 0, all others -100
        labels = torch.full((B, S), -100, dtype=torch.long)
        labels[0, 0] = 5

        loss = _causal_loss(logits, labels)

        # After left-shift: shifted_labels = labels[:,1:] = all -100
        # So loss should be 0 (no supervised positions)
        emit_payload(
            "test_position_0_never_predicted",
            "Label at position 0 only ⇒ after left-shift, no targets remain ⇒ loss ≈ 0.",
            {"loss": float(loss.item())},
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    def test_position_1_is_first_target(self):
        """Position 1 is the first target: logits[0] predicts labels[1]."""
        B, S, V = 1, 4, 16
        torch.manual_seed(1)
        logits = torch.randn(B, S, V)
        labels = torch.tensor([[-100, 7, -100, -100]], dtype=torch.long)

        got = _causal_loss(logits, labels)

        # shifted: logits[0] → labels[1]=7, rest ignored
        ref_logit = logits[0, 0, :]  # position 0
        ref_target = torch.tensor([7], dtype=torch.long)
        expected = F.cross_entropy(ref_logit.unsqueeze(0), ref_target)

        emit_payload(
            "test_position_1_is_first_target",
            "logits[0] should predict labels[1] = 7; positions 2,3 are -100 after shift.",
            {"got": float(got.item()), "expected": float(expected.item())},
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_short_sequence_returns_zero(self):
        """Sequence of length 1 cannot have left-shift → loss = 0."""
        logits = torch.randn(1, 1, 10)
        labels = torch.tensor([[3]], dtype=torch.long)
        loss = _causal_loss(logits, labels)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    # ---- accuracy shift alignment ----

    def test_accuracy_shift_alignment(self):
        """Masked-token accuracy must use shifted_preds=preds[:,:-1] vs shifted_labels=m2t_y[:,1:]."""
        B, S, V = 1, 8, 32
        torch.manual_seed(2)
        logits = torch.randn(B, S, V)
        preds = logits.argmax(-1)

        # Simulate labels: mask positions 2,4,6
        labels = torch.full((B, S), -100, dtype=torch.long)
        labels[0, 2] = preds[0, 1].item()  # logits[1] predicts position 2 → correct match
        labels[0, 4] = 0                     # logits[3] predicts position 4 → maybe wrong
        labels[0, 6] = preds[0, 5].item()  # logits[5] predicts position 6 → correct match

        # Apply left-shift
        shifted_preds = preds[:, :-1]
        shifted_labels = labels[:, 1:]
        shifted_mask = shifted_labels.ne(-100)

        correct = (shifted_preds[shifted_mask] == shifted_labels[shifted_mask]).float()
        acc = float(correct.mean().item()) if shifted_mask.any() else None
        n_pred_pos = int(shifted_mask.sum().item())

        emit_payload(
            "test_accuracy_shift_alignment",
            "Accuracy should use shifted_preds[:,:-1] vs shifted_labels[:,1:].",
            {"accuracy": acc, "pred_positions_count": n_pred_pos, "shifted_mask_sum": n_pred_pos},
        )
        # We know positions 2 and 6 should be correct (by construction)
        self.assertIsNotNone(acc)
        self.assertEqual(n_pred_pos, 3, "3 label positions (2,4,6) → 3 shifted positions")
        # At least 2 of 3 should be correct
        self.assertGreaterEqual(acc, 0.6)

    def test_pred_positions_count_matches_labels(self):
        """pred_positions_count should equal the number of non-(-100) in shifted_labels."""
        B, S = 1, 16
        labels = torch.full((B, S), -100, dtype=torch.long)
        # Set labels at positions 3, 7, 11
        labels[0, 3] = 10
        labels[0, 7] = 20
        labels[0, 11] = 30

        shifted_labels = labels[:, 1:]
        shifted_mask = shifted_labels.ne(-100)
        count = int(shifted_mask.sum().item())

        emit_payload(
            "test_pred_positions_count_matches_labels",
            "pred_positions_count = count of non-(-100) entries in shifted_labels.",
            {"labels_at": [3, 7, 11], "shifted_count": count},
        )
        self.assertEqual(count, 3)

    def test_no_labels_means_zero_pred_positions(self):
        """If all labels are -100, pred_positions_count = 0."""
        labels = torch.full((1, 10), -100, dtype=torch.long)
        shifted_labels = labels[:, 1:]
        count = int(shifted_labels.ne(-100).sum().item())
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
