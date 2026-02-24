# Threshold decode invariants for LLaDA2.1 Gamma/Delta logic.
# Entrypoint: unittest.
# Pure torch tests for llada21_sets and llada21_apply.
from __future__ import annotations
import unittest
import torch
from hildanext.formulas import llada21_sets,llada21_apply
from reporting import emit_payload

class ThresholdDecodeInvariantTests(unittest.TestCase):
    def test_gamma_delta_membership_and_disjointness(self):
        mask_id=99
        tokens=torch.tensor([[99,5,8,99,7,4]],dtype=torch.long)
        pred=torch.tensor([[11,6,8,10,5,4]],dtype=torch.long)
        conf=torch.tensor([[0.91,0.83,0.77,0.88,0.79,0.55]],dtype=torch.float32)
        sets=llada21_sets(tokens,pred,conf,mask_id=mask_id,tau_mask=0.9,tau_edit=0.8)
        emit_payload(
            "test_gamma_delta_membership_and_disjointness",
            "Validates Gamma/Delta set membership and disjointness constraints.",
            {
                "tokens":tokens.tolist(),
                "pred_ids":pred.tolist(),
                "confidence":conf.tolist(),
                "tau_mask":0.9,
                "tau_edit":0.8,
                "gamma_idx":torch.nonzero(sets.gamma,as_tuple=False).tolist(),
                "delta_idx":torch.nonzero(sets.delta,as_tuple=False).tolist()
            }
        )
        self.assertTrue(sets.gamma[0,0].item())
        self.assertFalse(sets.gamma[0,3].item())
        self.assertTrue(sets.delta[0,1].item())
        self.assertFalse(sets.delta[0,2].item())
        self.assertFalse(torch.any(sets.gamma & sets.delta).item())
    def test_apply_semantics(self):
        mask_id=99
        tokens=torch.tensor([[99,5,8,99,7,4]],dtype=torch.long)
        pred=torch.tensor([[11,6,8,10,5,9]],dtype=torch.long)
        conf=torch.tensor([[0.91,0.83,0.77,0.88,0.79,0.55]],dtype=torch.float32)
        out,sets=llada21_apply(tokens,pred,conf,mask_id=mask_id,tau_mask=0.9,tau_edit=0.8)
        emit_payload(
            "test_apply_semantics",
            "Validates token update semantics for Gamma and Delta application.",
            {
                "tokens_before":tokens.tolist(),
                "pred_ids":pred.tolist(),
                "confidence":conf.tolist(),
                "tokens_after":out.tolist(),
                "gamma_count":int(sets.gamma_count),
                "delta_count":int(sets.delta_count)
            }
        )
        self.assertEqual(int(out[0,0].item()),11)
        self.assertEqual(int(out[0,1].item()),6)
        self.assertEqual(int(out[0,2].item()),8)
        self.assertEqual(int(out[0,3].item()),99)
        self.assertEqual(int(out[0,4].item()),7)
        self.assertEqual(int(out[0,5].item()),4)
        self.assertGreaterEqual(sets.gamma_count,1)
    def test_delta_monotonicity_vs_tau_edit(self):
        mask_id=99
        tokens=torch.tensor([[5,6,7,8,9]],dtype=torch.long)
        pred=torch.tensor([[4,6,6,1,3]],dtype=torch.long)
        conf=torch.tensor([[0.95,0.94,0.70,0.86,0.62]],dtype=torch.float32)
        low=llada21_sets(tokens,pred,conf,mask_id=mask_id,tau_mask=0.9,tau_edit=0.6)
        high=llada21_sets(tokens,pred,conf,mask_id=mask_id,tau_mask=0.9,tau_edit=0.85)
        emit_payload(
            "test_delta_monotonicity_vs_tau_edit",
            "Checks that Delta shrinks when tau_edit increases.",
            {"delta_low_tau":int(low.delta_count),"delta_high_tau":int(high.delta_count),"tau_low":0.6,"tau_high":0.85}
        )
        self.assertGreaterEqual(int(low.delta_count),int(high.delta_count))

if __name__=="__main__":
    unittest.main()
