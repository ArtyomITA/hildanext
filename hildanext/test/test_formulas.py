# Formula tests for LLaDA, LLaDA2.0 and LLaDA2.1.
# Entrypoints: unittest test methods.
# Verifies equation logic used in SAFE implementation.
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
from hildanext.formulas import llada_m2t_loss,llada2_wsd_block,llada21_sets,llada21_apply
from reporting import emit_payload

class FormulaTests(unittest.TestCase):
    def test_llada_m2t_loss_matches_masked_ce(self):
        torch.manual_seed(1)
        logits=torch.randn(1,5,11)
        target=torch.tensor([[1,2,3,4,5]],dtype=torch.long)
        mask=torch.tensor([[0,1,0,1,0]],dtype=torch.bool)
        got=llada_m2t_loss(logits,target,mask)
        labels=torch.full_like(target,-100)
        labels[mask]=target[mask]
        exp=F.cross_entropy(logits[:,:-1,:].reshape(-1,11),labels[:,1:].reshape(-1),ignore_index=-100)
        emit_payload(
            "test_llada_m2t_loss_matches_masked_ce",
            "Checks M2T masked CE implementation against explicit CE computation.",
            {
                "got_loss":float(got.item()),
                "expected_loss":float(exp.item()),
                "abs_diff":float(torch.abs(got-exp).item()),
                "mask_positions":torch.nonzero(mask,as_tuple=False).tolist()
            }
        )
        self.assertTrue(torch.allclose(got,exp,atol=1e-6))
    def test_llada2_wsd_phase_boundaries(self):
        p0,b0=llada2_wsd_block(0,4,3,4,1,64,16)
        p1,b1=llada2_wsd_block(3,4,3,4,1,64,16)
        p2,b2=llada2_wsd_block(5,4,3,4,1,64,16)
        p3,b3=llada2_wsd_block(9,4,3,4,1,64,16)
        emit_payload(
            "test_llada2_wsd_phase_boundaries",
            "Checks WSD warmup/stable/decay phases and block bounds.",
            {"samples":[{"step":0,"phase":p0,"block":b0},{"step":3,"phase":p1,"block":b1},{"step":5,"phase":p2,"block":b2},{"step":9,"phase":p3,"block":b3}]}
        )
        self.assertEqual(p0,"warmup")
        self.assertEqual(p1,"warmup")
        self.assertEqual(p2,"stable")
        self.assertEqual(p3,"decay")
        self.assertTrue(b0<=b1<=64)
        self.assertTrue(16<=b3<=64)
    def test_llada21_gamma_delta_sets(self):
        mask_id=99
        tokens=torch.tensor([[99,5,6,99]],dtype=torch.long)
        pred=torch.tensor([[7,8,6,4]],dtype=torch.long)
        conf=torch.tensor([[0.9,0.8,0.95,0.5]],dtype=torch.float32)
        sets=llada21_sets(tokens,pred,conf,mask_id=mask_id,tau_mask=0.7,tau_edit=0.75)
        self.assertEqual(sets.gamma_count,1)
        self.assertEqual(sets.delta_count,1)
        out,sets2=llada21_apply(tokens,pred,conf,mask_id=mask_id,tau_mask=0.7,tau_edit=0.75)
        emit_payload(
            "test_llada21_gamma_delta_sets",
            "Checks Gamma/Delta set creation and apply behavior.",
            {
                "tokens_before":tokens.tolist(),
                "pred_ids":pred.tolist(),
                "confidence":conf.tolist(),
                "tau_mask":0.7,
                "tau_edit":0.75,
                "gamma_count":sets.gamma_count,
                "delta_count":sets.delta_count,
                "tokens_after":out.tolist()
            }
        )
        self.assertEqual(sets2.gamma_count,1)
        self.assertEqual(int(out[0,0].item()),7)
        self.assertEqual(int(out[0,1].item()),8)

if __name__=="__main__":
    unittest.main()
