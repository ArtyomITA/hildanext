# Remasking invariants for confidence-based remask step.
# Entrypoint: unittest.
# Ensures output domain, low-confidence behavior and count bounds.
from __future__ import annotations
import unittest
import torch
from hildanext.config import RemaskConfig
from hildanext.diffusion import apply_remask
from reporting import emit_payload

class RemaskInvariantTests(unittest.TestCase):
    def test_output_domain_and_shape(self):
        mask_id=99
        x=torch.tensor([[1,2,3,99,5,6,7,8]],dtype=torch.long)
        conf=torch.tensor([[0.8,0.2,0.4,0.1,0.9,0.3,0.7,0.6]],dtype=torch.float32)
        cfg=RemaskConfig(target_ratio=0.25,min_ratio=0.1,block_size=4,block_stride=2,percentile_safety=0.95)
        y=apply_remask(x,conf,mask_id=mask_id,cfg=cfg)
        emit_payload(
            "test_output_domain_and_shape",
            "Checks remask output domain and shape consistency.",
            {"tokens_before":x.tolist(),"confidence":conf.tolist(),"tokens_after":y.tolist(),"mask_id":mask_id}
        )
        self.assertEqual(tuple(y.shape),tuple(x.shape))
        vals=set(int(v) for v in y.view(-1).tolist())
        allowed=set(int(v) for v in x.view(-1).tolist())|{mask_id}
        self.assertTrue(vals.issubset(allowed))
    def test_low_confidence_prefers_remask(self):
        mask_id=99
        x=torch.tensor([[10,11,12,13,14,15]],dtype=torch.long)
        conf=torch.tensor([[0.01,0.02,0.99,0.98,0.97,0.96]],dtype=torch.float32)
        cfg=RemaskConfig(target_ratio=0.34,min_ratio=0.1,block_size=4,block_stride=2,percentile_safety=0.95)
        y=apply_remask(x,conf,mask_id=mask_id,cfg=cfg)
        emit_payload(
            "test_low_confidence_prefers_remask",
            "Checks low-confidence positions are remasked first.",
            {"tokens_before":x.tolist(),"confidence":conf.tolist(),"tokens_after":y.tolist(),"mask_id":mask_id}
        )
        self.assertEqual(int(y[0,0].item()),mask_id)
        self.assertEqual(int(y[0,1].item()),mask_id)
    def test_remask_count_bound(self):
        mask_id=99
        x=torch.tensor([[1,2,3,4,5,6,99,8,9,10]],dtype=torch.long)
        conf=torch.tensor([[0.9,0.8,0.2,0.1,0.3,0.7,0.4,0.5,0.6,0.95]],dtype=torch.float32)
        cfg=RemaskConfig(target_ratio=0.3,min_ratio=0.1,block_size=4,block_stride=2,percentile_safety=0.95)
        y=apply_remask(x,conf,mask_id=mask_id,cfg=cfg)
        cand=int((x!=mask_id).sum().item())
        k=max(int(cand*cfg.min_ratio),int(cand*cfg.target_ratio))
        existing=int((x==mask_id).sum().item())
        total_after=int((y==mask_id).sum().item())
        emit_payload(
            "test_remask_count_bound",
            "Checks remask count upper bound against configured ratio cap.",
            {"candidate_count":cand,"k_bound":k,"mask_before":existing,"mask_after":total_after}
        )
        self.assertLessEqual(total_after,existing+k)

if __name__=="__main__":
    unittest.main()
