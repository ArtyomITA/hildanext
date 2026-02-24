# Optional dInfer one-step parity check against local threshold apply.
# Entrypoint: unittest.
# Runs only when HILDANEXT_ENABLE_DINFER_PARITY=1 and dInfer internals are importable.
from __future__ import annotations
from pathlib import Path
import os
import sys
import unittest
import torch
from hildanext.formulas import llada21_apply
from hildanext.config import load_config,clone_with_updates
from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace,exception_with_stack
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_VENDOR=ROOT/"vendor"/"dinfer"/"python"
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

class DInferParityTests(unittest.TestCase):
    def test_one_step_threshold_parity(self):
        if os.environ.get("HILDANEXT_ENABLE_DINFER_PARITY","0")!="1":
            self.skipTest("set HILDANEXT_ENABLE_DINFER_PARITY=1 to enable")
        cfg=clone_with_updates(load_config(CFG_PATH),{"runtime":{"strict_fallbacks":False}})
        tr=trace_from_cfg(cfg)
        tk=set_active_trace(tr)
        if not CFG_VENDOR.exists():
            tr.record_fallback(event="fallback",module="test_infer_parity_dinfer",func="test_one_step_threshold_parity",action="skip",reason="dinfer_missing",extra_dict={"engine_requested":"dinfer","path":str(CFG_VENDOR)})
            tr.flush()
            reset_active_trace(tk)
            self.skipTest("vendor/dinfer/python missing")
        sys.path.insert(0,str(CFG_VENDOR))
        try:
            from dinfer.decoding.parallel_strategy import get_transfer_index_threshold
        except Exception as e:
            tr.record_fallback(event="fallback",module="test_infer_parity_dinfer",func="test_one_step_threshold_parity",action="skip",reason="dinfer_missing",exception_str=exception_with_stack(e),extra_dict={"engine_requested":"dinfer"})
            tr.flush()
            reset_active_trace(tk)
            self.skipTest(f"dinfer threshold path unavailable: {e}")
        try:
            torch.manual_seed(11)
            b,l,v=1,12,64
            mask_id=63
            x=torch.randint(0,v-1,(b,l),dtype=torch.long)
            mask_index=torch.zeros((b,l),dtype=torch.bool)
            mask_index[:,3:6]=True
            mask_index[:,8:11]=True
            x[mask_index]=mask_id
            logits=torch.randn(b,l,v,dtype=torch.float32)
            x0,transfer=get_transfer_index_threshold(logits=logits,temperature=0.0,remasking='low_confidence',mask_index=mask_index,x=x.clone(),num_transfer_tokens=None,mask_id=mask_id,threshold=0.0)
            p=torch.softmax(logits.to(torch.float32),dim=-1)
            pred=torch.argmax(logits,dim=-1)
            conf=torch.gather(p,dim=-1,index=pred.unsqueeze(-1)).squeeze(-1)
            local_updated,sets=llada21_apply(tokens=x,pred_ids=pred,confidence=conf,mask_id=mask_id,tau_mask=0.0,tau_edit=2.0)
            dinfer_updated=torch.where(transfer,x0,x)
            self.assertTrue(torch.equal(local_updated,dinfer_updated))
            self.assertEqual(int(sets.gamma_count),int(transfer.sum().item()))
            emit_payload(
                "test_one_step_threshold_parity",
                "Optional dInfer one-step threshold parity against local apply on synthetic tensors.",
                {
                    "shape":[b,l,v],
                    "mask_count":int(mask_index.sum().item()),
                    "local_gamma":int(sets.gamma_count),
                    "dinfer_transfer":int(transfer.sum().item())
                }
            )
        finally:
            tr.flush()
            reset_active_trace(tk)

if __name__=="__main__":
    unittest.main()
