# Doc-gating tests for LLaDA2.0 composite attention mask.
# Entrypoint: unittest.
# Verifies cross-doc and padding positions are always blocked.
from __future__ import annotations
import unittest
import torch
from hildanext.masks import batch_doc_attention_mask
from reporting import emit_payload

class LLaDA20CompositeMaskDocGatingTests(unittest.TestCase):
    def test_doc_gating_with_padding(self):
        l=16
        bsize=4
        base=torch.tensor([0,0,0,0,0,0,1,1,1,1,1,1,-1,-1,-1,-1],dtype=torch.long)
        docs=torch.cat([base,base],dim=0).unsqueeze(0)
        m=batch_doc_attention_mask(docs,causal=False,mask_mode="composite_llada20",block_size=bsize,base_len=l)[0]
        valid=docs[0].ge(0)
        for i in range(2*l):
            for j in range(2*l):
                di=int(docs[0,i].item())
                dj=int(docs[0,j].item())
                if di<0 or dj<0 or di!=dj:
                    self.assertFalse(bool(m[i,j].item()),msg=f"cross-doc/pad leakage at ({i},{j}) di={di} dj={dj}")
        self.assertFalse(bool(m[0,6].item()))
        self.assertFalse(bool(m[6,0].item()))
        self.assertTrue(bool(valid[:l].sum().item())>0)
        self.assertTrue(bool(valid[l:].sum().item())>0)
        emit_payload(
            "test_doc_gating_with_padding",
            "Checks composite_llada20 doc gating with two docs + padding sentinel -1.",
            {"shape":[2*l,2*l],"total_true":int(m.sum().item()),"cross_doc_checks":"passed","pad_blocked":"passed"}
        )

if __name__=="__main__":
    unittest.main()
