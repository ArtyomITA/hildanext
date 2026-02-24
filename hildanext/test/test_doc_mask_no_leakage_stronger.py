# Stronger doc-boundary attention tests for packed multi-doc batches.
# Entrypoint: unittest.
# Verifies no cross-doc leakage and causal behavior inside docs.
from __future__ import annotations
import unittest
import torch
from hildanext.masks import batch_doc_attention_mask
from reporting import emit_payload

class DocMaskNoLeakageStrongerTests(unittest.TestCase):
    def test_strict_no_cross_doc_leakage(self):
        docs=torch.tensor([[0,0,1,1,2,2,-1,-1],[4,4,4,5,5,-1,-1,-1]],dtype=torch.long)
        m=batch_doc_attention_mask(docs,causal=False)
        b,s=docs.shape
        for bi in range(b):
            for i in range(s):
                for j in range(s):
                    di=int(docs[bi,i].item())
                    dj=int(docs[bi,j].item())
                    allowed=bool(di>=0 and dj>=0 and di==dj)
                    self.assertEqual(bool(m[bi,i,j].item()),allowed)
        emit_payload(
            "test_strict_no_cross_doc_leakage",
            "Strong no-leakage check across 3-doc packed sequences.",
            {"doc_ids_batch":docs.tolist(),"mask_batch":m.int().tolist()}
        )
    def test_causal_inside_same_doc(self):
        docs=torch.tensor([[0,0,1,1,2,2]],dtype=torch.long)
        m=batch_doc_attention_mask(docs,causal=True)[0]
        s=docs.shape[1]
        for i in range(s):
            for j in range(s):
                di=int(docs[0,i].item())
                dj=int(docs[0,j].item())
                same=di==dj
                if same and j<=i:
                    self.assertTrue(m[i,j].item())
                else:
                    self.assertFalse(m[i,j].item())
        emit_payload(
            "test_causal_inside_same_doc",
            "Causal mask check inside each document segment only.",
            {"doc_ids":docs.tolist(),"causal_mask":m.int().tolist()}
        )

if __name__=="__main__":
    unittest.main()
