# Doc-boundary mask tests for packed sequences.
# Entrypoints: unittest test methods.
# Verifies no cross-document attention leakage.
from __future__ import annotations
import unittest
import torch
from hildanext.masks import doc_attention_mask,batch_doc_attention_mask
from reporting import emit_payload

class MaskTests(unittest.TestCase):
    def test_doc_attention_mask_blocks_cross_doc(self):
        docs=torch.tensor([0,0,1,1,-1],dtype=torch.long)
        m=doc_attention_mask(docs,causal=False)
        emit_payload(
            "test_doc_attention_mask_blocks_cross_doc",
            "Checks no attention across different doc_ids and padded ids.",
            {"doc_ids":docs.tolist(),"mask_matrix":m.int().tolist()}
        )
        self.assertTrue(m[0,1].item())
        self.assertFalse(m[0,2].item())
        self.assertFalse(m[2,0].item())
        self.assertFalse(m[4,0].item())
    def test_batch_doc_attention_mask_shape(self):
        docs=torch.tensor([[0,0,1],[5,5,5]],dtype=torch.long)
        m=batch_doc_attention_mask(docs,causal=True)
        emit_payload(
            "test_batch_doc_attention_mask_shape",
            "Checks batch doc mask shape and causal lower-triangular behavior.",
            {"doc_ids_batch":docs.tolist(),"mask_batch":m.int().tolist()}
        )
        self.assertEqual(tuple(m.shape),(2,3,3))
        self.assertTrue(m[0,1,0].item())
        self.assertFalse(m[0,0,1].item())

if __name__=="__main__":
    unittest.main()
