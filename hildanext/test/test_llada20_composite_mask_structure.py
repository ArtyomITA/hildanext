# Structural tests for LLaDA2.0 composite attention mask.
# Entrypoint: unittest.
# Validates xt-xt,xt-x0,x0-x0 and x0->xt forbidden region.
from __future__ import annotations
import unittest
import torch
from hildanext.masks import batch_doc_attention_mask
from reporting import emit_payload

class LLaDA20CompositeMaskStructureTests(unittest.TestCase):
    def test_composite_mask_regions(self):
        l=16
        bsize=4
        docs=torch.tensor([[0]*l+[0]*l],dtype=torch.long)
        m=batch_doc_attention_mask(docs,causal=False,mask_mode="composite_llada20",block_size=bsize,base_len=l)[0]
        self.assertEqual(tuple(m.shape),(2*l,2*l))
        def blk(x:int)->int:
            return x//bsize
        for i in range(2*l):
            for j in range(2*l):
                if i<l and j<l:
                    exp=blk(i)==blk(j)
                elif i<l and j>=l:
                    exp=blk(i)>blk(j-l)
                elif i>=l and j>=l:
                    exp=blk(i-l)>=blk(j-l)
                else:
                    exp=False
                self.assertEqual(bool(m[i,j].item()),bool(exp),msg=f"mismatch at ({i},{j})")
        self.assertFalse(bool(torch.any(m[l:, :l]).item()))
        emit_payload(
            "test_composite_mask_regions",
            "Checks full composite_llada20 region equations on L=16 block=4.",
            {"shape":[2*l,2*l],"xt_xt_true":int(m[:l,:l].sum().item()),"xt_x0_true":int(m[:l,l:].sum().item()),"x0_x0_true":int(m[l:,l:].sum().item()),"x0_to_xt_true":int(m[l:,:l].sum().item())}
        )

if __name__=="__main__":
    unittest.main()
