# T2T corruption correctness and finite recovery loss checks.
# Entrypoint: unittest.
# Validates perturb ratio, labels, and forward loss finiteness.
from __future__ import annotations
import math
import unittest
import torch
import torch.nn.functional as F
from hildanext.diffusion import t2t_corrupt_tokens
from hildanext.utils import TinyCausalLM
from reporting import emit_payload

class T2TCorruptionTests(unittest.TestCase):
    def test_t2t_corruption_and_recovery(self):
        torch.manual_seed(5)
        b,s,v=2,40,256
        ids=torch.randint(0,v,(b,s),dtype=torch.long)
        attn=torch.ones((b,s),dtype=torch.long)
        resp=torch.ones((b,s),dtype=torch.long)
        p_edit=0.25
        noisy,labels,mask=t2t_corrupt_tokens(ids,attn,resp,ratio=p_edit,vocab_size=v)
        perturbed=int(mask.sum().item())
        total=int(attn.sum().item())
        ratio=float(perturbed/max(1,total))
        self.assertGreater(ratio,0.05)
        self.assertLess(ratio,0.45)
        self.assertTrue(torch.equal(labels[mask],ids[mask]))
        self.assertTrue(torch.all(labels[~mask].eq(-100)).item())
        model=TinyCausalLM(vocab_size=v,hidden_size=64)
        out=model(input_ids=noisy).logits
        loss=F.cross_entropy(out[:,:-1,:].reshape(-1,v),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertTrue(math.isfinite(float(loss.item())))
        emit_payload(
            "test_t2t_corruption_and_recovery",
            "Checks T2T corruption mask ratio and finite recovery loss.",
            {
                "p_edit":p_edit,
                "perturbed":perturbed,
                "total":total,
                "ratio":ratio,
                "loss":float(loss.item())
            }
        )

if __name__=="__main__":
    unittest.main()
