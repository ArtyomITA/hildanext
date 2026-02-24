# WSD ladder traversal and divisibility enforcement checks.
# Entrypoint: unittest.
# Validates warmup ladder, stable MDLM block, and decay behavior.
from __future__ import annotations
import unittest
from hildanext.config import WSDConfig
from hildanext.diffusion import wsd_block
from hildanext.formulas import llada2_wsd_block
from reporting import emit_payload

class WSDLadderTests(unittest.TestCase):
    def test_wsd_ladder_and_divisibility(self):
        seq_len=256
        cfg=WSDConfig(
            warmup_steps=5,
            stable_steps=3,
            decay_steps=4,
            start_block_size=1,
            max_block_size=256,
            end_block_size=32,
            ladder_blocks=[1,4,32,64,256],
            enforce_divisibility=True
        )
        samples=[]
        for step in range(0,12):
            x=wsd_block(step,cfg,seq_len=seq_len)
            self.assertGreaterEqual(int(x.block_size),1)
            self.assertEqual(seq_len%int(x.block_size),0)
            samples.append({"step":step,"phase":x.phase,"block":x.block_size})
        warmup=[x["block"] for x in samples if x["phase"]=="warmup"]
        self.assertTrue(all(warmup[i]<=warmup[i+1] for i in range(max(0,len(warmup)-1))))
        stable=[x["block"] for x in samples if x["phase"]=="stable"]
        self.assertTrue(all(int(x)==seq_len for x in stable))
        decay=[x["block"] for x in samples if x["phase"]=="decay"]
        if len(decay)>=2:
            self.assertGreaterEqual(decay[0],decay[-1])
        p,b=llada2_wsd_block(7,5,3,4,1,256,32,seq_len=seq_len,ladder_blocks=[1,4,32,64,256],enforce_divisibility=True)
        self.assertIn(p,{"stable","decay"})
        self.assertEqual(seq_len%int(b),0)
        emit_payload(
            "test_wsd_ladder_and_divisibility",
            "Checks WSD ladder progression and block-size divisibility by seq_len.",
            {"seq_len":seq_len,"samples":samples}
        )

if __name__=="__main__":
    unittest.main()
