# AR generation tests on fallback runtime.
# Entrypoints: unittest test methods.
# Validates AR output path for model sanity checks.
from __future__ import annotations
from pathlib import Path
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.ar import generate_ar
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
SMOKE_CFG=ROOT/"runs"/"configs"/"smoke.json"

class ARTests(unittest.TestCase):
    def test_ar_generation_dummy(self):
        cfg=load_config(SMOKE_CFG)
        cfg=clone_with_updates(cfg,{"runtime":{"force_dummy_model":True,"device":"cuda" if torch.cuda.is_available() else "cpu"}})
        prompt="Hello AR baseline."
        out=generate_ar(cfg,prompt=prompt,max_new_tokens=12,seed=3)
        emit_payload(
            "test_ar_generation_dummy",
            "Dummy AR baseline generation on CUDA-first runtime.",
            {
                "prompt":prompt,
                "max_new_tokens":12,
                "seed":3,
                "result":out
            }
        )
        self.assertIn("text",out)
        self.assertTrue(bool(out["text"].strip()))
        self.assertGreaterEqual(out["tokens_generated"],1)

if __name__=="__main__":
    unittest.main()
