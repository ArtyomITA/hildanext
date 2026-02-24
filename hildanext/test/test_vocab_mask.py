# Tokenizer vocab and mask token tests.
# Entrypoints: unittest test methods.
# Ensures mask id is valid and in vocabulary length bounds.
from __future__ import annotations
from pathlib import Path
import unittest
from hildanext.config import load_config,clone_with_updates
from hildanext.tokenization import load_tokenizer,ensure_mask_token
from hildanext.utils import SimpleTokenizer
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
SMOKE_CFG=ROOT/"runs"/"configs"/"smoke.json"

class VocabMaskTests(unittest.TestCase):
    def test_simple_tokenizer_mask(self):
        tok=SimpleTokenizer(vocab_size=64)
        mask_id=ensure_mask_token(tok,"<|mask|>")
        emit_payload(
            "test_simple_tokenizer_mask",
            "Checks mask token id insertion on SimpleTokenizer.",
            {"mask_id":int(mask_id),"vocab_len":int(len(tok))}
        )
        self.assertGreaterEqual(mask_id,0)
        self.assertLess(mask_id,len(tok))
    def test_local_tokenizer_mask(self):
        cfg=load_config(SMOKE_CFG)
        tok=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code)
        mask_id=ensure_mask_token(tok,cfg.model.mask_token)
        emit_payload(
            "test_local_tokenizer_mask",
            "Checks mask token id on local tokenizer assets.",
            {"model_dir":cfg.paths.model_dir,"mask_token":cfg.model.mask_token,"mask_id":int(mask_id),"vocab_len":int(max(1,len(tok)))}
        )
        self.assertGreaterEqual(mask_id,0)
        self.assertLess(mask_id,max(1,len(tok)))

if __name__=="__main__":
    unittest.main()
