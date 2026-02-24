# Tokenizer + mask token checks against local HF model assets.
# Entrypoint: unittest.
# Embedding resize check is skipped if real model cannot be loaded.
from __future__ import annotations
from pathlib import Path
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.tokenization import load_tokenizer,ensure_mask_token
from hildanext.inference import load_model_bundle
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_model_dir(p:str)->bool:
    d=Path(p)
    return d.exists() and (d/"config.json").exists()

class TokenizerMaskRealTests(unittest.TestCase):
    def test_mask_token_id_and_embedding_resize(self):
        cfg=load_config(CFG_PATH)
        if not _has_model_dir(cfg.paths.model_dir):
            self.skipTest("model_dir missing")
        tok=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code)
        mask_id=ensure_mask_token(tok,cfg.model.mask_token)
        payload={"model_dir":cfg.paths.model_dir,"mask_token":cfg.model.mask_token,"tokenizer_mask_id":int(mask_id),"tokenizer_vocab_len":int(len(tok))}
        self.assertGreaterEqual(mask_id,0)
        self.assertLess(mask_id,max(1,len(tok)))
        self.assertEqual(int(tok.convert_tokens_to_ids(cfg.model.mask_token)),int(mask_id))
        cfg2=clone_with_updates(cfg,{"runtime":{"force_dummy_model":False,"use_dinfer":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},"train":{"dtype":"float32"}})
        bundle=load_model_bundle(cfg2,for_training=False)
        if bundle.is_dummy:
            emit_payload("test_mask_token_id_and_embedding_resize","Tokenizer check with real model unavailable for embedding-resize verification.",payload|{"bundle_is_dummy":True,"load_error":bundle.load_error})
            self.skipTest(f"embedding check skipped, fallback dummy: {bundle.load_error}")
        mask_id2=ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
        self.assertEqual(int(mask_id2),int(bundle.tokenizer.convert_tokens_to_ids(cfg.model.mask_token)))
        emb=bundle.model.get_input_embeddings() if hasattr(bundle.model,"get_input_embeddings") else None
        if emb is None:
            emit_payload("test_mask_token_id_and_embedding_resize","Tokenizer check with model missing embedding accessor.",payload|{"bundle_is_dummy":False,"has_embedding_accessor":False})
            self.skipTest("model has no input embeddings accessor")
        payload=payload|{"bundle_is_dummy":False,"device":str(bundle.device),"model_vocab":int(emb.num_embeddings),"tokenizer_vocab_after_resize":int(len(bundle.tokenizer))}
        emit_payload("test_mask_token_id_and_embedding_resize","Tokenizer mask and embedding resize consistency check.",payload)
        self.assertEqual(int(len(bundle.tokenizer)),int(emb.num_embeddings))

if __name__=="__main__":
    unittest.main()
