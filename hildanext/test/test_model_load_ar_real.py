# Real model load + AR generation smoke for deterministic greedy path.
# Entrypoint: unittest.
# Skips cleanly if local HF model folder is missing or load fails.
from __future__ import annotations
from pathlib import Path
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.inference import load_model_bundle
from hildanext.ar import generate_ar
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_model_dir(p:str)->bool:
    d=Path(p)
    if not d.exists():
        return False
    marks=["model.safetensors","pytorch_model.bin","config.json","tokenizer.json"]
    return any((d/m).exists() for m in marks)

class RealModelARTests(unittest.TestCase):
    def test_model_load_and_ar_greedy_determinism(self):
        cfg=load_config(CFG_PATH)
        if not _has_model_dir(cfg.paths.model_dir):
            self.skipTest("local model dir not present")
        cfg=clone_with_updates(cfg,{"runtime":{"force_dummy_model":False,"use_dinfer":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},"train":{"dtype":"float32"}})
        bundle=load_model_bundle(cfg,for_training=False)
        if bundle.is_dummy:
            self.skipTest(f"real model unavailable, fallback dummy: {bundle.load_error}")
        dtype="unknown"
        try:
            dtype=str(next(bundle.model.parameters()).dtype)
        except Exception:
            pass
        print(f"[real_ar] device={bundle.device} dtype={dtype} vocab={bundle.vocab_size} mask_id={bundle.mask_id}")
        prompts=[
            "Say one safe sentence.",
            "List three colors.",
            "Q: 2+2? A: 4. Q: 3+3? A:"
        ]
        a1=generate_ar(cfg,prompt=prompts[0],max_new_tokens=12,seed=123)
        a2=generate_ar(cfg,prompt=prompts[0],max_new_tokens=12,seed=123)
        b1=generate_ar(cfg,prompt=prompts[1],max_new_tokens=12,seed=123)
        c1=generate_ar(cfg,prompt=prompts[2],max_new_tokens=12,seed=123)
        emit_payload(
            "test_model_load_and_ar_greedy_determinism",
            "Loads real model and runs deterministic AR generation on multiple prompts.",
            {
                "device":str(bundle.device),
                "dtype":dtype,
                "vocab_size":int(bundle.vocab_size),
                "mask_id":int(bundle.mask_id),
                "prompts":prompts,
                "outputs":{
                    "prompt1_seed123_run1":a1,
                    "prompt1_seed123_run2":a2,
                    "prompt2_seed123":b1,
                    "fewshot_prompt_seed123":c1
                }
            }
        )
        self.assertTrue(bool(a1["text"].strip()))
        self.assertTrue(bool(a2["text"].strip()))
        self.assertTrue(bool(b1["text"].strip()))
        self.assertEqual(a1["text"],a2["text"])

if __name__=="__main__":
    unittest.main()
