# Real dLLM decode invariants on local Qwen model.
# Entrypoint: unittest.
# Fails on degenerate decoding when real model is available.
from __future__ import annotations
from pathlib import Path
import math
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.inference import build_engine
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_model(model_dir:str)->bool:
    p=Path(model_dir)
    return p.exists() and (p/"config.json").exists() and any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

class RealDLLMDecodeTests(unittest.TestCase):
    def test_real_decode_non_degenerate(self):
        cfg=load_config(CFG_PATH)
        if not _has_model(cfg.paths.model_dir):
            self.skipTest("local model dir missing")
        cfg=clone_with_updates(cfg,{
            "runtime":{"use_dinfer":False,"force_dummy_model":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},
            "inference":{
                "strict_decode_invariants":True,
                "allow_tau_fallback_on_degenerate":False,
                "max_steps":8,
                "s_mode_tau_mask":0.08,
                "s_mode_tau_edit":0.08
            }
        })
        eng=build_engine(cfg)
        prompts=["Write one sentence about rain.","Q: 5+7? A:","Complete safely: The quick brown fox"]
        rows=[]
        for i,p in enumerate(prompts):
            text=eng.generate(prompt=p,mode="S_MODE",max_new_tokens=16,seed=100+i)
            st=eng.last_stats
            logs=st.get("logs",[])
            self.assertFalse(bool(st.get("dummy_model",True)))
            self.assertTrue(bool(text.strip()))
            self.assertNotEqual(text.strip(),"dummy-output")
            self.assertGreater(len(logs),0)
            self.assertTrue(any(int(x.get("gamma_count",0))>0 for x in logs))
            dec=any(float(logs[j]["mask_ratio"])<float(logs[j-1]["mask_ratio"]) for j in range(1,len(logs)))
            self.assertTrue(dec)
            for x in logs:
                a=x.get("avg_conf_masked")
                b=x.get("avg_conf_tokens")
                if a is not None:
                    self.assertTrue(math.isfinite(float(a)))
                if b is not None:
                    self.assertTrue(math.isfinite(float(b)))
            rows.append({"prompt":p,"text":text,"stats":st})
        emit_payload(
            "test_real_decode_non_degenerate",
            "Runs real decode on multiple prompts and checks non-degenerate Gamma/Delta dynamics.",
            {"rows":rows}
        )
        eng.close()

if __name__=="__main__":
    unittest.main()
