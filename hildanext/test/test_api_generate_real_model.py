# API real-model generation invariants for non-degenerate decode.
# Entrypoint: unittest.
# Requires local model dir; skips only when model assets are missing.
from __future__ import annotations
from pathlib import Path
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.api import create_app
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_model(model_dir:str)->bool:
    p=Path(model_dir)
    return p.exists() and (p/"config.json").exists() and any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

class APIRealModelTests(unittest.TestCase):
    def test_api_generate_real_model(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as e:
            self.skipTest(f"fastapi TestClient missing: {e}")
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
        app=create_app(cfg)
        c=TestClient(app)
        h=c.get("/health")
        self.assertEqual(h.status_code,200)
        self.assertFalse(bool(h.json().get("dummy_model",True)))
        req={"prompt":"Write one sentence about stars.","mode":"S_MODE","max_new_tokens":16,"seed":5}
        r=c.post("/generate",json=req)
        self.assertEqual(r.status_code,200,msg=r.text)
        data=r.json()
        text=str(data.get("text",""))
        st=data.get("stats",{})
        logs=st.get("logs",[])
        self.assertFalse(bool(st.get("dummy_model",True)))
        self.assertTrue(bool(text.strip()))
        self.assertNotEqual(text.strip(),"dummy-output")
        self.assertTrue(any(int(x.get("gamma_count",0))>0 for x in logs))
        dec=any(float(logs[i]["mask_ratio"])<float(logs[i-1]["mask_ratio"]) for i in range(1,len(logs)))
        self.assertTrue(dec)
        emit_payload(
            "test_api_generate_real_model",
            "Calls /generate on real model and checks non-degenerate decode trajectory.",
            {"request":req,"response":data}
        )

if __name__=="__main__":
    unittest.main()
