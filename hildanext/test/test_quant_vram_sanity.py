# Quant bench sanity tests for non-crash and report schema checks.
# Entrypoint: unittest.
# Allows unavailable quant modes with ok:false and structured reason.
from __future__ import annotations
from pathlib import Path
import json
import math
import tempfile
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.quant import run_quant_bench
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_model_dir(p:str)->bool:
    d=Path(p)
    return d.exists() and (d/"config.json").exists()

def _finite_or_none(x):
    if x is None:
        return True
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

class QuantVRAMSanityTests(unittest.TestCase):
    def test_quant_bench_report_schema(self):
        cfg=load_config(CFG_PATH)
        if not _has_model_dir(cfg.paths.model_dir):
            self.skipTest("model_dir missing")
        cfg=clone_with_updates(cfg,{"runtime":{"device":"cuda" if torch.cuda.is_available() else "cpu","use_dinfer":False}})
        with tempfile.TemporaryDirectory(prefix="hildanext_quant_") as td:
            out_json=Path(td)/"quant_vram.json"
            rep=run_quant_bench(cfg,modes=["fp16","nf4","int8"],prompt="Write one short safe line.",max_new_tokens=8,engine_name="transformers",seed=11,out_json=out_json,train_probe=True)
            self.assertTrue(out_json.exists())
            data=json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("results",data)
            self.assertGreaterEqual(len(data["results"]),1)
            for row in data["results"]:
                for k in ["mode","device","dtype","ok","elapsed","tokens_per_sec","peak_mem"]:
                    self.assertIn(k,row)
                self.assertTrue(_finite_or_none(row.get("elapsed")))
                tps=row.get("tokens_per_sec",{})
                self.assertTrue(_finite_or_none(tps.get("ar")))
                self.assertTrue(_finite_or_none(tps.get("dllm")))
                self.assertTrue(_finite_or_none(row.get("peak_mem")))
                self.assertIn("train_probe",row)
                tp=row.get("train_probe")
                if tp is not None:
                    self.assertIn("ok",tp)
                    self.assertTrue(_finite_or_none(tp.get("elapsed")))
                    self.assertTrue(_finite_or_none(tp.get("peak_mem")))
            emit_payload(
                "test_quant_bench_report_schema",
                "Runs quant bench and validates metric schema/finite values.",
                {"device":cfg.runtime.device,"modes":["fp16","nf4","int8"],"results":data["results"]}
            )

if __name__=="__main__":
    unittest.main()
