# Fallback tracing invariants for strict reproducibility.
# Entrypoint: unittest.
# Forces dInfer-missing path and validates JSONL+stats evidence.
from __future__ import annotations
from pathlib import Path
import json
import tempfile
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.inference import build_engine
from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _read_jsonl(p:Path):
    rows=[]
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

class FallbackTracingMandatoryTests(unittest.TestCase):
    def test_dinfer_missing_is_traced_in_logs_and_stats(self):
        base=load_config(CFG_PATH)
        with tempfile.TemporaryDirectory(prefix="hildanext_trace_") as td:
            t=Path(td)
            cfg=clone_with_updates(base,{
                "paths":{"logs_dir":str(t/"logs"),"vendor_dinfer":str(t/"missing_dinfer")},
                "runtime":{"use_dinfer":True,"strict_fallbacks":False,"force_dummy_model":True,"device":"cuda" if torch.cuda.is_available() else "cpu"}
            })
            tr=trace_from_cfg(cfg)
            tk=set_active_trace(tr)
            try:
                eng=build_engine(cfg,trace=tr)
                out=eng.generate(prompt="hello",mode="S_MODE",max_new_tokens=8,seed=7)
                stats=dict(eng.last_stats or {})
                eng.close()
            finally:
                tr.flush()
                reset_active_trace(tk)
            fb_path=Path(cfg.paths.logs_dir)/"fallbacks.jsonl"
            self.assertTrue(fb_path.exists())
            rows=_read_jsonl(fb_path)
            self.assertGreater(len(rows),0)
            dinfer=[r for r in rows if str(r.get("reason",""))=="dinfer_missing"]
            self.assertGreater(len(dinfer),0)
            row=dinfer[-1]
            for k in ["run_id","ts_utc","module","func","event_type","action","reason","extra"]:
                self.assertIn(k,row)
            self.assertIn("fallbacks",stats)
            self.assertTrue(any(str(x.get("reason",""))=="dinfer_missing" for x in stats.get("fallbacks",[])))
            emit_payload(
                "test_dinfer_missing_is_traced_in_logs_and_stats",
                "Forces dinfer-missing fallback and checks JSONL+engine stats trace payload.",
                {"sample_text":out,"engine":eng.name,"fallback_row":row,"stats_fallback_tail":stats.get("fallbacks",[])[-4:]}
            )

if __name__=="__main__":
    unittest.main()
