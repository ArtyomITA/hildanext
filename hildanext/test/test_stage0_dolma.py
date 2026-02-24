# Stage0 Dolma real-data tests for manifest,prep,tokenized artifacts.
# Entrypoint: unittest.
# Skips cleanly when no real Dolma source is available.
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
from hildanext.config import load_config,clone_with_updates
from hildanext.wsd_stage0 import dolma_manifest,prepare_dolma_only,verify_dolma_only,stream_docs
from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace
try:
    from reporting import emit_payload
except Exception:
    from test.reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"
ALT=Path("E:/DIFFUSION/model2/work/data/dolma_full_1766185610/raw/shard_0000.jsonl")

def _find_real_path(cfg)->str:
    cand=[]
    if ALT.exists():
        cand.append(ALT)
    for d in sorted(Path("E:/DIFFUSION/HildaNext").glob("dolma_v1_6_sample_*")):
        cand.append(d)
    if cfg.data.dolma_path:
        cand.append(Path(cfg.data.dolma_path))
    for p in cand:
        if p.is_file() and p.suffix.lower()==".jsonl":
            try:
                first=p.read_text(encoding="utf-8",errors="ignore").splitlines()[0]
                if "Dolma seed document" in first:
                    continue
            except Exception:
                pass
            return str(p)
        if p.is_dir():
            has_supported=any(x.suffix.lower() in {".jsonl",".txt",".parquet"} or x.name.lower().endswith(".jsonl.gz") or x.name.lower().endswith(".zst") for x in p.rglob("*") if x.is_file())
            if has_supported:
                return str(p)
    return ""

class Stage0DolmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=load_config(CFG_PATH)
        cls.real_path=_find_real_path(base)
        cls._tmp=None
        cls.cfg=None
        cls.prep=None
        cls.verify=None
        cls.events=[]
        if not cls.real_path:
            return
        cls._tmp=tempfile.TemporaryDirectory(prefix="hildanext_stage0_dolma_")
        t=Path(cls._tmp.name)
        cls.cfg=clone_with_updates(base,{
            "data":{"dolma_path":cls.real_path,"tinystories_path":"","max_samples":256,"seq_len":128,"eval_pct_stage0":0.01},
            "paths":{"raw_dir":str(t/"raw"),"processed_dir":str(t/"processed"),"tokenized_dir":str(t/"tokenized"),"logs_dir":str(t/"logs"),"checkpoints_dir":str(t/"ckpt")},
            "runtime":{"strict_fallbacks":False,"device":"cuda"}
        })
        tr=trace_from_cfg(cls.cfg)
        tk=set_active_trace(tr)
        try:
            cls.prep=prepare_dolma_only(cls.cfg,trace=tr)
            cls.verify=verify_dolma_only(cls.cfg,trace=tr)
            cls.events=tr.all_events()
        finally:
            tr.flush()
            reset_active_trace(tk)
        emit_payload("stage0_dolma_setup","Stage0 Dolma setup with manifest/prep/verify payload.",{"real_path":cls.real_path,"prep":cls.prep,"verify":cls.verify,"events":cls.events[-32:]})

    @classmethod
    def tearDownClass(cls):
        if cls._tmp is not None:
            cls._tmp.cleanup()

    def test_dolma_manifest_real_ok(self):
        if not self.real_path:
            self.skipTest("real dolma source missing")
        rep=dolma_manifest(self.cfg)
        emit_payload("test_dolma_manifest_real_ok","Manifest verdict and file stats for Dolma source.",rep)
        self.assertEqual(rep.get("verdict"),"REAL_OK")
        self.assertGreater(int(rep.get("file_count",0)),0)

    def test_no_synthetic_dolma_allowed(self):
        if not self.real_path:
            self.skipTest("real dolma source missing")
        acts=[str((x or {}).get("action","")) for x in self.events]
        self.assertNotIn("synthetic_dolma",acts)

    def test_tokenized_artifacts_exist(self):
        if not self.real_path:
            self.skipTest("real dolma source missing")
        art=((self.prep or {}).get("verify_artifacts") or {})
        self.assertTrue(bool(art.get("ok")))
        self.assertGreater(int(art.get("tokens_shards",0)),0)
        self.assertGreater(int(art.get("doc_index_shards",0)),0)

    def test_doc_boundary_signal_real(self):
        if not self.real_path:
            self.skipTest("real dolma source missing")
        b=((self.verify or {}).get("boundary") or {})
        self.assertTrue(bool(b.get("doc_boundary_changed")))

    def test_stream_read_first_k_docs(self):
        if not self.real_path:
            self.skipTest("real dolma source missing")
        p=Path(self.real_path)
        k=16
        got=0
        for doc_id,text in stream_docs(p,max_docs=k):
            self.assertTrue(str(doc_id).strip())
            self.assertTrue(str(text).strip())
            got+=1
        emit_payload("test_stream_read_first_k_docs","First K stream docs preview count.",{"k":k,"got":got,"path":self.real_path})
        self.assertGreaterEqual(got,min(4,k))

if __name__=="__main__":
    unittest.main()
