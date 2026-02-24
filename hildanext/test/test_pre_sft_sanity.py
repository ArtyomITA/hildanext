# Pre-SFT data/tokenize/doc-id sanity plus one-step CPT smoke.
# Entrypoint: unittest.
# Uses tiny temporary outputs and skips real-step if model load fails.
from __future__ import annotations
from pathlib import Path
import math
import tempfile
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.datasets import prepare_data
from hildanext.tokenization import tokenize_all
from hildanext.training import run_wsd_conversion
from hildanext.inference import load_model_bundle
from hildanext.io_utils import read_jsonl
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"runs"/"configs"/"default.json"

def _has_real_model(cfg)->bool:
    p=Path(cfg.paths.model_dir)
    return p.exists() and (p/"config.json").exists() and ((p/"model.safetensors").exists() or (p/"pytorch_model.bin").exists())

class PreSFTSanityTests(unittest.TestCase):
    def test_prepare_tokenize_and_one_step_training(self):
        base=load_config(CFG_PATH)
        with tempfile.TemporaryDirectory(prefix="hildanext_pre_sft_") as td:
            t=Path(td)
            cfg=clone_with_updates(base,{
                "paths":{
                    "raw_dir":str(t/"raw"),
                    "processed_dir":str(t/"processed"),
                    "tokenized_dir":str(t/"tokenized"),
                    "logs_dir":str(t/"logs"),
                    "checkpoints_dir":str(t/"ckpt")
                },
                "runtime":{"use_dinfer":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},
                "data":{"max_samples":48,"seq_len":96},
                "train":{"max_steps":1,"batch_size":1,"accum_steps":1,"max_tokens":20000,"dtype":"float32"}
            })
            prepare_data(cfg,download=False,max_samples=cfg.data.max_samples)
            rep=tokenize_all(cfg,max_records=cfg.data.max_samples)
            train_path=Path(rep["train"]["output"])
            self.assertTrue(train_path.exists())
            rows=read_jsonl(train_path,max_rows=16)
            self.assertGreater(len(rows),0)
            found_boundary=False
            for r in rows:
                self.assertIn("input_ids",r)
                self.assertIn("attention_mask",r)
                self.assertIn("doc_ids",r)
                self.assertEqual(len(r["input_ids"]),len(r["attention_mask"]))
                self.assertEqual(len(r["input_ids"]),len(r["doc_ids"]))
                non_pad=[int(d) for d in r["doc_ids"] if int(d)>=0]
                self.assertTrue(all(non_pad[i]<=non_pad[i+1] for i in range(max(0,len(non_pad)-1))))
                for i in range(max(0,len(non_pad)-1)):
                    if non_pad[i]!=non_pad[i+1]:
                        found_boundary=True
                        break
            self.assertTrue(found_boundary)
            real_cfg=clone_with_updates(cfg,{"runtime":{"force_dummy_model":False}})
            real_present=_has_real_model(real_cfg)
            do_real=False
            if real_present:
                b=load_model_bundle(real_cfg,for_training=False)
                do_real=not b.is_dummy
            train_cfg=real_cfg if do_real else clone_with_updates(cfg,{"runtime":{"force_dummy_model":True,"device":"cuda" if torch.cuda.is_available() else "cpu"}})
            try:
                out=run_wsd_conversion(train_cfg,steps=1)
            except Exception as e:
                if do_real:
                    self.skipTest(f"real one-step cpt unavailable: {e}")
                raise
            emit_payload(
                "test_prepare_tokenize_and_one_step_training",
                "Pre-SFT sanity: prepare/tokenize/doc-ids and one-step CPT training summary.",
                {
                    "config":{"device":train_cfg.runtime.device,"seq_len":train_cfg.data.seq_len,"max_samples":train_cfg.data.max_samples},
                    "tokenized_records_out":{k:v.get("records_out",0) for k,v in rep.items() if isinstance(v,dict)},
                    "sample_record_preview":rows[0] if rows else {},
                    "found_doc_boundary":bool(found_boundary),
                    "used_real_model":bool(do_real),
                    "cpt_summary":out
                }
            )
            self.assertGreaterEqual(int(out.get("steps",0)),1)
            self.assertTrue(math.isfinite(float(out.get("loss_last",float("nan")))))

if __name__=="__main__":
    unittest.main()
