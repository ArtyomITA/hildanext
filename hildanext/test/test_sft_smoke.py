# SFT smoke test for one-step SAFE objective.
# Entrypoints: unittest test methods.
# Runs prepare->tokenize->sft and checks finite metrics.
from __future__ import annotations
from pathlib import Path
import math
import unittest
import torch
from hildanext.config import load_config,clone_with_updates
from hildanext.datasets import prepare_data
from hildanext.tokenization import tokenize_all
from hildanext.training import run_sft_training
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CFG=ROOT/"runs"/"configs"/"default.json"

def _model_exists(model_dir:str)->bool:
    p=Path(model_dir)
    return p.exists() and (p/"config.json").exists() and any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

class SFTSmokeTests(unittest.TestCase):
    def test_sft_one_step(self):
        cfg=load_config(DEFAULT_CFG)
        cfg=clone_with_updates(cfg,{
            "runtime":{"force_dummy_model":False,"use_dinfer":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},
            "data":{"max_samples":48,"seq_len":96},
            "train":{"max_steps":1,"batch_size":1,"accum_steps":1,"dtype":"fp16"}
        })
        prepare_data(cfg,download=False,max_samples=cfg.data.max_samples)
        tok=tokenize_all(cfg,max_records=cfg.data.max_samples)
        rep=run_sft_training(cfg,steps=1)
        emit_payload(
            "test_sft_one_step",
            "Runs one-step SFT smoke (prepare->tokenize->train).",
            {"config":{"device":cfg.runtime.device,"max_samples":cfg.data.max_samples,"seq_len":cfg.data.seq_len},"tokenize_out":{k:v.get("records_out",0) for k,v in tok.items() if isinstance(v,dict)},"train_summary":rep}
        )
        self.assertGreaterEqual(rep["steps"],1)
        self.assertFalse(math.isnan(float(rep["loss_last"])))
        if _model_exists(cfg.paths.model_dir):
            self.assertFalse(bool(rep.get("dummy_model",True)))

if __name__=="__main__":
    unittest.main()
