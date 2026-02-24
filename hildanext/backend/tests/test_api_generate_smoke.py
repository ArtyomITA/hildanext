# API smoke test for /generate endpoint with local app factory.
# Entrypoint: pytest.
# Skips if TestClient dependencies are missing.
from pathlib import Path
import pytest
import torch
import json
from hildanext.config import load_config,clone_with_updates
from hildanext.api import create_app

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"runs"/"configs"/"smoke.json"
API_LOG=ROOT/"runs"/"reports"/"logs"/"api_generate_smoke_mdm.json"

def _has_model(model_dir:str)->bool:
    p=Path(model_dir)
    return p.exists() and (p/"config.json").exists() and any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

def test_api_generate_smoke():
    tc=pytest.importorskip("fastapi.testclient")
    cfg=load_config(CFG)
    has_model=_has_model(cfg.paths.model_dir)
    cfg=clone_with_updates(cfg,{
        "runtime":{"force_dummy_model":False if has_model else True,"use_dinfer":False,"device":"cuda" if torch.cuda.is_available() else "cpu"},
        "inference":{"strict_decode_invariants":True if has_model else False,"allow_tau_fallback_on_degenerate":False,"s_mode_tau_mask":0.08,"s_mode_tau_edit":0.08}
    })
    app=create_app(cfg)
    c=tc.TestClient(app)
    h=c.get("/health")
    assert h.status_code==200
    r=c.post("/generate",json={"prompt":"Write one short line.","mode":"S_MODE","max_new_tokens":12,"seed":3})
    assert r.status_code==200
    data=r.json()
    assert isinstance(data.get("text",""),str)
    assert data.get("text","").strip()
    if has_model:
        assert bool(data.get("stats",{}).get("dummy_model",True)) is False
    API_LOG.parent.mkdir(parents=True,exist_ok=True)
    API_LOG.write_text(json.dumps({"health":h.json(),"request":{"prompt":"Write one short line.","mode":"S_MODE","max_new_tokens":12,"seed":3},"response":data,"has_model":has_model},ensure_ascii=True,indent=2),encoding="utf-8")
