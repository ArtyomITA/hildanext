# API smoke test for /generate endpoint with local app factory.
# Entrypoint: pytest.
# Skips if TestClient dependencies are missing.
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import json
from hildanext.config import load_config,clone_with_updates
from hildanext.api import create_app
import hildanext.api as api_mod
import hildanext.inference as inf
from hildanext.toon import write_toon

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"runs"/"configs"/"smoke.json"
API_LOG=ROOT/"runs"/"reports"/"logs"/"api_generate_smoke_mdm.json"
API_LOG_TOON=ROOT/"runs"/"reports"/"logs"/"api_generate_smoke_mdm.toon"

def _has_model(model_dir:str)->bool:
    p=Path(model_dir)
    return p.exists() and (p/"config.json").exists() and any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

def test_api_generate_smoke():
    tc=pytest.importorskip("fastapi.testclient")
    cfg=load_config(CFG)
    has_model=_has_model(cfg.paths.model_dir)
    use_real_model=bool(has_model and torch.cuda.is_available())
    cfg=clone_with_updates(cfg,{
        "runtime":{
            "force_dummy_model":False if use_real_model else True,
            "use_dinfer":False,
            "device":"cuda" if use_real_model else "cpu",
            "require_cuda_for_inference":False,
        },
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
    if use_real_model:
        assert bool(data.get("stats",{}).get("dummy_model",True)) is False
    API_LOG.parent.mkdir(parents=True,exist_ok=True)
    report={"health":h.json(),"request":{"prompt":"Write one short line.","mode":"S_MODE","max_new_tokens":12,"seed":3},"response":data,"has_model":has_model,"use_real_model":use_real_model}
    API_LOG.write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding="utf-8")
    write_toon(API_LOG_TOON,report,root_key="api_generate_smoke")


def _cfg_for_inference_log_tests():
    cfg=load_config(CFG)
    return clone_with_updates(
        cfg,
        {
            "runtime":{
                "force_dummy_model":True,
                "use_dinfer":False,
                "device":"cpu",
                "require_cuda_for_inference":False,
                "inference_log_ring_size":3000,
                "inference_log_preview_chars":140,
                "inference_sse_keepalive_s":10,
                "dllm_stop_eos_enabled":True,
                "dllm_stop_plateau_patience":2,
                "dllm_stop_plateau_delta_ratio":0.01,
                "dllm_stop_cycle_enabled":True,
            },
            "inference":{
                "strict_decode_invariants":False,
                "allow_tau_fallback_on_degenerate":False,
                "max_steps":8,
                "remask":{"target_ratio":0.0,"min_ratio":0.0},
            },
        },
    )


class _FakeTokenizer:
    vocab_size=4096
    eos_token_id=2

    def __len__(self):
        return self.vocab_size

    def encode(self,text,add_special_tokens=False):
        base=str(text or "")
        return [((ord(ch)%255)+3) for ch in base] or [11]

    def decode(self,ids,skip_special_tokens=True):
        if isinstance(ids,torch.Tensor):
            ids=ids.detach().cpu().tolist()
        return "".join("x" for _ in list(ids or []))

    def __call__(self,texts,return_tensors="pt"):
        encoded=[self.encode((texts or [""])[0])]
        return {"input_ids":torch.tensor(encoded,dtype=torch.long)}


class _FakeModel:
    def eval(self):
        return self

    def train(self,mode=False):
        return self

    def __call__(self,input_ids,use_cache=False):
        b,l=input_ids.shape
        vocab=4096
        logits=torch.zeros((b,l,vocab),dtype=torch.float32,device=input_ids.device)
        logits[...,7]=1.0
        return SimpleNamespace(logits=logits)


class _FakeEngine:
    name="transformers"

    def __init__(self,cfg):
        self.cfg=cfg
        tok=_FakeTokenizer()
        self.bundle=SimpleNamespace(
            tokenizer=tok,
            is_dummy=True,
            load_reason="test",
            env_issues={},
            device=torch.device("cpu"),
            actual_dtype="float32",
        )
        self.last_stats={}

    def generate(self,**kwargs):
        max_new=int(kwargs.get("max_new_tokens",16) or 16)
        self.last_stats={
            "engine":"transformers",
            "mode":str(kwargs.get("mode","S_MODE")),
            "effort":str(kwargs.get("effort","medium")),
            "steps":2,
            "steps_to_converge":2,
            "logs":[
                {
                    "step":1,
                    "mask_ratio":0.5,
                    "gamma_count":8,
                    "delta_count":4,
                    "avg_conf_masked":0.61,
                    "avg_conf_tokens":0.77,
                    "tau_mask":0.08,
                    "tau_edit":0.08,
                    "decode_policy":"current_base",
                    "use_bidirectional_effective":False,
                    "tau_mask_active":False,
                    "tau_edit_active":False,
                    "delta_edit_enabled":False,
                    "remask_enabled":False,
                    "eligible_gamma_count":0,
                    "applied_gamma_count":8,
                    "eligible_delta_count":4,
                    "applied_delta_count":0,
                    "remask_applied":False,
                    "remask_count":0,
                    "low_conf_fallback_applied":False,
                    "low_conf_fallback_count":0,
                    "delta_stage_ready":True,
                    "delta_gate_reason":"",
                    "accept_strategy":"topk_budget",
                    "block_mask_ratio_before":1.0,
                    "dominant_token":{"id":7,"text":"x"},
                    "dominant_token_share":0.75,
                    "topk_tokens":[{"id":7,"text":"x","count":8,"share":0.75}],
                    "noop_reason":"",
                },
                {
                    "step":2,
                    "mask_ratio":0.0,
                    "gamma_count":1,
                    "delta_count":0,
                    "avg_conf_masked":0.88,
                    "avg_conf_tokens":0.93,
                    "tau_mask":0.08,
                    "tau_edit":0.08,
                    "decode_policy":"current_base",
                    "use_bidirectional_effective":False,
                    "tau_mask_active":False,
                    "tau_edit_active":False,
                    "delta_edit_enabled":False,
                    "remask_enabled":False,
                    "eligible_gamma_count":0,
                    "applied_gamma_count":1,
                    "eligible_delta_count":0,
                    "applied_delta_count":0,
                    "remask_applied":False,
                    "remask_count":0,
                    "low_conf_fallback_applied":False,
                    "low_conf_fallback_count":0,
                    "delta_stage_ready":True,
                    "delta_gate_reason":"",
                    "accept_strategy":"topk_budget",
                    "block_mask_ratio_before":0.5,
                    "dominant_token":{"id":7,"text":"x"},
                    "dominant_token_share":1.0,
                    "topk_tokens":[{"id":7,"text":"x","count":1,"share":1.0}],
                    "noop_reason":"",
                },
            ],
            "tokens_generated":max_new,
            "tokens_per_sec":120.0,
            "vram_peak_bytes":None,
            "dummy_model":True,
            "load_reason":"test",
            "device":"cpu",
            "actual_dtype":"float32",
            "finish_reason":"converged",
            "truncated":False,
            "stop_guard_triggered":False,
            "stop_guard_reason":"",
            "env_issues":{},
            "fallbacks":[],
        }
        cb=getattr(self.cfg.runtime,"inference_step_callback",None)
        if callable(cb):
            for row in self.last_stats["logs"]:
                cb(dict(row))
        if "problem:" in str(kwargs.get("prompt","")).lower():
            return "#### 2"
        return "Ciao!"

    def close(self):
        return


def test_inference_logs_tail_after_and_benchmark_tags(monkeypatch:pytest.MonkeyPatch):
    tc=pytest.importorskip("fastapi.testclient")
    monkeypatch.setattr(api_mod,"build_engine",lambda cfg,trace=None:_FakeEngine(cfg))
    app=create_app(_cfg_for_inference_log_tests())
    c=tc.TestClient(app)

    r=c.post("/generate",json={"prompt":"Write a short greeting.","mode":"S_MODE","max_new_tokens":24,"seed":7})
    assert r.status_code==200
    payload=r.json()
    assert payload["stats"]["finish_reason"] in {"converged","length","eos","plateau","cycle"}

    logs_resp=c.get("/inference/logs?tail=400")
    assert logs_resp.status_code==200
    logs=logs_resp.json().get("events",[])
    assert any(row.get("event")=="INFER_REQ_START" for row in logs)
    assert any(row.get("event")=="DLLM_REQ_DONE" for row in logs)
    assert any(row.get("event")=="DLLM_STEP" for row in logs)
    done=next(row for row in logs if row.get("event")=="DLLM_REQ_DONE")
    assert len(str((done.get("meta") or {}).get("answer_preview","")))<=140

    last_id=str(logs[-1].get("id","0"))
    bench=c.post(
        "/stage0/validate/gsm8k-item",
        json={
            "question":"If you have 1 apple and buy 1 more, how many apples do you have?",
            "answer_target":"#### 2",
            "scope":"DLLM",
            "mode":"S_MODE",
            "effort":"low",
            "max_new_tokens":64,
        },
    )
    assert bench.status_code==200
    after_resp=c.get(f"/inference/logs?after_id={last_id}&tail=200")
    assert after_resp.status_code==200
    after_rows=after_resp.json().get("events",[])
    assert after_rows
    assert any(row.get("benchmark")=="gsm8k" for row in after_rows)


def test_inference_logs_stream_route_is_registered(monkeypatch:pytest.MonkeyPatch):
    monkeypatch.setattr(api_mod,"build_engine",lambda cfg,trace=None:_FakeEngine(cfg))
    app=create_app(_cfg_for_inference_log_tests())
    paths={getattr(route,"path","") for route in app.routes}
    assert "/inference/logs/stream" in paths


def _make_engine_cfg(**runtime_updates):
    cfg=_cfg_for_inference_log_tests()
    cfg.runtime.dllm_stop_eos_enabled=bool(runtime_updates.get("dllm_stop_eos_enabled",cfg.runtime.dllm_stop_eos_enabled))
    cfg.runtime.dllm_stop_plateau_patience=int(runtime_updates.get("dllm_stop_plateau_patience",cfg.runtime.dllm_stop_plateau_patience))
    cfg.runtime.dllm_stop_plateau_delta_ratio=float(runtime_updates.get("dllm_stop_plateau_delta_ratio",cfg.runtime.dllm_stop_plateau_delta_ratio))
    cfg.runtime.dllm_stop_cycle_enabled=bool(runtime_updates.get("dllm_stop_cycle_enabled",cfg.runtime.dllm_stop_cycle_enabled))
    cfg.runtime.dllm_shadow_enabled=bool(runtime_updates.get("dllm_shadow_enabled",False))
    return cfg


def _fake_bundle():
    return inf.ModelBundle(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
        mask_id=0,
        vocab_size=4096,
        is_dummy=True,
        load_reason="test",
        env_issues={},
        model_name_or_path="test",
        requested_dtype="fp16",
        actual_dtype="float32",
        fallbacks=[],
    )


def test_stop_guard_eos(monkeypatch:pytest.MonkeyPatch):
    cfg=_make_engine_cfg(dllm_stop_eos_enabled=True,dllm_shadow_enabled=False)
    monkeypatch.setattr(inf,"load_model_bundle",lambda cfg,for_training=False,trace=None:_fake_bundle())
    engine=inf.TransformersEngine(cfg)
    eos=getattr(engine.bundle.tokenizer,"eos_token_id",None)
    if eos is None:
        eos=2
        setattr(engine.bundle.tokenizer,"eos_token_id",eos)

    def fake_predict(model,seq,prompt_len,max_new,mask_id,temperature=0.0,use_bidirectional=False):
        pred=torch.full((1,max_new),int(eos),dtype=torch.long,device=seq.device)
        conf=torch.full((1,max_new),0.95,dtype=torch.float32,device=seq.device)
        return pred,conf

    monkeypatch.setattr(inf,"_predict_bidirectional",fake_predict)

    _=engine.generate(prompt="ciao",mode="S_MODE",max_new_tokens=32,effort="high",seed=13)
    st=dict(engine.last_stats or {})
    assert st.get("finish_reason")=="eos"
    assert bool(st.get("stop_guard_triggered")) is True
    assert st.get("stop_guard_reason")=="eos"


def test_stop_guard_plateau(monkeypatch:pytest.MonkeyPatch):
    cfg=_make_engine_cfg(dllm_stop_eos_enabled=False,dllm_stop_plateau_patience=2,dllm_stop_plateau_delta_ratio=0.01,dllm_stop_cycle_enabled=False,dllm_shadow_enabled=False)
    monkeypatch.setattr(inf,"load_model_bundle",lambda cfg,for_training=False,trace=None:_fake_bundle())
    engine=inf.TransformersEngine(cfg)

    def fake_predict(model,seq,prompt_len,max_new,mask_id,temperature=0.0,use_bidirectional=False):
        pred=torch.full((1,max_new),7,dtype=torch.long,device=seq.device)
        conf=torch.full((1,max_new),0.9,dtype=torch.float32,device=seq.device)
        return pred,conf

    def fake_step(**kwargs):
        tokens=kwargs["tokens"]
        return tokens.clone(),{
            "decode_policy":"current_base",
            "use_bidirectional_effective":False,
            "tau_mask_active":False,
            "tau_edit_active":False,
            "delta_edit_enabled":False,
            "remask_enabled":False,
            "eligible_gamma_count":0,
            "applied_gamma_count":0,
            "eligible_delta_count":0,
            "applied_delta_count":0,
            "gamma_count":0,
            "delta_count":0,
            "remask_applied":False,
            "remask_count":0,
            "dominant_token":{"id":7,"text":"x"},
            "dominant_token_share":1.0,
            "topk_tokens":[{"id":7,"text":"x","count":1,"share":1.0}],
            "low_conf_fallback_applied":False,
            "low_conf_fallback_count":0,
            "delta_stage_ready":True,
            "delta_gate_reason":"",
            "accept_strategy":"forced_noop",
            "block_mask_ratio_before":1.0,
            "noop_reason":"forced_noop",
        }

    monkeypatch.setattr(inf,"_predict_bidirectional",fake_predict)
    monkeypatch.setattr(inf,"_apply_decode_policy_step",fake_step)

    _=engine.generate(prompt="ciao",mode="S_MODE",max_new_tokens=64,effort="high",seed=5)
    st=dict(engine.last_stats or {})
    assert st.get("finish_reason")=="plateau"
    assert bool(st.get("stop_guard_triggered")) is True
    assert st.get("stop_guard_reason")=="plateau"


def test_stop_guard_cycle(monkeypatch:pytest.MonkeyPatch):
    cfg=_make_engine_cfg(dllm_stop_eos_enabled=False,dllm_stop_cycle_enabled=True,dllm_stop_plateau_patience=3,dllm_stop_plateau_delta_ratio=0.01,dllm_shadow_enabled=False)
    monkeypatch.setattr(inf,"load_model_bundle",lambda cfg,for_training=False,trace=None:_fake_bundle())
    engine=inf.TransformersEngine(cfg)

    def fake_predict(model,seq,prompt_len,max_new,mask_id,temperature=0.0,use_bidirectional=False):
        pred=torch.full((1,max_new),11,dtype=torch.long,device=seq.device)
        conf=torch.full((1,max_new),0.88,dtype=torch.float32,device=seq.device)
        return pred,conf

    def fake_step(**kwargs):
        tokens=kwargs["tokens"]
        return tokens.clone(),{
            "decode_policy":"current_base",
            "use_bidirectional_effective":False,
            "tau_mask_active":False,
            "tau_edit_active":False,
            "delta_edit_enabled":False,
            "remask_enabled":False,
            "eligible_gamma_count":0,
            "applied_gamma_count":0,
            "eligible_delta_count":0,
            "applied_delta_count":0,
            "gamma_count":0,
            "delta_count":0,
            "remask_applied":False,
            "remask_count":0,
            "dominant_token":{"id":11,"text":"x"},
            "dominant_token_share":1.0,
            "topk_tokens":[{"id":11,"text":"x","count":1,"share":1.0}],
            "low_conf_fallback_applied":False,
            "low_conf_fallback_count":0,
            "delta_stage_ready":True,
            "delta_gate_reason":"",
            "accept_strategy":"forced_noop",
            "block_mask_ratio_before":1.0,
            "noop_reason":"forced_noop",
        }

    monkeypatch.setattr(inf,"_predict_bidirectional",fake_predict)
    monkeypatch.setattr(inf,"_apply_decode_policy_step",fake_step)

    _=engine.generate(prompt="ciao",mode="S_MODE",max_new_tokens=64,effort="high",seed=17)
    st=dict(engine.last_stats or {})
    assert st.get("finish_reason")=="cycle"
    assert bool(st.get("stop_guard_triggered")) is True
    assert st.get("stop_guard_reason")=="cycle"


def test_policy_step_current_base_reports_tau_inactive():
    tok=_FakeTokenizer()
    tokens=torch.tensor([[0,0,5]],dtype=torch.long)
    pred=torch.tensor([[9,8,7]],dtype=torch.long)
    conf=torch.tensor([[0.01,0.02,0.95]],dtype=torch.float32)
    updated,row=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.90,
        tau_edit=0.90,
        budget=1,
        policy="current_base",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
    )
    assert int(updated.ne(tokens).sum().item())==1
    assert row["tau_mask_active"] is False
    assert row["tau_edit_active"] is False
    assert row["eligible_gamma_count"]==0
    assert row["applied_gamma_count"]==1
    assert row["applied_delta_count"]==0


def test_policy_step_shadow_tau_mask_controls_reveal():
    tok=_FakeTokenizer()
    tokens=torch.tensor([[0,0]],dtype=torch.long)
    pred=torch.tensor([[9,8]],dtype=torch.long)
    conf=torch.tensor([[0.2,0.4]],dtype=torch.float32)
    updated_hi,row_hi=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.5,
        tau_edit=0.5,
        budget=2,
        policy="shadow_llada21",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
    )
    updated_lo,row_lo=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.1,
        tau_edit=0.5,
        budget=2,
        policy="shadow_llada21",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
    )
    assert int(updated_hi.ne(tokens).sum().item())==0
    assert int(updated_lo.ne(tokens).sum().item())==2
    assert row_hi["applied_gamma_count"]==0
    assert row_lo["applied_gamma_count"]==2


def test_policy_step_shadow_tau_edit_and_remask():
    tok=_FakeTokenizer()
    remask_cfg=SimpleNamespace(target_ratio=1.0,min_ratio=1.0)
    tokens=torch.tensor([[4,5]],dtype=torch.long)
    pred=torch.tensor([[9,8]],dtype=torch.long)
    conf=torch.tensor([[0.95,0.10]],dtype=torch.float32)
    updated_mid,row_mid=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.5,
        tau_edit=0.5,
        budget=2,
        policy="shadow_llada21",
        use_bidirectional_effective=False,
        remask_cfg=remask_cfg,
        use_remask=True,
        is_last_step=False,
    )
    updated_last,row_last=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.5,
        tau_edit=0.5,
        budget=2,
        policy="shadow_llada21",
        use_bidirectional_effective=False,
        remask_cfg=remask_cfg,
        use_remask=True,
        is_last_step=True,
    )
    assert row_mid["applied_delta_count"]==1
    assert row_mid["remask_enabled"] is True
    assert row_mid["remask_applied"] is True
    assert int(updated_mid.eq(0).sum().item())>=1
    assert row_last["applied_delta_count"]==1
    assert row_last["remask_enabled"] is False
    assert row_last["remask_applied"] is False


def test_policy_step_threshold_cap_uses_threshold_then_fallback():
    tok=_FakeTokenizer()
    tokens=torch.tensor([[0,0,0]],dtype=torch.long)
    pred=torch.tensor([[3,4,5]],dtype=torch.long)
    conf=torch.tensor([[0.99,0.20,0.10]],dtype=torch.float32)
    updated,row=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.95,
        tau_edit=0.5,
        budget=2,
        policy="threshold_cap",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
    )
    assert int(updated.ne(tokens).sum().item())==2
    assert row["tau_mask_active"] is True
    assert row["eligible_gamma_count"]==1
    assert row["applied_gamma_count"]==2
    assert row["low_conf_fallback_applied"] is True
    assert row["low_conf_fallback_count"]==2


def test_policy_step_shadow_cap_topup_reduces_threshold_stall():
    tok=_FakeTokenizer()
    tokens=torch.tensor([[0,0,7]],dtype=torch.long)
    pred=torch.tensor([[9,8,6]],dtype=torch.long)
    conf=torch.tensor([[0.92,0.30,0.91]],dtype=torch.float32)
    updated,row=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.90,
        tau_edit=0.90,
        budget=2,
        policy="shadow_llada21_cap",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
        step_in_block=1,
        block_mask_ratio=0.66,
    )
    assert int(updated.ne(tokens).sum().item())==3
    assert row["applied_gamma_count"]==2
    assert row["applied_delta_count"]==1
    assert row["low_conf_fallback_applied"] is True
    assert row["low_conf_fallback_count"]==1


def test_policy_step_delayed_edit_gates_delta_on_first_step():
    tok=_FakeTokenizer()
    tokens=torch.tensor([[0,5]],dtype=torch.long)
    pred=torch.tensor([[9,8]],dtype=torch.long)
    conf=torch.tensor([[0.95,0.96]],dtype=torch.float32)
    updated,row=inf._apply_decode_policy_step(
        tok=tok,
        tokens=tokens,
        pred_ids=pred,
        confidence=conf,
        mask_id=0,
        tau_mask=0.90,
        tau_edit=0.90,
        budget=1,
        policy="shadow_llada21_delayed_edit",
        use_bidirectional_effective=False,
        remask_cfg=None,
        use_remask=False,
        is_last_step=False,
        step_in_block=1,
        block_mask_ratio=1.0,
    )
    assert int(updated[0,0].item())==9
    assert int(updated[0,1].item())==5
    assert row["delta_stage_ready"] is False
    assert row["delta_gate_reason"]=="await_initial_draft"
    assert row["applied_delta_count"]==0
