# Builds a single exhaustive backend_readiness.md with inline logs and metrics.
# Main entrypoint: python hildanext/test/build_backend_readiness.py.
# Consumes mdm test artifacts, audit/quant reports, and formula mapping.
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from typing import Any,Dict,List
import json
import math

ROOT=Path(__file__).resolve().parents[1]
LOG_DIR=ROOT/"runs"/"reports"/"logs"
DOC_PATH=ROOT/"docs"/"reports"/"backend_readiness.md"
RUN_DOC_PATH=ROOT/"runs"/"reports"/"backend_readiness.md"
UNIT_RESULT=LOG_DIR/"unittest_result_mdm.json"
PAYLOADS=LOG_DIR/"test_payloads_mdm.jsonl"
PYTEST_LOG=LOG_DIR/"pytest_backend_mdm.log"
AUDIT_JSON=ROOT/"runs"/"reports"/"formula_audit.json"
AUDIT_MD=ROOT/"runs"/"reports"/"formula_audit.md"
QUANT_JSON=ROOT/"runs"/"reports"/"quant_vram.json"
API_JSON=LOG_DIR/"api_generate_smoke_mdm.json"
PREFLIGHT_JSON=ROOT/"runs"/"reports"/"preflight.json"
RECIPE_JSON=ROOT/"runs"/"reports"/"recipe_llada21.json"
DINFER_JSON=ROOT/"runs"/"reports"/"dinfer_smoke.json"
FALLBACKS_JSONL=ROOT/"runs"/"logs"/"fallbacks.jsonl"
METRICS_JSONL=ROOT/"runs"/"logs"/"metrics.jsonl"
SFT_PLAYBOOK=ROOT/"docs"/"reports"/"sft_vram_playbook.md"

def _read_text(p:Path)->str:
    if not p.exists():
        return f"[missing] {p}"
    return p.read_text(encoding="utf-8",errors="ignore").strip()

def _read_json(p:Path)->Dict[str,Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8",errors="ignore"))
    except Exception:
        return {}

def _read_jsonl(p:Path)->List[Dict[str,Any]]:
    if not p.exists():
        return []
    out=[]
    for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"raw":line})
    return out

def _code_block(txt:str,lang:str="text")->str:
    return f"```{lang}\n{txt}\n```"

def _strip_leading_json_block(txt:str)->str:
    s=txt or ""
    i=s.find("{")
    if i<0:
        return s
    depth=0
    in_str=False
    esc=False
    end=-1
    for j,ch in enumerate(s[i:],start=i):
        if in_str:
            if esc:
                esc=False
            elif ch=="\\":
                esc=True
            elif ch=="\"":
                in_str=False
            continue
        if ch=="\"":
            in_str=True
            continue
        if ch=="{":
            depth+=1
            continue
        if ch=="}":
            depth-=1
            if depth==0:
                end=j
                break
    if end<0:
        return s
    out=(s[:i]+s[end+1:]).strip()
    return out

def _toon_scalar(v:Any)->str:
    if v is None:
        return "null"
    if isinstance(v,bool):
        return "true" if v else "false"
    if isinstance(v,(int,float)):
        if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):
            return "null"
        return str(v)
    s=str(v)
    if not s:
        return "\"\""
    bad=set(",:[]{}\n\t\r")
    needs_quote=any(ch in bad for ch in s) or s[0].isspace() or s[-1].isspace() or "#" in s
    return json.dumps(s,ensure_ascii=True) if needs_quote else s

def _uniform_obj_array(arr:List[Any])->List[str]:
    if not arr:
        return []
    if not all(isinstance(x,dict) for x in arr):
        return []
    keys=list(arr[0].keys())
    if not keys:
        return []
    for x in arr:
        if list(x.keys())!=keys:
            return []
        if not all(not isinstance(x[k],(dict,list)) for k in keys):
            return []
    return [str(k) for k in keys]

def _to_toon(v:Any,indent:int=0,key:str|None=None)->List[str]:
    pre="\t"*indent
    if isinstance(v,dict):
        out=[]
        if key is not None and not v:
            return [f"{pre}{key}: {{}}"]
        if key is not None:
            out.append(f"{pre}{key}:")
            pre="\t"*(indent+1)
            indent+=1
        for k,val in v.items():
            ks=str(k)
            if isinstance(val,dict):
                if not val:
                    out.append(f"{pre}{ks}: {{}}")
                else:
                    out.extend(_to_toon(val,indent,ks))
            elif isinstance(val,list):
                out.extend(_to_toon(val,indent,ks))
            else:
                out.append(f"{pre}{ks}: {_toon_scalar(val)}")
        return out
    if isinstance(v,list):
        n=len(v)
        if key is not None:
            fields=_uniform_obj_array(v)
            if fields:
                out=[f"{pre}{key}[{n}]{{{','.join(fields)}}}:"]
                for row in v:
                    out.append(f"{pre}\t{','.join(_toon_scalar(row[f]) for f in fields)}")
                return out
            if all(not isinstance(x,(dict,list)) for x in v):
                return [f"{pre}{key}[{n}]: {','.join(_toon_scalar(x) for x in v)}"]
            out=[f"{pre}{key}[{n}]:"]
            for i,x in enumerate(v):
                if isinstance(x,(dict,list)):
                    out.extend(_to_toon(x,indent+1,f"[{i}]"))
                else:
                    out.append(f"{pre}\t[{i}]: {_toon_scalar(x)}")
            return out
        out=[f"{pre}[{n}]:"]
        for i,x in enumerate(v):
            if isinstance(x,(dict,list)):
                out.extend(_to_toon(x,indent+1,f"[{i}]"))
            else:
                out.append(f"{pre}\t[{i}]: {_toon_scalar(x)}")
        return out
    if key is None:
        return [f"{pre}{_toon_scalar(v)}"]
    return [f"{pre}{key}: {_toon_scalar(v)}"]

def _as_toon(v:Any)->str:
    return "\n".join(_to_toon(v))

def _formula_map()->List[Dict[str,str]]:
    return [
        {
            "component":"LLaDA M2T loss",
            "paper":"L = -sum_{i in M} log p_theta(x_i | x_not_M)",
            "code":"labels[masked_pos]=target_ids[masked_pos]; loss=CE(logits[:,:-1,:],labels[:,1:],ignore_index=-100)",
            "symbol":"formulas.llada_m2t_loss"
        },
        {
            "component":"LLaDA2.0 WSD",
            "paper":"Warmup ladder (1->4->32->64->seq_len), stable at seq_len, decay toward end_block",
            "code":"llada2_wsd_block(step,...,ladder_blocks,enforce_divisibility) then diffusion.wsd_block wrapper",
            "symbol":"formulas.llada2_wsd_block + diffusion.wsd_block"
        },
        {
            "component":"LLaDA2.0 doc mask",
            "paper":"Composite M(2Lx2L): xt->xt same block, xt->x0 strictly previous blocks, x0->x0 block-causal, x0->xt forbidden + doc gating",
            "code":"batch_doc_attention_mask(mask_mode=composite_llada20,base_len=L,block_size=B) + doc_id gating + padding gating",
            "symbol":"masks.batch_doc_attention_mask"
        },
        {
            "component":"LLaDA2.1 Gamma set",
            "paper":"Gamma_t={i | x_i is [MASK] and conf_i>=tau_mask}",
            "code":"gamma=(tokens==mask_id) & (confidence>=tau_mask)",
            "symbol":"formulas.llada21_sets"
        },
        {
            "component":"LLaDA2.1 Delta set",
            "paper":"Delta_t={i | x_i not [MASK], y_i!=x_i, conf_i>=tau_edit}",
            "code":"delta=(tokens!=mask_id) & (pred_ids!=tokens) & (confidence>=tau_edit)",
            "symbol":"formulas.llada21_sets"
        },
        {
            "component":"LLaDA2.1 apply",
            "paper":"x_{t+1}[Gamma_t U Delta_t] <- y_t",
            "code":"out=tokens.clone(); out[gamma]=pred_ids[gamma]; out[delta]=pred_ids[delta]",
            "symbol":"formulas.llada21_apply"
        },
        {
            "component":"LLaDA2.1 M2T+T2T mixture",
            "paper":"L = w_m2t*L_m2t + w_t2t*L_t2t",
            "code":"loss=cfg.m2t_weight*loss_m2t + cfg.t2t_weight*loss_t2t",
            "symbol":"diffusion.compute_m2t_t2t_losses"
        }
    ]

def _table(rows:List[List[str]])->str:
    if not rows:
        return ""
    head=rows[0]
    body=rows[1:]
    out=["| "+" | ".join(head)+" |","| "+" | ".join(["---"]*len(head))+" |"]
    for r in body:
        out.append("| "+" | ".join(r)+" |")
    return "\n".join(out)

def _status_of(unit:Dict[str,Any],needle:str)->str:
    for r in unit.get("tests",[]):
        tid=str(r.get("test_id",""))
        if needle in tid:
            return str(r.get("status","missing"))
    return "missing"

def build()->str:
    unit=_read_json(UNIT_RESULT)
    payloads=_read_jsonl(PAYLOADS)
    audit=_read_json(AUDIT_JSON)
    quant=_read_json(QUANT_JSON)
    audit_md=_read_text(AUDIT_MD)
    sft_playbook=_read_text(SFT_PLAYBOOK)
    api=_read_json(API_JSON)
    preflight=_read_json(PREFLIGHT_JSON)
    recipe=_read_json(RECIPE_JSON)
    dinfer=_read_json(DINFER_JSON)
    fallbacks_rows=_read_jsonl(FALLBACKS_JSONL)
    metrics_rows=_read_jsonl(METRICS_JSONL)
    ts=datetime.now(timezone.utc).isoformat()
    lines=[]
    lines.append("# Backend Readiness Full Evidence Pack")
    lines.append(f"Generated at: `{ts}`")
    lines.append("")
    lines.append("## Execution policy")
    lines.append("- Required env: `mdm`")
    lines.append("- CUDA-first runtime for all integration tests and benches")
    lines.append("- CPU allowed only for explicit CPU unit checks or architecture limits")
    lines.append("")
    lines.append("## Environment logs")
    lines.append("### conda envs")
    lines.append(_code_block(_read_text(LOG_DIR/"conda_envs.log")))
    lines.append("### base env torch/cuda")
    lines.append(_code_block(_read_text(LOG_DIR/"env_cuda_current.log")))
    lines.append("### mdm env torch/cuda")
    lines.append(_code_block(_read_text(LOG_DIR/"env_cuda_mdm.log")))
    lines.append("")
    lines.append("## Installation logs (mdm)")
    lines.append("### pip install -e backend")
    lines.append(_code_block(_read_text(LOG_DIR/"install_editable_mdm.log")))
    lines.append("### pip install pytest")
    lines.append(_code_block(_read_text(LOG_DIR/"install_pytest_mdm.log")))
    lines.append("")
    lines.append("## Test runner summary (unittest mdm)")
    lines.append(_code_block(_as_toon(unit.get("summary",{})),"toon"))
    rows=[["test_id","status","duration_sec","note"]]
    for r in unit.get("tests",[]):
        rows.append([str(r.get("test_id","")),str(r.get("status","")),str(r.get("duration_sec","")),str(r.get("skip_reason",r.get("error",""))).replace("\n"," ")[:180]])
    lines.append(_table(rows))
    lines.append("")
    lines.append("### unittest raw log")
    lines.append(_code_block(_read_text(LOG_DIR/"unittest_full_mdm.log")))
    lines.append("")
    lines.append("## Backend readiness checklist")
    crow=[["check","evidence","status"]]
    crow.append(["Model load real + AR smoke","test_model_load_and_ar_greedy_determinism","`"+_status_of(unit,"test_model_load_and_ar_greedy_determinism")+"`"])
    crow.append(["Mask token + embedding resize","test_mask_token_id_and_embedding_resize","`"+_status_of(unit,"test_mask_token_id_and_embedding_resize")+"`"])
    crow.append(["Doc mask no leakage","test_strict_no_cross_doc_leakage","`"+_status_of(unit,"test_strict_no_cross_doc_leakage")+"`"])
    crow.append(["WSD ladder + divisibility","test_wsd_ladder_and_divisibility","`"+_status_of(unit,"test_wsd_ladder_and_divisibility")+"`"])
    crow.append(["M2T/T2T objective sanity","test_t2t_corruption_and_recovery","`"+_status_of(unit,"test_t2t_corruption_and_recovery")+"`"])
    crow.append(["Threshold decode non-degenerate","test_real_decode_non_degenerate","`"+_status_of(unit,"test_real_decode_non_degenerate")+"`"])
    crow.append(["API generate real model","test_api_generate_real_model","`"+_status_of(unit,"test_api_generate_real_model")+"`"])
    crow.append(["Fallback tracing mandatory","test_dinfer_missing_is_traced_in_logs_and_stats","`"+_status_of(unit,"test_dinfer_missing_is_traced_in_logs_and_stats")+"`"])
    crow.append(["Composite mask structure","test_composite_mask_regions","`"+_status_of(unit,"test_composite_mask_regions")+"`"])
    crow.append(["Composite doc gating","test_doc_gating_with_padding","`"+_status_of(unit,"test_doc_gating_with_padding")+"`"])
    crow.append(["Quant modes + VRAM + train probe","test_quant_bench_report_schema","`"+_status_of(unit,"test_quant_bench_report_schema")+"`"])
    crow.append(["dInfer parity optional","test_one_step_threshold_parity","`"+_status_of(unit,"test_one_step_threshold_parity")+"`"])
    lines.append(_table(crow))
    lines.append("")
    lines.append("## Detailed per-test payloads")
    for item in payloads:
        lines.append(f"### {item.get('test_id','unknown')}")
        lines.append(f"- Description: {item.get('description','')}")
        lines.append(f"- Timestamp: `{item.get('ts','')}`")
        lines.append(_code_block(_as_toon(item.get("payload",{})),"toon"))
    if not payloads:
        lines.append("_No structured payloads found._")
    lines.append("")
    lines.append("## Backend pytest (mdm)")
    lines.append(_code_block(_read_text(PYTEST_LOG)))
    lines.append("")
    lines.append("## API generate smoke evidence")
    lines.append(_code_block(_as_toon(api),"toon"))
    lines.append("")
    lines.append("## Formula alignment: paper vs code")
    frows=[["component","paper formula","code implementation","symbol"]]
    for f in _formula_map():
        frows.append([f["component"],f["paper"],f["code"],f["symbol"]])
    lines.append(_table(frows))
    lines.append("")
    lines.append("### formula audit json")
    lines.append(_code_block(_as_toon(audit),"toon"))
    lines.append("### formula audit md")
    lines.append(_code_block(audit_md,"md"))
    lines.append("")
    lines.append("## Quantization and VRAM evidence (mdm CUDA)")
    lines.append(_code_block(_as_toon(quant),"toon"))
    lines.append("")
    lines.append("### quant-bench raw cli log")
    lines.append(_code_block(_strip_leading_json_block(_read_text(LOG_DIR/"quant_bench_cli_mdm.log"))))
    lines.append("")
    lines.append("## Preflight strict (mdm)")
    lines.append("### preflight report")
    lines.append(_code_block(_as_toon(preflight),"toon"))
    lines.append("### preflight raw cli log")
    lines.append(_code_block(_strip_leading_json_block(_read_text(LOG_DIR/"preflight_strict_mdm.log"))))
    lines.append("")
    lines.append("## Recipe Runner Evidence")
    rmini={
        "ok":recipe.get("ok"),
        "run_id":recipe.get("run_id"),
        "duration_sec":recipe.get("duration_sec"),
        "fallbacks_count":recipe.get("fallbacks_count"),
        "fallbacks_blocking_count":recipe.get("fallbacks_blocking_count"),
        "stage0":{"steps":recipe.get("stage0",{}).get("steps"),"token_seen":recipe.get("stage0",{}).get("token_seen"),"resumed_from":recipe.get("stage0",{}).get("resumed_from"),"loss_last":recipe.get("stage0",{}).get("loss_last")},
        "stage1":{"steps":recipe.get("stage1",{}).get("steps"),"token_seen":recipe.get("stage1",{}).get("token_seen"),"resumed_from":recipe.get("stage1",{}).get("resumed_from"),"loss_last":recipe.get("stage1",{}).get("loss_last")}
    }
    lines.append("### recipe summary")
    lines.append(_code_block(_as_toon(rmini),"toon"))
    lines.append("### recipe raw log run1")
    lines.append(_code_block(_strip_leading_json_block(_read_text(LOG_DIR/"recipe_verify_run.log"))))
    lines.append("### recipe raw log run2 (resume)")
    lines.append(_code_block(_strip_leading_json_block(_read_text(LOG_DIR/"recipe_verify_run2.log"))))
    lines.append("")
    lines.append("## dInfer optional smoke")
    lines.append(_code_block(_as_toon(dinfer),"toon"))
    lines.append("### dinfer raw cli log")
    lines.append(_code_block(_strip_leading_json_block(_read_text(LOG_DIR/"dinfer_smoke_mdm.log"))))
    lines.append("")
    lines.append("## Fallback Trace JSONL")
    lines.append("### fallbacks tail")
    lines.append(_code_block(_as_toon(fallbacks_rows[-48:] if len(fallbacks_rows)>48 else fallbacks_rows),"toon"))
    lines.append("### metrics tail")
    lines.append(_code_block(_as_toon(metrics_rows[-64:] if len(metrics_rows)>64 else metrics_rows),"toon"))
    lines.append("")
    lines.append("## SFT VRAM playbook")
    lines.append(_code_block(sft_playbook,"md"))
    lines.append("")
    lines.append("## Environment issue remediation")
    lines.append("- `env_issue:numpy_dll`: PyTorch can run CUDA kernels but torch->numpy bridge raises DLL warning in this env.")
    lines.append("- Remediation: reinstall NumPy in `mdm` (`pip install --force-reinstall numpy==1.26.4`) and ensure VC runtime is available; then rerun `conda run -n mdm python -c \"import torch;print(torch.tensor([1.0]).numpy())\"`.")
    lines.append("")
    lines.append("## Non-mdm guard evidence")
    lines.append(_code_block(_read_text(LOG_DIR/"unittest_base_blocked.log")))
    return "\n".join(lines).strip()+"\n"

def main()->int:
    DOC_PATH.parent.mkdir(parents=True,exist_ok=True)
    RUN_DOC_PATH.parent.mkdir(parents=True,exist_ok=True)
    txt=build()
    DOC_PATH.write_text(txt,encoding="utf-8")
    RUN_DOC_PATH.write_text(txt,encoding="utf-8")
    print(str(DOC_PATH))
    print(str(RUN_DOC_PATH))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
