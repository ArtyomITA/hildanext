# Formula audit for LLaDA/LLaDA2.0/LLaDA2.1 alignment.
# Main entrypoints: collect_formula_impl,paper_map,run_audit.
# Produces compact md/json reports with invariant checks.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Tuple
import inspect
import json
import math
import torch
from . import formulas
from . import diffusion
from . import masks
from .config import TrainConfig
from .utils import TinyCausalLM

def collect_formula_impl()->Dict[str,Dict[str,Any]]:
    symbols={
        "formulas.llada_m2t_loss":getattr(formulas,"llada_m2t_loss",None),
        "formulas.llada2_wsd_block":getattr(formulas,"llada2_wsd_block",None),
        "formulas.llada21_sets":getattr(formulas,"llada21_sets",None),
        "formulas.llada21_apply":getattr(formulas,"llada21_apply",None),
        "diffusion.compute_m2t_t2t_losses":getattr(diffusion,"compute_m2t_t2t_losses",None),
        "masks.doc_attention_mask":getattr(masks,"doc_attention_mask",None),
        "masks.batch_doc_attention_mask":getattr(masks,"batch_doc_attention_mask",None)
    }
    out:Dict[str,Dict[str,Any]]={}
    for name,obj in symbols.items():
        out[name]={
            "exists":callable(obj),
            "module":obj.__module__ if callable(obj) else "",
            "qualname":obj.__qualname__ if callable(obj) else "",
            "signature":str(inspect.signature(obj)) if callable(obj) else ""
        }
    return out

def paper_map()->Dict[str,Dict[str,Any]]:
    return {
        "formulas.llada_m2t_loss":{
            "component":"LLaDA base M2T loss",
            "paper_name":"LLaDA",
            "section":"masked objective / M2T",
            "keyword":"masked",
            "file_source":"llada2_0.html",
            "tests":["test_formulas.py::FormulaTests.test_llada_m2t_loss_matches_masked_ce"]
        },
        "formulas.llada2_wsd_block":{
            "component":"LLaDA2.0 WSD schedule",
            "paper_name":"LLaDA2.0",
            "section":"WSD warmup-stable-decay",
            "keyword":"warmup",
            "file_source":"llada2_0.html",
            "tests":["test_formulas.py::FormulaTests.test_llada2_wsd_phase_boundaries","test_pre_sft_sanity.py::PreSFTSanityTests.test_prepare_tokenize_and_one_step_training"]
        },
        "formulas.llada21_sets":{
            "component":"LLaDA2.1 Gamma/Delta sets",
            "paper_name":"LLaDA2.1",
            "section":"threshold-based unmasking and editing",
            "keyword":"threshold",
            "file_source":"llada2_1.html",
            "tests":["test_inference_threshold_decode_invariants.py::ThresholdDecodeInvariantTests.test_gamma_delta_membership_and_disjointness","test_formulas.py::FormulaTests.test_llada21_gamma_delta_sets"]
        },
        "formulas.llada21_apply":{
            "component":"LLaDA2.1 apply sets",
            "paper_name":"LLaDA2.1",
            "section":"Gamma_t and Delta_t application",
            "keyword":"editing",
            "file_source":"llada2_1.html",
            "tests":["test_inference_threshold_decode_invariants.py::ThresholdDecodeInvariantTests.test_apply_semantics"]
        },
        "diffusion.compute_m2t_t2t_losses":{
            "component":"LLaDA2.1 M2T+T2T training mixture",
            "paper_name":"LLaDA2.1",
            "section":"mixture objective for CPT/SFT",
            "keyword":"mixture",
            "file_source":"llada2_1.html",
            "tests":["test_pre_sft_sanity.py::PreSFTSanityTests.test_prepare_tokenize_and_one_step_training","test_audit_report.py::AuditReportTests.test_audit_outputs_exist_and_have_expected_keys"]
        },
        "masks.doc_attention_mask":{
            "component":"LLaDA2.0 doc-level attention",
            "paper_name":"LLaDA2.0",
            "section":"document boundary masking for packing",
            "keyword":"document",
            "file_source":"llada2_0.html",
            "tests":["test_masks.py::MaskTests.test_doc_attention_mask_blocks_cross_doc","test_doc_mask_no_leakage_stronger.py::DocMaskNoLeakageStrongerTests.test_strict_no_cross_doc_leakage"]
        },
        "masks.batch_doc_attention_mask":{
            "component":"LLaDA2.0 composite mask (MBD/MOBC/MBC)",
            "paper_name":"LLaDA2.0",
            "section":"vectorized composite mask families",
            "keyword":"block",
            "file_source":"llada2_0.html",
            "tests":[
                "test_llada20_composite_mask_structure.py::LLaDA20CompositeMaskStructureTests.test_composite_mask_regions",
                "test_llada20_composite_mask_doc_gating.py::LLaDA20CompositeMaskDocGatingTests.test_doc_gating_with_padding"
            ]
        }
    }

def _safe_find_text(path:Path,needle:str)->bool:
    try:
        if not path.exists():
            return False
        if path.suffix.lower()==".pdf":
            return False
        txt=path.read_text(encoding="utf-8",errors="ignore").lower()
        return needle.lower() in txt
    except Exception:
        return False

def _find_test_refs(root:Path,symbol:str)->List[str]:
    test_root=root/"test"
    if not test_root.exists():
        return []
    hits:List[str]=[]
    pattern=symbol.split(".")[-1]
    for fp in sorted(test_root.glob("test_*.py")):
        try:
            txt=fp.read_text(encoding="utf-8",errors="ignore")
        except Exception:
            continue
        if pattern in txt:
            hits.append(fp.name)
    return hits

def _check_invariants(symbol:str)->Tuple[bool,str]:
    try:
        if symbol=="formulas.llada_m2t_loss":
            logits=torch.randn(1,6,13)
            y=torch.tensor([[1,2,3,4,5,6]],dtype=torch.long)
            m=torch.tensor([[0,1,0,1,0,1]],dtype=torch.bool)
            loss=formulas.llada_m2t_loss(logits,y,m)
            return bool(torch.isfinite(loss).item()),"finite masked CE"
        if symbol=="formulas.llada2_wsd_block":
            p0,b0=formulas.llada2_wsd_block(0,3,2,3,1,64,16)
            p1,b1=formulas.llada2_wsd_block(2,3,2,3,1,64,16)
            p2,b2=formulas.llada2_wsd_block(4,3,2,3,1,64,16)
            ok=p0=="warmup" and p1=="warmup" and p2 in {"stable","decay"} and 1<=b0<=b1<=64 and 16<=b2<=64
            return ok,"phase and block bounds"
        if symbol=="formulas.llada21_sets":
            mask_id=99
            x=torch.tensor([[99,5,6,99]],dtype=torch.long)
            pred=torch.tensor([[7,8,6,4]],dtype=torch.long)
            conf=torch.tensor([[0.9,0.8,0.95,0.5]],dtype=torch.float32)
            sets=formulas.llada21_sets(x,pred,conf,mask_id,0.7,0.75)
            disjoint=not torch.any(sets.gamma & sets.delta).item()
            return bool(disjoint and sets.gamma_count==1 and sets.delta_count==1),"Gamma/Delta disjointness"
        if symbol=="formulas.llada21_apply":
            mask_id=99
            x=torch.tensor([[99,5,6,99]],dtype=torch.long)
            pred=torch.tensor([[7,8,6,4]],dtype=torch.long)
            conf=torch.tensor([[0.9,0.8,0.95,0.5]],dtype=torch.float32)
            out,sets=formulas.llada21_apply(x,pred,conf,mask_id,0.7,0.75)
            ok=int(out[0,0].item())==7 and int(out[0,1].item())==8 and int(out[0,2].item())==6 and int(out[0,3].item())==99 and sets.gamma_count==1 and sets.delta_count==1
            return ok,"set application semantics"
        if symbol=="diffusion.compute_m2t_t2t_losses":
            model=TinyCausalLM(vocab_size=128,hidden_size=32)
            ids=torch.randint(0,128,(2,12),dtype=torch.long)
            attn=torch.ones_like(ids)
            docs=torch.tensor([[0]*6+[1]*6,[2]*12],dtype=torch.long)
            resp=torch.tensor([[0]*4+[1]*8,[0]*6+[1]*6],dtype=torch.long)
            cfg=TrainConfig(batch_size=1,accum_steps=1,m2t_weight=0.5,t2t_weight=0.5,mask_ratio=0.2,t2t_noise_ratio=0.2,multi_turn_t2t=2)
            out=diffusion.compute_m2t_t2t_losses(model,ids,attn,docs,resp,mask_id=3,vocab_size=128,cfg=cfg,focus_response=True)
            ok=all(torch.isfinite(out[k]).item() for k in ["loss","loss_m2t","loss_t2t"])
            return ok,"finite M2T/T2T losses"
        if symbol=="masks.doc_attention_mask":
            docs=torch.tensor([0,0,1,1,-1],dtype=torch.long)
            m=masks.doc_attention_mask(docs,causal=False)
            ok=m[0,1].item() and (not m[0,2].item()) and (not m[4,0].item())
            return bool(ok),"no cross-document leakage"
        if symbol=="masks.batch_doc_attention_mask":
            l=16
            bsize=4
            docs=torch.tensor([[0]*l+[0]*l],dtype=torch.long)
            m=masks.batch_doc_attention_mask(docs,causal=False,mask_mode="composite_llada20",block_size=bsize,base_len=l)[0]
            ok_xt_xt=bool(m[1,2].item()) and (not bool(m[1,6].item()))
            ok_xt_x0=bool(m[8,16].item()) and (not bool(m[2,20].item()))
            ok_x0_x0=bool(m[21,17].item()) and (not bool(m[17,21].item()))
            ok_x0_xt=not bool(m[20,2].item())
            return bool(ok_xt_xt and ok_xt_x0 and ok_x0_x0 and ok_x0_xt),"composite_llada20 structure"
        return False,"unknown symbol"
    except Exception as e:
        return False,f"invariant error: {e}"

def _status(implemented:bool,invariant_ok:bool,keyword_found:bool,test_found:bool)->str:
    if not implemented or not invariant_ok:
        return "FAIL"
    if not keyword_found or not test_found:
        return "WARN"
    return "PASS"

def _render_md(rows:List[Dict[str,Any]],summary:Dict[str,int])->str:
    lines=[
        "# Formula Audit",
        "",
        "| Component | Code symbol | Paper | Section/Eq | Tests | Status | Notes |",
        "|---|---|---|---|---|---|---|"
    ]
    for r in rows:
        tests="<br>".join(r["tests"]) if r["tests"] else "-"
        notes=r["notes"].replace("|","/")
        lines.append(f"| {r['component']} | `{r['code_symbol']}` | {r['paper']} | {r['section_eq']} | {tests} | {r['status']} | {notes} |")
    lines.append("")
    lines.append(f"Summary: PASS={summary['PASS']} WARN={summary['WARN']} FAIL={summary['FAIL']}")
    return "\n".join(lines)+"\n"

def run_audit(root_dir:str|Path,out_md:str|Path,out_json:str|Path)->Dict[str,Any]:
    root=Path(root_dir)
    impl=collect_formula_impl()
    pmap=paper_map()
    docs_root=(root/"docs") if (root/"docs").exists() else root
    rows:List[Dict[str,Any]]=[]
    for code_symbol,meta in pmap.items():
        i=impl.get(code_symbol,{})
        implemented=bool(i.get("exists",False))
        inv_ok,inv_note=_check_invariants(code_symbol)
        source_file=docs_root/meta["file_source"]
        keyword_found=_safe_find_text(source_file,meta["keyword"])
        mapped_tests=list(meta.get("tests",[]))
        discovered=_find_test_refs(root,code_symbol)
        test_found=bool(mapped_tests or discovered)
        status=_status(implemented,inv_ok,keyword_found,test_found)
        notes=[]
        notes.append(inv_note if inv_ok else f"invariant failed: {inv_note}")
        if not keyword_found:
            notes.append(f"keyword '{meta['keyword']}' not found in {meta['file_source']}")
        if discovered and not mapped_tests:
            mapped_tests=discovered
        row={
            "component":meta["component"],
            "code_symbol":code_symbol,
            "paper":meta["paper_name"],
            "section_eq":meta["section"],
            "tests":mapped_tests,
            "status":status,
            "notes":"; ".join(notes),
            "implemented":implemented,
            "keyword_found":keyword_found,
            "file_source":meta["file_source"]
        }
        rows.append(row)
    summary={"PASS":0,"WARN":0,"FAIL":0}
    for r in rows:
        summary[r["status"]]+=1
    payload={
        "summary":summary,
        "components":rows,
        "all_pass":summary["FAIL"]==0
    }
    out_md_path=Path(out_md)
    out_json_path=Path(out_json)
    out_md_path.parent.mkdir(parents=True,exist_ok=True)
    out_json_path.parent.mkdir(parents=True,exist_ok=True)
    out_md_path.write_text(_render_md(rows,summary),encoding="utf-8")
    out_json_path.write_text(json.dumps(payload,ensure_ascii=True,indent=2),encoding="utf-8")
    return payload
