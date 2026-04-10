import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from hildanext.tokenization import load_tokenizer
except ImportError:
    import sys
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    from hildanext.tokenization import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("audit_wsd_dataset")

def analyze_schema(record: Dict[str, Any]) -> str:
    if "source_type" in record:
        return record["source_type"]
    if "prompt" in record and "response" in record:
        return "prompt_response"
    if "messages" in record:
        return "messages_chat"
    if "text" in record:
        text = record["text"]
        if "<think>" in text:
            return "think"
        if "/no_think" in text or "<think>\n\n</think>" in text:
            return "no_think"
        return "raw_text"
    return "unknown"

def check_segmentation(text: str, s_type: str) -> List[str]:
    issues = []
    if "<|im_start|>" in text and "<|im_end|>" not in text:
        issues.append("Missing <|im_end|>")
    if "<think>" in text and "</think>" not in text:
        issues.append("Missing </think>")
    
# Heuristic for generic blind 120-word chunking
    if s_type == "raw_text":
        words = text.split()
        if len(words) in [119, 120, 121]:
            issues.append("Suspicious 120-word chunk boundary")

        stripped = text.strip()
        if stripped and stripped[-1] in [',', 'and', 'or', 'but']:
            issues.append("Ends with conjunction/comma")
            
    return issues

def audit_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--processed_path", type=str, default="")
    parser.add_argument("--tokenized_path", type=str, default="")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--out_name", type=str, default="audit_report")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail_on_warning", action="store_true")
    parser.add_argument("--audit_legacy", action="store_true")
    parser.add_argument("--audit_qwen_prep", action="store_true")
    args = parser.parse_args()

    reports_dir = Path("E:/DIFFUSION/HildaNext/hildanext/runs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # 1. Source Discovery
    logger.info("PHASE 1: SOURCE DISCOVERY")
    proc_path = None
    tok_path = None
    
    if args.dataset_root:
        root = Path(args.dataset_root)
        proc_path = root.parent.parent / "processed_qwen_wsd" / root.name / "train.jsonl"
        if not proc_path.exists():
             proc_path = root / "train.jsonl"
        tok_path = root / "train.jsonl" if "tokenized" in str(root) else root.parent.parent / "tokenized_qwen_wsd" / root.name / "train.jsonl"
    elif args.processed_path and args.tokenized_path:
        proc_path = Path(args.processed_path)
        tok_path = Path(args.tokenized_path)
    
    if not proc_path or not proc_path.exists() or not tok_path or not tok_path.exists():
        logger.error(f"Cannot find dataset artifacts! Proc: {proc_path}, Tok: {tok_path}")
        return

    logger.info(f"Processed: {proc_path} ({proc_path.stat().st_size} bytes)")
    logger.info(f"Tokenized: {tok_path} ({tok_path.stat().st_size} bytes)")
    
    # Load Tokenizer
    tokenizer = load_tokenizer(args.model_dir)
    
    # 2. Schema Audit & 3. Rendering & 4. Think Audit & 5. Segmentation Audit
    logger.info("PHASES 2-5: SCHEMA, RENDER, THINK AND SEGMENTATION AUDIT")
    stats = {"raw_text": 0, "messages_chat": 0, "prompt_response": 0, "think": 0, "no_think": 0, "unknown": 0}
    
    samples = []
    seg_ok = 0
    seg_warn = 0
    seg_err = 0
    
    with open(proc_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            rec = json.loads(line)
            s_type = analyze_schema(rec)
            text = rec.get("formatted", rec.get("text", ""))
            
            if s_type not in stats:
                stats[s_type] = 0
            stats[s_type] += 1
            
            issues = check_segmentation(text, s_type)
            if "Missing <|im_end|>" in issues or "Missing </think>" in issues:
                seg_err += 1
            elif issues:
                seg_warn += 1
            else:
                seg_ok += 1
                
            if len(samples) < args.max_samples * 5:
                # Need to run tokenizer to get counts and rendering
                if s_type == "raw_text":
                    final_render = text
                else:
                    try:
                        # Try to handle messages or prompt_response
                        messages = rec.get("messages", [])
                        if not messages and "prompt" in rec and "response" in rec:
                            messages = [{"role": "user", "content": rec["prompt"]}, {"role": "assistant", "content": rec["response"]}]
                        if messages:
                            final_render = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        else:
                            final_render = text
                    except Exception:
                         final_render = text

                orig_short = text[:200] + ("..." if len(text)>200 else "")
                enc = tokenizer(final_render, add_special_tokens=False)
                t_ids = enc["input_ids"] if isinstance(enc["input_ids"], list) else enc["input_ids"].tolist()

                samples.append({
                    "id": i,
                    "source_type": s_type,
                    "source_name": rec.get("source_name", rec.get("source", "unknown")),
                    "original_excerpt": orig_short,
                    "rendered_qwen": final_render,
                    "first_tokens": str(t_ids[:10]),
                    "token_len": len(t_ids),
                    "flags": {
                        "has_im_start": "<|im_start|>" in final_render,
                        "has_im_end": "<|im_end|>" in final_render,
                        "has_think_open": "<think>" in final_render,
                        "has_think_close": "</think>" in final_render,
                        "has_no_think_switch": "/no_think" in final_render,
                        "contains_literal_User_Assistant": ("User:" in final_render and "Assistant:" in final_render),
                        "contains_broken_turns": bool(issues)
                    }
                })

    # Pick samples to print
    random.shuffle(samples)
    printed_samples = ""
    for s in samples[:args.max_samples]:
        printed_samples += f"=== SAMPLE {s['id']} ===\n"
        printed_samples += f"source_type: {s['source_type']}\n"
        printed_samples += f"source_name: {s['source_name']}\n"
        printed_samples += f"token_len: {s['token_len']}\n"
        printed_samples += f"first_tokens: {s['first_tokens']}\n"
        printed_samples += f"flags: {json.dumps(s['flags'])}\n"
        printed_samples += f"original_excerpt:\n{s['original_excerpt']}\n"
        printed_samples += f"rendered_qwen:\n{s['rendered_qwen'][:500]}...\n"
        printed_samples += "===================\n\n"
        
        # 6. Packing Audit
    logger.info("PHASE 6: PACKING AND INITIAL IDs")
    pack_samples_raw_boundary = ""
    pack_samples_chat_boundary = ""
    pack_samples_pad = ""
    packing_ok = True
    
    # Try to find a sequence with cross-doc boundary, and one with padding
    b_raw_found = 0
    b_chat_found = 0
    padding_found = 0
    
    raw_boundary_tokens = set()
    chat_boundary_tokens = set()
    pad_tokens_obs = set()
    
    with open(tok_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples * 20 and b_raw_found > 0 and b_chat_found > 0 and padding_found > 0: 
                break
            
            rec = json.loads(line)
            ids = rec.get("input_ids", [])
            docs = rec.get("doc_ids", [])
            source = str(rec.get("source", "misc"))
            
            if len(ids) != args.seq_len:
                packing_ok = False
                
            has_boundary = False
            for j in range(1, len(docs)):
                if docs[j] != docs[j-1] and docs[j-1] != -1:
                    has_boundary = True
                    break
                    
            has_padding = -1 in docs
            
            if has_boundary:
                for j in range(1, len(docs)):
                    if docs[j] != docs[j-1]:
                        bound_tok = ids[j-1]
                        bound_tok_str = tokenizer.convert_ids_to_tokens(bound_tok) or str(bound_tok)
                        
                        start_idx = max(0, j - 15)
                        end_idx = min(len(ids), start_idx + 30)
                        decoded = tokenizer.decode(ids[start_idx:end_idx])
                        doc_str = " ".join(map(str, docs[start_idx:end_idx]))
                        tok_str = str(ids[start_idx:end_idx])

                        if source == "raw" or "raw" in source:
                            raw_boundary_tokens.add(bound_tok_str)
                            if b_raw_found == 0:
                                pack_samples_raw_boundary += f"=== RAW BOUNDARY SAMPLE ===\ndoc_ids around boundary: {doc_str}\ntokens around boundary: {tok_str}\ntransition_token: {bound_tok_str}\ndecoded excerpt: {repr(decoded)}\n=====================\n\n"
                            b_raw_found += 1
                        else:
                            chat_boundary_tokens.add(bound_tok_str)
                            if b_chat_found == 0:
                                pack_samples_chat_boundary += f"=== CHAT BOUNDARY SAMPLE ===\ndoc_ids around boundary: {doc_str}\ntokens around boundary: {tok_str}\ntransition_token: {bound_tok_str}\ndecoded excerpt: {repr(decoded)}\n=====================\n\n"
                            b_chat_found += 1
                        break
                        
            if has_padding:
                for j in range(1, len(docs)):
                    if docs[j] == -1:
                        # doc[j] is padding, so ids[j] is pad token
                        pad_tok = ids[j]
                        pad_tok_str = tokenizer.convert_ids_to_tokens(pad_tok) or str(pad_tok)
                        pad_tokens_obs.add(pad_tok_str)
                        
                        if padding_found == 0:
                            start_idx = max(0, j - 10)
                            end_idx = min(len(ids), start_idx + 20)
                            decoded = tokenizer.decode(ids[start_idx:end_idx])
                            doc_str = " ".join(map(str, docs[start_idx:end_idx]))
                            tok_str = str(ids[start_idx:end_idx])
                            
                            pack_samples_pad += f"=== PADDING SAMPLE ===\ndoc_ids tail: {doc_str}\ntokens tail: {tok_str}\nobserved_pad_token: {pad_tok_str}\ndecoded excerpt: {repr(decoded)}\n=====================\n\n"
                        padding_found += 1
                        break
                        
    # 7. Token Semantics Audit Validation
    # Raw boundaries should use <|endoftext|>
    # Chat boundaries should use <|im_end|>
    semantics_ok = True
    sem_verdict = "GO_100"
    
    if "<|im_end|>" in raw_boundary_tokens:
        semantics_ok = False
        sem_verdict = "GO_WITH_WARNINGS" if p_score > 0 else "NO_GO"
        
    sem_text = (
        "## Token Semantics Audit\n"
        f"- tokenizer.eos_token: {tokenizer.eos_token} (id {tokenizer.eos_token_id})\n"
        f"- tokenizer.pad_token: {getattr(tokenizer, 'pad_token', 'None')} (id {getattr(tokenizer, 'pad_token_id', 'None')})\n"
        f"- eot_id_check: {tokenizer.convert_tokens_to_ids('<|endoftext|>')}\n"
        f"- im_end_id_check: {tokenizer.convert_tokens_to_ids('<|im_end|>')}\n"
        f"- raw_document_boundary_token_detected: {list(raw_boundary_tokens)}\n"
        f"- chat_turn_boundary_token_detected: {list(chat_boundary_tokens)}\n"
        f"- padding_token_detected: {list(pad_tokens_obs)}\n"
        f"- semantics_verdict: {sem_verdict}\n\n"
    )
    
    # 8. Scores
    q_score = 100 if "User:" not in printed_samples else 40
    t_score = 100 if seg_err == 0 else 60
    s_score = max(0, 100 - (seg_warn * 2) - (seg_err * 10))
    p_score = 100 if packing_ok else 0
    overall = (q_score + t_score + s_score + p_score) / 4
    
    verdict = "GO"
    if seg_err > 0 or p_score == 0 or sem_verdict == "NO_GO":
        verdict = "NO_GO"
    elif seg_warn > 0 or q_score < 100 or sem_verdict == "GO_WITH_WARNINGS":
        verdict = "GO_WITH_WARNINGS"
        
    if sem_verdict == "GO_100" and verdict == "GO":
        verdict = "GO_100"
        
    if args.strict and verdict not in ("GO", "GO_100"):
        verdict = "NO_GO"

    report = {
        "verdict": verdict,
        "scores": {
            "qwen_format": q_score,
            "think_nothink": t_score,
            "segmentation": s_score,
            "packing": p_score,
            "overall": overall
        },
        "stats": stats,
        "segmentation_health": {
            "ok": seg_ok,
            "warn": seg_warn,
            "err": seg_err
        }
    }
    
    # 8. Report out
    json_path = reports_dir / f"{timestamp}_{args.out_name}_wsd_dataset_audit_qwen.json"
    md_path = reports_dir / f"{timestamp}_{args.out_name}_wsd_dataset_audit_qwen.md"
    
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# QWEN WSD DATASET AUDIT REPORT\n\n")
        f.write(f"**Verdict:** {verdict}\n")
        f.write(f"**Overall Score:** {overall}/100\n\n")
        f.write(f"## Segmentation Health\n- OK: {seg_ok}\n- Warn: {seg_warn}\n- Err: {seg_err}\n\n")
        f.write(f"## Dataset Stats\n{json.dumps(stats, indent=2)}\n\n")
        f.write(f"{sem_text}")
        f.write(f"## Rendered Samples\n```\n{printed_samples}\n```\n\n")
        f.write(f"## Packing Samples\n```\n{pack_samples_raw_boundary}{pack_samples_chat_boundary}{pack_samples_pad}\n```\n\n")
        if verdict != "GO":
            f.write(f"## Minimum Required Fixes\n")
            if seg_err > 0: f.write("- P0: Resolve missing </think> or <|im_end|> tags.\n")
            if p_score == 0: f.write("- P0: Fix sequence chunk packing lengths.\n")
            if q_score < 100: f.write("- P1: Remove hardcoded 'User:' templates.\n")

    logger.info(f"Audit completed. Verdict: {verdict}")
    logger.info(f"Report: {md_path}")

if __name__ == "__main__":
    audit_dataset()
