import argparse
import collections
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# Enable running as a script even if not installed
try:
    from hildanext.tokenization import ensure_mask_token, load_tokenizer, _pack_streaming
    from hildanext.io_utils import write_json, ensure_dir
except ImportError:
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    from hildanext.tokenization import ensure_mask_token, load_tokenizer, _pack_streaming
    from hildanext.io_utils import write_json, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dataset_prep_qwen")

def download_tiny_overlays(base_dir: Path) -> Tuple[Path, Path]:
    """Downloads tiny subset from OpenThoughts-114k for think traces."""
    cached_dir = base_dir / "data" / "cached_hf_overlays"
    ensure_dir(cached_dir)
    think_path = cached_dir / "openthoughts_tiny.jsonl"
    
    if not think_path.exists():
        logger.info(f"Downloading tiny OpenThoughts subset to {think_path}...")
        try:
            from datasets import load_dataset
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
            count = 0
            with open(think_path, "w", encoding="utf-8") as f:
                for row in ds:
                    if count >= 2000:
                        break
                    # OpenThoughts usually has 'conversations' with 'value' and 'from' (or 'role')
                    messages = []
                    for turn in row.get("conversations", []):
                        role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                        content = turn.get("value", "")
                        messages.append({"role": role, "content": content})
                    if messages:
                        f.write(json.dumps({"messages": messages}) + "\n")
                        count += 1
            logger.info(f"Downloaded {count} think examples.")
        except ImportError:
            logger.warning("Failed to download HF overlays: 'datasets' module not found. Run: pip install datasets")
        except Exception as e:
            logger.warning(f"Failed to download HF overlays: {e}")
            
    return think_path

def stream_local_curated(dir_path: Path) -> Iterator[Dict[str, Any]]:
    if not dir_path.exists() or not dir_path.is_dir():
        return
    for fp in dir_path.glob("*.jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def stream_dolma_raw(raw_path: Path, max_docs: int) -> Iterator[Dict[str, Any]]:
    if not raw_path.exists():
        return
    import gzip
    count = 0
    for fp in raw_path.rglob("*.json.gz"):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                if count >= max_docs:
                    return
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        yield {"text": text}
                        count += 1

def format_structured_qwen(tokenizer, record: Dict[str, Any], source_type: str, strategy: str) -> str:
    messages = []
    
    # Support both SCHEMA A and SCHEMA B
    if "messages" in record:
        messages = record["messages"]
    elif "prompt" in record and "response" in record:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]}
        ]
    else:
        return ""

    if source_type == "no_think":
        # Apply strict no_think strategy
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m["role"] == "user":
                last_user_idx = i
                
        # Handle slash_no_think
        if strategy in ["slash_no_think", "both"] and last_user_idx >= 0:
            if not messages[last_user_idx]["content"].endswith(" /no_think"):
                messages[last_user_idx]["content"] += " /no_think"
                
        # NOTE: manual empty_think prepend removed.
        # The Qwen3 chat template automatically wraps the last assistant
        # turn with <think>\n\n</think> when enable_thinking=True (default),
        # so prepending manually was redundant.

    elif source_type == "think":
        # Ensure we don't fabricate false reasoning. If no <think>, we should have skipped it,
        # but if we didn't, leave it alone (or we could force skip).
        pass

    try:
        # standard strict Qwen formatting
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return formatted
    except Exception as e:
        logger.warning(f"Error applying chat template: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="raw_both")
    parser.add_argument("--raw_weight", type=float, default=0.90)
    parser.add_argument("--no_think_weight", type=float, default=0.08)
    parser.add_argument("--think_weight", type=float, default=0.02)
    parser.add_argument("--max_raw_docs", type=int, default=100000)
    parser.add_argument("--max_nothink_examples", type=int, default=4000)
    parser.add_argument("--max_think_examples", type=int, default=1000)
    parser.add_argument("--nothink_strategy", default="both", choices=["empty_think", "slash_no_think", "both"])
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--download_tiny_overlays", action="store_true")
    parser.add_argument("--use_local_curated_only", action="store_true")
    parser.add_argument("--out_name", default="qwen_wsd_run")
    parser.add_argument("--workspace_root", default="E:/DIFFUSION/HildaNext")
    parser.add_argument("--model_dir", default="E:/DIFFUSION/HildaNext/hildanext/models/qwen3-0.6b")
    parser.add_argument("--dolma_dir", default="E:/DIFFUSION/HildaNext/dolma_v1_6_sample_1767050862/raw")

    args = parser.parse_args()

    ws_root = Path(args.workspace_root)
    hilda_dir = ws_root / "hildanext"
    data_dir = hilda_dir / "data"
    curated_dir = data_dir / "curated_sources"
    
    ensure_dir(curated_dir / "chat")
    ensure_dir(curated_dir / "no_think")
    ensure_dir(curated_dir / "think")

    out_base = args.out_name
    proc_dir = data_dir / "processed_qwen_wsd" / out_base
    tok_dir = data_dir / "tokenized_qwen_wsd" / out_base
    reports_dir = hilda_dir / "runs" / "reports"
    ensure_dir(reports_dir)

    if args.dry_run:
        logger.info("DRY RUN MODE. Artifacts will not be fully built.")

    # Load Tokenizer
    logger.info(f"Loading Qwen3 tokenizer from {args.model_dir}...")
    tokenizer = load_tokenizer(args.model_dir, trust_remote_code=True)
    mask_token_id = ensure_mask_token(tokenizer, "<|mask|>")
    
    # Collect sources
    think_sources: List[Iterator[Dict[str, Any]]] = []
    nothink_sources: List[Iterator[Dict[str, Any]]] = []
    
    if args.download_tiny_overlays:
        think_path = download_tiny_overlays(ws_root)
        if think_path.exists():
            def _th():
                with open(think_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line)
            think_sources.append(_th())

    nothink_sources.append(stream_local_curated(curated_dir / "no_think"))
    think_sources.append(stream_local_curated(curated_dir / "think"))
    
    # Reservoir sampling buffer structure
    # To keep it memory efficient without pure reservoir, we'll collect up to max, then shuffle/mix
    
    logger.info(f"Collecting up to {args.max_think_examples} think examples...")
    think_examples = []
    for src in think_sources:
        for rec in src:
            if len(think_examples) >= args.max_think_examples:
                break
            think_examples.append(rec)
            
    logger.info(f"Collecting up to {args.max_nothink_examples} no_think examples...")
    nothink_examples = []
    for src in nothink_sources:
        for rec in src:
            if len(nothink_examples) >= args.max_nothink_examples:
                break
            nothink_examples.append(rec)
            
    raw_examples = []
    if not args.use_local_curated_only and args.mode in ["raw_only", "raw_no_think", "raw_think", "raw_both"]:
        logger.info(f"Collecting up to {args.max_raw_docs} raw docs from Dolma...")
        dolma_raw_path = Path(args.dolma_dir)
        for rec in stream_dolma_raw(dolma_raw_path, args.max_raw_docs):
            raw_examples.append(rec)

    # Format arrays
    def format_all(arr, s_type):
        out = []
        for rec in arr:
            if s_type == "raw":
                txt = rec.get("text", "")
                if txt:
                    out.append({"text": txt, "source_type": "raw", "formatted": txt})
            else:
                fmt = format_structured_qwen(tokenizer, rec, s_type, args.nothink_strategy)
                if fmt:
                    out.append({"text": fmt, "source_type": s_type, "formatted": fmt})
        return out

    fmt_think = format_all(think_examples, "think")
    fmt_nothink = format_all(nothink_examples, "no_think")
    fmt_raw = format_all(raw_examples, "raw")

    # Construct Mix
    mixed = []
    if args.mode == "raw_only":
        mixed = fmt_raw
    elif args.mode == "raw_no_think":
        mixed = fmt_raw + fmt_nothink
    elif args.mode == "raw_think":
        mixed = fmt_raw + fmt_think
    elif args.mode == "raw_both":
        mixed = fmt_raw + fmt_nothink + fmt_think
        
    random.shuffle(mixed)
    
    total = len(mixed)
    eval_count = max(1, int(total * args.eval_ratio))
    if total == 0:
        logger.warning("No examples collected! Check your sources.")
        eval_count = 0
        
    eval_set = mixed[:eval_count]
    train_set = mixed[eval_count:]

    # Report
    report = {
        "mode": args.mode,
        "weights": {"raw": args.raw_weight, "no_think": args.no_think_weight, "think": args.think_weight},
        "examples_found": {
            "raw": len(fmt_raw),
            "no_think": len(fmt_nothink),
            "think": len(fmt_think)
        },
        "samples": {
            "raw": [x["formatted"] for x in fmt_raw[:3]],
            "no_think": [x["formatted"] for x in fmt_nothink[:3]],
            "think": [x["formatted"] for x in fmt_think[:3]]
        },
        "output_paths": {}
    }
    
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{timestamp}_{args.out_name}_prep_report.json"

    if args.dry_run:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Dry run complete. Report saved to {report_path}")
        return

    # Write processed
    ensure_dir(proc_dir)
    ensure_dir(tok_dir)
    
    with open(proc_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for m in train_set:
            f.write(json.dumps(m) + "\n")
            
    with open(proc_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for m in eval_set:
            f.write(json.dumps(m) + "\n")
            
    # Tokenization and packing step
    def _do_pack_and_write(dataset, out_path_jsonl):
        # Determine actual token IDs for semantics separation
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eot_id is None or eot_id < 0:
            eot_id = tokenizer.eos_token_id  # Fallback if somehow missing
        
        # We will tokenize each and use _pack_streaming logic inline
        carry_ids, carry_docs, carry_resp, carry_src = [], [], [], ""
        doc_offset = 0
        
        with open(out_path_jsonl, "w", encoding="utf-8") as fw:
            for rec in dataset:
                doc_offset += 1
                s_type = rec["source_type"]
                text = rec["formatted"]
                
                # tokenize
                enc = tokenizer(text, add_special_tokens=False)
                input_ids = enc["input_ids"] if isinstance(enc["input_ids"], list) else enc["input_ids"].tolist()
                
                resp_mask = [1] * len(input_ids)
                
                # Semantic boundaries!
                # All doc types get <|endoftext|> appended as document boundary.
                # Raw docs: eot separates packed documents (per Qwen spec).
                # Chat docs: eot follows the final <|im_end|>, matching Qwen's
                #   pre-training format: ...<|im_end|><|endoftext|>
                use_eos_id = eot_id
                
                packed, carry_ids, carry_docs, carry_resp, carry_src, doc_offset = _pack_streaming(
                    [(input_ids, resp_mask, s_type)],
                    args.seq_len,
                    pad_id=eot_id,
                    eos_id=use_eos_id,
                    carry_ids=carry_ids,
                    carry_docs=carry_docs,
                    carry_resp=carry_resp,
                    carry_src=carry_src,
                    doc_offset=doc_offset,
                    trunc_prob=0.01
                )
                
                for p in packed:
                    fw.write(json.dumps({
                        "input_ids": p["input_ids"],
                        "doc_ids": p["doc_ids"],
                        "attention_mask": p.get("attention_mask", [1 if d >= 0 else 0 for d in p["doc_ids"]]),
                        "response_mask": p["response_mask"],
                        "source": p["source"]
                    }) + "\n")
            
            # Flush final carry manually if any
            if carry_ids:
                need = args.seq_len - len(carry_ids)
                final_doc_ids = carry_docs + [-1] * need
                p = {
                    "input_ids": carry_ids + [eot_id] * need,
                    "doc_ids": final_doc_ids,
                    "attention_mask": [1 if d >= 0 else 0 for d in final_doc_ids],
                    "response_mask": carry_resp + [0] * need,
                    "source": carry_src
                }
                fw.write(json.dumps(p) + "\n")

    logger.info("Packing train set...")
    _do_pack_and_write(train_set, tok_dir / "train.jsonl")
    logger.info("Packing eval set...")
    _do_pack_and_write(eval_set, tok_dir / "eval.jsonl")

    report["output_paths"] = {
        "processed_train": str(proc_dir / "train.jsonl"),
        "processed_eval": str(proc_dir / "eval.jsonl"),
        "tokenized_train": str(tok_dir / "train.jsonl"),
        "tokenized_eval": str(tok_dir / "eval.jsonl"),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Done. Report: {report_path}")

if __name__ == "__main__":
    main()
