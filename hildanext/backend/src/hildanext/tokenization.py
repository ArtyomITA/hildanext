# Tokenization and sequence packing with document boundaries.
# Main entrypoints: load_tokenizer,tokenize_all,ensure_mask_token.
# Output includes doc_ids for doc-level attention masks.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Tuple
import json
import sys
import time
from .config import AppConfig
from .io_utils import read_jsonl,write_jsonl,ensure_dir,write_json
from .utils import SimpleTokenizer
from .trace import use_trace,exception_with_stack

def load_tokenizer(model_dir:str,trust_remote_code:bool=True,trace=None,cfg=None):
    tr=use_trace(cfg,trace)
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=trust_remote_code)
        if tok.pad_token_id is None:
            tok.pad_token=tok.eos_token
        return tok
    except Exception as e:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="tokenization",
                func="load_tokenizer",
                action="simple_tokenizer_fallback",
                reason="tokenizer_load_failed",
                exception_str=exception_with_stack(e),
                extra_dict={"model_dir":model_dir}
            )
        return SimpleTokenizer()

def ensure_mask_token(tokenizer,mask_token:str,model=None)->int:
    mid=getattr(tokenizer,"mask_token_id",None)
    if mid is not None and int(mid)>=0:
        return int(mid)
    vocab=tokenizer.get_vocab() if hasattr(tokenizer,"get_vocab") else {}
    if mask_token in vocab:
        mid=int(vocab[mask_token])
        try:
            tokenizer.mask_token=mask_token
            tokenizer.mask_token_id=mid
        except Exception:
            pass
        return mid
    added=tokenizer.add_special_tokens({"additional_special_tokens":[mask_token]})
    mid=int(tokenizer.convert_tokens_to_ids(mask_token))
    try:
        tokenizer.mask_token=mask_token
        tokenizer.mask_token_id=mid
    except Exception:
        pass
    if model is not None and added>0 and hasattr(model,"resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    return mid

def _ids_within_vocab(ids:List[int],vocab_size:int)->List[int]:
    if vocab_size<=0:
        return ids
    out=[]
    for x in ids:
        xi=int(x)
        if xi<0:
            xi=0
        if xi>=vocab_size:
            xi=xi%vocab_size
        out.append(xi)
    return out

def _encode_text(tokenizer,text:str)->List[int]:
    enc=tokenizer(text,return_tensors=None)
    ids=enc["input_ids"][0] if isinstance(enc["input_ids"],list) and enc["input_ids"] and isinstance(enc["input_ids"][0],list) else enc["input_ids"]
    if isinstance(ids,list):
        return [int(x) for x in ids]
    return [int(x) for x in ids.tolist()]

def _encode_record(tokenizer,record:Dict[str,Any],vocab_size:int)->Tuple[List[int],List[int],str]:
    src=str(record.get("source","unknown"))
    if isinstance(record.get("token_ids"),list) and record["token_ids"]:
        ids=_ids_within_vocab([int(x) for x in record["token_ids"]],vocab_size)
        resp=[0]*len(ids)
        return ids,resp,src
    prompt=str(record.get("prompt","")).strip()
    response=str(record.get("response","")).strip()
    if prompt and response:
        prefix=f"User: {prompt}\nAssistant:"
        full=f"{prefix} {response}"
        pids=_encode_text(tokenizer,prefix)
        fids=_encode_text(tokenizer,full)
        fids=_ids_within_vocab(fids,vocab_size)
        resp=[0]*min(len(pids),len(fids))+[1]*max(0,len(fids)-len(pids))
        resp=resp[:len(fids)]
        if len(resp)<len(fids):
            resp=resp+[1]*(len(fids)-len(resp))
        return fids,resp,src
    text=str(record.get("text","")).strip()
    ids=_encode_text(tokenizer,text) if text else []
    ids=_ids_within_vocab(ids,vocab_size)
    resp=[0]*len(ids)
    return ids,resp,src

def _pack(encoded:List[Tuple[List[int],List[int],str]],seq_len:int,pad_id:int,eos_id:int)->List[Dict[str,Any]]:
    out=[]
    cur_ids:List[int]=[]
    cur_docs:List[int]=[]
    cur_resp:List[int]=[]
    cur_src="mixed"
    doc_i=0
    def flush():
        if not cur_ids:
            return
        need=seq_len-len(cur_ids)
        ids=cur_ids+[pad_id]*need
        docs=cur_docs+[-1]*need
        resp=cur_resp+[0]*need
        attn=[1 if d>=0 else 0 for d in docs]
        out.append({"input_ids":ids,"doc_ids":docs,"response_mask":resp,"attention_mask":attn,"source":cur_src})
    for ids,resp,src in encoded:
        if not ids:
            continue
        pos=0
        while pos<len(ids):
            room=max(1,seq_len-len(cur_ids))
            take=max(1,room-1)
            chunk_ids=ids[pos:pos+take]
            chunk_resp=resp[pos:pos+take] if resp else [0]*len(chunk_ids)
            pos+=len(chunk_ids)
            if pos>=len(ids):
                chunk_ids=chunk_ids+[eos_id]
                chunk_resp=chunk_resp+[0]
            if len(cur_ids)+len(chunk_ids)>seq_len and cur_ids:
                flush()
                cur_ids=[]
                cur_docs=[]
                cur_resp=[]
            if len(chunk_ids)>seq_len:
                chunk_ids=chunk_ids[:seq_len]
                chunk_resp=chunk_resp[:seq_len]
            cur_ids.extend(chunk_ids)
            cur_docs.extend([doc_i]*len(chunk_ids))
            cur_resp.extend(chunk_resp)
            cur_src=src
            if len(cur_ids)==seq_len:
                flush()
                cur_ids=[]
                cur_docs=[]
                cur_resp=[]
        doc_i+=1
    if cur_ids:
        flush()
    return out

def _pack_streaming(
    encoded:List[Tuple[List[int],List[int],str]],
    seq_len:int,pad_id:int,eos_id:int,
    carry_ids:list,carry_docs:list,carry_resp:list,carry_src:str,doc_offset:int
)->Tuple[list,list,list,list,str,int]:
    """Like _pack but accepts and returns carry-over partial sequence for streaming across chunks."""
    out:list=[]
    cur_ids=list(carry_ids)
    cur_docs=list(carry_docs)
    cur_resp=list(carry_resp)
    cur_src=carry_src
    doc_i=doc_offset
    def flush()->None:
        if not cur_ids:
            return
        need=seq_len-len(cur_ids)
        ids=cur_ids+[pad_id]*need
        docs=cur_docs+[-1]*need
        resp=cur_resp+[0]*need
        attn=[1 if d>=0 else 0 for d in docs]
        out.append({"input_ids":ids,"doc_ids":docs,"response_mask":resp,"attention_mask":attn,"source":cur_src})
    for ids,resp,src in encoded:
        if not ids:
            continue
        pos=0
        while pos<len(ids):
            room=max(1,seq_len-len(cur_ids))
            take=max(1,room-1)
            chunk_ids=ids[pos:pos+take]
            chunk_resp=resp[pos:pos+take] if resp else [0]*len(chunk_ids)
            pos+=len(chunk_ids)
            if pos>=len(ids):
                chunk_ids=chunk_ids+[eos_id]
                chunk_resp=chunk_resp+[0]
            if len(cur_ids)+len(chunk_ids)>seq_len and cur_ids:
                flush()
                cur_ids=[]
                cur_docs=[]
                cur_resp=[]
            if len(chunk_ids)>seq_len:
                chunk_ids=chunk_ids[:seq_len]
                chunk_resp=chunk_resp[:seq_len]
            cur_ids.extend(chunk_ids)
            cur_docs.extend([doc_i]*len(chunk_ids))
            cur_resp.extend(chunk_resp)
            cur_src=src
            if len(cur_ids)==seq_len:
                flush()
                cur_ids=[]
                cur_docs=[]
                cur_resp=[]
        doc_i+=1
    return out,cur_ids,cur_docs,cur_resp,cur_src,doc_i

def tokenize_split(cfg:AppConfig,input_path:str,output_path:str,max_records:int|None=None,trace=None)->Dict[str,Any]:
    """Stream-based tokenization: processes CHUNK_ROWS rows at a time to avoid OOM on large files."""
    tr=use_trace(cfg,trace)
    tok=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code,trace=tr,cfg=cfg)
    vocab_size=len(tok) if hasattr(tok,"__len__") else 0
    pad_id=int(getattr(tok,"pad_token_id",0) or 0)
    eos_id=int(getattr(tok,"eos_token_id",1) or 1)
    seq_len=cfg.data.seq_len
    CHUNK_ROWS=5000  # process 5k rows at a time (~30 MB peak RAM per chunk)
    records_in=0
    records_out=0
    # Carry-over: partial sequence that didn't fill seq_len in previous chunk
    carry_ids:list=[]
    carry_docs:list=[]
    carry_resp:list=[]
    carry_src:str="mixed"
    doc_offset=0  # global doc counter across chunks to keep doc_ids unique
    ensure_dir(str(Path(output_path).parent))
    input_size=Path(input_path).stat().st_size if Path(input_path).exists() else 0
    t0=time.time()
    last_log=t0
    LOG_EVERY=30  # seconds between progress lines
    print(f"[tokenize] START  input={Path(input_path).name}  ({input_size//1024//1024} MB)  chunk={CHUNK_ROWS} rows",flush=True)
    with open(output_path,"w",encoding="utf-8") as out_f, \
         open(input_path,"r",encoding="utf-8",errors="ignore") as in_f:
        chunk:list=[]
        def flush_chunk():
            nonlocal records_out,carry_ids,carry_docs,carry_resp,carry_src,doc_offset,last_log
            if not chunk:
                return
            enc=[_encode_record(tok,r,vocab_size) for r in chunk]
            out_rows,carry_ids,carry_docs,carry_resp,carry_src,doc_offset=_pack_streaming(
                enc,seq_len,pad_id,eos_id,
                carry_ids,carry_docs,carry_resp,carry_src,doc_offset)
            for row in out_rows:
                out_f.write(json.dumps(row,separators=(",",":"))+"\n")
                records_out+=1
            chunk.clear()
            now=time.time()
            if now-last_log>=LOG_EVERY:
                elapsed=now-t0
                out_mb=Path(output_path).stat().st_size//1024//1024 if Path(output_path).exists() else 0
                rate=records_in/elapsed if elapsed>0 else 0
                eta=int((input_size/max(1,Path(input_path).stat().st_size-input_size+1))) if input_size>0 else 0
                print(f"[tokenize] rows_in={records_in:,}  seqs_out={records_out:,}  out={out_mb} MB  rate={rate:.0f} rows/s  elapsed={int(elapsed)}s",flush=True)
                last_log=now
        for line in in_f:
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except Exception:
                continue
            chunk.append(row)
            records_in+=1
            if max_records and records_in>=max_records:
                flush_chunk()
                break
            if len(chunk)>=CHUNK_ROWS:
                flush_chunk()
        flush_chunk()
        # Flush any remaining carry-over partial sequence
        if carry_ids:
            need=seq_len-len(carry_ids)
            ids=carry_ids+[pad_id]*need
            docs=carry_docs+[-1]*need
            resp=carry_resp+[0]*need
            attn=[1 if d>=0 else 0 for d in docs]
            row={"input_ids":ids,"doc_ids":docs,"response_mask":resp,"attention_mask":attn,"source":carry_src}
            out_f.write(json.dumps(row,separators=(",",":"))+"\n")
            records_out+=1
    rep={"input":input_path,"output":output_path,"records_in":records_in,"records_out":records_out,"seq_len":seq_len}
    if tr is not None:
        rep["fallbacks"]=tr.snapshot_fallbacks(limit=16)
    return rep

def tokenize_all(cfg:AppConfig,max_records:int|None=None,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    ensure_dir(cfg.paths.tokenized_dir)
    in_train=str(Path(cfg.paths.processed_dir)/"train.jsonl")
    in_eval=str(Path(cfg.paths.processed_dir)/"eval.jsonl")
    in_sft_train=str(Path(cfg.paths.processed_dir)/"sft_train.jsonl")
    in_sft_eval=str(Path(cfg.paths.processed_dir)/"sft_eval.jsonl")
    out_train=str(Path(cfg.paths.tokenized_dir)/"train.jsonl")
    out_eval=str(Path(cfg.paths.tokenized_dir)/"eval.jsonl")
    out_sft_train=str(Path(cfg.paths.tokenized_dir)/"sft_train.jsonl")
    out_sft_eval=str(Path(cfg.paths.tokenized_dir)/"sft_eval.jsonl")
    rep={
        "train":tokenize_split(cfg,in_train,out_train,max_records=max_records,trace=tr),
        "eval":tokenize_split(cfg,in_eval,out_eval,max_records=max_records,trace=tr),
        "sft_train":tokenize_split(cfg,in_sft_train,out_sft_train,max_records=max_records,trace=tr),
        "sft_eval":tokenize_split(cfg,in_sft_eval,out_sft_eval,max_records=max_records,trace=tr)
    }
    rep["fallbacks"]=tr.snapshot_fallbacks(limit=32) if tr is not None else []
    write_json(Path(cfg.paths.tokenized_dir)/"manifest.tokenize.json",rep)
    return rep
