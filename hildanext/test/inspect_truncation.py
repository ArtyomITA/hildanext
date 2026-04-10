"""Inspect tokenized data to understand truncation at position 512."""
import json, sys
sys.path.insert(0, 'backend/src')
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('../Qwen3-0.6B')
print(f"EOS={tok.eos_token_id}  PAD={tok.pad_token_id}")

# Key special token IDs
SPECIAL = {151643:'<|endoftext|>', 151644:'<|im_start|>', 151645:'<|im_end|>',
           151667:'<think>', 151668:'</think>'}

with open('data/tokenized_qwen_wsd/qwen_wsd_run/train.jsonl','r',encoding='utf-8') as f:
    stats = {'mid_doc_cut':0, 'mid_think_cut':0, 'mid_chat_cut':0, 'safe_cut':0, 'total':0}
    for i, line in enumerate(f):
        if i >= 100: break
        row = json.loads(line)
        ids = row['input_ids']
        docs = row['doc_ids']
        half = len(ids) // 2  # 512

        stats['total'] += 1

        # Check if truncation at 512 cuts mid-document
        if docs[half-1] == docs[half] and docs[half] >= 0:
            stats['mid_doc_cut'] += 1

        # Check if truncation cuts inside a <think>...</think> block
        think_opens = sum(1 for t in ids[:half] if t == 151667)
        think_closes = sum(1 for t in ids[:half] if t == 151668)
        if think_opens > think_closes:
            stats['mid_think_cut'] += 1

        # Check if truncation cuts inside a chat turn (im_start without matching im_end)
        im_starts = sum(1 for t in ids[:half] if t == 151644)
        im_ends = sum(1 for t in ids[:half] if t == 151645)
        if im_starts > im_ends:
            stats['mid_chat_cut'] += 1

        if docs[half-1] != docs[half] or docs[half] < 0:
            stats['safe_cut'] += 1

        # Print detailed info for first 5
        if i < 5:
            # Find doc boundaries
            boundaries = [j for j in range(1, len(docs)) if docs[j] != docs[j-1]]
            pad_start = next((j for j in range(len(ids)-1, -1, -1) if docs[j] >= 0), -1) + 1
            unique_docs = len(set(d for d in docs if d >= 0))

            print(f"\n=== Row {i} ===")
            print(f"  len={len(ids)}  docs={unique_docs}  pad={len(ids)-pad_start}")
            print(f"  boundaries: {boundaries[:8]}")
            print(f"  doc_id[511]={docs[half-1]}  doc_id[512]={docs[half]}")

            # Show special tokens in first 512
            specials_in_half = [(j, ids[j], SPECIAL.get(ids[j], f'?{ids[j]}'))
                               for j in range(half) if ids[j] in SPECIAL]
            print(f"  specials in [:512]: {len(specials_in_half)} tokens")
            for j, tid, name in specials_in_half[-5:]:
                print(f"    pos {j}: {name}")

            # Decode around cut point
            snippet = tok.decode(ids[max(0,half-8):min(len(ids),half+8)])
            print(f"  text around 512: ...{repr(snippet[:120])}...")

    print(f"\n{'='*60}")
    print(f"TRUNCATION SAFETY REPORT (first {stats['total']} rows)")
    print(f"{'='*60}")
    print(f"  Mid-document cuts:  {stats['mid_doc_cut']}/{stats['total']}  "
          f"({100*stats['mid_doc_cut']/max(1,stats['total']):.0f}%)")
    print(f"  Mid-<think> cuts:   {stats['mid_think_cut']}/{stats['total']}  "
          f"({100*stats['mid_think_cut']/max(1,stats['total']):.0f}%)")
    print(f"  Mid-chat-turn cuts: {stats['mid_chat_cut']}/{stats['total']}  "
          f"({100*stats['mid_chat_cut']/max(1,stats['total']):.0f}%)")
    print(f"  Safe cuts (at doc boundary/pad): {stats['safe_cut']}/{stats['total']}  "
          f"({100*stats['safe_cut']/max(1,stats['total']):.0f}%)")
