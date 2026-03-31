"""Bootstrap a .ckpt resume file from an interrupted tokenize_split run.

Usage (after stopping the tokenizer process):
    python scripts/make_resume_ckpt.py

What it does:
  1. Counts lines already written in data/tokenized/train.jsonl
  2. Estimates input rows consumed (using observed seqs/row ratio)
  3. Rounds DOWN conservatively to the nearest chunk boundary (5000 rows)
  4. Truncates train.jsonl to a safe seq count (may re-do last chunk)
  5. Writes data/tokenized/train.jsonl.ckpt so tokenize_split can resume

After running this script, re-launch dolma-prep and it will automatically
resume from the checkpoint with batch encoding (5-8x faster).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
TOKENIZED_DIR = BASE / "data" / "tokenized"
TRAIN_OUT = TOKENIZED_DIR / "train.jsonl"
CKPT_OUT  = Path(str(TRAIN_OUT) + ".ckpt")

# ── constants ─────────────────────────────────────────────────────────────────
CHUNK_ROWS   = 5000          # must match tokenize_split
# Conservative seqs-per-row ratio observed from the interrupted run
# (10_269_888 seqs / 19_005_000 rows = 0.5404)
# Using slightly lower value to be extra conservative (never over-skip)
SEQS_PER_ROW = 0.535

# ── count current output lines ────────────────────────────────────────────────
if not TRAIN_OUT.exists():
    print(f"ERROR: {TRAIN_OUT} not found — nothing to resume.")
    sys.exit(1)

print(f"Counting lines in {TRAIN_OUT} ({TRAIN_OUT.stat().st_size // 1_000_000} MB) ...")
seqs_written = 0
with open(TRAIN_OUT, "rb") as f:
    for _ in f:
        seqs_written += 1
print(f"  seqs already written: {seqs_written:,}")

# ── estimate safe checkpoint ──────────────────────────────────────────────────
rows_consumed_est = seqs_written / SEQS_PER_ROW
# floor to chunk boundary for safety
chunks_done = int(rows_consumed_est / CHUNK_ROWS)
rows_consumed_safe = chunks_done * CHUNK_ROWS

# seqs to keep: floor conservatively (re-do last chunk, ~2700 seqs)
seqs_to_keep = int(rows_consumed_safe * SEQS_PER_ROW)
seqs_to_keep = max(0, seqs_to_keep - 2700)  # extra safety margin = 1 chunk

print(f"  estimated rows consumed : {rows_consumed_est:,.0f}")
print(f"  safe rows (chunk-aligned): {rows_consumed_safe:,}  ({chunks_done} chunks)")
print(f"  seqs to keep            : {seqs_to_keep:,}  (dropping last {seqs_written - seqs_to_keep:,} seqs)")

if seqs_to_keep <= 0:
    print("ERROR: seqs_to_keep <= 0 — file too small to resume, delete it and restart fresh.")
    sys.exit(1)

if seqs_to_keep >= seqs_written:
    seqs_to_keep = seqs_written
    print("  (no truncation needed)")

# ── truncate train.jsonl to seqs_to_keep ─────────────────────────────────────
if seqs_to_keep < seqs_written:
    print(f"Truncating {TRAIN_OUT.name} to {seqs_to_keep:,} lines ...")
    tmp = Path(str(TRAIN_OUT) + ".truncating")
    lines_written = 0
    with open(TRAIN_OUT, "rb") as fin, open(tmp, "wb") as fout:
        for line in fin:
            fout.write(line)
            lines_written += 1
            if lines_written >= seqs_to_keep:
                break
    os.replace(tmp, TRAIN_OUT)
    print(f"  truncated to {lines_written:,} lines  ({TRAIN_OUT.stat().st_size // 1_000_000} MB)")

# ── write checkpoint ──────────────────────────────────────────────────────────
ckpt = {
    "rows_consumed": rows_consumed_safe,
    "seqs_written":  seqs_to_keep,
    "carry_ids":     [],   # fresh carry = start fresh at chunk boundary
    "carry_docs":    [],
    "carry_resp":    [],
    "carry_src":     "mixed",
    "doc_offset":    rows_consumed_safe,
}
CKPT_OUT.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
print(f"Checkpoint written: {CKPT_OUT}")
print()
print("=== READY TO RESUME ===")
print(f"  will skip first {rows_consumed_safe:,} input rows")
print(f"  will append after {seqs_to_keep:,} existing seqs")
print()
print("Run:")
print(f'  cd {BASE}')
print( '  C:\\Users\\Administrator\\.conda\\envs\\mdm\\python.exe -u -m hildanext.cli dolma-prep --config "runs/configs/llada21_dolma_wsd_only.json"')
