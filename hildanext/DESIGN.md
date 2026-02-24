# HildaNext SAFE Design
Date: 2026-02-22

## Scope
- Implemented SAFE-only backend for AR->dLLM conversion, SFT, inference, and serving.
- Excluded speculative/non-Markovian components from FULL architecture.
- Built for CPU demo and Pascal-safe GPU workflows.

## SAFE Decisions
- Backend language: Python.
- CLI: argparse.
- Config format: JSON.
- API server: FastAPI with `/health`, `/generate`, `/jobs/*`.
- Inference engines:
  - primary attempt: dInfer adapter (when runtime deps and supported model are available),
  - fallback: Transformers threshold-edit loop,
  - final fallback: tiny dummy LM for smoke/demo continuity.

## Data Pipeline
- Sources: local Dolma sample path + TinyStories local/download/synthetic fallback.
- Processed outputs:
  - `data/processed/train.jsonl`, `eval.jsonl`
  - `data/processed/sft_train.jsonl`, `sft_eval.jsonl`
  - `data/processed/humaneval_dummy.jsonl`
- Tokenization outputs:
  - `data/tokenized/train.jsonl`, `eval.jsonl`
  - `data/tokenized/sft_train.jsonl`, `sft_eval.jsonl`
- Packing includes `doc_ids`; doc-level attention masks are built from `doc_ids`.

## WSD Conversion
- WSD schedule implemented as:
  - warmup: block size linearly from 1 to full sequence.
  - stable: full-sequence block.
  - decay: linearly back to compact block.
- CPT objective:
  - M2T loss on masked positions.
  - T2T loss on noised observed positions.

## SFT Objective
- SFT uses mixture M2T+T2T with response-focused masking.
- Multi-turn forward minimal implemented as two T2T noising passes per step.

## Inference Logic
- Threshold-edit decode loop alternates:
  - `Γt` unmasking (`MASK->token`) under `tau_mask`,
  - `Δt` editing (`token->token`) under `tau_edit`.
- Presets:
  - `S_MODE`: lower `tau_mask`, more correction passes.
  - `Q_MODE`: higher `tau_mask`, conservative drafting.
- Loop logs steps, mask ratio, edit counts, throughput estimate.

## TODO/Uncertain Items
- dInfer currently targets LLaDA-family runtime stacks and imports vLLM/sglang; generic Qwen dLLM serving via dInfer is not guaranteed.
- Dolma sample in this workspace is largely pretokenized shards; token ID space may not perfectly match Qwen tokenizer. SAFE fallback normalizes out-of-range IDs for demo continuity.
- Top-k checkpoint merge is implemented for checkpoints produced by this backend; external heterogeneous checkpoint merges are not guaranteed.
