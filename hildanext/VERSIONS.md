# HildaNext Dependency Pins
Date: 2026-04-10
Workspace: e:\DIFFUSION\HildaNext

## Local Environment Check
- conda env: `mdm`
- python: `3.10.18`
- torch: `2.4.0+cu121`
- torch cuda runtime: `12.1`
- transformers: `4.57.3`
- note: backend package targets Python `>=3.11`; smoke/test was executed with system Python 3.13.

## Vendored Repositories
- `vendor/llada`
  - remote: `https://github.com/ML-GSAI/LLaDA`
  - commit: `570f29032d6824ea14977c89a8eb402e6eb25f96`
- `vendor/dinfer`
  - remote: `https://github.com/inclusionAI/dInfer`
  - commit: `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`

## Local Model/Data Inputs
- Qwen local HF dir: `models/qwen3-0.6b` (junction to `Qwen3-0.6B`)
- Dolma sample local dir: `../dolma_v1_6_sample_1767050862`
- LLaDA docs copied into `docs/` from workspace HTML sources.

## Notes
- dInfer API usage is based on real repo symbols: `dinfer.DiffusionLLMServing` and `dinfer.SamplingParams`.
- dInfer hard-imports vLLM/sglang in current codebase; backend implements guarded fallback to Transformers/dummy engine.

## Changelog
### April 2026
- WSD instability root cause identified: `composite_llada20` doubles seq to 2048, `lm_head` logits cause VRAM thrashing on GTX 1080. See `docs/WSD_INVESTIGATION_REPORT.md §11`.
- Fix: Option 3 (slim lm_head on x_t only) + Option 2 (halve to S=512 for composite phases). Both in `diffusion.py` and `training.py`.
- SDPA backend: `EFFICIENT_ATTENTION` (not MATH) confirmed on Pascal via `torch._fused_sdp_choice`.
- WSD schedule updated: W=1000/S=3000/D=1000 (5000 steps), accum_steps=8, multi_turn_t2t=2.
- Doc-boundary-aware batch truncation added to `training.py`.
- Dataset: 60,638 rows × 1024 tokens (Dolma), no retokenization needed.
- Frontend routes refactored: `/chat`, `/inferenceplus`, `/benchmark`, `/legacy/wsd`.
- Stale docs deleted: test.md, safe_architecture.md, plan_stage_0.md, full_architecture_reference.md.

### February 2026
- Initial pin of environment and vendored repos.
