# HildaNext Dependency Pins
Date: 2026-02-22
Workspace: e:\DIFFUSION\HildaNext

## Local Environment Check
- conda env: `mdm`
- python: `3.10.18`
- torch: `2.4.0+cu121`
- torch cuda runtime: `12.1`
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
