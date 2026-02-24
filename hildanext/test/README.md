# HildaNext Test Pack

Folder: `hildanext/test`

Execution policy:
- Use only conda env `mdm`.
- Use CUDA-first runtime (`runtime.device:auto` or `cuda`).
- CPU fallback is allowed only for architecture/runtime limits or explicit CPU-only unit checks.

Commands:
```bash
conda run -n mdm python hildanext/test/run_tests.py
conda run -n mdm python hildanext/test/build_inventory.py
conda run -n mdm python -m pytest hildanext/backend/tests -q
conda run -n mdm python -m hildanext.cli audit --config e:/DIFFUSION/HildaNext/hildanext/runs/configs/default.json
conda run -n mdm python -m hildanext.cli quant-bench --config e:/DIFFUSION/HildaNext/hildanext/runs/configs/default.json --modes "fp16,nf4,int8"
```

Main checks:
- precision fp16/fp32 paths
- vocab length and mask token
- doc-boundary masks
- AR baseline generation
- SFT smoke run
- formula checks for LLaDA/LLaDA2/LLaDA2.1
