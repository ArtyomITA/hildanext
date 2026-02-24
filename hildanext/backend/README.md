# HildaNext Backend

SAFE backend for AR->dLLM conversion workflow with WSD, M2T+T2T training, threshold-edit decode, FastAPI serving, and dInfer fallback integration.

## Install
```bash
pip install -e hildanext/backend
```

## Default Config
- `hildanext/runs/configs/default.json`
- `hildanext/runs/configs/smoke.json`

## CLI
```bash
hildanext prepare-data --config hildanext/runs/configs/default.json
hildanext tokenize --config hildanext/runs/configs/default.json
hildanext convert-wsd --config hildanext/runs/configs/default.json --steps 10
hildanext sft --config hildanext/runs/configs/default.json --steps 10
hildanext serve --config hildanext/runs/configs/default.json --host 127.0.0.1 --port 8080
hildanext smoke-test --config hildanext/runs/configs/smoke.json
```

## API
- `GET /health`
- `POST /generate`
- `POST /jobs/submit`
- `GET /jobs/{job_id}`

## Test Pack
```bash
python hildanext/test/run_tests.py
python hildanext/test/build_inventory.py
```
