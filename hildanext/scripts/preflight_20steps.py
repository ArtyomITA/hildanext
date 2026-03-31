"""Preflight: run exactly 20 WSD training steps and populate runs/logs/.

Output files consumed by the FastAPI /frontend/wsd endpoint:
  runs/logs/cpt.jsonl       – one metric row per step
  runs/logs/fallbacks.jsonl – fallback/notice events from the trace
  runs/logs/cpt_run.log     – human-readable console log

Usage (from repo root):
    conda run -n hilda python scripts/preflight_20steps.py
    conda run -n hilda python scripts/preflight_20steps.py --config runs/configs/preflight_strict.json
    conda run -n hilda python scripts/preflight_20steps.py --steps 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from hildanext.config import load_config  # noqa: E402
from hildanext.training import run_wsd_conversion  # noqa: E402
from hildanext.trace import trace_from_cfg  # noqa: E402

_LOG = logging.getLogger("preflight")


def _clear_logs(logs_dir: Path) -> None:
    """Remove previous run artefacts so the frontend sees only preflight data."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("cpt.jsonl", "fallbacks.jsonl", "cpt_run.log", "cpt.eval.jsonl"):
        p = logs_dir / fname
        if p.exists():
            p.unlink()
            _LOG.info("cleared %s", p)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run a 20-step WSD preflight.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "runs" / "configs" / "llada21_dolma_wsd_only.json"),
        help="Path to AppConfig JSON (default: llada21_dolma_wsd_only.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of optimiser steps to run (default: 20)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear existing log files before running",
    )
    args = parser.parse_args()

    _LOG.info("loading config: %s", args.config)
    cfg = load_config(args.config)

    logs_dir = Path(cfg.paths.logs_dir)
    _LOG.info("logs_dir: %s", logs_dir)

    if not args.no_clear:
        _clear_logs(logs_dir)

    trace = trace_from_cfg(cfg)

    _LOG.info("starting preflight: %d steps → %s/cpt.jsonl", args.steps, logs_dir)
    result = run_wsd_conversion(
        cfg,
        steps=args.steps,
        trace=trace,
        resume=False,
        ckpt_every=99999,   # no checkpoints during preflight
        eval_every=99999,   # no eval during preflight
    )
    trace.flush()

    n_rows = logs_dir / "cpt.jsonl"
    count = sum(1 for _ in n_rows.open()) if n_rows.exists() else 0
    _LOG.info("preflight done. %d rows written to cpt.jsonl", count)
    _LOG.info("result summary: %s", {k: v for k, v in (result or {}).items() if k != "rows"})

    # Verify the FastAPI endpoint will find the file
    if count == 0:
        _LOG.error("cpt.jsonl is empty — check training config and data paths.")
        sys.exit(1)

    _LOG.info(
        "✓  Start the API with:  cd hildanext && "
        "conda run -n hilda python -m hildanext.api serve "
        "--config runs/configs/llada21_dolma_wsd_only.json"
    )


if __name__ == "__main__":
    main()
