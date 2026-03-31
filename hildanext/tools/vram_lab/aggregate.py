#!/usr/bin/env python
"""aggregate.py — Aggregate VRAM bench JSONL logs into reports.

Reads all logs/*.jsonl and *.summary.json files,
produces:
  - reports/summary.md          (Markdown comparison table)
  - reports/charts.md           (Mermaid timeline + bar chart)
  - reports/recommended_config.json  (best config + fallback)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summaries(logs_dir: Path) -> List[Dict[str, Any]]:
    """Load all *.summary.json files from logs dir."""
    summaries: List[Dict[str, Any]] = []
    for f in sorted(logs_dir.glob("*.summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summaries.append(data)
        except Exception as e:
            print(f"[aggregate] WARN: cannot read {f}: {e}")
    return summaries


def _load_jsonl_peaks(logs_dir: Path) -> Dict[str, Dict[str, float]]:
    """Read *.jsonl files, extract peak VRAM and avg tok/s per run_id."""
    peaks: Dict[str, Dict[str, float]] = {}
    for f in sorted(logs_dir.glob("run_*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8").strip().split("\n")
            if not lines:
                continue
            peak_alloc = 0.0
            peak_res = 0.0
            tok_per_s_sum = 0.0
            tok_per_s_count = 0
            run_id = ""
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                run_id = row.get("run_id", "")
                m = row.get("metrics", {})
                peak_alloc = max(peak_alloc, m.get("cuda_peak_alloc_mb", 0))
                peak_res = max(peak_res, m.get("cuda_peak_res_mb", 0))
                tps = m.get("tok_per_s", 0)
                if tps > 0:
                    tok_per_s_sum += tps
                    tok_per_s_count += 1
            if run_id:
                peaks[run_id] = {
                    "peak_alloc_mb": round(peak_alloc, 1),
                    "peak_res_mb": round(peak_res, 1),
                    "avg_tok_per_s": round(tok_per_s_sum / max(1, tok_per_s_count), 1),
                }
        except Exception as e:
            print(f"[aggregate] WARN: cannot read {f}: {e}")
    return peaks


def _stability_score(s: Dict) -> int:
    """Higher is more stable. 0 = OOM, 1 = NaN, 2 = ok."""
    if s.get("oom"):
        return 0
    if s.get("total_nan_inf", 0) > 0:
        return 1
    return 2


def generate_reports(logs_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(logs_dir)
    peaks = _load_jsonl_peaks(logs_dir)

    if not summaries:
        print("[aggregate] No summaries found — nothing to aggregate.")
        return

    # Merge peaks into summaries
    for s in summaries:
        rid = s.get("run_id", "")
        if rid in peaks:
            s["peak_alloc_mb"] = peaks[rid]["peak_alloc_mb"]
            s["peak_res_mb"] = peaks[rid]["peak_res_mb"]
            s["avg_tok_per_s"] = peaks[rid]["avg_tok_per_s"]
        else:
            s.setdefault("peak_alloc_mb", 0)
            s.setdefault("peak_res_mb", 0)
            s.setdefault("avg_tok_per_s", 0)

    # Sort: stability desc, peak VRAM asc, tok/s desc
    summaries.sort(key=lambda x: (-_stability_score(x), x.get("peak_alloc_mb", 99999), -x.get("avg_tok_per_s", 0)))

    # --- summary.md ---
    md_lines: List[str] = []
    md_lines.append("# VRAM Lab — GTX 1080 Micro-Test Results\n")
    md_lines.append(f"Runs aggregated: {len(summaries)}\n")
    md_lines.append("")
    md_lines.append("## Comparison Table\n")
    md_lines.append("| # | Run ID | Precision | Optimizer | Seq Len | Grad Acc | Checkpoint | Steps | OOM | NaN/Inf | Peak Alloc MB | Peak Res MB | Avg tok/s | Loss | Stable |")
    md_lines.append("|---|--------|-----------|-----------|---------|----------|------------|-------|-----|---------|---------------|-------------|-----------|------|--------|")
    for i, s in enumerate(summaries, 1):
        cfg = s.get("config", {})
        loss_s = f"{s['last_loss']:.4f}" if s.get("last_loss") is not None else "N/A"
        stable = "YES" if _stability_score(s) == 2 else ("NaN" if _stability_score(s) == 1 else "OOM")
        md_lines.append(
            f"| {i} "
            f"| {s.get('run_id', 'N/A')} "
            f"| {cfg.get('precision', 'N/A')} "
            f"| {cfg.get('optimizer', 'N/A')} "
            f"| {cfg.get('seq_len', 'N/A')} "
            f"| {cfg.get('grad_acc', 'N/A')} "
            f"| {cfg.get('checkpoint', 'N/A')} "
            f"| {s.get('steps_done', 0)} "
            f"| {s.get('oom', False)} "
            f"| {s.get('total_nan_inf', 0)} "
            f"| {s.get('peak_alloc_mb', 0):.0f} "
            f"| {s.get('peak_res_mb', 0):.0f} "
            f"| {s.get('avg_tok_per_s', 0):.0f} "
            f"| {loss_s} "
            f"| {stable} |"
        )
    md_lines.append("")
    md_lines.append("### Legend\n")
    md_lines.append("- **Stable**: YES = no OOM, no NaN/Inf; NaN = had NaN/Inf issues; OOM = out of memory crash")
    md_lines.append("- **Peak Alloc MB**: `torch.cuda.max_memory_allocated()` across all steps")
    md_lines.append("- **Peak Res MB**: `torch.cuda.max_memory_reserved()` across all steps")
    md_lines.append("")

    summary_md = reports_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[aggregate] Wrote {summary_md}")

    # --- charts.md ---
    chart_lines: List[str] = []
    chart_lines.append("# VRAM Lab — Charts\n")

    # Timeline
    chart_lines.append("## Test Sequence Timeline\n")
    chart_lines.append("```mermaid")
    chart_lines.append("timeline")
    chart_lines.append("  title Sequenza micro-test GTX1080 (prima di WSD)")
    chart_lines.append("  Compatibilità : Verifica PyTorch sm_61 OK; bitsandbytes OK")
    chart_lines.append("  Baseline : AMP fp16 + AdamW; seq_len 1024; 20 step")
    chart_lines.append("  Checkpointing : AMP fp16 + checkpoint; misura delta VRAM")
    chart_lines.append("  Optimizer : AdamW vs bnb_adamw8bit vs bnb_paged_adamw8bit")
    chart_lines.append("  SDPA : forza backend MATH; test maschera causale nascosta")
    chart_lines.append("  Sweep : sweep seq_len e grad_acc; trova soglia OOM")
    chart_lines.append("  Report : tabella comparativa + grafici + config raccomandata")
    chart_lines.append("```\n")

    # Bar chart: peak VRAM per config
    stable_runs = [s for s in summaries if not s.get("oom")]
    if stable_runs:
        labels = []
        values = []
        for s in stable_runs[:8]:  # limit bars
            rid = s.get("run_id", "?")
            # Shorten run_id for display
            short = rid.replace("amp_fp16__", "").replace("bnb_", "").replace("__ckpt_full", "+ckpt")
            if len(short) > 30:
                short = short[:27] + "..."
            labels.append(f'"{short}"')
            values.append(int(s.get("peak_alloc_mb", 0)))

        max_val = max(values) if values else 8500
        y_max = int(math.ceil(max_val / 500) * 500 + 500)

        chart_lines.append("## Peak VRAM per Configuration (MB)\n")
        chart_lines.append("```mermaid")
        chart_lines.append("xychart-beta")
        chart_lines.append('  title "Peak VRAM per configurazione (MB)"')
        chart_lines.append(f"  x-axis [{', '.join(labels)}]")
        chart_lines.append(f'  y-axis "MB" 0 --> {y_max}')
        chart_lines.append(f"  bar [{', '.join(str(v) for v in values)}]")
        chart_lines.append("```\n")

    charts_md = reports_dir / "charts.md"
    charts_md.write_text("\n".join(chart_lines), encoding="utf-8")
    print(f"[aggregate] Wrote {charts_md}")

    # --- recommended_config.json ---
    best = None
    fallbacks: List[Dict] = []
    for s in summaries:
        if _stability_score(s) == 2:
            if best is None:
                best = s
            else:
                fallbacks.append(s)

    rec: Dict[str, Any] = {}
    if best:
        rec["recommended"] = {
            "run_id": best.get("run_id"),
            "config": best.get("config"),
            "peak_alloc_mb": best.get("peak_alloc_mb"),
            "peak_res_mb": best.get("peak_res_mb"),
            "avg_tok_per_s": best.get("avg_tok_per_s"),
        }
    else:
        rec["recommended"] = None
        rec["warning"] = "No stable configuration found"

    rec["fallbacks"] = [
        {
            "run_id": f.get("run_id"),
            "config": f.get("config"),
            "peak_alloc_mb": f.get("peak_alloc_mb"),
        }
        for f in fallbacks[:3]
    ]

    rec_path = reports_dir / "recommended_config.json"
    rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[aggregate] Wrote {rec_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate VRAM bench results")
    parser.add_argument("--logs_dir", type=str, default="")
    parser.add_argument("--reports_dir", type=str, default="")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    logs_dir = Path(args.logs_dir) if args.logs_dir else base / "logs"
    reports_dir = Path(args.reports_dir) if args.reports_dir else base / "reports"

    generate_reports(logs_dir, reports_dir)


if __name__ == "__main__":
    main()
