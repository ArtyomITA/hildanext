# Overhead Benchmark Suite

Strumenti di benchmarking per misurare VRAM, throughput, qualità training,
e overhead per-item su GTX 1080 8GB.

## Come eseguire

```powershell
conda activate mdm
cd E:\DIFFUSION\HildaNext\hildanext

# 1. VRAM matrix: tutti gli optimizer × grad_ckpt × seq_len
python -m test.overhead_bench.bench_vram_matrix --opt-steps 5

# 2. Overhead micro-benchmarks: .item() sync, nvidia-smi vs pynvml, empty_cache, wsd_block caching
python -m test.overhead_bench.bench_overhead_items

# 3. Qualità training: loss convergence, NaN rate, accuracy per 50 opt steps
python -m test.overhead_bench.bench_optimizer_quality --opt-steps 30

# 4. Seq-len sweep con re-tokenizzazione on-the-fly a 2048
python -m test.overhead_bench.bench_seq_len --seq-lens 1024,1536,2048
```

## Output

Ogni benchmark scrive:
- `results/<nome>.jsonl` — metriche per-step
- `results/<nome>.summary.json` — aggregati
- Tabella comparativa su stdout

## File

| File | Misura |
|------|--------|
| `bench_vram_matrix.py` | Peak VRAM, tok/s, OOM, NaN per ogni combo (optimizer, grad_ckpt, seq_len) |
| `bench_overhead_items.py` | Costo in μs di ogni overhead item nel hot path |
| `bench_optimizer_quality.py` | Loss curve, grad norms, NaN rate per optimizer su 50 step |
| `bench_seq_len.py` | Throughput e VRAM per seq_len 1024/1536/2048 con re-tokenizzazione |

## Dopo aver scelto

Cancella questa cartella: `rm -rf hildanext/test/overhead_bench/`
