# Adaptive KV Compression Evaluation (H100)

## Summary Metrics
| Config | Compression× | Throughput× | Attention Coverage | Key Cosine | Selected Cosine |
| --- | --- | --- | --- | --- | --- |
| baseline_full.yaml | 1.51 | 1.00 | 0.98 | 0.93 | 1.00 |
| hybrid_6x.yaml | 4.38 | 1.83 | 0.89 | 0.80 | 1.00 |
| hybrid_8x.yaml | 3.41 | 2.09 | 0.91 | 0.88 | 0.99 |
| hybrid_12x.yaml | 3.47 | 3.11 | 0.96 | 0.92 | 0.98 |

## Hyperparameter Sweep (Hybrid 8×)
Guard/bit combinations show the trade-off between compression and coverage.
| Guard | Bits | Compression× | Attention Coverage | Key Cosine |
| --- | --- | --- | --- | --- |
| 0.40 | 2 | 4.68 | 0.91 | 0.88 |
| 0.40 | 3 | 4.55 | 0.91 | 0.88 |
| 0.40 | 4 | 4.30 | 0.91 | 0.89 |
| 0.50 | 2 | 3.95 | 0.91 | 0.88 |
| 0.50 | 3 | 3.85 | 0.91 | 0.88 |
| 0.50 | 4 | 3.68 | 0.91 | 0.89 |
| 0.60 | 2 | 3.41 | 0.91 | 0.88 |
| 0.60 | 3 | 3.34 | 0.91 | 0.88 |
| 0.60 | 4 | 3.21 | 0.91 | 0.89 |
| 0.70 | 2 | 3.01 | 0.91 | 0.88 |
| 0.70 | 3 | 2.95 | 0.91 | 0.88 |
| 0.70 | 4 | 2.85 | 0.91 | 0.89 |

## Reproducibility
- System specs recorded in `reports/system_specs.txt`.
- Commands to regenerate experiments: `reports/repro_commands.sh`.
- Evaluation outputs live under `reports/` (JSON + tables).
- Paper tables exported via `scripts/generate_benchmark_tables.py` → `reports/benchmark_tables.md`.

## Benchmarking Workflow
1. Generate benchmark prompts (LongBench/RULER) and run the baseline model to capture accuracy metrics (store under `reports/benchmarks/baseline/*.json`).
2. Replay the prompts using compressed configs (hybrid 6× / 8× / 12×) and save metrics to `reports/benchmarks/{hybrid_6x,hybrid_8x,hybrid_12x}/*.json`.
3. Use `scripts/generate_benchmark_tables.py --eval-path reports/latest_eval.json` to refresh summary tables, then extend the script with benchmark metrics once available.
4. Update the MLSys draft with both proxy metrics (coverage/cosine) and benchmark accuracy deltas to justify compression choices.
