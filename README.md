# Adaptive KV Cache Compression for Long-Context LLM Serving

This repository provides a modular research codebase for experimenting with **layer-aware KV cache sparsification** and **mixed-precision quantization** on a single NVIDIA H100 GPU. The design follows MLSys best practices—production-conscious abstractions, reproducible evaluation, and extensible kernels—while focusing on the hybrid compression setting suggested in the project brief.

## Key Components

- `adaptive_kv.cache`: lightweight tensor abstractions describing per-layer KV caches and paged layouts compatible with vLLM/SGLang.
- `adaptive_kv.selection`: token and layer budget policies including Pyramid-style layer funnels, SnapKV-inspired clustering, and StreamingLLM sliding windows.
- `adaptive_kv.quantization`: configurable per-layer bit allocation with support for asymmetric 2–4-bit quantization and higher-precision guard bands.
- `adaptive_kv.pipeline`: orchestration utilities that integrate the components above into prefill/decode compression loops and expose metrics hooks.
- `experiments/`: example YAML configs targeting 5×, 8×, and 12× compression regimes plus evaluation scripts for LongBench, RULER, and throughput sweeps.
- `tests/`: lightweight unit tests covering cache layout transforms, selector correctness on synthetic attention, and quantization error bounds.
- `scripts/run_quality_benchmarks.py`: proxy LongBench/RULER harness computing reconstruction error, attention retention, and guard-band coverage for a given KV artifact.

## Quick Start

```bash
python -m pip install -e .
python scripts/run_experiment.py --config experiments/longbench_8x.yaml
```

The default configuration assumes access to H100-class hardware and relies on PyTorch 2.3+, flash-attn ≥2.5, and vLLM ≥0.4.0. Each experiment generates:

- Memory footprint vs. compression ratio curves
- Throughput (tokens/s), TTFT, and TPOT traces across batch sizes
- LongBench quality metrics with the full-cache baseline for comparison

## Research Roadmap

1. **Ablate hybrid compression**: demonstrate multiplicative gains from combining layer-aware sparsification with adaptive quantization.
2. **System optimization**: plug custom selective-attention kernels (MiniKV/FlashInfer style) into `adaptive_kv.runtime` to remove prefill slowdowns.
3. **Workload adaptation**: integrate task-sensitive policies that profile attention in prefill and retune budgets per request.

## Repository Layout

```
.
├── README.md
├── pyproject.toml
├── scripts/
│   └── run_experiment.py
├── src/
│   └── adaptive_kv/
│       ├── __init__.py
│       ├── cache.py
│       ├── config.py
│       ├── pipeline.py
│       ├── quantization.py
│       ├── selection.py
│       └── metrics.py
├── experiments/
│   ├── baseline_full.yaml
│   ├── hybrid_8x.yaml
│   └── hybrid_12x.yaml
└── tests/
    ├── test_cache.py
    ├── test_quantization.py
    └── test_selection.py
```

Each module contains docstrings pointing to relevant papers and implementation notes aligning the code with MLSys conference expectations.

## License

MIT License (see `LICENSE`)
