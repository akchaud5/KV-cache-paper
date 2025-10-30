#!/usr/bin/env python
"""Entry point for running adaptive KV compression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import yaml
from rich.console import Console
from rich.table import Table

from adaptive_kv.cache import KVCache
from adaptive_kv.config import ExperimentConfig
from adaptive_kv.metrics import metric_report, summarize_memory, summarize_performance
from adaptive_kv.pipeline import CompressionPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive KV cache compression experiment runner."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Optional .pt file containing KV cache tensors with keys/values entries.",
    )
    parser.add_argument(
        "--attention-stats",
        type=Path,
        help="Optional .pt file containing aggregated attention scores tensor.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate synthetic cache and attention statistics if real data unavailable.",
    )
    parser.add_argument("--report-path", type=Path, help="Write JSON summary to this path.")
    return parser.parse_args()


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(data)


def load_cache(path: Path) -> KVCache:
    payload = torch.load(path, map_location="cpu")
    if "keys" not in payload or "values" not in payload:
        msg = "Cache checkpoint must contain 'keys' and 'values' tensors."
        raise ValueError(msg)
    return KVCache(keys=payload["keys"], values=payload["values"])


def load_attention(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 2:
        msg = "Attention statistics must be 2-D (layers, tokens)."
        raise ValueError(msg)
    return tensor


def simulate_cache(layers: int = 32, tokens: int = 4096, heads: int = 32, head_dim: int = 128):
    keys = torch.randn(layers, tokens, heads, head_dim, dtype=torch.float16)
    values = torch.randn_like(keys)
    attention = torch.rand(layers, tokens)
    return KVCache(keys=keys, values=values), attention


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.cache_path and args.attention_stats:
        cache = load_cache(args.cache_path)
        attention = load_attention(args.attention_stats)
    elif args.simulate or not (args.cache_path and args.attention_stats):
        layers = int((config.metrics or {}).get("layers", 32))
        tokens = int(config.max_sequence_length // 4)
        cache, attention = simulate_cache(layers=layers, tokens=tokens)
    else:
        msg = "Must provide both --cache-path and --attention-stats, or use --simulate."
        raise ValueError(msg)

    pipeline = CompressionPipeline(config)
    compressed = pipeline.compress(cache, attention)
    reconstructed = pipeline.reconstruct(compressed, cache)

    memory_stats = summarize_memory(compressed.original_bytes, compressed.compressed_bytes)
    perf_stats = summarize_performance(
        baseline_tps=config.metrics.get("baseline_tps", 1.0),
        compressed_tps=config.metrics.get("compressed_tps", 1.0),
        baseline_ttft_ms=config.metrics.get("baseline_ttft_ms", 0.0),
        compressed_ttft_ms=config.metrics.get("compressed_ttft_ms", 0.0),
        baseline_tpot_ms=config.metrics.get("baseline_tpot_ms", 0.0),
        compressed_tpot_ms=config.metrics.get("compressed_tpot_ms", 0.0),
    )

    table = Table(title="Adaptive KV Compression Report")
    table.add_column("Metric")
    table.add_column("Value")
    report = metric_report(memory_stats, perf_stats)
    report["compression_ratio"] = compressed.compression_ratio
    for key, value in report.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)
    console.print(f"Reconstructed cache dtype: {reconstructed.keys.dtype}")

    if args.report_path:
        import json

        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
