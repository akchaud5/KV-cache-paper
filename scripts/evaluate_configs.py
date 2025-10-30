#!/usr/bin/env python
"""Batch evaluation script combining compression and quality metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import yaml
from rich.console import Console
from rich.table import Table

from adaptive_kv.cache import KVCache
from adaptive_kv.config import ExperimentConfig
from adaptive_kv.metrics import metric_report, summarize_memory, summarize_performance
from adaptive_kv.pipeline import CompressionPipeline
from adaptive_kv.quality import build_quality_report

console = Console()


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(data)


def load_cache(path: Path) -> KVCache:
    payload = torch.load(path, map_location="cpu")
    return KVCache(keys=payload["keys"], values=payload["values"])


def load_attention(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 2:
        msg = "Attention tensor must be 2-D (layers, tokens)."
        raise ValueError(msg)
    return tensor


def evaluate_config(
    cfg_path: Path, cache_path: Path, attention_path: Path
) -> dict[str, float | str]:
    config = load_config(cfg_path)
    cache = load_cache(cache_path)
    attention = load_attention(attention_path)

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
    quality = build_quality_report(cache, reconstructed, compressed, attention)

    report = metric_report(memory_stats, perf_stats)
    report["compression_ratio"] = compressed.compression_ratio
    report.update(
        {
            "key_mse": quality.key_mse,
            "value_mse": quality.value_mse,
            "key_cosine": quality.key_cosine,
            "value_cosine": quality.value_cosine,
            "key_cosine_selected": quality.key_cosine_selected,
            "value_cosine_selected": quality.value_cosine_selected,
            "attention_coverage": quality.attention_coverage,
            "selected_fraction": quality.selected_fraction,
            "high_precision_fraction": quality.high_precision_fraction,
        }
    )
    report["config"] = cfg_path.name
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple configs using shared artifacts.")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            Path("experiments/baseline_full.yaml"),
            Path("experiments/hybrid_6x.yaml"),
            Path("experiments/hybrid_8x.yaml"),
            Path("experiments/hybrid_12x.yaml"),
        ],
        help="List of experiment YAML configs to evaluate.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/cache/prefill_cache.pt"),
        help="Path to KV cache artifact (.pt).",
    )
    parser.add_argument(
        "--attention-path",
        type=Path,
        default=Path("artifacts/attn/prefill_attn.pt"),
        help="Path to attention statistics artifact (.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/latest_eval.json"),
        help="Where to store the aggregated evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = Table(title="Adaptive KV Evaluation Summary")
    table.add_column("Config")
    table.add_column("Compression", justify="right")
    table.add_column("Throughput×", justify="right")
    table.add_column("Attn Cov.", justify="right")
    table.add_column("Key Cosine", justify="right")
    table.add_column("Selected Cosine", justify="right")

    results: List[dict[str, float | str]] = []
    for cfg in args.configs:
        report = evaluate_config(cfg, args.cache_path, args.attention_path)
        results.append(report)
        table.add_row(
            report["config"],
            f"{report['compression_ratio']:.2f}",
            f"{report['throughput_gain']:.2f}",
            f"{report['attention_coverage']:.2f}",
            f"{report['key_cosine']:.2f}",
            f"{report['key_cosine_selected']:.2f}",
        )

    console.print(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
