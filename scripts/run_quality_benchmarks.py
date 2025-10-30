#!/usr/bin/env python
"""Compute quality metrics for adaptive KV compression runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from adaptive_kv.cache import KVCache
from adaptive_kv.config import ExperimentConfig
from adaptive_kv.quality import build_quality_report
from adaptive_kv.pipeline import CompressionPipeline


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        msg = f"Unsupported config extension: {path.suffix}"
        raise ValueError(msg)
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


def simulate_cache(
    layers: int = 32, tokens: int = 4096, heads: int = 32, head_dim: int = 128
) -> tuple[KVCache, torch.Tensor]:
    keys = torch.randn(layers, tokens, heads, head_dim, dtype=torch.float16)
    values = torch.randn_like(keys)
    attention = torch.rand(layers, tokens)
    return KVCache(keys=keys, values=values), attention

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run proxy LongBench/RULER quality metrics for compressed KV caches."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Optional .pt file with KV cache tensors (keys/values).",
    )
    parser.add_argument(
        "--attention-stats",
        type=Path,
        help="Optional .pt file with attention statistics.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Synthesize cache/attention if real artifacts are unavailable.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional JSON file to persist metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline = CompressionPipeline(config)

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

    attention = attention.float()
    compressed = pipeline.compress(cache, attention)
    reconstructed = pipeline.reconstruct(compressed, cache)
    report = build_quality_report(cache, reconstructed, compressed, attention)

    table = Table(title="Adaptive KV Quality Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("key_mse", f"{report.key_mse:.6e}")
    table.add_row("value_mse", f"{report.value_mse:.6e}")
    table.add_row("key_cosine", f"{report.key_cosine:.4f}")
    table.add_row("value_cosine", f"{report.value_cosine:.4f}")
    table.add_row("key_cosine_selected", f"{report.key_cosine_selected:.4f}")
    table.add_row("value_cosine_selected", f"{report.value_cosine_selected:.4f}")
    table.add_row("attention_coverage", f"{report.attention_coverage:.4f}")
    table.add_row("selected_fraction", f"{report.selected_fraction:.4f}")
    table.add_row("high_precision_fraction", f"{report.high_precision_fraction:.4f}")
    console.print(table)

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key_mse": report.key_mse,
            "value_mse": report.value_mse,
            "key_cosine": report.key_cosine,
            "value_cosine": report.value_cosine,
            "key_cosine_selected": report.key_cosine_selected,
            "value_cosine_selected": report.value_cosine_selected,
            "attention_coverage": report.attention_coverage,
            "selected_fraction": report.selected_fraction,
            "high_precision_fraction": report.high_precision_fraction,
            "config": args.config.name,
            "cache_path": str(args.cache_path) if args.cache_path else None,
            "attention_path": str(args.attention_stats) if args.attention_stats else None,
        }
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
