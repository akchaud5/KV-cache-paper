#!/usr/bin/env python
"""Hyper-parameter sweep for guard ratios and bit-widths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
import yaml
from rich.console import Console
from rich.table import Table

from adaptive_kv.cache import KVCache
from adaptive_kv.config import ExperimentConfig, QuantizationConfig
from adaptive_kv.pipeline import CompressionPipeline
from adaptive_kv.quality import build_quality_report

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep guard ratios / bits for a configuration.")
    parser.add_argument("--config", type=Path, required=True, help="Base experiment config.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/cache/prefill_cache.pt"),
        help="KV cache artifact (.pt).",
    )
    parser.add_argument(
        "--attention-path",
        type=Path,
        default=Path("artifacts/attn/prefill_attn.pt"),
        help="Attention artifact (.pt).",
    )
    parser.add_argument(
        "--guards",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7],
        help="Guard band fractions to sweep.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Default bit-widths to sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/hyperparam_sweep.json"),
        help="Output JSON file for sweep results.",
    )
    return parser.parse_args()


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(data)


def load_artifacts(cache_path: Path, attention_path: Path) -> tuple[KVCache, torch.Tensor]:
    cache_payload = torch.load(cache_path, map_location="cpu")
    cache = KVCache(keys=cache_payload["keys"], values=cache_payload["values"])
    attention = torch.load(attention_path, map_location="cpu")
    return cache, attention


def sweep(
    config: ExperimentConfig,
    guards: Iterable[float],
    bits: Iterable[int],
    cache: KVCache,
    attention: torch.Tensor,
) -> List[dict[str, float]]:
    results: List[dict[str, float]] = []
    for guard in guards:
        for bit in bits:
            cfg = config.model_copy(deep=True)
            quant = cfg.quantization.model_copy(update={"high_precision_guard": guard, "default_bits": bit})
            if quant.key_bits and quant.key_bits < bit:
                quant.key_bits = bit
            if quant.value_bits and quant.value_bits < bit:
                quant.value_bits = bit
            cfg.quantization = QuantizationConfig.model_validate(quant.model_dump())

            pipeline = CompressionPipeline(cfg)
            compressed = pipeline.compress(cache, attention)
            reconstructed = pipeline.reconstruct(compressed, cache)
            quality = build_quality_report(cache, reconstructed, compressed, attention)
            results.append(
                {
                    "guard": guard,
                    "bits": bit,
                    "compression_ratio": compressed.compression_ratio,
                    "key_cosine": quality.key_cosine,
                    "attention_coverage": quality.attention_coverage,
                    "selected_fraction": quality.selected_fraction,
                    "config": cfg.model_name,
                }
            )
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cache, attention = load_artifacts(args.cache_path, args.attention_path)
    results = sweep(config, args.guards, args.bits, cache, attention)
    table = Table(title=f"Hyperparameter Sweep ({args.config.name})")
    table.add_column("Guard")
    table.add_column("Bits")
    table.add_column("Compression", justify="right")
    table.add_column("Attn Cov.", justify="right")
    table.add_column("Key Cosine", justify="right")

    for row in results:
        table.add_row(
            f"{row['guard']:.2f}",
            f"{row['bits']}",
            f"{row['compression_ratio']:.2f}",
            f"{row['attention_coverage']:.2f}",
            f"{row['key_cosine']:.2f}",
        )
    console.print(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.print(f"Sweep results saved to {args.output}")


if __name__ == "__main__":
    main()
