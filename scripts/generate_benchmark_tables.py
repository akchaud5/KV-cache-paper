#!/usr/bin/env python
"""Generate paper-ready benchmark tables from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_EVAL_PATH = Path("reports/latest_eval.json")
DEFAULT_SWEEP_PATH = Path("reports/hyperparam_sweep.json")
DEFAULT_BENCHMARK_DIR = Path("reports/benchmarks")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        msg = f"Required report file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    msg = f"Unexpected JSON structure in {path}"
    raise TypeError(msg)


def format_summary(eval_path: Path) -> pd.DataFrame:
    records = load_jsonl(eval_path)
    df = pd.DataFrame.from_records(records)
    df = df[
        [
            "config",
            "compression_ratio",
            "throughput_gain",
            "attention_coverage",
            "key_cosine",
            "value_cosine",
            "key_cosine_selected",
        ]
    ]
    df = df.rename(
        columns={
            "config": "Config",
            "compression_ratio": "Compression×",
            "throughput_gain": "Throughput×",
            "attention_coverage": "Attention Coverage",
            "key_cosine": "Key Cosine",
            "value_cosine": "Value Cosine",
            "key_cosine_selected": "Selected Cosine",
        }
    )
    return df


def format_sweep(sweep_path: Path) -> pd.DataFrame:
    records = load_jsonl(sweep_path)
    df = pd.DataFrame.from_records(records)
    df = df[
        ["guard", "bits", "compression_ratio", "attention_coverage", "key_cosine"]
    ].rename(
        columns={
            "guard": "Guard",
            "bits": "Bits",
            "compression_ratio": "Compression×",
            "attention_coverage": "Attention Coverage",
            "key_cosine": "Key Cosine",
        }
    )
    return df


def write_markdown(table: pd.DataFrame, header: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(f"## {header}\n\n")
        handle.write(table.to_markdown(index=False))
        handle.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert evaluation JSON into paper-ready benchmark tables."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to latest evaluation JSON file.",
    )
    parser.add_argument(
        "--sweep-path",
        type=Path,
        default=DEFAULT_SWEEP_PATH,
        help="Path to hyperparameter sweep JSON file.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing benchmark JSON reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmark_tables.md"),
        help="Destination markdown file for tables.",
    )
    args = parser.parse_args()

    if args.output.exists():
        args.output.unlink()

    summary = format_summary(args.eval_path)
    write_markdown(summary, "Evaluation Summary", args.output)

    if args.sweep_path.exists():
        sweep = format_sweep(args.sweep_path)
        write_markdown(sweep, "Hyperparameter Sweep (Hybrid 8×)", args.output)

    if args.benchmark_dir.exists():
        rows = []
        for path in sorted(args.benchmark_dir.glob("*.json")):
            data = json.loads(path.read_text())
            base_scores = data.get("baseline_scores", data.get("scores", {}))
            scores = data.get("scores", base_scores)
            dataset = data.get("dataset", "Benchmark")
            name = path.stem
            for task, base_score in base_scores.items():
                task_score = scores.get(task, base_score)
                delta = task_score - base_score
                rows.append(
                    {
                        "Config": name,
                        "Dataset": dataset,
                        "Task": task,
                        "Baseline": base_score,
                        "Score": task_score,
                        "Delta": delta,
                    }
                )
        if rows:
            benchmark_df = pd.DataFrame(rows)
            write_markdown(benchmark_df, "Benchmark Accuracy (Synthetic)", args.output)

    print(f"Wrote benchmark tables to {args.output}")


if __name__ == "__main__":
    main()
