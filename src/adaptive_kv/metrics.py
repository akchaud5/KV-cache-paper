"""Metrics helpers for reporting compression efficacy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MemoryStats:
    """Memory usage summary."""

    original_mb: float
    compressed_mb: float
    reduction_ratio: float


@dataclass
class PerformanceStats:
    """Latency and throughput deltas."""

    throughput_gain: float
    ttft_delta_ms: float
    tpot_delta_ms: float


def summarize_memory(original_bytes: int, compressed_bytes: int) -> MemoryStats:
    """Return memory summary in MB."""
    mb = 1024 * 1024
    compressed = max(compressed_bytes, 1)
    reduction = original_bytes / compressed
    return MemoryStats(
        original_mb=original_bytes / mb,
        compressed_mb=compressed_bytes / mb,
        reduction_ratio=reduction,
    )


def summarize_performance(
    baseline_tps: float,
    compressed_tps: float,
    baseline_ttft_ms: float,
    compressed_ttft_ms: float,
    baseline_tpot_ms: float,
    compressed_tpot_ms: float,
) -> PerformanceStats:
    """Return throughput and latency deltas."""
    throughput_gain = compressed_tps / max(baseline_tps, 1e-6)
    ttft_delta = compressed_ttft_ms - baseline_ttft_ms
    tpot_delta = compressed_tpot_ms - baseline_tpot_ms
    return PerformanceStats(
        throughput_gain=throughput_gain,
        ttft_delta_ms=ttft_delta,
        tpot_delta_ms=tpot_delta,
    )


def metric_report(memory: MemoryStats, perf: PerformanceStats) -> Dict[str, float]:
    """Return dictionary ready for logging or serialization."""
    return {
        "memory_reduction": memory.reduction_ratio,
        "original_mb": memory.original_mb,
        "compressed_mb": memory.compressed_mb,
        "throughput_gain": perf.throughput_gain,
        "ttft_delta_ms": perf.ttft_delta_ms,
        "tpot_delta_ms": perf.tpot_delta_ms,
    }

