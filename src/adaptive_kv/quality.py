"""Quality evaluation helpers for KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .cache import KVCache
from .pipeline import CompressedCache


@dataclass
class QualityReport:
    """Aggregate collection of quality metrics."""

    key_mse: float
    value_mse: float
    key_cosine: float
    value_cosine: float
    attention_coverage: float
    selected_fraction: float
    high_precision_fraction: float
    key_cosine_selected: float
    value_cosine_selected: float


def cache_error_metrics(original: KVCache, reconstructed: KVCache) -> Dict[str, float]:
    """Return MSE and cosine similarity between original and reconstructed caches."""
    orig_keys = original.keys.float().view(original.keys.shape[0], -1)
    orig_values = original.values.float().view(original.values.shape[0], -1)
    rec_keys = reconstructed.keys.float().view(reconstructed.keys.shape[0], -1)
    rec_values = reconstructed.values.float().view(reconstructed.values.shape[0], -1)

    key_mse = torch.mean((orig_keys - rec_keys) ** 2).item()
    value_mse = torch.mean((orig_values - rec_values) ** 2).item()

    key_cosine = F.cosine_similarity(orig_keys, rec_keys, dim=1).mean().item()
    value_cosine = F.cosine_similarity(orig_values, rec_values, dim=1).mean().item()
    return {
        "key_mse": key_mse,
        "value_mse": value_mse,
        "key_cosine": key_cosine,
        "value_cosine": value_cosine,
    }


def attention_retention_metrics(
    compressed: CompressedCache, attention_scores: torch.Tensor
) -> Dict[str, float]:
    """Return metrics describing how well selection keeps attention mass."""
    layers, tokens = attention_scores.shape
    total_attention = attention_scores.sum().clamp(min=1e-8)
    retained_attention = 0.0
    selected_tokens = 0
    high_precision_tokens = 0

    for layer_idx, mask in enumerate(compressed.selection.masks):
        layer_attention = attention_scores[layer_idx]
        retained_attention += layer_attention[mask].sum().item()
        selected_tokens += int(mask.sum().item())

    for layer in compressed.layers:
        high_precision_tokens += int(layer.quantized.high_precision_mask.sum().item())

    attention_coverage = float(retained_attention / total_attention.item())
    selected_fraction = float(selected_tokens / (layers * tokens))
    high_precision_fraction = (
        float(high_precision_tokens) / max(selected_tokens, 1)
        if selected_tokens
        else 0.0
    )
    return {
        "attention_coverage": attention_coverage,
        "selected_fraction": selected_fraction,
        "high_precision_fraction": high_precision_fraction,
    }


def build_quality_report(
    original: KVCache, reconstructed: KVCache, compressed: CompressedCache, attention: torch.Tensor
) -> QualityReport:
    """Compute combined quality metrics."""
    cache_metrics = cache_error_metrics(original, reconstructed)
    attention_metrics = attention_retention_metrics(compressed, attention)
    key_selected, value_selected = _selected_similarity(original, reconstructed, compressed)
    return QualityReport(
        key_mse=cache_metrics["key_mse"],
        value_mse=cache_metrics["value_mse"],
        key_cosine=cache_metrics["key_cosine"],
        value_cosine=cache_metrics["value_cosine"],
        attention_coverage=attention_metrics["attention_coverage"],
        selected_fraction=attention_metrics["selected_fraction"],
        high_precision_fraction=attention_metrics["high_precision_fraction"],
        key_cosine_selected=key_selected,
        value_cosine_selected=value_selected,
    )


def _selected_similarity(
    original: KVCache, reconstructed: KVCache, compressed: CompressedCache
) -> tuple[float, float]:
    """Return cosine similarity restricted to retained tokens."""
    key_scores = []
    value_scores = []
    for layer_idx, mask in enumerate(compressed.selection.masks):
        if not mask.any():
            continue
        orig_keys = original.keys[layer_idx][mask].float().reshape(mask.sum(), -1)
        rec_keys = reconstructed.keys[layer_idx][mask].float().reshape(mask.sum(), -1)
        orig_values = original.values[layer_idx][mask].float().reshape(mask.sum(), -1)
        rec_values = reconstructed.values[layer_idx][mask].float().reshape(mask.sum(), -1)
        key_scores.append(F.cosine_similarity(orig_keys, rec_keys, dim=1))
        value_scores.append(F.cosine_similarity(orig_values, rec_values, dim=1))
    if not key_scores:
        return 1.0, 1.0
    key_mean = torch.cat(key_scores).mean().item()
    value_mean = torch.cat(value_scores).mean().item()
    return key_mean, value_mean
