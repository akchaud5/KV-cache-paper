import pytest
import torch

from adaptive_kv.cache import KVCache
from adaptive_kv.pipeline import CompressedCache, CompressedLayer
from adaptive_kv.quality import build_quality_report, cache_error_metrics


def _make_cache(layers: int = 2, tokens: int = 8, heads: int = 2, head_dim: int = 4) -> KVCache:
    keys = torch.randn(layers, tokens, heads, head_dim, dtype=torch.float16)
    values = torch.randn_like(keys)
    return KVCache(keys=keys, values=values)


def test_cache_error_metrics_zero_when_identical():
    cache = _make_cache()
    metrics = cache_error_metrics(cache, cache)
    assert metrics["key_mse"] == 0.0
    assert metrics["value_mse"] == 0.0
    assert metrics["key_cosine"] == pytest.approx(1.0)
    assert metrics["value_cosine"] == pytest.approx(1.0)


def test_quality_report_tracks_attention_and_high_precision():
    cache = _make_cache()
    attention = torch.rand(cache.layers, cache.tokens)

    compressed_layers = []
    masks = []
    retain_counts = []
    importance_scores = []
    for layer_idx in range(cache.layers):
        token_indices = torch.arange(cache.tokens)
        hp_mask = torch.zeros(cache.tokens, dtype=torch.bool)
        hp_mask[: cache.tokens // 2] = True
        quantized = type("QuantizedOutput", (), {})()
        quantized.keys = type("QuantizedTensor", (), {})()
        quantized.values = type("QuantizedTensor", (), {})()
        quantized.high_precision_mask = hp_mask
        quantized.high_precision_keys = cache.keys[layer_idx, hp_mask]
        quantized.high_precision_values = cache.values[layer_idx, hp_mask]
        compressed_layers.append(
            CompressedLayer(layer_index=layer_idx, token_indices=token_indices, quantized=quantized)
        )
        mask = torch.ones(cache.tokens, dtype=torch.bool)
        masks.append(mask)
        retain_counts.append(cache.tokens)
        importance_scores.append(torch.rand(cache.tokens))

    selection = type("SelectionOutput", (), {})()
    selection.masks = masks
    selection.retain_counts = retain_counts
    selection.importance_scores = importance_scores
    compressed = CompressedCache(
        layers=compressed_layers,
        selection=selection,
        original_bytes=cache.storage_bytes,
        compressed_bytes=cache.storage_bytes // 4,
    )

    report = build_quality_report(cache, cache, compressed, attention)
    assert 0.0 <= report.attention_coverage <= 1.0 + 1e-6
    assert report.selected_fraction == 1.0
    assert report.high_precision_fraction == 0.5
    assert report.key_cosine_selected == pytest.approx(1.0)
    assert report.value_cosine_selected == pytest.approx(1.0)
