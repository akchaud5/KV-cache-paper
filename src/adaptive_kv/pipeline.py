"""Compression pipeline orchestrating selection and quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from .cache import KVCache
from .config import ExperimentConfig
from .quantization import QuantizationOutput, QuantizedTensor, build_quantizer
from .selection import SelectionOutput, build_selector


@dataclass
class CompressedLayer:
    """Compressed representation for a single transformer layer."""

    layer_index: int
    token_indices: torch.Tensor
    quantized: QuantizationOutput


@dataclass
class CompressedCache:
    """Aggregate compressed cache including diagnostics."""

    layers: List[CompressedLayer]
    selection: SelectionOutput
    original_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return float("inf")
        return self.original_bytes / self.compressed_bytes


class CompressionPipeline:
    """End-to-end pipeline for adaptive KV cache compression."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.selector = build_selector(config.selector)
        self.quantizer = build_quantizer(config.quantization)

    def compress(self, cache: KVCache, attention_scores: torch.Tensor) -> CompressedCache:
        """Compress KV cache using configured selector and quantizer."""

        if attention_scores.shape[0] != cache.layers:
            msg = (
                f"Attention score layer count {attention_scores.shape[0]} "
                f"does not match cache layers {cache.layers}"
            )
            raise ValueError(msg)

        selection = self.selector.compute(attention_scores)
        compressed_layers: List[CompressedLayer] = []
        total_compressed_bits = 0

        for layer_idx, mask in enumerate(selection.masks):
            layer_view = cache.layer(layer_idx)
            token_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if token_indices.numel() == 0:
                continue
            selected_keys = layer_view.keys[token_indices]
            selected_values = layer_view.values[token_indices]
            importance = selection.importance_scores[layer_idx][token_indices]
            quantized = self.quantizer.quantize(selected_keys, selected_values, importance)
            compressed_layers.append(
                CompressedLayer(
                    layer_index=layer_idx,
                    token_indices=token_indices,
                    quantized=quantized,
                )
            )
            total_compressed_bits += _estimate_bits(quantized)

        original_bits = cache.storage_bytes * 8
        compressed_bytes = (total_compressed_bits + 7) // 8
        return CompressedCache(
            layers=compressed_layers,
            selection=selection,
            original_bytes=cache.storage_bytes,
            compressed_bytes=compressed_bytes,
        )

    def reconstruct(self, compressed: CompressedCache, template: KVCache) -> KVCache:
        """Reconstruct an approximate KV cache from compressed form."""

        restored = template.clone_empty()
        for layer in compressed.layers:
            target_keys = restored.keys[layer.layer_index]
            target_values = restored.values[layer.layer_index]
            token_indices = layer.token_indices

            # Restore high-precision tokens directly.
            hp_mask = layer.quantized.high_precision_mask
            hp_indices = token_indices[hp_mask]
            target_keys[hp_indices] = layer.quantized.high_precision_keys.to(target_keys.dtype)
            target_values[hp_indices] = layer.quantized.high_precision_values.to(
                target_values.dtype
            )

            # Dequantize remaining tokens.
            lp_mask = ~hp_mask
            if lp_mask.sum() == 0:
                continue
            lp_indices = token_indices[lp_mask]
            keys_lp = _dequantize(layer.quantized.keys)
            values_lp = _dequantize(layer.quantized.values)
            target_keys[lp_indices] = keys_lp[lp_mask].to(target_keys.dtype)
            target_values[lp_indices] = values_lp[lp_mask].to(target_values.dtype)
        return restored


def _dequantize(tensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize tensor back to floating point."""

    data = tensor.data.to(torch.float32)
    original_shape = tensor.original_shape
    group_size = tensor.group_size
    # Expand per-group parameters back to per-dimension.
    scale = tensor.scale.unsqueeze(-1).repeat_interleave(group_size, dim=-1)
    zero = tensor.zero_point.unsqueeze(-1).repeat_interleave(group_size, dim=-1)
    scale = scale[..., : original_shape[-1]].reshape(original_shape)
    zero = zero[..., : original_shape[-1]].reshape(original_shape)
    return (data - zero) * scale


def _estimate_bits(output: QuantizationOutput) -> int:
    """Estimate bit usage including metadata."""
    data_bits = output.keys.bits * output.keys.data.numel()
    data_bits += output.values.bits * output.values.data.numel()
    metadata_bits = (
        output.keys.scale.numel() * output.keys.scale.element_size() * 8
        + output.keys.zero_point.numel() * output.keys.zero_point.element_size() * 8
        + output.values.scale.numel() * output.values.scale.element_size() * 8
        + output.values.zero_point.numel() * output.values.zero_point.element_size() * 8
    )
    high_precision_bits = (
        output.high_precision_keys.numel() * output.high_precision_keys.element_size() * 8
        + output.high_precision_values.numel() * output.high_precision_values.element_size() * 8
    )
    return data_bits + metadata_bits + high_precision_bits
