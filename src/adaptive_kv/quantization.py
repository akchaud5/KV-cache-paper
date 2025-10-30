"""KV cache quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Size

from .config import QuantizationConfig


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data and metadata."""

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    original_shape: Size
    group_size: int


@dataclass
class QuantizationOutput:
    """Quantized keys/values with additional metadata for bookkeeping."""

    keys: QuantizedTensor
    values: QuantizedTensor
    high_precision_mask: torch.Tensor
    high_precision_keys: torch.Tensor
    high_precision_values: torch.Tensor


class BaseQuantizer:
    """Interface for quantizing KV tensors."""

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(
        self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor
    ) -> QuantizationOutput:
        raise NotImplementedError

    @staticmethod
    def _quantize_uniform_asymmetric(
        tensor: torch.Tensor,
        bits: int,
        group_size: int,
        stochastic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor using per-group asymmetric affine mapping."""
        qmin = 0
        qmax = 2**bits - 1
        orig_shape = tensor.shape
        last_dim = orig_shape[-1]
        pad = (group_size - last_dim % group_size) % group_size
        if pad > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad), mode="constant", value=0.0)
        group_count = tensor.shape[-1] // group_size
        groups = tensor.view(-1, group_size)
        min_vals, _ = groups.min(dim=1, keepdim=True)
        max_vals, _ = groups.max(dim=1, keepdim=True)
        ranges = (max_vals - min_vals).clamp(min=1e-6)
        scales = ranges / max(qmax - qmin, 1)
        zero_points = qmin - min_vals / scales
        zero_points = zero_points.clamp(qmin, qmax)
        normalized = groups / scales + zero_points
        if stochastic:
            noise = torch.empty_like(normalized).uniform_(-0.5, 0.5)
            normalized = normalized + noise
        quantized = torch.round(normalized).clamp(qmin, qmax)
        quantized = quantized.view(*tensor.shape)
        if pad > 0:
            quantized = quantized[..., :last_dim]
        quantized = quantized.view(*orig_shape)
        scale_view = scales.view(*orig_shape[:-1], group_count)
        zero_view = zero_points.view(*orig_shape[:-1], group_count)
        if pad > 0:
            effective_groups = (last_dim + group_size - 1) // group_size
            scale_view = scale_view[..., :effective_groups]
            zero_view = zero_view[..., :effective_groups]
        return quantized.to(torch.int32), scale_view, zero_view


class UniformQuantizer(BaseQuantizer):
    """Uniform quantizer with optional mixed precision for keys and values."""

    def quantize(
        self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor
    ) -> QuantizationOutput:
        config = self.config
        key_bits = config.key_bits or config.default_bits
        value_bits = config.value_bits or config.default_bits

        guard_fraction = config.high_precision_guard
        retain = max(1, int(keys.shape[0] * guard_fraction))
        importance_flat = importance.view(-1)
        retain = min(retain, importance_flat.numel())
        topk = torch.topk(importance_flat, retain, dim=0).indices
        high_precision_mask = torch.zeros(
            importance_flat.shape[0], dtype=torch.bool, device=keys.device
        )
        high_precision_mask.scatter_(0, topk.unique(), True)

        q_keys = self._quantize_uniform_asymmetric(
            keys, key_bits, config.group_size, config.stochastic_rounding
        )
        q_values = self._quantize_uniform_asymmetric(
            values, value_bits, config.group_size, config.stochastic_rounding
        )

        keys_tensor = QuantizedTensor(
            data=q_keys[0],
            scale=q_keys[1],
            zero_point=q_keys[2],
            bits=key_bits,
            original_shape=keys.shape,
            group_size=config.group_size,
        )
        values_tensor = QuantizedTensor(
            data=q_values[0],
            scale=q_values[1],
            zero_point=q_values[2],
            bits=value_bits,
            original_shape=values.shape,
            group_size=config.group_size,
        )
        return QuantizationOutput(
            keys=keys_tensor,
            values=values_tensor,
            high_precision_mask=high_precision_mask,
            high_precision_keys=keys[high_precision_mask].contiguous(),
            high_precision_values=values[high_precision_mask].contiguous(),
        )


class MixedPrecisionQuantizer(BaseQuantizer):
    """Importance-aware bit allocation for keys/values."""

    def quantize(
        self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor
    ) -> QuantizationOutput:
        config = self.config
        default_bits = config.default_bits
        key_bits = config.key_bits or default_bits
        value_bits = config.value_bits or default_bits

        guard_fraction = config.high_precision_guard
        num_tokens = keys.shape[0]
        retain_high_precision = max(1, int(num_tokens * guard_fraction))
        importance_flat = importance.reshape(num_tokens, -1).mean(dim=1)
        ranks = torch.argsort(importance_flat, descending=True)
        high_precision_indices = ranks[:retain_high_precision]
        high_precision_mask = torch.zeros(num_tokens, dtype=torch.bool, device=keys.device)
        high_precision_mask[high_precision_indices] = True

        q_keys = self._quantize_uniform_asymmetric(
            keys, key_bits, config.group_size, config.stochastic_rounding
        )
        q_values = self._quantize_uniform_asymmetric(
            values, value_bits, config.group_size, config.stochastic_rounding
        )

        keys_tensor = QuantizedTensor(
            data=q_keys[0],
            scale=q_keys[1],
            zero_point=q_keys[2],
            bits=key_bits,
            original_shape=keys.shape,
            group_size=config.group_size,
        )
        values_tensor = QuantizedTensor(
            data=q_values[0],
            scale=q_values[1],
            zero_point=q_values[2],
            bits=value_bits,
            original_shape=values.shape,
            group_size=config.group_size,
        )
        return QuantizationOutput(
            keys=keys_tensor,
            values=values_tensor,
            high_precision_mask=high_precision_mask,
            high_precision_keys=keys[high_precision_mask].contiguous(),
            high_precision_values=values[high_precision_mask].contiguous(),
        )


def build_quantizer(config: QuantizationConfig) -> BaseQuantizer:
    """Factory for quantizer instances."""

    if not config.enabled:
        return NoOpQuantizer(config)
    if config.high_precision_guard > 0.0:
        return MixedPrecisionQuantizer(config)
    return UniformQuantizer(config)


class NoOpQuantizer(BaseQuantizer):
    """Returns tensors unchanged while still returning metadata."""

    def quantize(
        self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor
    ) -> QuantizationOutput:
        metadata = torch.ones(keys.shape[:-1], device=keys.device)
        tensor = QuantizedTensor(
            data=keys.to(torch.float32),
            scale=torch.ones_like(metadata),
            zero_point=torch.zeros_like(metadata),
            bits=16,
            original_shape=keys.shape,
            group_size=1,
        )
        return QuantizationOutput(
            keys=tensor,
            values=QuantizedTensor(
                data=values.to(torch.float32),
                scale=torch.ones_like(metadata),
                zero_point=torch.zeros_like(metadata),
                bits=16,
                original_shape=values.shape,
                group_size=1,
            ),
            high_precision_mask=torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device),
            high_precision_keys=keys.clone(),
            high_precision_values=values.clone(),
        )
