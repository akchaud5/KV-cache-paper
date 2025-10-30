"""Tensor abstractions describing transformer KV caches.

The implementation assumes standard decoder-only transformer layout where keys and values
are stored per layer with shape `(num_layers, seq_len, num_heads, head_dim)`.
These utilities intentionally mirror vLLM/SGLang conventions so they can be swapped into
existing paged-attention runtimes with minimal glue code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _validate_cache_shapes(keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.ndim != 4 or values.ndim != 4:
        msg = "Keys and values must be 4-D tensors: (layers, tokens, heads, head_dim)."
        raise ValueError(msg)
    if keys.shape != values.shape:
        msg = f"Key/value shape mismatch: {keys.shape} vs {values.shape}"
        raise ValueError(msg)


@dataclass
class LayerCacheView:
    """Convenience wrapper exposing per-layer cache utilities."""

    keys: torch.Tensor
    values: torch.Tensor

    def select_tokens(self, token_indices: torch.Tensor) -> "LayerCacheView":
        """Return a new view containing only the selected token positions."""
        if token_indices.dtype not in (torch.int32, torch.int64):
            msg = "token_indices must be int32 or int64 tensor."
            raise ValueError(msg)
        return LayerCacheView(
            keys=self.keys.index_select(dim=0, index=token_indices),
            values=self.values.index_select(dim=0, index=token_indices),
        )

    def retain_mask(self, mask: torch.Tensor) -> "LayerCacheView":
        """Return a view filtered by a boolean mask."""
        if mask.dtype != torch.bool:
            msg = "mask must be boolean tensor."
            raise ValueError(msg)
        if mask.shape[0] != self.keys.shape[0]:
            msg = "mask length must match layer token dimension."
            raise ValueError(msg)
        return LayerCacheView(keys=self.keys[mask], values=self.values[mask])

    @property
    def token_count(self) -> int:
        return int(self.keys.shape[0])

    @property
    def storage_bytes(self) -> int:
        bytes_per_element = self.keys.element_size() + self.values.element_size()
        return int(self.keys.numel() + self.values.numel()) * bytes_per_element


@dataclass
class KVCache:
    """Represents the full transformer KV cache with layer access convenience."""

    keys: torch.Tensor
    values: torch.Tensor
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        _validate_cache_shapes(self.keys, self.values)
        if self.device is None:
            self.device = self.keys.device

    @property
    def layers(self) -> int:
        return int(self.keys.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.keys.shape[1])

    @property
    def heads(self) -> int:
        return int(self.keys.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.keys.shape[3])

    def layer(self, index: int) -> LayerCacheView:
        """Return a LayerCacheView for the specified layer index."""
        if index < 0 or index >= self.layers:
            msg = f"Layer index {index} out of range for {self.layers} layers."
            raise IndexError(msg)
        return LayerCacheView(
            keys=self.keys[index],
            values=self.values[index],
        )

    def clone_empty(self, tokens: Optional[int] = None) -> "KVCache":
        """Return an empty cache with the same metadata and optional token length."""
        token_len = tokens or self.tokens
        shape = (self.layers, token_len, self.heads, self.head_dim)
        keys = torch.zeros(shape, dtype=self.keys.dtype, device=self.device)
        values = torch.zeros(shape, dtype=self.values.dtype, device=self.device)
        return KVCache(keys=keys, values=values, device=self.device)

    def select_tokens(self, mask_per_layer: Iterable[torch.Tensor]) -> "KVCache":
        """Select tokens per layer and return a compacted cache."""
        compact_keys: List[torch.Tensor] = []
        compact_values: List[torch.Tensor] = []
        for layer_idx, mask in enumerate(mask_per_layer):
            layer_view = self.layer(layer_idx).retain_mask(mask)
            compact_keys.append(layer_view.keys)
            compact_values.append(layer_view.values)
        keys = torch.nn.utils.rnn.pad_sequence(
            compact_keys, batch_first=False
        )  # shape (max_tokens, layers, heads, head_dim)
        values = torch.nn.utils.rnn.pad_sequence(compact_values, batch_first=False)
        # Rearrange back to (layers, tokens, heads, head_dim)
        keys = keys.permute(1, 0, 2, 3).contiguous()
        values = values.permute(1, 0, 2, 3).contiguous()
        return KVCache(keys=keys, values=values, device=self.device)

    def to(self, device: torch.device) -> "KVCache":
        """Move cache tensors to the target device."""
        return KVCache(
            keys=self.keys.to(device=device, non_blocking=True),
            values=self.values.to(device=device, non_blocking=True),
            device=device,
        )

    @property
    def storage_bytes(self) -> int:
        bytes_per_element = self.keys.element_size() + self.values.element_size()
        return int(self.keys.numel() + self.values.numel()) * bytes_per_element

    def apply_updates(
        self, layer_indices: Iterable[int], token_indices: Iterable[int], new_values: torch.Tensor
    ) -> None:
        """In-place update for modified cache entries."""
        for offset, (layer_idx, token_idx) in enumerate(zip(layer_indices, token_indices)):
            self.keys[layer_idx, token_idx] = new_values[offset, 0]
            self.values[layer_idx, token_idx] = new_values[offset, 1]


class PagedKVCache:
    """Logical view matching paged-attention memory layouts.

    Attributes
    ----------
    cache : KVCache
        Underlying contiguous cache representation.
    page_size : int
        Number of tokens per page.
    page_map : Dict[Tuple[int, int], int]
        Mapping from (request_id, token_idx) to page id.
    """

    def __init__(self, cache: KVCache, page_size: int = 128):
        if page_size <= 0:
            msg = "page_size must be positive."
            raise ValueError(msg)
        self.cache = cache
        self.page_size = page_size
        self.page_map: Dict[Tuple[int, int], int] = {}

    def register_request(self, request_id: int, token_count: int) -> None:
        """Assign pages for a given request."""
        required_pages = (token_count + self.page_size - 1) // self.page_size
        start_page = len(self.page_map)
        for page_offset in range(required_pages):
            page_id = start_page + page_offset
            for token_idx in range(self.page_size):
                global_token = page_offset * self.page_size + token_idx
                if global_token >= token_count:
                    break
                self.page_map[(request_id, global_token)] = page_id

    def page_usage(self) -> Dict[int, int]:
        """Return token counts per page."""
        usage: Dict[int, int] = {}
        for page_id in self.page_map.values():
            usage[page_id] = usage.get(page_id, 0) + 1
        return usage

    def gather_tokens(self, request_id: int, token_indices: torch.Tensor) -> KVCache:
        """Gather selected tokens for a request, preserving layer structure."""
        layers = []
        for layer_idx in range(self.cache.layers):
            layer_view = self.cache.layer(layer_idx)
            layers.append(layer_view.select_tokens(token_indices))
        keys = torch.stack([layer.keys for layer in layers])
        values = torch.stack([layer.values for layer in layers])
        return KVCache(keys=keys, values=values, device=self.cache.device)
