"""Token selection policies for adaptive KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .config import LayerBudget, SelectorConfig


@dataclass
class SelectionOutput:
    """Container for selection masks and auxiliary diagnostics."""

    masks: List[torch.Tensor]
    retain_counts: List[int]
    importance_scores: List[torch.Tensor]


class BaseSelector:
    """Abstract interface for computing per-layer token retention masks."""

    def __init__(self, config: SelectorConfig):
        self.config = config

    def compute(self, attention_scores: torch.Tensor) -> SelectionOutput:
        """Return SelectionOutput given aggregated attention scores.

        Parameters
        ----------
        attention_scores:
            Tensor of shape (layers, tokens) containing accumulated attention weights or
            comparable importance signals.
        """
        raise NotImplementedError

    def _base_mask(self, tokens: int, retain_ratio: float, min_tokens: int) -> torch.Tensor:
        """Return default mask retaining most recent tokens."""
        retain = max(min_tokens, int(tokens * retain_ratio))
        retain = min(retain, tokens)
        mask = torch.zeros(tokens, dtype=torch.bool)
        mask[-retain:] = True
        return mask


def _layer_budget_map(
    config: SelectorConfig, num_layers: int, default_ratio: float
) -> List[float]:
    ratios = [default_ratio for _ in range(num_layers)]
    for budget in config.layer_budgets:
        ratios[budget.layer_index] = budget.retain_ratio
    return ratios


class PyramidSelector(BaseSelector):
    """Layer-aware allocation inspired by PyramidKV."""

    def compute(self, attention_scores: torch.Tensor) -> SelectionOutput:
        layers, tokens = attention_scores.shape
        default_ratio = self.config.global_retain_ratio
        layer_ratios = _layer_budget_map(self.config, layers, default_ratio)

        # If no manual budgets provided, shape a pyramid funnel (more budget for lower layers).
        if not self.config.layer_budgets:
            indices = torch.arange(layers, dtype=torch.float32)
            # Linear decay from 1.5x to 0.5x of base ratio
            scaling = torch.linspace(1.5, 0.5, layers)
            layer_ratios = (scaling * default_ratio).tolist()

        masks: List[torch.Tensor] = []
        retain_counts: List[int] = []
        importance: List[torch.Tensor] = []
        for layer_idx in range(layers):
            scores = attention_scores[layer_idx]
            temp = self.config.attention_temperature
            adjusted_scores = F.softmax(scores / temp, dim=-1)
            retain_ratio = min(max(layer_ratios[layer_idx], 0.0), 1.0)
            retain = max(self.config.min_tokens, int(tokens * retain_ratio))
            retain = min(retain, tokens)
            topk = torch.topk(adjusted_scores, retain, dim=-1).indices
            mask = torch.zeros(tokens, dtype=torch.bool)
            mask.scatter_(0, topk, True)
            masks.append(mask)
            retain_counts.append(retain)
            importance.append(adjusted_scores)
        return SelectionOutput(masks=masks, retain_counts=retain_counts, importance_scores=importance)


class SnapSelector(BaseSelector):
    """Approximation of SnapKV clustering-based contiguous selection."""

    def compute(self, attention_scores: torch.Tensor) -> SelectionOutput:
        layers, tokens = attention_scores.shape
        masks: List[torch.Tensor] = []
        retain_counts: List[int] = []
        importance: List[torch.Tensor] = []
        cluster_count = min(self.config.clustering_k, max(1, tokens // 64))

        positions = torch.arange(tokens, dtype=torch.float32)
        for layer_idx in range(layers):
            scores = attention_scores[layer_idx]
            temp = self.config.attention_temperature
            adjusted_scores = F.softmax(scores / temp, dim=-1)
            retain_ratio = self.config.global_retain_ratio
            retain = max(self.config.min_tokens, int(tokens * retain_ratio))
            retain = min(retain, tokens)

            # Simple k-means on 1D positions weighted by attention scores.
            weights = adjusted_scores + 1e-6
            centroids = torch.linspace(0, tokens - 1, cluster_count)
            for _ in range(5):
                dists = torch.abs(positions.unsqueeze(1) - centroids.unsqueeze(0))
                assignment = torch.argmin(dists, dim=1)
                for c in range(cluster_count):
                    mask = assignment == c
                    if mask.sum() == 0:
                        continue
                    centroids[c] = (positions[mask] * weights[mask]).sum() / weights[mask].sum()
            regions: List[torch.Tensor] = []
            for centroid in centroids:
                center = int(torch.clamp(centroid.round(), 0, tokens - 1))
                regions.append(torch.tensor([center], dtype=torch.int64))
            candidates = torch.cat(regions).unique()
            _, top_indices = torch.topk(adjusted_scores[candidates], min(len(candidates), retain))
            selected_positions = candidates[top_indices]
            mask = torch.zeros(tokens, dtype=torch.bool)
            mask.scatter_(0, selected_positions, True)
            masks.append(mask)
            retain_counts.append(mask.sum().item())
            importance.append(adjusted_scores)
        return SelectionOutput(masks=masks, retain_counts=retain_counts, importance_scores=importance)


class StreamingSelector(BaseSelector):
    """StreamingLLM-inspired policy retaining sink tokens and a sliding window."""

    def __init__(self, config: SelectorConfig, sink_tokens: int = 4):
        super().__init__(config)
        self.sink_tokens = sink_tokens

    def compute(self, attention_scores: torch.Tensor) -> SelectionOutput:
        layers, tokens = attention_scores.shape
        window = self.config.window_size or max(self.sink_tokens * 4, tokens // 8)
        masks: List[torch.Tensor] = []
        retain_counts: List[int] = []
        importance: List[torch.Tensor] = []

        for _layer_idx in range(layers):
            mask = torch.zeros(tokens, dtype=torch.bool)
            mask[: self.sink_tokens] = True
            mask[-window:] = True
            retain_counts.append(mask.sum().item())
            masks.append(mask)
            importance.append(torch.zeros_like(attention_scores[0]))
        return SelectionOutput(masks=masks, retain_counts=retain_counts, importance_scores=importance)


class HybridSelector(BaseSelector):
    """Combines Pyramid layer awareness with StreamingLLM recency guarantees."""

    def __init__(self, config: SelectorConfig):
        super().__init__(config)
        self.pyramid = PyramidSelector(config)
        self.streaming = StreamingSelector(config)

    def compute(self, attention_scores: torch.Tensor) -> SelectionOutput:
        pyramid = self.pyramid.compute(attention_scores)
        streaming = self.streaming.compute(attention_scores)
        masks: List[torch.Tensor] = []
        retain_counts: List[int] = []
        importance: List[torch.Tensor] = []
        for layer_idx, mask in enumerate(pyramid.masks):
            combined = mask | streaming.masks[layer_idx]
            masks.append(combined)
            retain_counts.append(int(combined.sum().item()))
            importance.append(pyramid.importance_scores[layer_idx])
        return SelectionOutput(masks=masks, retain_counts=retain_counts, importance_scores=importance)


def build_selector(config: SelectorConfig) -> BaseSelector:
    """Factory returning the requested selector implementation."""

    if config.strategy == "pyramid":
        return PyramidSelector(config)
    if config.strategy == "snap":
        return SnapSelector(config)
    if config.strategy == "streaming":
        return StreamingSelector(config)
    if config.strategy == "hybrid":
        return HybridSelector(config)
    msg = f"Unknown selection strategy: {config.strategy}"
    raise ValueError(msg)

