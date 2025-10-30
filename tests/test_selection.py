import torch

from adaptive_kv.config import LayerBudget, SelectorConfig
from adaptive_kv.selection import build_selector


def make_attention(layers: int = 4, tokens: int = 128) -> torch.Tensor:
    base = torch.arange(tokens, dtype=torch.float32)
    attention = torch.stack([torch.sin(base / (idx + 1)) for idx in range(1, layers + 1)], dim=0)
    return attention.abs()


def test_pyramid_selector_allocates_more_to_lower_layers():
    config = SelectorConfig(strategy="pyramid", global_retain_ratio=0.25, min_tokens=16)
    selector = build_selector(config)
    scores = make_attention()
    output = selector.compute(scores)
    assert len(output.masks) == scores.shape[0]
    retain_first = output.retain_counts[0]
    retain_last = output.retain_counts[-1]
    assert retain_first >= retain_last


def test_hybrid_selector_preserves_sink_tokens():
    config = SelectorConfig(
        strategy="hybrid",
        global_retain_ratio=0.2,
        min_tokens=16,
        window_size=32,
        layer_budgets=[LayerBudget(layer_index=0, retain_ratio=0.4)],
    )
    selector = build_selector(config)
    scores = make_attention(tokens=256)
    output = selector.compute(scores)
    mask = output.masks[0]
    assert mask[:4].all()  # sink tokens retained
    assert mask[-32:].all()  # sliding window retained

