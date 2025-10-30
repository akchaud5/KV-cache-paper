import torch

from adaptive_kv.cache import KVCache, PagedKVCache


def make_cache(layers: int = 4, tokens: int = 64, heads: int = 8, head_dim: int = 32) -> KVCache:
    keys = torch.randn(layers, tokens, heads, head_dim, dtype=torch.float16)
    values = torch.randn_like(keys)
    return KVCache(keys=keys, values=values)


def test_clone_empty_preserves_shape():
    cache = make_cache()
    empty = cache.clone_empty()
    assert empty.keys.shape == cache.keys.shape
    assert empty.values.shape == cache.values.shape


def test_layer_view_retains_tokens():
    cache = make_cache(tokens=16)
    layer = cache.layer(0)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[-4:] = True
    subset = layer.retain_mask(mask)
    assert subset.keys.shape[0] == 4
    assert subset.values.shape == subset.keys.shape


def test_paged_cache_registers_requests():
    cache = make_cache(tokens=32)
    paged = PagedKVCache(cache, page_size=8)
    paged.register_request(request_id=1, token_count=20)
    usage = paged.page_usage()
    assert sum(usage.values()) == 20
    assert len(usage) == 3

