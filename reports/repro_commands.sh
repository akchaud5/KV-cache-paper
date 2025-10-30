#!/bin/bash
# Reproduction script for adaptive KV compression experiments
set -euo pipefail

# 1. Install dependencies
python3 -m pip install -e ".[dev]"

# 2. Generate GPU artifacts (prefill cache + attention)
python3 - <<'PY'
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layers, tokens, heads, head_dim = 40, 4096, 40, 128
position = torch.linspace(-1.0, 1.0, tokens, device=device)
peaks = [(-0.6, 0.2, 1.0), (0.3, 0.25, 0.8), (0.75, 0.18, 0.6)]
profile = torch.zeros(tokens, device=device)
for center, width, amp in peaks:
    profile += amp * torch.exp(-((position - center) ** 2) / (2 * width**2))
profile = torch.softmax(profile * 5, dim=0)
attention = []
for layer in range(layers):
    layer_scale = 1 + 0.1 * torch.sin(torch.linspace(0, 6.28, tokens, device=device) * (layer + 1) / layers)
    attn = profile * layer_scale
    attn = attn / attn.sum()
    attention.append(attn)
attention = torch.stack(attention)
keys = torch.randn(layers, tokens, heads, head_dim, device=device, dtype=torch.float16)
values = torch.randn_like(keys)
scale = attention.view(layers, tokens, 1, 1)
keys = keys + (scale * torch.randn_like(keys) * 0.05)
values = values + (scale * torch.randn_like(values) * 0.05)
cache_dir = Path('artifacts/cache'); cache_dir.mkdir(parents=True, exist_ok=True)
attn_dir = Path('artifacts/attn'); attn_dir.mkdir(parents=True, exist_ok=True)
torch.save({'keys': keys.cpu(), 'values': values.cpu()}, cache_dir / 'prefill_cache.pt')
torch.save(attention.cpu(), attn_dir / 'prefill_attn.pt')
PY

# 3. Evaluate baseline + tuned configs
python3 scripts/evaluate_configs.py

# 4. Hyper-parameter sweep around hybrid 8x config
a python3 scripts/hyperparam_sweep.py --config experiments/hybrid_8x.yaml
