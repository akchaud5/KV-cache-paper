import torch

from adaptive_kv.config import QuantizationConfig
from adaptive_kv.quantization import build_quantizer


def test_quantizer_outputs_expected_shapes():
    config = QuantizationConfig(
        enabled=True,
        default_bits=4,
        key_bits=4,
        value_bits=2,
        high_precision_guard=0.1,
        group_size=16,
    )
    quantizer = build_quantizer(config)
    keys = torch.randn(128, 16, 64)
    values = torch.randn_like(keys)
    importance = torch.rand(128)
    output = quantizer.quantize(keys, values, importance)
    assert output.keys.data.shape == keys.shape
    assert output.values.data.shape == values.shape
    assert output.high_precision_mask.shape[0] == keys.shape[0]
    assert output.high_precision_keys.shape[0] == output.high_precision_mask.sum()


def test_quantization_reduces_value_variance():
    config = QuantizationConfig(
        enabled=True,
        default_bits=3,
        key_bits=4,
        value_bits=2,
        high_precision_guard=0.05,
        group_size=32,
    )
    quantizer = build_quantizer(config)
    keys = torch.randn(64, 8, 32)
    values = torch.randn_like(keys)
    importance = torch.rand(64)
    baseline_var = values.var()
    output = quantizer.quantize(keys, values, importance)
    quantized_var = output.values.data.float().var()
    assert quantized_var <= baseline_var

