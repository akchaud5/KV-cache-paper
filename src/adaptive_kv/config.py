"""Configuration models for adaptive KV cache compression experiments."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)


class LayerBudget(BaseModel):
    """Cache budget specification for a single transformer layer."""

    layer_index: NonNegativeInt = Field(description="Zero-based layer index.")
    retain_ratio: PositiveFloat = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of tokens to retain for this layer after sparsification.",
    )
    priority: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance weighting for budget redistribution.",
    )


class SelectorConfig(BaseModel):
    """Configuration for token selection / sparsification strategies."""

    strategy: Literal["pyramid", "snap", "streaming", "hybrid"] = Field(
        description="Selection policy to use."
    )
    global_retain_ratio: PositiveFloat = Field(
        gt=0.0,
        le=1.0,
        default=0.25,
        description="Overall fraction of tokens to retain before layer-specific overrides.",
    )
    window_size: Optional[PositiveInt] = Field(
        default=None, description="Sliding window length for streaming-style policies."
    )
    min_tokens: int = Field(
        default=32,
        ge=0,
        description="Minimum tokens to retain in any layer regardless of ratios.",
    )
    layer_budgets: List[LayerBudget] = Field(
        default_factory=list,
        description="Optional per-layer budget overrides (e.g., PyramidKV profile).",
    )
    clustering_k: int = Field(
        default=8,
        ge=1,
        description="Number of clusters to use when forming contiguous token regions.",
    )
    attention_temperature: PositiveFloat = Field(
        default=1.0,
        description="Temperature applied to attention scores when computing importance.",
    )

    @field_validator("layer_budgets")
    @classmethod
    def ensure_unique_layers(cls, value: List[LayerBudget]) -> List[LayerBudget]:
        """Ensure no duplicate layer indices exist."""
        indices = {budget.layer_index for budget in value}
        if len(indices) != len(value):
            msg = "Duplicate layer_index entries detected in layer_budgets."
            raise ValueError(msg)
        return value


class QuantizationConfig(BaseModel):
    """Configuration for KV cache quantization."""

    enabled: bool = Field(default=True)
    default_bits: Literal[2, 3, 4, 8, 16] = Field(
        default=4, description="Nominal bit-width for KV tensors."
    )
    key_bits: Optional[Literal[2, 3, 4, 8, 16]] = Field(
        default=None, description="Override bit-width for key tensors."
    )
    value_bits: Optional[Literal[2, 3, 4, 8, 16]] = Field(
        default=None, description="Override bit-width for value tensors."
    )
    high_precision_guard: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Fraction of highest-importance tokens kept at full precision as a guard band."
        ),
    )
    asymmetric: bool = Field(
        default=True, description="Use asymmetric quantization with per-channel zero-points."
    )
    group_size: int = Field(
        default=16,
        ge=1,
        description="Tokens per quantization group for sharing scale/zero-point metadata.",
    )
    stochastic_rounding: bool = Field(default=True)
    calibration_steps: int = Field(
        default=256,
        ge=1,
        description="Number of running statistics updates collected during calibration.",
    )

    @field_validator("high_precision_guard")
    @classmethod
    def validate_guard_fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = "high_precision_guard must be between 0 and 1."
            raise ValueError(msg)
        return value


class CompressionBudget(BaseModel):
    """Overall budget targets for reporting and validation."""

    target_compression_ratio: PositiveFloat = Field(
        description="Desired overall compression ratio (full cache size / compressed size)."
    )
    min_throughput_gain: PositiveFloat = Field(
        default=1.0,
        description="Minimum acceptable throughput speedup versus baseline.",
    )
    max_accuracy_drop: float = Field(
        default=0.02,
        ge=0.0,
        description="Maximum permissible drop in evaluation metrics (fractional).",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_name: str = Field(description="HF or local model identifier used for the experiment.")
    max_sequence_length: PositiveInt = Field(default=128000)
    selector: SelectorConfig
    quantization: QuantizationConfig
    budget: CompressionBudget
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional baseline metrics (accuracy, throughput) for comparisons.",
    )
