"""Quantization options reserved for model-loading and kernel backends."""
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationConfig:
    method: str  # e.g. awq, gptq, bitsandbytes; backend support is future work
    bits: int = 4
    group_size: int = 128

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
