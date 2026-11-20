"""Model architecture implementations for the inference backend."""
from .base import ModelAdapter
from .registry import (
    register_model_adapter,
    registered_model_adapters,
    resolve_model_adapter,
)

__all__ = [
    "ModelAdapter",
    "register_model_adapter",
    "registered_model_adapters",
    "resolve_model_adapter",
]
