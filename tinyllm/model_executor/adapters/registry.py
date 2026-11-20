"""Architecture registry for model-specific inference implementations."""
from __future__ import annotations

from typing import Any

from .base import ModelAdapter
from .qwen2 import Qwen2Adapter
from .qwen3 import Qwen3Adapter

_ADAPTERS: list[type[ModelAdapter]] = [Qwen2Adapter, Qwen3Adapter]


def register_model_adapter(adapter_type: type[ModelAdapter]) -> None:
    """Register an architecture implementation for future model loading."""
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, ModelAdapter):
        raise TypeError("adapter_type must be a ModelAdapter subclass")
    if adapter_type not in _ADAPTERS:
        _ADAPTERS.append(adapter_type)


def resolve_model_adapter(config: Any) -> ModelAdapter | None:
    """Resolve by HF architecture name first, then by model_type as fallback."""
    architectures = tuple(getattr(config, "architectures", None) or ())
    model_type = getattr(config, "model_type", None)
    for adapter_type in _ADAPTERS:
        if any(name in adapter_type.architectures for name in architectures):
            return adapter_type()
    for adapter_type in _ADAPTERS:
        if model_type in adapter_type.model_types:
            return adapter_type()
    return None


def registered_model_adapters() -> tuple[str, ...]:
    """Return architecture adapter names in registry order."""
    return tuple(adapter_type.name for adapter_type in _ADAPTERS)


__all__ = [
    "ModelAdapter",
    "register_model_adapter",
    "registered_model_adapters",
    "resolve_model_adapter",
]
