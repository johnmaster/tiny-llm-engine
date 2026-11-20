"""Interfaces between model architectures and shared inference components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """Model-family implementation used by the shared paged attention path."""

    architectures: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    name: str

    @abstractmethod
    def find_attention_modules(self, model: Any) -> list[tuple[str, Any]]:
        """Find the attention layers that use this adapter's forward contract."""

    @abstractmethod
    def project_qkv(self, backend: Any, module: Any, hidden_states: Any,
                    kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Project and position-encode Q/K/V as ``[batch, heads, seq, dim]``."""

    @abstractmethod
    def layer_index(self, module: Any) -> int:
        """Return the model layer index used to address the KV cache."""

    @abstractmethod
    def attention_scale(self, module: Any) -> float:
        """Return the scale applied to QK scores."""

    @abstractmethod
    def project_output(self, module: Any, attention_output: Any) -> Any:
        """Apply the model family's output projection."""

    @abstractmethod
    def format_attention_output(self, module: Any, attention_output: Any) -> Any:
        """Return the value expected by the Transformers decoder layer."""
