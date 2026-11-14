"""Interface for custom attention kernels.

The Transformers backend currently delegates attention to the selected model's
implementation. FlashAttention and paged attention will plug into model
execution through this interface once the engine owns its KV block layout.
"""
from __future__ import annotations
from typing import Any, Protocol


class AttentionBackend(Protocol):
    def prefill(self, query: Any, key: Any, value: Any, **metadata: Any) -> Any: ...
    def decode(self, query: Any, key: Any, value: Any, **metadata: Any) -> Any: ...
