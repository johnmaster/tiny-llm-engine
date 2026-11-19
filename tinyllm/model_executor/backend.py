"""Interface between scheduling and model execution."""
from __future__ import annotations
from typing import Protocol, Sequence

from ..kv_cache import PrefixMatch


class ModelBackend(Protocol):
    """Execute batched decode and chunked-prefill operations selected by scheduler."""
    eos_token_id: int | None

    def encode(self, prompt: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def prefill(self, prompts: Sequence[Sequence[int]], *,
                prefix_matches: Sequence[PrefixMatch] | None = None) -> list[tuple[int, object]]: ...
    def prefill_chunk(self, token_chunks: Sequence[Sequence[int]],
                      states: Sequence[object | None],
                      prefix_matches: Sequence[PrefixMatch | None],
                      final_chunks: Sequence[bool]) -> list[tuple[int | None, object]]: ...
    def decode_step(self, tokens: Sequence[int], states: Sequence[object]) -> list[tuple[int, object, bool]]: ...
