"""Contracts for paged KV storage and prefix-cache implementations.

The current Hugging Face backend keeps per-request KV tensors itself. A block
manager will replace that ownership when prefix caching and paged attention are
implemented.
"""
from __future__ import annotations
from typing import Protocol, Sequence

BlockId = int


class KVCacheManager(Protocol):
    def allocate(self, num_tokens: int) -> list[BlockId]: ...
    def append(self, block_table: list[BlockId], token_count: int) -> list[BlockId]: ...
    def fork(self, block_table: Sequence[BlockId]) -> list[BlockId]: ...
    def release(self, block_table: Sequence[BlockId]) -> None: ...


class PrefixCache(Protocol):
    def match_prefix(self, token_ids: Sequence[int]) -> tuple[int, Sequence[BlockId]]: ...
    def insert(self, token_ids: Sequence[int], block_table: Sequence[BlockId]) -> None: ...
