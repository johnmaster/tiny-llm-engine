"""Contracts for paged KV storage and prefix-cache implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

BlockId = int
CopyBlock = Callable[[BlockId, BlockId, int], None]


@dataclass
class BlockTable:
    """Logical-to-physical KV mapping owned by one request.

    ``token_count`` is the number of valid tokens represented by this table.
    ``block_ids[i]`` stores logical tokens ``i * block_size`` onward.
    Tables are mutable only through their manager.
    """

    block_ids: list[BlockId]
    token_count: int
    _owner: object = field(repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


class KVCacheManager(Protocol):
    def allocate(self, num_tokens: int) -> BlockTable: ...
    def append(self, table: BlockTable, num_tokens: int) -> BlockTable: ...
    def fork(self, table: BlockTable) -> BlockTable: ...
    def ensure_writable(self, table: BlockTable, logical_block: int,
                        copy_block: CopyBlock | None = None) -> BlockId: ...
    def release(self, table: BlockTable) -> None: ...
    def table_from_blocks(self, block_ids: list[BlockId], token_count: int) -> BlockTable: ...
    def retain_block(self, block_id: BlockId) -> None: ...
    def release_block(self, block_id: BlockId) -> None: ...


class PrefixCache(Protocol):
    def match(self, token_ids: Sequence[int]) -> object: ...
    def insert(self, token_ids: Sequence[int], block_table: BlockTable) -> None: ...
