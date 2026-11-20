"""Reference-counted fixed-size KV block allocator.

This module owns block IDs and request block tables. The model executor owns
the actual per-layer K/V tensors and supplies a callback when a shared block
must be copied before writing (copy-on-write).
"""
from __future__ import annotations

from collections import deque
from threading import RLock
from .interfaces import BlockId, BlockTable, CopyBlock


class KVCacheOutOfBlocks(RuntimeError):
    """Raised when the configured physical block pool is exhausted."""


class PagedKVCacheManager:
    """Manage fixed-size physical pages and per-sequence block tables.

    Args:
        num_blocks: Total physical blocks available to this manager.
        block_size: Number of token positions in each physical block.

    ``append(table, n)`` appends ``n`` logical tokens and reserves enough
    physical blocks. Before modifying a shared partial block, it uses the
    configured copy callback to detach that block. The callback receives
    ``(source_block, destination_block, valid_tokens_in_source)`` and must copy
    the K/V data for every model layer.
    """

    def __init__(self, num_blocks: int, block_size: int, *, copy_block: CopyBlock | None = None):
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._copy_block = copy_block
        self._owner = object()
        # Keep never-used pages separate so recently released pages are reused
        # first. This lets the extensible KV backing track peak live demand
        # instead of growing through the entire pool across short requests.
        self._free: deque[BlockId] = deque(range(num_blocks))
        self._recycled: list[BlockId] = []
        self._ref_counts = [0] * num_blocks
        self._lock = RLock()

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free) + len(self._recycled)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    def ref_count(self, block_id: BlockId) -> int:
        self._validate_block_id(block_id)
        with self._lock:
            return self._ref_counts[block_id]

    def allocate(self, num_tokens: int) -> BlockTable:
        """Allocate enough blocks for a sequence with ``num_tokens`` tokens."""
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        count = self._blocks_for(num_tokens)
        with self._lock:
            ids = self._take_blocks(count)
            return BlockTable(ids, num_tokens, self._owner)

    def append(self, table: BlockTable, num_tokens: int) -> BlockTable:
        """Append a token count to a table and reserve capacity for the result."""
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        with self._lock:
            self._validate_table(table)
            if num_tokens == 0:
                return table

            old_count = table.token_count
            new_count = old_count + num_tokens
            required = self._blocks_for(new_count)
            extra = required - len(table.block_ids)
            reserved = self._take_blocks(extra) if extra > 0 else []
            # Appending writes into the current last page when it is partial.
            # Detach it first if it is shared, leaving the fork's prefix intact.
            try:
                if old_count and old_count % self.block_size:
                    logical_block = (old_count - 1) // self.block_size
                    block_id = table.block_ids[logical_block]
                    if self._ref_counts[block_id] > 1:
                        self._copy_on_write(table, logical_block, old_count % self.block_size)
            except Exception:
                self._return_unpublished(reserved)
                raise
            table.block_ids.extend(reserved)
            table.token_count = new_count
            return table

    def fork(self, table: BlockTable) -> BlockTable:
        """Create another sequence table sharing all existing physical blocks."""
        with self._lock:
            self._validate_table(table)
            for block_id in table.block_ids:
                self._ref_counts[block_id] += 1
            return BlockTable(list(table.block_ids), table.token_count, self._owner)

    def table_from_blocks(self, block_ids: list[BlockId], token_count: int) -> BlockTable:
        """Create an owned table referencing existing blocks."""
        if token_count < 0 or len(block_ids) != self._blocks_for(token_count):
            raise ValueError("block count does not match token_count")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("a physical block cannot appear twice in one block table")
        with self._lock:
            for block_id in block_ids:
                self._validate_block_id(block_id)
                if self._ref_counts[block_id] <= 0:
                    raise ValueError(f"block {block_id} is not allocated")
            for block_id in block_ids:
                self._ref_counts[block_id] += 1
            return BlockTable(list(block_ids), token_count, self._owner)

    def retain_block(self, block_id: BlockId) -> None:
        """Add one owner for an allocated physical block."""
        self._validate_block_id(block_id)
        with self._lock:
            if self._ref_counts[block_id] <= 0:
                raise ValueError(f"block {block_id} is not allocated")
            self._ref_counts[block_id] += 1

    def release_block(self, block_id: BlockId) -> None:
        """Drop one owner for a physical block without releasing a whole table."""
        self._validate_block_id(block_id)
        with self._lock:
            if self._ref_counts[block_id] <= 0:
                raise ValueError(f"block {block_id} is not allocated")
            self._ref_counts[block_id] -= 1
            if self._ref_counts[block_id] == 0:
                self._recycled.append(block_id)

    def ensure_writable(self, table: BlockTable, logical_block: int,
                        copy_block: CopyBlock | None = None) -> BlockId:
        """Return an unshared block for writing, copying shared contents as needed."""
        with self._lock:
            self._validate_table(table)
            if logical_block < 0 or logical_block >= len(table.block_ids):
                raise IndexError("logical_block is outside this block table")
            valid_tokens = min(
                self.block_size,
                max(0, table.token_count - logical_block * self.block_size),
            )
            return self._copy_on_write(table, logical_block, valid_tokens, copy_block)

    def release(self, table: BlockTable) -> None:
        """Release a table and return blocks with no remaining references."""
        with self._lock:
            self._validate_table(table)
            for block_id in table.block_ids:
                self._ref_counts[block_id] -= 1
                if self._ref_counts[block_id] == 0:
                    self._recycled.append(block_id)
            table._released = True

    def block_for_token(self, table: BlockTable, token_index: int) -> tuple[BlockId, int]:
        """Map a logical token index to ``(physical_block_id, offset_in_block)``."""
        with self._lock:
            self._validate_table(table)
            if token_index < 0 or token_index >= table.token_count:
                raise IndexError("token_index is outside the valid token range")
            logical_block, offset = divmod(token_index, self.block_size)
            return table.block_ids[logical_block], offset

    def snapshot(self) -> dict[str, int]:
        """Return a small allocator status snapshot suitable for scheduler stats."""
        with self._lock:
            return {
                "total_blocks": self.num_blocks,
                "free_blocks": len(self._free) + len(self._recycled),
                "used_blocks": self.num_blocks - len(self._free) - len(self._recycled),
                "block_size": self.block_size,
            }

    def _copy_on_write(self, table: BlockTable, logical_block: int, valid_tokens: int,
                       copy_block: CopyBlock | None = None) -> BlockId:
        source = table.block_ids[logical_block]
        if self._ref_counts[source] == 1:
            return source
        copier = copy_block or self._copy_block
        if copier is None:
            raise RuntimeError(
                "copy-on-write needs a copy_block callback to copy K/V data for every layer"
            )
        destination = self._take_blocks(1)[0]
        try:
            copier(source, destination, valid_tokens)
        except Exception:
            self._ref_counts[destination] = 0
            self._recycled.append(destination)
            raise
        table.block_ids[logical_block] = destination
        self._ref_counts[source] -= 1
        return destination

    def _take_blocks(self, count: int) -> list[BlockId]:
        free_count = len(self._free) + len(self._recycled)
        if count > free_count:
            raise KVCacheOutOfBlocks(
                f"KV block pool exhausted: requested {count} blocks, "
                f"only {free_count} free ({self.num_blocks} total)"
            )
        ids = []
        for _ in range(count):
            ids.append(self._recycled.pop() if self._recycled else self._free.popleft())
        for block_id in ids:
            self._ref_counts[block_id] = 1
        return ids

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _validate_block_id(self, block_id: BlockId) -> None:
        if not isinstance(block_id, int) or block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"invalid block id: {block_id!r}")

    def _validate_table(self, table: BlockTable) -> None:
        if not isinstance(table, BlockTable) or table._owner is not self._owner:
            raise ValueError("block table does not belong to this KV cache manager")
        if table._released:
            raise ValueError("block table has already been released")
        expected = self._blocks_for(table.token_count)
        if len(table.block_ids) != expected:
            raise ValueError("block table capacity does not match its token count")
        if len(set(table.block_ids)) != len(table.block_ids):
            raise ValueError("a physical block cannot appear twice in one block table")
        for block_id in table.block_ids:
            self._validate_block_id(block_id)
            if self._ref_counts[block_id] <= 0:
                raise ValueError(f"block {block_id} is not allocated")

    def _return_unpublished(self, block_ids: list[BlockId]) -> None:
        for block_id in block_ids:
            self._ref_counts[block_id] = 0
            self._recycled.append(block_id)
