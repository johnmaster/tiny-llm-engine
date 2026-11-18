"""Bounded exact-token prefix cache over immutable, complete KV blocks."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from struct import pack
from typing import Sequence

from .block_manager import PagedKVCacheManager
from .interfaces import BlockId, BlockTable


@dataclass
class PrefixMatch:
    matched_tokens: int
    block_table: BlockTable | None


class ExactPrefixCache:
    """Cache full-block KV prefixes, keyed by cumulative token-prefix hashes.

    The cache owns one reference to each cached block. A match creates another
    owned block table for the request. Partial blocks are never cached, so a
    request can only append into private storage.
    """

    def __init__(self, manager: PagedKVCacheManager, max_cached_blocks: int = 256):
        if max_cached_blocks < 1:
            raise ValueError("max_cached_blocks must be at least 1")
        self.manager = manager
        self.block_size = manager.block_size
        self.max_cached_blocks = max_cached_blocks
        self._entries: OrderedDict[bytes, BlockId] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def match(self, token_ids: Sequence[int]) -> PrefixMatch:
        # Leave at least one prompt token for the model forward that produces
        # next-token logits; the cache currently stores KV but not logits.
        block_count = min(len(token_ids) // self.block_size,
                          max(0, (len(token_ids) - 1) // self.block_size))
        digest = b"tinyllm-prefix-v1"
        block_ids: list[BlockId] = []
        for block_index in range(block_count):
            start = block_index * self.block_size
            chunk = token_ids[start:start + self.block_size]
            digest = _extend_digest(digest, chunk)
            block_id = self._entries.get(digest)
            if block_id is None:
                break
            self._entries.move_to_end(digest)
            block_ids.append(block_id)

        matched_tokens = len(block_ids) * self.block_size
        if block_ids:
            table = self.manager.table_from_blocks(block_ids, matched_tokens)
            self.hits += 1
            self.tokens_saved += matched_tokens
            return PrefixMatch(matched_tokens, table)
        self.misses += 1
        return PrefixMatch(0, None)

    def insert(self, token_ids: Sequence[int], block_table: BlockTable) -> None:
        full_blocks = min(len(token_ids) // self.block_size,
                          len(block_table.block_ids), self.max_cached_blocks)
        digest = b"tinyllm-prefix-v1"
        for block_index in range(full_blocks):
            start = block_index * self.block_size
            digest = _extend_digest(
                digest, token_ids[start:start + self.block_size]
            )
            existing = self._entries.get(digest)
            if existing is not None:
                self._entries.move_to_end(digest)
                continue

            block_id = block_table.block_ids[block_index]
            self.manager.retain_block(block_id)
            self._entries[digest] = block_id
            while len(self._entries) > self.max_cached_blocks:
                _, evicted_block = self._entries.popitem(last=False)
                self.manager.release_block(evicted_block)

    def clear(self) -> None:
        while self._entries:
            _, block_id = self._entries.popitem(last=False)
            self.manager.release_block(block_id)

    def snapshot(self) -> dict[str, int]:
        return {
            "cached_blocks": len(self._entries),
            "max_cached_blocks": self.max_cached_blocks,
            "hits": self.hits,
            "misses": self.misses,
            "tokens_saved": self.tokens_saved,
        }


def _extend_digest(parent: bytes, token_ids: Sequence[int]) -> bytes:
    encoded = bytearray()
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id < 0 or token_id >= 2**32:
            raise ValueError(f"token id is outside uint32 range: {token_id}")
        encoded.extend(pack("<I", token_id))
    return blake2b(parent + encoded, digest_size=24).digest()
