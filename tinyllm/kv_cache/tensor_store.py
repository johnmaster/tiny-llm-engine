"""Per-layer contiguous storage for GPU block-table KV attention."""
from __future__ import annotations

from typing import Any

from .block_manager import PagedKVCacheManager
from .interfaces import BlockId, BlockTable
from ..attention.triton_paged import paged_decode_attention, paged_prefill_attention


class PagedKVTensorStore:
    """Store each layer's keys and values in fixed-size physical blocks.

    Each layer's page pool is allocated lazily and kept contiguous so the Triton
    decode kernel can follow physical page IDs without gathering KV sequences.
    ``gather_layer`` remains for backends that require a conventional dense cache.
    """

    def __init__(self, *, torch: Any, num_blocks: int, block_size: int,
                 num_layers: int, num_kv_heads: int, head_dim: int,
                 dtype: Any, device: Any):
        self.torch = torch
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        # One contiguous page pool per layer is required by the GPU paged
        # attention kernel. The pools are allocated lazily on first write.
        self._storage: list[tuple[Any, Any] | None] = [None] * num_layers
        self.manager = PagedKVCacheManager(
            num_blocks, block_size, copy_block=self._copy_block
        )

    def allocate(self, num_tokens: int) -> BlockTable:
        return self.manager.allocate(num_tokens)

    def append(self, table: BlockTable, num_tokens: int) -> BlockTable:
        return self.manager.append(table, num_tokens)

    def release(self, table: BlockTable) -> None:
        self.manager.release(table)

    @property
    def materialized_blocks(self) -> int:
        """Number of physical pages currently held by active sequences."""
        return self.manager.used_blocks

    def device_page_table(self, tables: list[BlockTable]) -> tuple[Any, Any]:
        """Pack request block tables and lengths into device tensors once per batch."""
        if not tables:
            raise ValueError("at least one block table is required")
        torch = self.torch
        max_pages = max(len(table.block_ids) for table in tables)
        page_table = torch.zeros((len(tables), max_pages), dtype=torch.int32,
                                 device=self.device)
        for row, table in enumerate(tables):
            if table._released:
                raise RuntimeError("cannot use a released KV block table")
            page_table[row, :len(table.block_ids)] = torch.tensor(
                table.block_ids, dtype=torch.int32, device=self.device
            )
        lengths = torch.tensor([table.token_count for table in tables],
                               dtype=torch.int32, device=self.device)
        return page_table, lengths

    def write_layer(self, table: BlockTable, layer_idx: int, start: int,
                    key: Any, value: Any) -> None:
        """Write [tokens, kv_heads, head_dim] K/V tensors at a logical offset."""
        self._validate_layer(layer_idx)
        if start < 0 or start + key.shape[0] > table.token_count:
            raise ValueError("K/V write is outside the table's valid token range")
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must have matching [tokens, heads, head_dim] shapes")
        if key.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError("K/V tensor dimensions do not match this cache's model layout")

        cursor = 0
        while cursor < key.shape[0]:
            token_pos = start + cursor
            logical_block, offset = divmod(token_pos, self.block_size)
            block_id = self.manager.ensure_writable(table, logical_block)
            count = min(key.shape[0] - cursor, self.block_size - offset)
            key_cache, value_cache = self._ensure_layer(layer_idx)
            key_block, value_block = key_cache[block_id], value_cache[block_id]
            key_block[offset:offset + count].copy_(key[cursor:cursor + count])
            value_block[offset:offset + count].copy_(value[cursor:cursor + count])
            cursor += count

    def gather_layer(self, table: BlockTable, layer_idx: int) -> tuple[Any, Any]:
        """Gather one table into HF layout [1, heads, tokens, head_dim]."""
        self._validate_layer(layer_idx)
        if table.token_count == 0:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            empty = self.torch.empty(shape, dtype=self.dtype, device=self.device)
            return empty, empty

        key_parts, value_parts = [], []
        remaining = table.token_count
        for block_id in table.block_ids:
            count = min(remaining, self.block_size)
            if count <= 0:
                break
            key_cache, value_cache = self._ensure_layer(layer_idx)
            key_block, value_block = key_cache[block_id], value_cache[block_id]
            key_parts.append(key_block[:count])
            value_parts.append(value_block[:count])
            remaining -= count
        if remaining:
            raise RuntimeError("block table is shorter than its valid token count")
        key = self.torch.cat(key_parts, dim=0).permute(1, 0, 2).unsqueeze(0).contiguous()
        value = self.torch.cat(value_parts, dim=0).permute(1, 0, 2).unsqueeze(0).contiguous()
        return key, value

    def paged_attention(self, layer_idx: int, query: Any, page_table: Any,
                        lengths: Any, scale: float) -> Any:
        """Run fused Triton attention directly over physical pages and block tables."""
        self._validate_layer(layer_idx)
        key_cache, value_cache = self._ensure_layer(layer_idx)
        return paged_decode_attention(query, key_cache, value_cache,
                                      page_table, lengths, scale)

    def paged_prefill_attention(self, layer_idx: int, query: Any, page_table: Any,
                                old_lengths: Any, query_lengths: Any,
                                scale: float) -> Any:
        """Run causal multi-token attention by following physical page IDs."""
        self._validate_layer(layer_idx)
        key_cache, value_cache = self._ensure_layer(layer_idx)
        return paged_prefill_attention(
            query, key_cache, value_cache, page_table, old_lengths, query_lengths, scale
        )

    def _ensure_layer(self, layer_idx: int) -> tuple[Any, Any]:
        storage = self._storage[layer_idx]
        if storage is None:
            shape = (self.manager.num_blocks, self.block_size,
                     self.num_kv_heads, self.head_dim)
            storage = (
                self.torch.empty(shape, dtype=self.dtype, device=self.device),
                self.torch.empty(shape, dtype=self.dtype, device=self.device),
            )
            self._storage[layer_idx] = storage
        return storage

    def _copy_block(self, source: BlockId, destination: BlockId, valid_tokens: int) -> None:
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = self._ensure_layer(layer_idx)
            source_key, source_value = key_cache[source], value_cache[source]
            destination_key = key_cache[destination]
            destination_value = value_cache[destination]
            destination_key[:valid_tokens].copy_(source_key[:valid_tokens])
            destination_value[:valid_tokens].copy_(source_value[:valid_tokens])

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError("layer_idx is outside the model's KV cache")
