"""Per-layer tensor storage for the reference paged KV cache."""
from __future__ import annotations

from typing import Any

from .block_manager import PagedKVCacheManager
from .interfaces import BlockId, BlockTable


class PagedKVTensorStore:
    """Store each layer's keys and values in fixed-size physical blocks.

    Storage is allocated lazily per block. ``gather_layer`` currently rebuilds
    dense tensors for Hugging Face attention; a paged attention kernel can later
    consume ``BlockTable.block_ids`` directly and remove that gather.
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
        self._storage: list[Any | None] = [None] * num_blocks
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
        """Number of block tensors allocated on the device so far."""
        return sum(block is not None for block in self._storage)

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
            block = self._ensure_block(block_id)
            key_block, value_block = block[layer_idx, 0], block[layer_idx, 1]
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
            block = self._storage[block_id]
            if block is None:
                raise RuntimeError(f"KV block {block_id} layer {layer_idx} has not been written")
            key_block, value_block = block[layer_idx, 0], block[layer_idx, 1]
            key_parts.append(key_block[:count])
            value_parts.append(value_block[:count])
            remaining -= count
        if remaining:
            raise RuntimeError("block table is shorter than its valid token count")
        key = self.torch.cat(key_parts, dim=0).permute(1, 0, 2).unsqueeze(0).contiguous()
        value = self.torch.cat(value_parts, dim=0).permute(1, 0, 2).unsqueeze(0).contiguous()
        return key, value

    def _ensure_block(self, block_id: BlockId) -> Any:
        block = self._storage[block_id]
        if block is None:
            shape = (self.num_layers, 2, self.block_size, self.num_kv_heads, self.head_dim)
            block = self.torch.empty(shape, dtype=self.dtype, device=self.device)
            self._storage[block_id] = block
        return block

    def _copy_block(self, source: BlockId, destination: BlockId, valid_tokens: int) -> None:
        for layer_idx in range(self.num_layers):
            source_block = self._storage[source]
            if source_block is None:
                raise RuntimeError(f"source KV block {source} layer {layer_idx} is uninitialized")
            destination_block = self._ensure_block(destination)
            source_key, source_value = source_block[layer_idx, 0], source_block[layer_idx, 1]
            destination_key = destination_block[layer_idx, 0]
            destination_value = destination_block[layer_idx, 1]
            destination_key[:valid_tokens].copy_(source_key[:valid_tokens])
            destination_value[:valid_tokens].copy_(source_value[:valid_tokens])

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError("layer_idx is outside the model's KV cache")
