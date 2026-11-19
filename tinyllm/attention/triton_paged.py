"""Triton single-token attention over a paged KV cache.

Each program computes one (request, query-head) output and follows that
request's block table while loading keys and values. KV pages are never
gathered into a dense sequence tensor.
"""
from __future__ import annotations

from typing import Any

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-GPU installs
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _paged_decode_kernel(
        q_ptr, k_ptr, v_ptr, table_ptr, length_ptr, out_ptr,
        q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
        k_stride_p: tl.constexpr, k_stride_t: tl.constexpr,
        k_stride_h: tl.constexpr, v_stride_p: tl.constexpr,
        v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
        table_stride_b: tl.constexpr, table_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr, out_stride_h: tl.constexpr,
        batch_size: tl.constexpr, query_heads: tl.constexpr,
        kv_heads: tl.constexpr, head_dim: tl.constexpr,
        page_size: tl.constexpr, scale: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        kv_head = query_head // (query_heads // kv_heads)
        d = tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + request * q_stride_b + query_head * q_stride_h + d,
                    d < head_dim, other=0).to(tl.float32)
        seq_len = tl.load(length_ptr + request)
        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        running_out = tl.full((BLOCK_D,), 0.0, tl.float32)
        t = tl.arange(0, BLOCK_T)

        for start in range(0, seq_len, BLOCK_T):
            positions = start + t
            valid = positions < seq_len
            logical_page = positions // page_size
            physical_page = tl.load(
                table_ptr + request * table_stride_b + logical_page * table_stride_p,
                valid, other=0,
            )
            page_offset = positions % page_size
            key_offsets = (
                physical_page[:, None] * k_stride_p
                + page_offset[:, None] * k_stride_t
                + kv_head * k_stride_h
                + d[None, :]
            )
            key = tl.load(k_ptr + key_offsets,
                          valid[:, None] & (d[None, :] < head_dim), other=0).to(tl.float32)
            scores = tl.sum(key * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            page_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, page_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max)
            value_offsets = (
                physical_page[:, None] * v_stride_p
                + page_offset[:, None] * v_stride_t
                + kv_head * v_stride_h
                + d[None, :]
            )
            value = tl.load(v_ptr + value_offsets,
                            valid[:, None] & (d[None, :] < head_dim), other=0).to(tl.float32)
            running_out = (running_out * old_scale
                           + tl.sum(weights[:, None] * value, axis=0))
            running_sum = running_sum * old_scale + tl.sum(weights, axis=0)
            running_max = new_max

        output = running_out / running_sum
        tl.store(out_ptr + request * out_stride_b + query_head * out_stride_h + d,
                 output, d < head_dim)


    @triton.jit
    def _paged_prefill_kernel(
        q_ptr, k_ptr, v_ptr, table_ptr, old_length_ptr, query_length_ptr, out_ptr,
        q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
        q_stride_q: tl.constexpr,
        k_stride_p: tl.constexpr, k_stride_t: tl.constexpr,
        k_stride_h: tl.constexpr, v_stride_p: tl.constexpr,
        v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
        table_stride_b: tl.constexpr, table_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr, out_stride_h: tl.constexpr,
        out_stride_q: tl.constexpr,
        query_heads: tl.constexpr, kv_heads: tl.constexpr,
        head_dim: tl.constexpr, page_size: tl.constexpr, scale: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        query_pos = tl.program_id(2)
        query_length = tl.load(query_length_ptr + request)
        output_ptr = (out_ptr + request * out_stride_b
                      + query_head * out_stride_h + query_pos * out_stride_q)
        if query_pos >= query_length:
            d = tl.arange(0, BLOCK_D)
            tl.store(output_ptr + d, 0.0, d < head_dim)
        else:
            kv_head = query_head // (query_heads // kv_heads)
            d = tl.arange(0, BLOCK_D)
            q = tl.load(
                q_ptr + request * q_stride_b + query_head * q_stride_h
                + query_pos * q_stride_q + d,
                d < head_dim, other=0,
            ).to(tl.float32)
            old_length = tl.load(old_length_ptr + request)
            kv_length = old_length + query_pos + 1
            running_max = tl.full((), -float("inf"), tl.float32)
            running_sum = tl.full((), 0.0, tl.float32)
            running_out = tl.full((BLOCK_D,), 0.0, tl.float32)
            t = tl.arange(0, BLOCK_T)

            for start in range(0, kv_length, BLOCK_T):
                positions = start + t
                valid = positions < kv_length
                logical_page = positions // page_size
                physical_page = tl.load(
                    table_ptr + request * table_stride_b + logical_page * table_stride_p,
                    valid, other=0,
                )
                page_offset = positions % page_size
                key_offsets = (
                    physical_page[:, None] * k_stride_p
                    + page_offset[:, None] * k_stride_t
                    + kv_head * k_stride_h
                    + d[None, :]
                )
                key = tl.load(
                    k_ptr + key_offsets,
                    valid[:, None] & (d[None, :] < head_dim), other=0,
                ).to(tl.float32)
                scores = tl.sum(key * q[None, :], axis=1) * scale
                scores = tl.where(valid, scores, -float("inf"))
                page_max = tl.max(scores, axis=0)
                new_max = tl.maximum(running_max, page_max)
                old_scale = tl.exp(running_max - new_max)
                weights = tl.exp(scores - new_max)
                value_offsets = (
                    physical_page[:, None] * v_stride_p
                    + page_offset[:, None] * v_stride_t
                    + kv_head * v_stride_h
                    + d[None, :]
                )
                value = tl.load(
                    v_ptr + value_offsets,
                    valid[:, None] & (d[None, :] < head_dim), other=0,
                ).to(tl.float32)
                running_out = (running_out * old_scale
                               + tl.sum(weights[:, None] * value, axis=0))
                running_sum = running_sum * old_scale + tl.sum(weights, axis=0)
                running_max = new_max

            result = running_out / running_sum
            tl.store(output_ptr + d, result, d < head_dim)


def paged_decode_attention(query: Any, key_cache: Any, value_cache: Any,
                           page_table: Any, lengths: Any, scale: float) -> Any:
    """Compute batched one-token attention using a device-side block table.

    Args:
        query: ``[batch, query_heads, head_dim]``.
        key_cache/value_cache: ``[pages, page_size, kv_heads, head_dim]``.
        page_table: padded ``[batch, max_pages]`` tensor of physical page IDs.
        lengths: ``[batch]`` tensor of valid KV token counts.
    """
    if triton is None:
        raise RuntimeError(
            "Triton is required for paged decode; install a CUDA PyTorch build "
            "that includes Triton (the RTX 5090 environment uses PyTorch 2.11)."
        )
    if query.device.type != "cuda":
        raise RuntimeError("Triton paged decode requires CUDA tensors")
    batch_size, query_heads, head_dim = query.shape
    pages, page_size, kv_heads, cache_head_dim = key_cache.shape
    if cache_head_dim != head_dim or value_cache.shape != key_cache.shape:
        raise ValueError("query and paged K/V dimensions do not match")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    output = query.new_empty(query.shape)
    _paged_decode_kernel[(batch_size, query_heads)](
        query, key_cache, value_cache, page_table, lengths, output,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        page_table.stride(0), page_table.stride(1),
        output.stride(0), output.stride(1),
        batch_size, query_heads, kv_heads, head_dim, page_size, scale,
        32, triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def paged_prefill_attention(query: Any, key_cache: Any, value_cache: Any,
                             page_table: Any, old_lengths: Any,
                             query_lengths: Any, scale: float) -> Any:
    """Compute causal multi-token attention directly from paged KV storage.

    Args:
        query: ``[batch, query_heads, query_width, head_dim]``.
        old_lengths: cached prefix length for each row before this chunk.
        query_lengths: valid query tokens in this row; padding queries are zeroed.
    """
    if triton is None:
        raise RuntimeError("Triton is required for paged prefill attention")
    if query.device.type != "cuda":
        raise RuntimeError("Triton paged prefill requires CUDA tensors")
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, query, head_dim]")
    batch_size, query_heads, query_width, head_dim = query.shape
    _, page_size, kv_heads, cache_head_dim = key_cache.shape
    if cache_head_dim != head_dim or value_cache.shape != key_cache.shape:
        raise ValueError("query and paged K/V dimensions do not match")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if old_lengths.shape != (batch_size,) or query_lengths.shape != (batch_size,):
        raise ValueError("old_lengths and query_lengths must have shape [batch]")
    output = query.new_empty(query.shape)
    _paged_prefill_kernel[(batch_size, query_heads, query_width)](
        query, key_cache, value_cache, page_table, old_lengths, query_lengths, output,
        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        page_table.stride(0), page_table.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        query_heads, kv_heads, head_dim, page_size, scale,
        32, triton.next_power_of_2(head_dim), num_warps=4,
    )
    return output
