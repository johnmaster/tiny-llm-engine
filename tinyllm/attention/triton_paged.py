"""Triton decode and causal prefill attention over a paged KV cache.

The kernels follow each request's block table directly; KV pages are never
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
    def _paged_prefill_tiled_kernel(
        q_ptr, k_ptr, v_ptr, table_ptr, old_length_ptr, query_length_ptr, out_ptr,
        q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
        q_stride_q: tl.constexpr,
        k_stride_p: tl.constexpr, k_stride_t: tl.constexpr,
        k_stride_h: tl.constexpr, v_stride_p: tl.constexpr,
        v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
        table_stride_b: tl.constexpr, table_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr, out_stride_h: tl.constexpr,
        out_stride_q: tl.constexpr,
        query_width: tl.constexpr,
        query_heads: tl.constexpr, kv_heads: tl.constexpr,
        head_dim: tl.constexpr, page_size: tl.constexpr, scale: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        query_tile = tl.program_id(2)
        query_start = query_tile * BLOCK_M
        query_length = tl.load(query_length_ptr + request)
        query_positions = query_start + tl.arange(0, BLOCK_M)
        d = tl.arange(0, BLOCK_D)
        output_ptr = (
            out_ptr + request * out_stride_b + query_head * out_stride_h
            + query_positions[:, None] * out_stride_q + d[None, :]
        )

        if query_start >= query_length:
            tl.store(
                output_ptr, 0.0,
                (query_positions[:, None] < query_width)
                & (d[None, :] < head_dim),
            )
        else:
            kv_head = query_head // (query_heads // kv_heads)
            old_length = tl.load(old_length_ptr + request)
            query_end = tl.minimum(query_length, query_start + BLOCK_M)
            kv_length = old_length + query_end
            query_valid = query_positions < query_length
            q = tl.load(
                q_ptr + request * q_stride_b + query_head * q_stride_h
                + query_positions[:, None] * q_stride_q + d[None, :],
                query_valid[:, None] & (d[None, :] < head_dim), other=0,
            )

            running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
            running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
            running_out = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
            key_positions_in_query = old_length + query_positions
            offsets = tl.arange(0, BLOCK_N)

            for start in range(0, kv_length, BLOCK_N):
                positions = start + offsets
                valid_key = positions < kv_length
                logical_pages = positions // page_size
                physical_pages = tl.load(
                    table_ptr + request * table_stride_b
                    + logical_pages * table_stride_p,
                    valid_key, other=0,
                )
                page_offsets = positions % page_size
                key = tl.load(
                    k_ptr
                    + physical_pages[:, None] * k_stride_p
                    + page_offsets[:, None] * k_stride_t
                    + kv_head * k_stride_h
                    + d[None, :],
                    valid_key[:, None] & (d[None, :] < head_dim), other=0,
                )
                value = tl.load(
                    v_ptr
                    + physical_pages[:, None] * v_stride_p
                    + page_offsets[:, None] * v_stride_t
                    + kv_head * v_stride_h
                    + d[None, :],
                    valid_key[:, None] & (d[None, :] < head_dim), other=0,
                )

                scores = tl.dot(q, tl.trans(key)).to(tl.float32) * scale
                causal = (
                    query_valid[:, None]
                    & valid_key[None, :]
                    & (positions[None, :] <= key_positions_in_query[:, None])
                )
                scores = tl.where(causal, scores, -float("inf"))
                tile_max = tl.max(scores, axis=1)
                new_max = tl.maximum(running_max, tile_max)
                alpha = tl.where(
                    running_max == -float("inf"), 0.0,
                    tl.exp(running_max - new_max),
                )
                probabilities = tl.where(
                    causal, tl.exp(scores - new_max[:, None]), 0.0
                )
                running_out = (
                    running_out * alpha[:, None]
                    + tl.dot(probabilities.to(value.dtype), value)
                )
                running_sum = (
                    running_sum * alpha + tl.sum(probabilities, axis=1)
                )
                running_max = new_max

            result = running_out / tl.maximum(running_sum[:, None], 1.0e-20)
            result = tl.where(query_valid[:, None], result, 0.0)
            tl.store(
                output_ptr, result,
                (query_positions[:, None] < query_width)
                & (d[None, :] < head_dim),
            )


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


def _validate_paged_prefill_inputs(query: Any, key_cache: Any, value_cache: Any,
                                   old_lengths: Any, query_lengths: Any
                                   ) -> tuple[int, int, int, int, int, int]:
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
    return batch_size, query_heads, query_width, head_dim, page_size, kv_heads


def paged_prefill_attention(
    query: Any, key_cache: Any, value_cache: Any, page_table: Any,
    old_lengths: Any, query_lengths: Any, scale: float,
    *, block_m: int = 16, block_n: int = 64,
) -> Any:
    """Tiled causal attention directly over the paged KV cache.

    Each Triton program handles a tile of query positions and reuses its loaded
    K/V tiles across those queries. ``page_table`` is read directly; KV pages
    are never gathered into a dense sequence tensor.
    """
    batch_size, query_heads, query_width, head_dim, page_size, kv_heads = (
        _validate_paged_prefill_inputs(
            query, key_cache, value_cache, old_lengths, query_lengths
        )
    )
    if block_m not in (16, 32) or block_n not in (64, 128):
        raise ValueError("paged prefill tiles must use block_m in {16, 32} and block_n in {64, 128}")
    output = query.new_empty(query.shape)
    _paged_prefill_tiled_kernel[
        (batch_size, query_heads, triton.cdiv(query_width, block_m))
    ](
        query, key_cache, value_cache, page_table, old_lengths, query_lengths, output,
        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        page_table.stride(0), page_table.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        query_width, query_heads, kv_heads, head_dim, page_size, scale,
        block_m, block_n, triton.next_power_of_2(head_dim),
        num_warps=4, num_stages=2,
    )
    return output
