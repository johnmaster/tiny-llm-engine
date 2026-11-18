"""Compare cached-prefix generation against uncached generation on a real model."""
from __future__ import annotations

import argparse
import sys

from tinyllm.config import EngineConfig
from tinyllm.engine import Engine
from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


def run_pair(backend: HuggingFaceBackend, prompts: tuple[str, str], *,
             enable_prefix_caching: bool, max_tokens: int,
             max_model_len: int, max_num_seqs: int,
             max_cached_blocks: int) -> tuple[object, object, Engine]:
    engine = Engine(
        backend,
        config=EngineConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            max_prefix_cache_blocks=max_cached_blocks,
        ),
    )
    try:
        first = engine.generate(prompts[0], max_tokens=max_tokens)
        second = engine.generate(prompts[1], max_tokens=max_tokens)
        return first, second, engine
    except Exception:
        engine.shutdown()
        raise


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify exact-token KV prefix reuse and generation equivalence"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_4",
                        choices=("sdpa", "flash_attention_2", "flash_attention_3",
                                 "flash_attention_4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-cached-blocks", type=int, default=256)
    parser.add_argument(
        "--shared-prefix",
        default=("请详细解释大型语言模型推理中的 KV cache、continuous batching "
                 "和 paged attention，并说明它们如何协同工作。"),
    )
    args = parser.parse_args()

    backend = HuggingFaceBackend(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attention_backend=args.attention_backend,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    prompts = (
        args.shared_prefix + "\n请补充解释 KV block 的生命周期。",
        args.shared_prefix + "\n请补充解释前缀命中如何减少计算。",
    )
    token_prompts = tuple(backend.encode(prompt) for prompt in prompts)
    common_tokens = common_prefix_length(*token_prompts)
    expected_hit_tokens = min(
        (common_tokens // backend.kv_block_size) * backend.kv_block_size,
        ((len(token_prompts[1]) - 1) // backend.kv_block_size) * backend.kv_block_size,
    )
    if expected_hit_tokens < backend.kv_block_size:
        raise RuntimeError(
            "the two tokenized prompts share less than one complete KV block; "
            "provide a longer --shared-prefix"
        )

    baseline_first, baseline_second, baseline_engine = run_pair(
        backend,
        prompts,
        enable_prefix_caching=False,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_cached_blocks=args.max_cached_blocks,
    )
    baseline_engine.shutdown()

    cached_first, cached_second, cached_engine = run_pair(
        backend,
        prompts,
        enable_prefix_caching=True,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_cached_blocks=args.max_cached_blocks,
    )
    try:
        print(f"Common token prefix: {common_tokens} tokens")
        print(f"Expected full-block cache hit: {expected_hit_tokens} tokens")
        print(f"Second request cache hit: {cached_second.prefix_cached_tokens} tokens")
        print("Uncached/cached greedy outputs:",
              baseline_first.output_tokens == cached_first.output_tokens,
              baseline_second.output_tokens == cached_second.output_tokens)
        print("Prefix cache stats:", cached_engine.scheduler.stats()["prefix_cache"])

        if cached_second.prefix_cached_tokens < expected_hit_tokens:
            raise RuntimeError("the second prompt did not hit the expected full KV blocks")
        if baseline_first.output_tokens != cached_first.output_tokens:
            raise RuntimeError("cached first-prompt output differs from uncached output")
        if baseline_second.output_tokens != cached_second.output_tokens:
            raise RuntimeError("cached second-prompt output differs from uncached output")

        prefix_cache = cached_engine.scheduler.prefix_cache
        if prefix_cache is not None:
            prefix_cache.clear()
        stats = backend.kv_cache.manager.snapshot()
        print("KV blocks after clearing the prefix cache:", stats)
        if stats["used_blocks"] != 0:
            raise RuntimeError("KV blocks remained allocated after clearing the prefix cache")
        print("Prefix cache correctness: PASS")
        return 0
    finally:
        cached_engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
