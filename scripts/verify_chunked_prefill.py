"""Check chunked-prefill equivalence and decode progress during a long prompt."""
from __future__ import annotations

import argparse
import sys
import time

from tinyllm.config import EngineConfig
from tinyllm.engine import Engine
from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


def wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {description}")


def make_engine(backend: HuggingFaceBackend, *, max_model_len: int,
                max_num_seqs: int, token_budget: int,
                chunk_size: int) -> Engine:
    return Engine(
        backend,
        config=EngineConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=token_budget,
            prefill_chunk_size=chunk_size,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare full and chunked prefill while a short request joins"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_4",
                        choices=("sdpa", "flash_attention_2", "flash_attention_3",
                                 "flash_attention_4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=32,
                        help="repeat the long-prompt paragraph this many times")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")

    backend = HuggingFaceBackend(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attention_backend=args.attention_backend,
        max_num_seqs=2,
        max_model_len=args.max_model_len,
    )
    long_prompt = (
        "请逐步说明大语言模型推理时，输入 token 如何经过 Transformer 层并写入 KV cache。"
        "请同时解释连续批处理、分页 KV block 和注意力计算之间的关系。\n"
    ) * args.repetitions
    short_prompt = "用一句话解释 KV cache。"
    long_length = len(backend.encode(long_prompt))
    short_length = len(backend.encode(short_prompt))
    if long_length + args.max_tokens > args.max_model_len:
        parser.error(
            f"long prompt has {long_length} tokens; increase --max-model-len "
            "or reduce --repetitions"
        )
    if long_length <= 2 * args.chunk_size:
        parser.error("long prompt must span at least three prefill chunks")
    if short_length + args.max_tokens > args.max_model_len:
        parser.error("short prompt plus output exceeds --max-model-len")

    full_engine = make_engine(
        backend,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        token_budget=args.max_model_len,
        chunk_size=args.max_model_len,
    )
    try:
        full_long = full_engine.generate(long_prompt, args.max_tokens, timeout=args.timeout)
        full_short = full_engine.generate(short_prompt, args.max_tokens, timeout=args.timeout)
    finally:
        full_engine.shutdown()

    chunked_engine = make_engine(
        backend,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        token_budget=args.token_budget,
        chunk_size=args.chunk_size,
    )
    try:
        chunked_engine.start()
        long_request = chunked_engine.submit(long_prompt, args.max_tokens)
        wait_until(
            lambda: 0 < long_request.prefill_cursor < len(long_request.prompt_tokens),
            args.timeout,
            "the long prompt to advance through a prefill chunk",
        )
        short_request = chunked_engine.submit(short_prompt, args.max_tokens)
        wait_until(
            lambda: bool(short_request.output_tokens)
            and long_request.prefill_cursor < len(long_request.prompt_tokens),
            args.timeout,
            "the short request to begin decode before long prefill finishes",
        )
        overlapped = True
        if not long_request.ready.wait(args.timeout):
            raise TimeoutError("chunked long request did not finish")
        if not short_request.ready.wait(args.timeout):
            raise TimeoutError("chunked short request did not finish")
        if long_request.status != "finished":
            raise RuntimeError(f"long request failed: {long_request.error}")
        if short_request.status != "finished":
            raise RuntimeError(f"short request failed: {short_request.error}")

        long_match = full_long.output_tokens == long_request.output_tokens
        short_match = full_short.output_tokens == short_request.output_tokens
        print(f"Long prompt tokens: {long_length}; chunk size: {args.chunk_size}")
        print(f"Long request prefill chunks: {long_request.prefill_chunks}")
        print(f"Short request prefill chunks: {short_request.prefill_chunks}")
        print(f"Short decode started before long prefill completed: {overlapped}")
        print(f"Full/chunked greedy tokens match: long={long_match}, short={short_match}")
        if not long_match or not short_match:
            raise RuntimeError("chunked and full-prefill outputs differ")
        print("Chunked prefill correctness: PASS")
        return 0
    finally:
        chunked_engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
