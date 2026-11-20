"""Check chunked-prefill equivalence and decode progress during a long prompt."""
from __future__ import annotations

import argparse
import statistics
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
                chunk_size: int, prefill_backend: str = "hf") -> Engine:
    return Engine(
        backend,
        config=EngineConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=token_budget,
            prefill_chunk_size=chunk_size,
            prefill_backend=prefill_backend,
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
    parser.add_argument("--prefill-backend", choices=("hf", "triton_paged"),
                        default="hf")
    parser.add_argument("--repetitions", type=int, default=32,
                        help="repeat the long-prompt paragraph this many times")
    parser.add_argument("--trials", type=int, default=3,
                        help="number of warmed interleaving measurements")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.trials < 1:
        parser.error("--trials must be at least 1")

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
        prefill_backend=args.prefill_backend,
    )
    try:
        # Warm the selected backend with both prompt shapes before measuring
        # responsiveness, so JIT compilation is not charged to the short request.
        chunked_engine.generate(long_prompt, max_tokens=2, timeout=args.timeout)
        chunked_engine.generate(short_prompt, max_tokens=2, timeout=args.timeout)
        print(f"Long prompt tokens: {long_length}; chunk size: {args.chunk_size}")
        print(f"Candidate prefill backend: {args.prefill_backend}")
        trial_results = []
        for trial_index in range(args.trials):
            chunked_engine.start()
            long_submitted_at = time.perf_counter()
            long_request = chunked_engine.submit(long_prompt, args.max_tokens)
            wait_until(
                lambda: 0 < long_request.prefill_cursor < len(long_request.prompt_tokens),
                args.timeout,
                "the long prompt to advance through a prefill chunk",
            )
            short_submitted_at = time.perf_counter()
            short_request = chunked_engine.submit(short_prompt, args.max_tokens)
            short_first_token_at: float | None = None
            long_prefill_incomplete_at_short_token: bool | None = None

            def short_has_output() -> bool:
                nonlocal short_first_token_at, long_prefill_incomplete_at_short_token
                if short_request.output_tokens and short_first_token_at is None:
                    short_first_token_at = time.perf_counter()
                    long_prefill_incomplete_at_short_token = (
                        long_request.prefill_cursor < len(long_request.prompt_tokens)
                    )
                return bool(short_request.output_tokens)

            wait_until(
                short_has_output,
                args.timeout,
                "the short request to produce its first token",
            )
            overlap = bool(long_prefill_incomplete_at_short_token)
            long_finished_at: float | None = None
            short_finished_at: float | None = None
            finish_deadline = time.monotonic() + args.timeout
            while long_finished_at is None or short_finished_at is None:
                now = time.perf_counter()
                if long_finished_at is None and long_request.ready.is_set():
                    long_finished_at = now
                if short_finished_at is None and short_request.ready.is_set():
                    short_finished_at = now
                if long_finished_at is not None and short_finished_at is not None:
                    break
                if now >= finish_deadline:
                    raise TimeoutError("interleaved requests did not finish")
                time.sleep(0.002)

            if long_request.status != "finished":
                raise RuntimeError(f"long request failed: {long_request.error}")
            if short_request.status != "finished":
                raise RuntimeError(f"short request failed: {short_request.error}")

            long_match = full_long.output_tokens == long_request.output_tokens
            short_match = full_short.output_tokens == short_request.output_tokens
            trial_results.append({
                "short_ttft_ms": (short_first_token_at - short_submitted_at) * 1000,
                "long_completion_ms": (long_finished_at - long_submitted_at) * 1000,
                "short_completion_ms": (short_finished_at - short_submitted_at) * 1000,
                "overlap": overlap,
                "tokens_match": long_match and short_match,
                "long_chunks": long_request.prefill_chunks,
                "short_chunks": short_request.prefill_chunks,
            })
            print(
                f"Trial {trial_index + 1}/{args.trials}: "
                f"short TTFT={trial_results[-1]['short_ttft_ms']:.2f} ms; "
                f"long completion={trial_results[-1]['long_completion_ms']:.2f} ms; "
                f"short completion={trial_results[-1]['short_completion_ms']:.2f} ms; "
                f"overlap={overlap}; tokens_match={long_match and short_match}; "
                f"chunks long/short={long_request.prefill_chunks}/"
                f"{short_request.prefill_chunks}"
            )

        for key, label in (
            ("short_ttft_ms", "Short request TTFT"),
            ("long_completion_ms", "Long request completion"),
            ("short_completion_ms", "Short request completion"),
        ):
            value = statistics.median(float(result[key]) for result in trial_results)
            print(f"{label} median: {value:.2f} ms")
        overlap_pass = all(bool(result["overlap"]) for result in trial_results)
        tokens_pass = all(bool(result["tokens_match"]) for result in trial_results)
        print(f"Short first token before long prefill finished: {overlap_pass}")
        print(f"Full/chunked greedy tokens match in every trial: {tokens_pass}")
        passed = overlap_pass and tokens_pass
        print("Chunked prefill interleaving:", "PASS" if passed else "FAIL")
        return 0 if passed else 1
    finally:
        chunked_engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
