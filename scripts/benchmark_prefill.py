"""Measure HF dense versus Triton paged prefill on the configured GPU model."""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

from tinyllm.config import EngineConfig
from tinyllm.engine import Engine
from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def print_summary(name: str, trials: list[dict[str, object]]) -> None:
    metrics = (
        ("ttft_p50_ms", "TTFT p50", "ms"),
        ("ttft_p95_ms", "TTFT p95", "ms"),
        ("itl_p50_ms", "decode ITL p50", "ms"),
        ("itl_p95_ms", "decode ITL p95", "ms"),
        ("prompt_tokens_per_s", "prompt throughput", "tokens/s"),
        ("output_tokens_per_s", "output throughput", "tokens/s"),
        ("peak_allocated_delta_mb", "peak allocated delta", "MiB"),
        ("peak_reserved_delta_mb", "peak reserved delta", "MiB"),
    )
    print(f"\n{name} ({len(trials)} trials; median):")
    print(f"  prompt tokens/request: {trials[0]['prompt_tokens']}")
    print(f"  prefill chunks/request: {trials[0]['prefill_chunks']}")
    for key, label, unit in metrics:
        value = statistics.median(float(trial[key]) for trial in trials)
        print(f"  {label}: {value:.2f} {unit}")


def run_trial(backend: HuggingFaceBackend, prompts: list[str], *, prefill_backend: str,
              max_tokens: int, max_model_len: int, token_budget: int,
              chunk_size: int, timeout: float) -> dict[str, object]:
    engine = Engine(
        backend,
        config=EngineConfig(
            max_num_seqs=len(prompts),
            max_model_len=max_model_len,
            max_num_batched_tokens=token_budget,
            prefill_chunk_size=chunk_size,
            prefill_backend=prefill_backend,
        ),
    )
    torch = backend.torch
    try:
        warmups = [engine.submit(prompt, max_tokens=2) for prompt in prompts]
        engine.start()
        warmup_deadline = time.monotonic() + timeout
        for request in warmups:
            remaining = warmup_deadline - time.monotonic()
            if remaining <= 0 or not request.ready.wait(remaining):
                raise TimeoutError("benchmark batch warmup did not finish")
            if request.status != "finished":
                raise RuntimeError(
                    f"warmup request {request.request_id} failed: {request.error}"
                )
        del warmups
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.device)
        allocated_before = torch.cuda.memory_allocated(backend.device)
        reserved_before = torch.cuda.memory_reserved(backend.device)

        engine.start()
        started = time.perf_counter()
        requests = []
        submitted_at = []
        for prompt in prompts:
            submitted_at.append(time.perf_counter())
            requests.append(engine.submit(prompt, max_tokens))
        token_times: list[list[float]] = [[] for _ in requests]
        seen_tokens = [0] * len(requests)
        deadline = started + timeout
        while not all(req.ready.is_set() for req in requests):
            now = time.perf_counter()
            for index, req in enumerate(requests):
                count = len(req.output_tokens)
                if count > seen_tokens[index]:
                    token_times[index].extend([now] * (count - seen_tokens[index]))
                    seen_tokens[index] = count
            if now >= deadline:
                unfinished = [req.request_id for req in requests if not req.ready.is_set()]
                raise TimeoutError(f"benchmark requests did not finish: {unfinished}")
            time.sleep(0.002)
        finished_at = time.perf_counter()
        torch.cuda.synchronize()

        for index, req in enumerate(requests):
            if req.status != "finished":
                raise RuntimeError(f"request {req.request_id} failed: {req.error}")
            if len(token_times[index]) < len(req.output_tokens):
                token_times[index].extend(
                    [finished_at] * (len(req.output_tokens) - len(token_times[index]))
                )

        ttft = [times[0] - submitted_at[index]
                for index, times in enumerate(token_times) if times]
        itl = [times[token] - times[token - 1]
               for times in token_times for token in range(1, len(times))]
        prompt_tokens = sum(len(req.prompt_tokens) for req in requests)
        output_tokens = sum(len(req.output_tokens) for req in requests)
        elapsed = max(1e-9, finished_at - started)
        first_token_wall = max(ttft, default=elapsed)
        return {
            "prefill_backend": prefill_backend,
            "prompt_tokens": [len(req.prompt_tokens) for req in requests],
            "generated_tokens": [len(req.output_tokens) for req in requests],
            "outputs": [list(req.output_tokens) for req in requests],
            "prefill_chunks": [req.prefill_chunks for req in requests],
            "elapsed_s": elapsed,
            "ttft_p50_ms": percentile(ttft, 0.50) * 1000,
            "ttft_p95_ms": percentile(ttft, 0.95) * 1000,
            "itl_p50_ms": percentile(itl, 0.50) * 1000,
            "itl_p95_ms": percentile(itl, 0.95) * 1000,
            "prompt_tokens_per_s": prompt_tokens / max(first_token_wall, 1e-9),
            "output_tokens_per_s": output_tokens / elapsed,
            "peak_allocated_delta_mb": (
                torch.cuda.max_memory_allocated(backend.device) - allocated_before
            ) / (1024 ** 2),
            "peak_reserved_delta_mb": (
                torch.cuda.max_memory_reserved(backend.device) - reserved_before
            ) / (1024 ** 2),
        }
    finally:
        engine.shutdown()


def benchmark_chunk_size(backend: HuggingFaceBackend, prompts: list[str], *,
                        chunk_size: int, kernel_tiles: list[tuple[int, int]],
                        trials: int, max_tokens: int, max_model_len: int,
                        token_budget: int, timeout: float) -> bool:
    dense_results: list[dict[str, object]] = []
    paged_results = {tile: [] for tile in kernel_tiles}
    for trial_index in range(trials):
        def run(prefill_backend: str, tile: tuple[int, int] | None = None
                ) -> dict[str, object]:
            if tile is not None:
                backend.paged_prefill_block_m, backend.paged_prefill_block_n = tile
            label = "Dense HF/FA4" if tile is None else f"Triton paged tile={tile[0]}x{tile[1]}"
            print(
                f"Chunk size {chunk_size}; trial {trial_index + 1}/{trials}: "
                f"{label} warmup + measurement"
            )
            return run_trial(
                backend,
                prompts,
                prefill_backend=prefill_backend,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                token_budget=token_budget,
                chunk_size=chunk_size,
                timeout=timeout,
            )

        if trial_index % 2 == 0:
            dense_results.append(run("hf"))
        for tile in kernel_tiles:
            paged_results[tile].append(run("triton_paged", tile))
        if trial_index % 2 == 1:
            dense_results.append(run("hf"))

    print_summary(f"hf, chunk_size={chunk_size}", dense_results)
    all_matches = True
    for tile in kernel_tiles:
        tile_results = paged_results[tile]
        print(f"\nKernel tile {tile[0]}x{tile[1]} output checks:")
        trial_matches = []
        for trial_index, (dense, paged) in enumerate(zip(dense_results, tile_results), start=1):
            matches = dense["outputs"] == paged["outputs"]
            trial_matches.append(matches)
            print(f"  trial {trial_index}/{trials}: greedy output tokens match: {matches}")
            if not matches:
                for request, (dense_tokens, paged_tokens) in enumerate(zip(
                    dense["outputs"], paged["outputs"]
                )):
                    first_mismatch = next((index for index, (dense_token, paged_token) in
                                           enumerate(zip(dense_tokens, paged_tokens))
                                           if dense_token != paged_token), None)
                    if first_mismatch is not None:
                        print(
                            f"    request={request} token_index={first_mismatch} "
                            f"dense={dense_tokens[first_mismatch]} "
                            f"paged={paged_tokens[first_mismatch]}"
                        )
                    elif len(dense_tokens) != len(paged_tokens):
                        print(
                            f"    request={request} output lengths differ: "
                            f"dense={len(dense_tokens)} paged={len(paged_tokens)}"
                        )
        print_summary(
            f"triton_paged, chunk_size={chunk_size}, tile={tile[0]}x{tile[1]}",
            tile_results,
        )
        matches_all_trials = all(trial_matches)
        all_matches &= matches_all_trials
        print(
            f"Chunk size {chunk_size}, tile {tile[0]}x{tile[1]}: "
            f"greedy output tokens match in every trial: {matches_all_trials}"
        )
    return all_matches


def parse_kernel_tile(value: str) -> tuple[int, int]:
    try:
        block_m_text, block_n_text = value.lower().split("x", maxsplit=1)
        tile = int(block_m_text), int(block_n_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("kernel tile must look like 16x64") from exc
    if tile[0] not in (16, 32) or tile[1] not in (64, 128):
        raise argparse.ArgumentTypeError(
            "supported kernel tiles are 16x64, 16x128, 32x64, and 32x128"
        )
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dense HF prefill against direct Triton paged prefill"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_4",
                        choices=("sdpa", "flash_attention_2", "flash_attention_3",
                                 "flash_attention_4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3,
                        help="number of warmed measurements per backend")
    parser.add_argument("--prompt-repetitions", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None,
                        help="sweep these chunk sizes instead of --chunk-size")
    parser.add_argument(
        "--kernel-tiles", type=parse_kernel_tile, nargs="+", default=[(16, 64)],
        metavar="MxN",
        help="sweep Triton kernel tiles separately from scheduler chunk sizes "
             "(e.g. 16x64 16x128 32x64 32x128)",
    )
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--prompt", default=(
        "请解释大语言模型推理中的 KV cache、分页注意力和连续批处理。"
    ))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.prompt_repetitions < 1:
        parser.error("--prompt-repetitions must be at least 1")
    chunk_sizes = args.chunk_sizes if args.chunk_sizes is not None else [args.chunk_size]
    if not chunk_sizes or any(chunk_size < 1 for chunk_size in chunk_sizes):
        parser.error("chunk sizes must all be at least 1")
    chunk_sizes = list(dict.fromkeys(chunk_sizes))
    kernel_tiles = list(dict.fromkeys(args.kernel_tiles))

    backend = HuggingFaceBackend(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attention_backend=args.attention_backend,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
    )
    base_prompt = (args.prompt + "\n") * args.prompt_repetitions
    prompts = [base_prompt + f"\n请求编号 {index}。" for index in range(args.batch_size)]
    for prompt in prompts:
        if len(backend.encode(prompt)) + args.max_tokens > args.max_model_len:
            parser.error("prompt plus output exceeds --max-model-len")

    outputs_match = True
    for chunk_size in chunk_sizes:
        print(f"\n=== Benchmarking scheduler chunk_size={chunk_size}; "
              f"kernel tiles={kernel_tiles} ===")
        outputs_match &= benchmark_chunk_size(
            backend,
            prompts,
            chunk_size=chunk_size,
            kernel_tiles=kernel_tiles,
            trials=args.trials,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            token_budget=args.token_budget,
            timeout=args.timeout,
        )
    print(f"\nGreedy output tokens match in every run: {outputs_match}")
    return 0 if outputs_match else 1


if __name__ == "__main__":
    sys.exit(main())
