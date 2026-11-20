"""Exercise staggered arrivals and KV-page reuse through the real engine scheduler."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

from tinyllm.config import EngineConfig
from tinyllm.engine import Engine
from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


DEFAULT_PROMPTS = [
    "你好，请用中文介绍一下大语言模型。",
    "请解释 KV cache 在自回归文本生成中的作用。",
    "说明 continuous batching 为什么可以提高 GPU 利用率。",
    "请描述 paged attention 如何通过 block table 读取 KV pages。",
]


def wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {description}")


def wait_requests(requests: Sequence[object], timeout: float) -> set[int]:
    deadline = time.monotonic() + timeout
    page_ids: set[int] = set()
    while not all(request.ready.is_set() for request in requests):
        for request in requests:
            if request.state is not None:
                page_ids.update(request.state.block_table.block_ids)
        if time.monotonic() >= deadline:
            unfinished = [r.request_id for r in requests if not r.ready.is_set()]
            raise TimeoutError(f"requests did not finish in time: {unfinished}")
        time.sleep(0.002)
    for request in requests:
        page_ids.update(request.released_block_ids)
        if request.status != "finished":
            raise RuntimeError(
                f"request {request.request_id} ended as {request.status}: {request.error}"
            )
        if not request.output_tokens:
            raise RuntimeError(f"request {request.request_id} produced no output tokens")
    return page_ids


def used_block_ids(requests: Sequence[object]) -> set[int]:
    ids: set[int] = set()
    for request in requests:
        state = request.state
        if state is not None:
            ids.update(state.block_table.block_ids)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check staggered scheduler admissions and paged KV block cleanup/reuse"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_4",
                        choices=("sdpa", "flash_attention_2", "flash_attention_3",
                                 "flash_attention_4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--reuse-max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prompts", nargs=4, default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    if args.max_num_seqs < 4:
        parser.error("--max-num-seqs must be at least 4 for this churn scenario")
    if args.max_tokens < 2 or args.reuse_max_tokens < 1:
        parser.error("token limits must be positive and --max-tokens must be at least 2")

    backend = HuggingFaceBackend(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attention_backend=args.attention_backend,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    engine = Engine(
        backend,
        config=EngineConfig(max_num_seqs=args.max_num_seqs,
                            max_model_len=args.max_model_len),
    )
    result_ok = False
    try:
        engine.start()
        wave1 = [engine.submit(prompt, args.max_tokens) for prompt in args.prompts[:2]]
        wait_until(
            lambda: any(req.status == "running" and req.state is not None for req in wave1),
            args.timeout,
            "the first wave to finish prefill and enter decode",
        )
        wave1_pages = used_block_ids(wave1)
        print("Wave 1 admitted:", [(r.status, len(r.prompt_tokens)) for r in wave1])

        wave2 = [engine.submit(prompt, args.max_tokens) for prompt in args.prompts[2:]]
        wait_until(
            lambda: (
                all(req.status == "running" and req.state is not None for req in wave1)
                and all(req.status == "running" and req.state is not None for req in wave2)
            ),
            args.timeout,
            "wave 2 to finish prefill while wave 1 is still decoding",
        )
        overlap = any(req.status == "running" for req in wave1) and any(
            req.status == "running" for req in wave2
        )
        if not overlap:
            raise RuntimeError("the two request waves did not overlap in the running batch")
        wave2_pages = used_block_ids(wave2)
        print("Wave 2 admitted during wave 1 decode:",
              [(r.status, len(r.prompt_tokens)) for r in wave2])
        all_wave_pages = wave1_pages | wave2_pages
        all_wave_pages |= wait_requests(wave1 + wave2, args.timeout)

        after_wave1 = backend.kv_cache.manager.snapshot()
        print("KV blocks after wave 1+2 completed:", after_wave1)
        if after_wave1["used_blocks"] != 0:
            raise RuntimeError("KV blocks were not fully released after request completion")

        # Released pages should be reused before untouched pool pages, keeping
        # the extensible cache's backing near the live-demand high-water mark.
        total_blocks = after_wave1["total_blocks"]
        if len(all_wave_pages) > total_blocks:
            raise RuntimeError("observed more unique request pages than the cache contains")

        reused_requests = [engine.submit(prompt, args.reuse_max_tokens)
                           for prompt in args.prompts]
        wait_until(
            lambda: any(req.state is not None for req in reused_requests),
            args.timeout,
            "the reuse wave to allocate KV pages",
        )
        reused_pages = used_block_ids(reused_requests)
        reused_page_ids = all_wave_pages & reused_pages
        print(f"Reused physical page IDs: {sorted(reused_page_ids)}")
        if not reused_page_ids:
            raise RuntimeError("the new wave did not reuse any page IDs from completed requests")
        wait_requests(reused_requests, args.timeout)

        final_stats = backend.kv_cache.manager.snapshot()
        print("KV blocks after reuse wave completed:", final_stats)
        if final_stats["used_blocks"] != 0:
            raise RuntimeError("KV blocks remained allocated after the reuse wave")
        print("KV storage after reuse wave:", backend.kv_cache_summary())
        print("Attention summary:", backend.attention_summary())
        print("Scheduler churn and KV reuse: PASS")
        result_ok = True
    finally:
        engine.shutdown()
        print("KV storage after Engine shutdown:", backend.kv_cache_summary())
    return 0 if result_ok else 1


if __name__ == "__main__":
    sys.exit(main())
