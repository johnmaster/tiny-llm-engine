from __future__ import annotations

import argparse

from .model_executor.mock import MockBackend
from .engine import Engine
from .config import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny continuous-batching inference skeleton")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512,
                        help="decode tokens plus prefill tokens allowed per scheduler iteration")
    parser.add_argument("--prefill-chunk-size", type=int, default=128,
                        help="maximum prompt tokens processed per sequence per iteration")
    parser.add_argument("--prefill-backend", choices=("hf", "triton_paged"), default="hf",
                        help="use HF/FA4 dense prefill or direct Triton block-table prefill")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="reuse exact full KV blocks from earlier prompts")
    parser.add_argument("--max-prefix-cache-blocks", type=int, default=256)
    parser.add_argument("--model", help="Hugging Face model name or local model path; enables CUDA inference")
    parser.add_argument("--device", default="cuda", help="device for --model (default: cuda)")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"),
        default="sdpa",
        help="Hugging Face attention implementation (RTX 5090: try flash_attention_4)",
    )
    args = parser.parse_args()

    if args.model:
        from .model_executor.hf_transformers import HuggingFaceBackend
        backend = HuggingFaceBackend(args.model, dtype=args.dtype, device=args.device,
                                     attention_backend=args.attention_backend,
                                     max_num_seqs=args.max_num_seqs,
                                     max_model_len=args.max_model_len)
        print(f"Attention backend: {backend.attention_backend}")
    else:
        backend = MockBackend()
        print("Using MockBackend; pass --model <path> on the GPU server for real inference.")
    engine = Engine(backend, config=EngineConfig(max_num_seqs=args.max_num_seqs,
                                                  max_model_len=args.max_model_len,
                                                  max_num_batched_tokens=args.max_num_batched_tokens,
                                                  prefill_chunk_size=args.prefill_chunk_size,
                                                  prefill_backend=args.prefill_backend,
                                                  enable_prefix_caching=args.enable_prefix_caching,
                                                  max_prefix_cache_blocks=args.max_prefix_cache_blocks))
    try:
        req = engine.generate(args.prompt, args.max_tokens)
        attention_summary = getattr(backend, "attention_summary", None)
        if attention_summary is not None:
            summary = attention_summary()
            if summary:
                print(summary)
        kv_cache_summary = getattr(backend, "kv_cache_summary", None)
        if kv_cache_summary is not None:
            print(kv_cache_summary())
        scheduler_stats = engine.stats()
        print("Scheduler:", {
            "max_num_batched_tokens": scheduler_stats["max_num_batched_tokens"],
            "prefill_chunk_size": scheduler_stats["prefill_chunk_size"],
            "prefill_backend": scheduler_stats["prefill_backend"],
        })
        prefix_stats = engine.stats().get("prefix_cache")
        if prefix_stats is not None:
            print(f"Prefix cache: {prefix_stats}")
        print(engine.backend.decode(req.output_tokens))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
