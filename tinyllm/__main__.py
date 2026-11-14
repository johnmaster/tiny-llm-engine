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
                                                  max_model_len=args.max_model_len))
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
        print(engine.backend.decode(req.output_tokens))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
