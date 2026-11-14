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
    args = parser.parse_args()

    if args.model:
        from .model_executor.hf_transformers import HuggingFaceBackend
        backend = HuggingFaceBackend(args.model, dtype=args.dtype, device=args.device)
    else:
        backend = MockBackend()
        print("Using MockBackend; pass --model <path> on the GPU server for real inference.")
    engine = Engine(backend, config=EngineConfig(max_num_seqs=args.max_num_seqs,
                                                  max_model_len=args.max_model_len))
    try:
        req = engine.generate(args.prompt, args.max_tokens)
        print(engine.backend.decode(req.output_tokens))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
