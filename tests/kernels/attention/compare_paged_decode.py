"""Compare the Qwen3 Triton paged decode path with HF dense-cache SDPA."""
from __future__ import annotations

import argparse
import sys

from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


def prompt_at_length(seed: list[int], length: int) -> list[int]:
    if length < 1:
        raise ValueError("prompt lengths must be positive")
    if not seed:
        raise ValueError("the prompt must tokenize to at least one token")
    if len(seed) >= length:
        return seed[:length]
    return seed + [seed[-1]] * (length - len(seed))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Triton paged decode logits against HF SDPA dense KV decode"
    )
    parser.add_argument("--model", required=True, help="local model directory or HF model ID")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_4",
                        choices=("sdpa", "flash_attention_2", "flash_attention_3",
                                 "flash_attention_4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--prompt-lengths", type=int, nargs="+",
                        default=[15, 16, 17, 31, 32, 33],
                        help="exact token lengths to test around page boundaries")
    parser.add_argument("--decode-steps", type=int, default=4,
                        help="decode steps per prompt; defaults to crossing the next page")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4],
                        help="mixed-length batch sizes to compare")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--kernel-atol", type=float, default=1e-2,
                        help="absolute tolerance for each layer's attention output")
    parser.add_argument("--kernel-rtol", type=float, default=1e-2,
                        help="relative tolerance for each layer's attention output")
    args = parser.parse_args()

    if args.decode_steps < 1:
        parser.error("--decode-steps must be at least 1")
    if any(length < 1 for length in args.prompt_lengths):
        parser.error("--prompt-lengths must contain only positive integers")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("--batch-sizes must contain only positive integers")
    if max(args.batch_sizes) > len(args.prompt_lengths):
        parser.error("each batch size must be no larger than --prompt-lengths count")
    largest = max(args.prompt_lengths) + args.decode_steps
    if largest > args.max_model_len:
        parser.error("max prompt length plus decode steps exceeds --max-model-len")

    backend = HuggingFaceBackend(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attention_backend=args.attention_backend,
        max_num_seqs=max(args.batch_sizes),
        max_model_len=args.max_model_len,
        kv_block_size=args.block_size,
    )
    seed = backend.encode(args.prompt)
    if not seed:
        parser.error("--prompt produced no tokens")

    all_tokens_match = True
    all_logits_close = True
    all_kernel_checks_close = True
    all_dense_control_tokens_match = True
    total_kernel_checks = 0
    close_kernel_checks = 0
    print("Reference: Hugging Face SDPA with dense KV cache")
    print("Dense control: Hugging Face eager with an independent dense KV cache")
    print("Candidate: Triton paged decode using per-request device block tables")
    print(f"Block size: {args.block_size}; dtype: {backend.model.dtype}")
    for batch_size in args.batch_sizes:
        if batch_size == 1:
            selected_indices = [len(args.prompt_lengths) // 2]
        else:
            selected_indices = [round(i * (len(args.prompt_lengths) - 1) / (batch_size - 1))
                                for i in range(batch_size)]
        selected_lengths = [args.prompt_lengths[index] for index in selected_indices]
        prompt_batch = [prompt_at_length(seed, length) for length in selected_lengths]
        result = backend.compare_paged_decode_batch(
            prompt_batch,
            steps=args.decode_steps,
            atol=args.atol,
            rtol=args.rtol,
            kernel_atol=args.kernel_atol,
            kernel_rtol=args.kernel_rtol,
        )
        requests = result["requests"]
        kernel_checks = result["kernel_checks"]
        token_match = all(row["token_match"] for request in requests
                          for row in request["steps"])
        logits_close = all(row["logits_close"] for request in requests
                           for row in request["steps"])
        dense_control_tokens_match = all(
            row["dense_backend_token_match"] for request in requests
            for row in request["steps"]
        )
        kernel_close = all(row["close"] for row in kernel_checks)
        all_tokens_match &= token_match
        all_logits_close &= logits_close
        all_kernel_checks_close &= kernel_close
        all_dense_control_tokens_match &= dense_control_tokens_match
        total_kernel_checks += len(kernel_checks)
        close_kernel_checks += sum(item["close"] for item in kernel_checks)
        print(f"\nBatch size {batch_size}, prompt lengths {selected_lengths}:")
        print(f"  Greedy tokens: {'MATCH' if token_match else 'MISMATCH'}")
        print(f"  SDPA/eager control tokens: "
              f"{'MATCH' if dense_control_tokens_match else 'MISMATCH'}")
        print(f"  Full logits allclose: {'YES' if logits_close else 'NO'} "
              f"(atol={args.atol}, rtol={args.rtol})")
        print(f"  Per-layer attention allclose: {sum(item['close'] for item in kernel_checks)}"
              f"/{len(kernel_checks)} "
              f"(atol={args.kernel_atol}, rtol={args.kernel_rtol})")
        for request in requests:
            print(f"  Request {request['index']} (prompt length {request['prompt_length']}):")
            for row in request["steps"]:
                print(
                    f"    step={row['step']} kv_length={row['kv_length']} "
                    f"max_abs={row['max_abs_error']:.6g} "
                    f"mean_abs={row['mean_abs_error']:.6g} "
                    f"rel_l2={row['relative_l2_error']:.6g} "
                    f"cosine={row['cosine_similarity']:.8f} "
                    f"top10={row['top10_overlap']:.2f} "
                    f"token={row['reference_token']}/{row['paged_token']} "
                    f"token_match={row['token_match']} logits_close={row['logits_close']}"
                )
                print(
                    f"      SDPA-vs-eager baseline: rel_l2="
                    f"{row['dense_backend_relative_l2_error']:.6g} "
                    f"cosine={row['dense_backend_cosine_similarity']:.8f} "
                    f"top10={row['dense_backend_top10_overlap']:.2f} "
                    f"token_match={row['dense_backend_token_match']}"
                )
        for item in kernel_checks:
            if not item["close"]:
                print(
                    f"  attention mismatch: request={item['request']} "
                    f"step={item['step']} layer={item['layer']} "
                    f"max_abs={item['max_abs_error']:.6g} "
                    f"mean_abs={item['mean_abs_error']:.6g} "
                    f"relative_l2={item['relative_l2_error']:.6g} "
                    f"cosine={item['cosine_similarity']:.8f}"
                )

    kernel_and_tokens_pass = all_kernel_checks_close and all_tokens_match
    print("\nKernel/token correctness:", "PASS" if kernel_and_tokens_pass else "FAIL")
    print("Dense SDPA/eager control tokens:",
          "MATCH" if all_dense_control_tokens_match else "MISMATCH")
    print("Full logits allclose:", "PASS" if all_logits_close else "FAIL (diagnostic)")
    print(f"Per-layer attention checks: {close_kernel_checks}/{total_kernel_checks} passed")
    return 0 if kernel_and_tokens_pass else 1


if __name__ == "__main__":
    sys.exit(main())
