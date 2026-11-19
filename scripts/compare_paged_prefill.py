"""Compare HF/FA4 dense prefill with direct block-table Triton prefill."""
from __future__ import annotations

import argparse
import sys

from tinyllm.model_executor.hf_transformers import HuggingFaceBackend


def prompt_at_length(seed: list[int], length: int) -> list[int]:
    if length < 1:
        raise ValueError("prompt length must be positive")
    if not seed:
        raise ValueError("the seed prompt must tokenize to at least one token")
    if len(seed) >= length:
        return seed[:length]
    repeats = (length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:length]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare full FA4 prefill with Triton paged, chunked prefill"
    )
    parser.add_argument("--model", required=True)
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
                        default=[15, 16, 17, 31, 32, 33, 63, 64, 65])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--kernel-atol", type=float, default=1e-2)
    parser.add_argument("--kernel-rtol", type=float, default=1e-2)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if any(length < 1 for length in args.prompt_lengths):
        parser.error("--prompt-lengths must contain positive integers")
    if any(size < 1 or size > len(args.prompt_lengths) for size in args.batch_sizes):
        parser.error("batch sizes must be positive and no larger than prompt-length count")
    if max(args.prompt_lengths) > args.max_model_len:
        parser.error("prompt length exceeds --max-model-len")

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
    all_kernel_checks_close = True
    total_checks = 0
    close_checks = 0
    total_output_checks = 0
    close_output_checks = 0
    print("Reference: Hugging Face dense prefill with configured attention backend")
    print("Candidate: Triton causal prefill reading the device block table")
    print(f"Block size: {args.block_size}; chunk size: {args.chunk_size}; "
          f"dtype: {backend.model.dtype}")

    for batch_size in args.batch_sizes:
        indices = ([len(args.prompt_lengths) // 2] if batch_size == 1 else
                   [round(i * (len(args.prompt_lengths) - 1) / (batch_size - 1))
                    for i in range(batch_size)])
        lengths = [args.prompt_lengths[index] for index in indices]
        prompts = [prompt_at_length(seed, length) for length in lengths]
        result = backend.compare_paged_prefill(
            prompts,
            chunk_size=args.chunk_size,
            kernel_atol=args.kernel_atol,
            kernel_rtol=args.kernel_rtol,
        )
        token_match = result["dense_tokens"] == result["paged_tokens"]
        checks = result["kernel_checks"]
        kernel_close = all(item["close"] for item in checks)
        all_tokens_match &= token_match
        all_kernel_checks_close &= kernel_close
        total_checks += len(checks)
        close_checks += sum(item["close"] for item in checks)
        print(f"\nBatch size {batch_size}, prompt lengths {lengths}:")
        print(f"  Greedy next tokens: {'MATCH' if token_match else 'MISMATCH'}")
        if not token_match:
            for row, (dense_token, paged_token) in enumerate(zip(
                result["dense_tokens"], result["paged_tokens"]
            )):
                if dense_token != paged_token:
                    dense_logits = result["dense_logits"][row]
                    paged_logits = result["paged_logits"][row]
                    dense_top = dense_logits.topk(5)
                    paged_top = paged_logits.topk(5)
                    logit_error = (dense_logits - paged_logits).abs()
                    print(
                        f"  token mismatch: request={row} prompt_length={lengths[row]} "
                        f"dense={dense_token} paged={paged_token}"
                    )
                    print(
                        f"  final logits: max_abs={logit_error.max().item():.6g} "
                        f"mean_abs={logit_error.mean().item():.6g}"
                    )
                    print(
                        "  dense top-5 (token, logit): "
                        + str(list(zip(dense_top.indices.tolist(),
                                       [round(value, 6) for value in dense_top.values.tolist()])))
                    )
                    print(
                        "  paged top-5 (token, logit): "
                        + str(list(zip(paged_top.indices.tolist(),
                                       [round(value, 6) for value in paged_top.values.tolist()])))
                    )
        print(f"  Paged chunks: {result['chunk_count']}")
        print(f"  Per-layer attention allclose: {sum(item['close'] for item in checks)}"
              f"/{len(checks)} (atol={args.kernel_atol}, rtol={args.kernel_rtol})")
        output_checks = result["attention_output_checks"]
        total_output_checks += len(output_checks)
        close_output_checks += sum(item["close"] for item in output_checks)
        print(
            "  Dense-vs-paged final-token attention outputs: "
            f"{sum(item['close'] for item in output_checks)}/{len(output_checks)} "
            f"(atol={args.kernel_atol}, rtol={args.kernel_rtol})"
        )
        for row in range(batch_size):
            row_checks = [item for item in output_checks if item["request"] == row]
            failed_layers = [item for item in row_checks if not item["close"]]
            if failed_layers:
                first = failed_layers[0]
                print(
                    f"  request={row} first dense/paged attention-output mismatch: "
                    f"layer={first['layer']} max_abs={first['max_abs_error']:.6g} "
                    f"mean_abs={first['mean_abs_error']:.6g}; "
                    f"failed_layers={len(failed_layers)}/{len(row_checks)}"
                )
        mismatches = [item for item in checks if not item["close"]]
        for item in mismatches[:12]:
            print(
                f"  mismatch: chunk={item['step']} request={item['request']} "
                f"layer={item['layer']} q_tokens={item['query_tokens']} "
                f"max_abs={item['max_abs_error']:.6g} "
                f"rel_l2={item['relative_l2_error']:.6g} "
                f"cosine={item['cosine_similarity']:.8f}"
            )

    all_intermediate_outputs_close = close_output_checks == total_output_checks
    print("\nGreedy next-token correctness:", "PASS" if all_tokens_match else "FAIL")
    print(f"Paged-kernel attention checks: {close_checks}/{total_checks} passed")
    print(
        "Dense-vs-paged attention-output parity (diagnostic): "
        f"{close_output_checks}/{total_output_checks} allclose"
    )
    functional_pass = all_tokens_match and all_kernel_checks_close
    print("Paged prefill functional check:", "PASS" if functional_pass else "FAIL")
    if not all_intermediate_outputs_close:
        print("Full intermediate numerical parity: NOT ESTABLISHED")
    return 0 if functional_pass else 1


if __name__ == "__main__":
    sys.exit(main())
