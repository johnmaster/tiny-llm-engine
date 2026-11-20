# tiny-llm-engine
A lightweight LLM inference engine built from scratch.

## GPU validation and benchmarks

Run these commands from the repository root. Replace `<model>` with a compatible
Hugging Face model ID or local model path.

| Area | Command |
| --- | --- |
| Paged prefill correctness | `python -m tests.kernels.attention.compare_paged_prefill --model <model>` |
| Paged decode correctness | `python -m tests.kernels.attention.compare_paged_decode --model <model>` |
| Chunked prefill and request interleaving | `python -m tests.e2e.verify_chunked_prefill --model <model>` |
| Prefix cache reuse | `python -m tests.e2e.verify_prefix_cache --model <model>` |
| Scheduler and KV page reuse | `python -m tests.e2e.verify_scheduler_churn --model <model>` |
| Prefill performance | `python -m benchmarks.benchmark_prefill --model <model>` |
