"""A small Hugging Face causal-LM backend with batched prefill and decode.

This is a correctness-first backend for learning and experimentation. It uses
greedy sampling, Hugging Face attention for the dense path, and registered
model implementations to connect supported architectures to paged attention.
It is not optimized like vLLM.
Install the dependencies listed in ``requirements-gpu.txt`` in a CUDA-enabled
PyTorch environment before running it on a GPU machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

from ..kv_cache import (BlockTable, KVCacheOutOfBlocks, PagedKVTensorStore,
                        PrefixMatch)
from .adapters import resolve_model_adapter


@dataclass
class _SequenceState:
    block_table: BlockTable

    @property
    def token_count(self) -> int:
        return self.block_table.token_count


class HuggingFaceBackend:
    def __init__(self, model_name_or_path: str, *, dtype: str = "auto", device: str = "cuda",
                 attention_backend: str = "sdpa", max_num_seqs: int = 8,
                 max_model_len: int = 8192, kv_block_size: int = 16,
                 paged_prefill_block_m: int = 16,
                 paged_prefill_block_n: int = 64):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("GPU backend requires: pip install -e '.[gpu]'") from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer needs a pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        supported_attention = {"sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"}
        if attention_backend not in supported_attention:
            choices = ", ".join(sorted(supported_attention))
            raise ValueError(f"attention_backend must be one of: {choices}")
        if attention_backend.startswith("flash_attention") and not device.startswith("cuda"):
            raise ValueError(f"{attention_backend} requires a CUDA device")
        self.device = torch.device(device)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype_map[dtype],
                attn_implementation=attention_backend,
                low_cpu_mem_usage=True,
            ).to(self.device)
        except (ImportError, RuntimeError, ValueError) as exc:
            if attention_backend == "sdpa":
                raise
            raise RuntimeError(
                f"Could not initialize attention backend '{attention_backend}'. "
                "Check that the installed Transformers release supports it and that its "
                "FlashAttention kernel is installed and compatible with this PyTorch/CUDA/GPU. "
                "For RTX 5090, use flash_attention_4 with a Transformers release that supports "
                "it and flash-attn-4 built for CUDA 13."
            ) from exc
        self.model.eval()
        if attention_backend.startswith("flash_attention") and self.model.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"{attention_backend} requires float16 or bfloat16 model weights; "
                f"loaded dtype is {self.model.dtype}. Set --dtype bfloat16 or --dtype float16."
            )
        self.attention_backend = attention_backend
        config = getattr(self.model.config, "text_config", None) or self.model.config
        self.model_adapter = resolve_model_adapter(config)
        if paged_prefill_block_m not in (16, 32) or paged_prefill_block_n not in (64, 128):
            raise ValueError(
                "paged prefill tiles must use block_m in {16, 32} "
                "and block_n in {64, 128}"
            )
        self.paged_prefill_block_m = paged_prefill_block_m
        self.paged_prefill_block_n = paged_prefill_block_n
        self._attention_calls: dict[str, int] = {}
        self._attention_output_capture: dict[str, Any] | None = None
        self._attention_layers: list[tuple[str, str]] = []
        self._attention_modules: list[tuple[str, Any]] = []
        if self.model_adapter is not None:
            candidate_modules = self.model_adapter.find_attention_modules(self.model)
        else:
            candidate_modules = [
                (name, module) for name, module in self.model.named_modules()
                if name.endswith(".self_attn") and hasattr(module, "config")
            ]
        for name, module in candidate_modules:
            configured_backend = getattr(module.config, "_attn_implementation", None)
            if configured_backend is None:
                continue
            self._attention_layers.append((name, configured_backend))
            self._attention_modules.append((name, module))
            self._attention_calls[name] = 0

            def count_attention_call(_module: Any, _inputs: Any, _output: Any,
                                     layer_name: str = name) -> None:
                self._attention_calls[layer_name] += 1
                if self._attention_output_capture is not None:
                    attention_output = _output[0] if isinstance(_output, tuple) else _output
                    if isinstance(attention_output, self.torch.Tensor):
                        self._attention_output_capture[layer_name] = attention_output.detach()

            module.register_forward_hook(count_attention_call)

        if self.model_adapter is not None and len(self._attention_modules) != len(candidate_modules):
            raise RuntimeError(
                f"The {self.model_adapter.name} adapter found attention layers without "
                "a configured Transformers attention backend"
            )

        if self._attention_layers:
            mismatched = [(name, backend) for name, backend in self._attention_layers
                          if backend != attention_backend]
            if mismatched:
                details = ", ".join(f"{name}={backend}" for name, backend in mismatched)
                raise RuntimeError(
                    f"Some attention layers did not select {attention_backend}: {details}"
                )

        if max_num_seqs < 1 or max_model_len < 1 or kv_block_size < 1:
            raise ValueError("max_num_seqs, max_model_len, and kv_block_size must be positive")
        num_layers = len(self._attention_layers) or int(config.num_hidden_layers)
        num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        max_blocks = max_num_seqs * ((max_model_len + kv_block_size - 1) // kv_block_size)
        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            bytes_per_block = (num_layers * 2 * num_kv_heads * head_dim * kv_block_size
                               * torch.empty((), dtype=self.model.dtype).element_size())
            # Leave room for temporary dense batches, logits, and CUDA workspaces.
            # The paged kernel uses fixed contiguous per-layer pools, so retain
            # extra headroom for prefill workspaces and logits.
            memory_limited_blocks = max(1, int(free_bytes * 0.45) // bytes_per_block)
            max_blocks = min(max_blocks, memory_limited_blocks)
        self.kv_block_size = kv_block_size
        self.kv_cache = PagedKVTensorStore(
            torch=torch,
            num_blocks=max_blocks,
            block_size=kv_block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.model.dtype,
            device=self.device,
        )
        self._paged_context: dict[str, Any] | None = None
        self._paged_decode_enabled = self.model_adapter is not None
        if self._paged_decode_enabled:
            self._paged_attention_calls = [0] * len(self._attention_modules)
            self._paged_prefill_calls = [0] * len(self._attention_modules)
            self._install_paged_attention()
        else:
            self._paged_attention_calls = []
            self._paged_prefill_calls = []
        if (getattr(config, "model_type", None) in ("qwen2", "qwen3")
                and not self._paged_decode_enabled):
            raise RuntimeError(
                f"This Transformers {config.model_type} implementation is not supported "
                "by the registered model adapter"
            )

    def attention_summary(self) -> str | None:
        """Summarize configured backends and layers actually called so far."""
        if not self._attention_layers:
            return None
        total = len(self._attention_layers)
        invoked = sum(self._attention_calls[name] > 0 for name, _ in self._attention_layers)
        prefill_layers = sum(count > 0 for count in self._paged_prefill_calls)
        if self._paged_decode_enabled:
            paged_layers = sum(count > 0 for count in self._paged_attention_calls)
            decode_mode = f"Triton paged decode {paged_layers}/{len(self._paged_attention_calls)} layers"
        else:
            decode_mode = "HF dense cache"
        prefill_mode = (f"Triton paged prefill {prefill_layers}/{total} layers"
                        if prefill_layers else self.attention_backend)
        adapter_name = self.model_adapter.name if self.model_adapter is not None else "HF"
        return (f"Model adapter: {adapter_name}; attention layers executed: {invoked}/{total}; "
                f"prefill backend: {prefill_mode}; decode: {decode_mode}")

    def kv_cache_summary(self) -> str:
        stats = self.kv_cache.manager.snapshot()
        return (f"Paged KV storage: block_size={self.kv_block_size}; "
                f"active_blocks={self.kv_cache.materialized_blocks}/"
                f"{stats['total_blocks']}")

    def release_state(self, state: object) -> None:
        if isinstance(state, _SequenceState):
            self.kv_cache.release(state.block_table)

    def _install_paged_attention(self) -> None:
        for _, module in self._attention_modules:
            original_forward = module.forward

            def paged_forward(attention_module: Any, *args: Any,
                              _original: Any = original_forward, **kwargs: Any) -> Any:
                context = self._paged_context
                if context is None:
                    return _original(*args, **kwargs)
                return self._paged_attention_forward(attention_module, context,
                                                     *args, **kwargs)

            module.forward = MethodType(paged_forward, module)

    def _paged_attention_forward(self, module: Any, context: dict[str, Any],
                                 hidden_states: Any, **kwargs: Any) -> tuple[Any, None]:
        """Run adapter-projected Q/K/V over the request's physical KV pages."""
        if self.model_adapter is None:
            raise RuntimeError("paged attention was called without a registered model adapter")
        torch = self.torch
        batch_size, query_length, _ = hidden_states.shape
        query, key, value = self.model_adapter.project_qkv(
            self, module, hidden_states, kwargs
        )

        layer_idx = self.model_adapter.layer_index(module)
        if context.get("mode", "decode") == "prefill":
            query_lengths = context["query_lengths"]
            for row, (table, old_length, valid_length) in enumerate(zip(
                context["tables"], context["old_lengths"], query_lengths
            )):
                self.kv_cache.write_layer(
                    table, layer_idx, old_length,
                    key[row, :, :valid_length, :].permute(1, 0, 2).contiguous(),
                    value[row, :, :valid_length, :].permute(1, 0, 2).contiguous(),
                )
            attention_output = self.kv_cache.paged_prefill_attention(
                layer_idx, query, context["page_table"],
                context["old_lengths_tensor"], context["query_lengths_tensor"],
                self.model_adapter.attention_scale(module),
                block_m=self.paged_prefill_block_m,
                block_n=self.paged_prefill_block_n,
            )
            if "kernel_checks" in context:
                query_heads = query.shape[1]
                groups = query_heads // self.kv_cache.num_kv_heads
                for row, (table, old_length, valid_length) in enumerate(zip(
                    context["tables"], context["old_lengths"], query_lengths
                )):
                    dense_key, dense_value = self.kv_cache.gather_layer(table, layer_idx)
                    dense_key = dense_key[0].repeat_interleave(groups, dim=0).float()
                    dense_value = dense_value[0].repeat_interleave(groups, dim=0).float()
                    query_fp32 = query[row, :, :valid_length, :].float()
                    valid_kv_length = old_length + valid_length
                    key_positions = torch.arange(valid_kv_length, device=self.device)
                    query_limits = old_length + torch.arange(valid_length, device=self.device)
                    scores = torch.einsum(
                        "hqd,hkd->hqk", query_fp32,
                        dense_key[:, :valid_kv_length, :],
                    ) * self.model_adapter.attention_scale(module)
                    causal_mask = key_positions[None, :] > query_limits[:, None]
                    scores = scores.masked_fill(causal_mask[None, :, :], float("-inf"))
                    probabilities = torch.softmax(scores, dim=-1)
                    reference_output = torch.einsum(
                        "hqk,hkd->hqd", probabilities,
                        dense_value[:, :valid_kv_length, :],
                    ).permute(1, 0, 2).to(dtype=attention_output.dtype)
                    actual_output = attention_output[row, :, :valid_length, :].permute(1, 0, 2)
                    error = (reference_output.float() - actual_output.float()).abs()
                    reference_norm = torch.linalg.vector_norm(reference_output.float())
                    relative_l2 = torch.linalg.vector_norm(error) / reference_norm.clamp_min(1e-12)
                    cosine = torch.nn.functional.cosine_similarity(
                        reference_output.float().reshape(1, -1),
                        actual_output.float().reshape(1, -1),
                    )[0]
                    context["kernel_checks"].append({
                        "mode": "prefill",
                        "step": context.get("chunk_index", 0),
                        "request": row,
                        "layer": layer_idx,
                        "query_tokens": valid_length,
                        "max_abs_error": float(error.max().item()),
                        "mean_abs_error": float(error.mean().item()),
                        "relative_l2_error": float(relative_l2.item()),
                        "cosine_similarity": float(cosine.item()),
                        "close": bool(torch.allclose(
                            reference_output, actual_output,
                            atol=context["kernel_atol"], rtol=context["kernel_rtol"],
                        )),
                    })
            attention_output = attention_output.transpose(1, 2).contiguous()
            self._paged_prefill_calls[layer_idx] += 1
        else:
            if query_length != 1:
                raise RuntimeError("paged decode only supports one query token per sequence")
            for row, (table, old_length) in enumerate(
                zip(context["tables"], context["old_lengths"])
            ):
                self.kv_cache.write_layer(
                    table, layer_idx, old_length,
                    key[row, :, 0, :].unsqueeze(0), value[row, :, 0, :].unsqueeze(0),
                )
            attention_output = self.kv_cache.paged_attention(
                layer_idx, query[:, :, 0, :], context["page_table"],
                context["lengths"], self.model_adapter.attention_scale(module),
            )
            if "kernel_checks" in context:
                query_heads = query.shape[1]
                groups = query_heads // self.kv_cache.num_kv_heads
                for row, table in enumerate(context["tables"]):
                    dense_key, dense_value = self.kv_cache.gather_layer(table, layer_idx)
                    dense_key = dense_key[0].repeat_interleave(groups, dim=0).float()
                    dense_value = dense_value[0].repeat_interleave(groups, dim=0).float()
                    query_fp32 = query[row, :, 0, :].float()
                    scores = torch.einsum("hd,hld->hl", query_fp32, dense_key)
                    probabilities = torch.softmax(
                        scores * self.model_adapter.attention_scale(module), dim=-1
                    )
                    reference_output = torch.einsum(
                        "hl,hld->hd", probabilities, dense_value
                    ).to(dtype=attention_output.dtype)
                    actual_output = attention_output[row]
                    error = (reference_output.float() - actual_output.float()).abs()
                    reference_norm = torch.linalg.vector_norm(reference_output.float())
                    relative_l2 = torch.linalg.vector_norm(error) / reference_norm.clamp_min(1e-12)
                    cosine = torch.nn.functional.cosine_similarity(
                        reference_output.float().reshape(1, -1),
                        actual_output.float().reshape(1, -1),
                    )[0]
                    context["kernel_checks"].append({
                        "step": context["kernel_step"],
                        "request": row,
                        "layer": layer_idx,
                        "max_abs_error": float(error.max().item()),
                        "mean_abs_error": float(error.mean().item()),
                        "relative_l2_error": float(relative_l2.item()),
                        "cosine_similarity": float(cosine.item()),
                        "close": bool(torch.allclose(
                            reference_output, actual_output,
                            atol=context["kernel_atol"], rtol=context["kernel_rtol"],
                        )),
                    })
            self._paged_attention_calls[layer_idx] += 1
            attention_output = attention_output.unsqueeze(1).contiguous()
        attention_output = attention_output.reshape(batch_size, query_length, -1).contiguous()
        projected = self.model_adapter.project_output(module, attention_output)
        return self.model_adapter.format_attention_output(module, projected)

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _legacy_cache(self, cache: Any) -> tuple[tuple[Any, Any], ...]:
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        elif hasattr(cache, "layers"):
            # Transformers 5 Cache objects expose per-layer key/value tensors
            # through ``layers`` and may no longer implement the legacy tuple
            # conversion helper.
            cache = tuple((layer.keys, layer.values) for layer in cache.layers)
        return tuple((key, value) for key, value in cache)

    @staticmethod
    def _cache_from_legacy(layers: Sequence[tuple[Any, Any]]) -> Any:
        """Build the Cache object required by recent Transformers releases."""
        from transformers import DynamicCache

        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(tuple(layers))

        # Transformers releases that removed from_legacy_cache accept the
        # same per-layer tensors through the cache constructor.
        try:
            return DynamicCache(ddp_cache_data=tuple(layers))
        except TypeError:
            # Keep compatibility with Cache implementations whose DynamicCache
            # constructor is empty but which still support incremental updates.
            cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(layers):
                cache.update(key, value, layer_idx)
            return cache

    @staticmethod
    def _last_valid_logits(logits: Any, lengths: list[int]) -> Any:
        # Prefill input rows are right-padded, so each row's last prompt token is
        # at length - 1 (not at the final padded column).
        return logits[range(len(lengths)), [length - 1 for length in lengths], :]

    def compare_paged_prefill(
        self,
        prompt_batches: Sequence[Sequence[int]],
        *,
        chunk_size: int = 32,
        kernel_atol: float = 1e-2,
        kernel_rtol: float = 1e-2,
    ) -> dict[str, Any]:
        """Compare FA4/HF dense prefill with direct Triton paged prefill."""
        if not self._paged_decode_enabled:
            raise RuntimeError(
                "paged prefill comparison requires a registered model adapter for this architecture"
            )
        prompts = [list(prompt) for prompt in prompt_batches]
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompt batch and each prompt must be nonempty")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        dense_logits_capture: list[Any] = []
        dense_attention_capture: dict[str, Any] = {}
        self._attention_output_capture = dense_attention_capture
        try:
            dense_outputs = self.prefill(prompts, logits_capture=dense_logits_capture)
        finally:
            self._attention_output_capture = None
        dense_tokens = [token for token, _ in dense_outputs]
        dense_logits = dense_logits_capture[0].cpu()
        dense_layer_outputs: dict[int, list[Any]] = {}
        for layer_idx, (name, _) in enumerate(self._attention_modules):
            layer_output = dense_attention_capture.get(name)
            if layer_output is not None:
                dense_layer_outputs[layer_idx] = [
                    layer_output[row, len(prompt) - 1].float().cpu()
                    for row, prompt in enumerate(prompts)
                ]
        del dense_attention_capture
        for _, state in dense_outputs:
            self.release_state(state)

        cursors = [0] * len(prompts)
        states: list[object | None] = [None] * len(prompts)
        paged_tokens: list[int | None] = [None] * len(prompts)
        paged_logits: list[Any | None] = [None] * len(prompts)
        paged_layer_outputs: list[dict[int, Any]] = [{} for _ in prompts]
        kernel_checks: list[dict[str, Any]] = []
        chunk_index = 0
        try:
            while any(cursor < len(prompt) for cursor, prompt in zip(cursors, prompts)):
                rows = [index for index, (cursor, prompt) in enumerate(zip(cursors, prompts))
                        if cursor < len(prompt)]
                chunks = [
                    prompts[index][cursors[index]:cursors[index] + chunk_size]
                    for index in rows
                ]
                final_chunks = [
                    cursors[index] + len(chunk) == len(prompts[index])
                    for index, chunk in zip(rows, chunks)
                ]
                chunk_logits: list[Any] = []
                chunk_attention_capture: dict[str, Any] = {}
                self._attention_output_capture = chunk_attention_capture
                try:
                    outputs = self.prefill_chunk(
                        chunks,
                        [states[index] for index in rows],
                        [None] * len(rows),
                        final_chunks,
                        prefill_backend="triton_paged",
                        kernel_checks=kernel_checks,
                        kernel_atol=kernel_atol,
                        kernel_rtol=kernel_rtol,
                        chunk_index=chunk_index,
                        logits_capture=chunk_logits,
                    )
                finally:
                    self._attention_output_capture = None
                for layer_idx, (name, _) in enumerate(self._attention_modules):
                    layer_output = chunk_attention_capture.get(name)
                    if layer_output is None:
                        continue
                    for local_row, (global_row, chunk, final) in enumerate(
                        zip(rows, chunks, final_chunks)
                    ):
                        if final:
                            paged_layer_outputs[global_row][layer_idx] = (
                                layer_output[local_row, len(chunk) - 1].float().cpu()
                            )
                del chunk_attention_capture
                for index, chunk, final, (token, state), row_logits in zip(
                    rows, chunks, final_chunks, outputs, chunk_logits
                ):
                    states[index] = state
                    cursors[index] += len(chunk)
                    if token is not None:
                        paged_tokens[index] = token
                    if final:
                        paged_logits[index] = row_logits
                chunk_index += 1
            attention_output_checks = []
            for request_idx in range(len(prompts)):
                for layer_idx, dense_rows in dense_layer_outputs.items():
                    paged_output = paged_layer_outputs[request_idx].get(layer_idx)
                    if paged_output is None:
                        continue
                    reference = dense_rows[request_idx]
                    error = (reference - paged_output).abs()
                    attention_output_checks.append({
                        "request": request_idx,
                        "layer": layer_idx,
                        "max_abs_error": float(error.max().item()),
                        "mean_abs_error": float(error.mean().item()),
                        "close": bool(self.torch.allclose(
                            reference, paged_output,
                            atol=kernel_atol, rtol=kernel_rtol,
                        )),
                    })
            return {
                "dense_tokens": dense_tokens,
                "paged_tokens": paged_tokens,
                "dense_logits": dense_logits,
                "paged_logits": self.torch.stack(paged_logits).cpu(),
                "kernel_checks": kernel_checks,
                "attention_output_checks": attention_output_checks,
                "chunk_count": chunk_index,
            }
        finally:
            self._paged_context = None
            for state in states:
                if state is not None:
                    self.release_state(state)

    def compare_paged_decode_batch(self, prompt_batches: Sequence[Sequence[Sequence[int]]],
                                   *, steps: int = 4, atol: float = 5e-2,
                                   rtol: float = 5e-2, kernel_atol: float = 1e-2,
                                   kernel_rtol: float = 1e-2) -> dict[str, Any]:
        """Compare mixed-length Triton paged decode against dense SDPA and eager."""
        if not self._paged_decode_enabled:
            raise RuntimeError(
                "paged decode comparison requires a registered model adapter for this architecture"
            )
        prompts = [list(prompt) for prompt in prompt_batches]
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompt batch and each prompt must be nonempty")
        if len(prompts) > self.kv_cache.manager.num_blocks:
            raise ValueError("batch is larger than the configured KV page pool")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        prompt_lengths = [len(prompt) for prompt in prompts]
        max_prompt_length = max(prompt_lengths)
        if max_prompt_length + steps > self.kv_cache.manager.num_blocks * self.kv_block_size:
            raise ValueError("prompt plus steps exceeds the allocated paged KV capacity")

        torch = self.torch
        batch_size = len(prompts)
        block_tables: list[BlockTable] = []
        requests = [{"index": row, "prompt_length": length, "steps": []}
                    for row, length in enumerate(prompt_lengths)]
        kernel_checks: list[dict[str, Any]] = []

        def clone_dense_cache(cache: Any) -> Any:
            return self._cache_from_legacy(tuple(
                (key.clone(), value.clone()) for key, value in self._legacy_cache(cache)
            ))

        def dense_decode(cache: Any, token_ids: Any, mask: Any, positions: Any,
                         attention_impl: str) -> Any:
            original_backends: dict[int, tuple[Any, Any]] = {}
            try:
                for _, module in self._attention_modules:
                    config = module.config
                    config_id = id(config)
                    if config_id not in original_backends:
                        original_backends[config_id] = (
                            config, getattr(config, "_attn_implementation", None)
                        )
                    config._attn_implementation = attention_impl
                self._paged_context = None
                with torch.inference_mode():
                    return self.model(
                        input_ids=token_ids,
                        attention_mask=mask,
                        position_ids=positions,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
            finally:
                for config, original in original_backends.values():
                    config._attn_implementation = original

        def logits_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
            difference = (reference - candidate).abs()
            relative_l2 = (torch.linalg.vector_norm(difference)
                           / torch.linalg.vector_norm(reference).clamp_min(1e-12))
            cosine = torch.nn.functional.cosine_similarity(
                reference.reshape(1, -1), candidate.reshape(1, -1)
            )[0]
            top_k = min(10, reference.numel())
            reference_top = torch.topk(reference, top_k)
            candidate_top = torch.topk(candidate, top_k)
            overlap = len(set(reference_top.indices.tolist())
                          & set(candidate_top.indices.tolist())) / top_k
            max_index = int(difference.argmax().item())
            top_two = torch.topk(reference, min(2, reference.numel())).values
            margin = top_two[0] - top_two[1] if top_two.numel() > 1 else top_two[0]
            return {
                "max_abs_error": float(difference[max_index].item()),
                "mean_abs_error": float(difference.mean().item()),
                "relative_l2_error": float(relative_l2.item()),
                "cosine_similarity": float(cosine.item()),
                "top10_overlap": float(overlap),
                "max_error_token_id": max_index,
                "reference_at_max_error": float(reference[max_index].item()),
                "candidate_at_max_error": float(candidate[max_index].item()),
                "reference_top1_margin": float(margin.item()),
                "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
            }

        try:
            width = max_prompt_length
            input_ids = torch.full((batch_size, width), self.tokenizer.pad_token_id,
                                   dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((batch_size, width), dtype=torch.long,
                                         device=self.device)
            for row, prompt in enumerate(prompts):
                length = prompt_lengths[row]
                input_ids[row, :length] = torch.tensor(prompt, dtype=torch.long,
                                                        device=self.device)
                attention_mask[row, :length] = 1
            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
            with torch.inference_mode():
                prefill_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )
            prefill_cache = prefill_output.past_key_values
            sdpa_cache = clone_dense_cache(prefill_cache)
            eager_cache = clone_dense_cache(prefill_cache)
            layer_cache = self._legacy_cache(prefill_cache)
            for row, length in enumerate(prompt_lengths):
                table = self.kv_cache.allocate(length)
                block_tables.append(table)
                for layer_idx, (key, value) in enumerate(layer_cache):
                    key_tokens = key[row, :, :length, :].permute(1, 0, 2).contiguous()
                    value_tokens = value[row, :, :length, :].permute(1, 0, 2).contiguous()
                    self.kv_cache.write_layer(table, layer_idx, 0, key_tokens, value_tokens)

            initial_logits = self._last_valid_logits(prefill_output.logits, prompt_lengths)
            next_tokens = initial_logits.argmax(dim=-1)
            del prefill_output, prefill_cache, layer_cache

            for step in range(steps):
                physical_width = width + step
                decode_ids = next_tokens.to(device=self.device, dtype=torch.long).unsqueeze(1)
                dense_mask = torch.zeros((batch_size, physical_width + 1), dtype=torch.long,
                                         device=self.device)
                for row, length in enumerate(prompt_lengths):
                    dense_mask[row, :length] = 1
                    dense_mask[row, width:physical_width + 1] = 1
                decode_positions = torch.tensor(
                    [[length + step] for length in prompt_lengths],
                    dtype=torch.long, device=self.device,
                )

                sdpa_output = dense_decode(
                    sdpa_cache, decode_ids, dense_mask, decode_positions, "sdpa"
                )
                sdpa_cache = sdpa_output.past_key_values
                reference_logits = sdpa_output.logits[:, -1, :].float()
                eager_output = dense_decode(
                    eager_cache, decode_ids, dense_mask, decode_positions, "eager"
                )
                eager_cache = eager_output.past_key_values
                eager_logits = eager_output.logits[:, -1, :].float()

                for table in block_tables:
                    self.kv_cache.append(table, 1)
                page_table, lengths_tensor = self.kv_cache.device_page_table(block_tables)
                self._paged_context = {
                    "tables": block_tables,
                    "old_lengths": [length + step for length in prompt_lengths],
                    "page_table": page_table,
                    "lengths": lengths_tensor,
                    "kernel_checks": kernel_checks,
                    "kernel_step": step + 1,
                    "kernel_atol": kernel_atol,
                    "kernel_rtol": kernel_rtol,
                }
                with torch.inference_mode():
                    paged_output = self.model(
                        input_ids=decode_ids,
                        attention_mask=None,
                        position_ids=decode_positions,
                        use_cache=False,
                        return_dict=True,
                    )
                self._paged_context = None
                paged_logits = paged_output.logits[:, -1, :].float()
                for row in range(batch_size):
                    ref = reference_logits[row]
                    eager = eager_logits[row]
                    paged = paged_logits[row]
                    paged_metrics = logits_metrics(ref, paged)
                    eager_metrics = logits_metrics(ref, eager)
                    reference_token = int(ref.argmax().item())
                    paged_token = int(paged.argmax().item())
                    requests[row]["steps"].append({
                        "step": step + 1,
                        "kv_length": prompt_lengths[row] + step + 1,
                        **paged_metrics,
                        "dense_backend_max_abs_error": eager_metrics["max_abs_error"],
                        "dense_backend_mean_abs_error": eager_metrics["mean_abs_error"],
                        "dense_backend_relative_l2_error": eager_metrics["relative_l2_error"],
                        "dense_backend_cosine_similarity": eager_metrics["cosine_similarity"],
                        "dense_backend_top10_overlap": eager_metrics["top10_overlap"],
                        "dense_backend_logits_close": eager_metrics["allclose"],
                        "dense_backend_token_match": (
                            reference_token == int(eager.argmax().item())
                        ),
                        "reference_token": reference_token,
                        "paged_token": paged_token,
                        "token_match": reference_token == paged_token,
                        "logits_close": paged_metrics["allclose"],
                    })
                next_tokens = reference_logits.argmax(dim=-1)

            return {"requests": requests, "kernel_checks": kernel_checks}
        finally:
            self._paged_context = None
            for table in block_tables:
                if not table._released:
                    self.kv_cache.release(table)

    def compare_paged_decode(self, prompt_tokens: Sequence[int], **kwargs: Any) -> dict[str, Any]:
        """Single-request convenience wrapper for ``compare_paged_decode_batch``."""
        result = self.compare_paged_decode_batch([prompt_tokens], **kwargs)
        return {"steps": result["requests"][0]["steps"],
                "kernel_checks": result["kernel_checks"]}

    def prefill_chunk(
        self,
        token_chunks: Sequence[Sequence[int]],
        states: Sequence[object | None],
        prefix_matches: Sequence[PrefixMatch | None],
        final_chunks: Sequence[bool],
        *,
        prefill_backend: str = "hf",
        kernel_checks: list[dict[str, Any]] | None = None,
        kernel_atol: float = 1e-2,
        kernel_rtol: float = 1e-2,
        chunk_index: int = 0,
        logits_capture: list[Any] | None = None,
    ) -> list[tuple[int | None, object]]:
        """Append one prefill chunk per sequence and return logits when complete."""
        if not token_chunks:
            return []
        batch_size = len(token_chunks)
        if not (len(states) == len(prefix_matches) == len(final_chunks) == batch_size):
            raise ValueError("prefill chunk metadata must match the batch size")
        torch = self.torch
        chunk_lengths = [len(chunk) for chunk in token_chunks]
        if any(length == 0 for length in chunk_lengths):
            raise ValueError("prefill chunks must contain at least one token")
        if prefill_backend not in ("hf", "triton_paged"):
            raise ValueError("prefill_backend must be 'hf' or 'triton_paged'")
        if prefill_backend == "triton_paged" and not self._paged_decode_enabled:
            raise RuntimeError(
                "Triton paged prefill requires a registered model adapter for this architecture"
            )
        old_lengths: list[int] = []
        tables: list[BlockTable | None] = []
        match_owned_tables: list[BlockTable] = []
        created_tables: list[BlockTable] = []
        for state, match in zip(states, prefix_matches):
            if state is not None and match is not None and match.block_table is not None:
                raise ValueError("a prefill chunk cannot use both state and prefix match")
            if state is not None:
                table = state.block_table
                old_length = state.token_count
            elif match is not None and match.block_table is not None:
                table = match.block_table
                old_length = match.matched_tokens
                match_owned_tables.append(table)
            else:
                table = None
                old_length = 0
            tables.append(table)
            old_lengths.append(old_length)

        try:
            # Fail before modifying any request table if the whole chunk batch
            # cannot reserve its required pages. The scheduler can then evict
            # cache-only references and retry this prefill from the full prompt.
            required_blocks = 0
            for table, old_length, chunk_length in zip(tables, old_lengths, chunk_lengths):
                new_length = old_length + chunk_length
                required = (new_length + self.kv_block_size - 1) // self.kv_block_size
                required_blocks += required - (0 if table is None else len(table.block_ids))
                if table is not None and old_length % self.kv_block_size:
                    last_block = table.block_ids[-1]
                    if self.kv_cache.manager.ref_count(last_block) > 1:
                        required_blocks += 1
            if required_blocks > self.kv_cache.manager.free_blocks:
                raise KVCacheOutOfBlocks(
                    f"prefill chunk needs {required_blocks} KV blocks, "
                    f"only {self.kv_cache.manager.free_blocks} are free"
                )

            max_past = max(old_lengths, default=0)
            width = max(chunk_lengths)
            if prefill_backend == "triton_paged":
                for row, chunk_length in enumerate(chunk_lengths):
                    table = tables[row]
                    if table is None:
                        table = self.kv_cache.allocate(chunk_length)
                        created_tables.append(table)
                        tables[row] = table
                    else:
                        self.kv_cache.append(table, chunk_length)
                page_table, _ = self.kv_cache.device_page_table(
                    [table for table in tables if table is not None]
                )
                self._paged_context = {
                    "mode": "prefill",
                    "tables": tables,
                    "old_lengths": old_lengths,
                    "query_lengths": chunk_lengths,
                    "page_table": page_table,
                    "old_lengths_tensor": torch.tensor(
                        old_lengths, dtype=torch.int32, device=self.device
                    ),
                    "query_lengths_tensor": torch.tensor(
                        chunk_lengths, dtype=torch.int32, device=self.device
                    ),
                    "chunk_index": chunk_index,
                }
                if kernel_checks is not None:
                    self._paged_context.update({
                        "kernel_checks": kernel_checks,
                        "kernel_atol": kernel_atol,
                        "kernel_rtol": kernel_rtol,
                    })

            past_key_values = None
            if max_past and prefill_backend == "hf":
                layer_caches = []
                for layer_idx in range(self.kv_cache.num_layers):
                    keys, values = [], []
                    for table, old_length in zip(tables, old_lengths):
                        if table is None:
                            shape = (1, self.kv_cache.num_kv_heads, 0,
                                     self.kv_cache.head_dim)
                            key = torch.empty(shape, dtype=self.model.dtype,
                                              device=self.device)
                            value = torch.empty_like(key)
                        else:
                            key, value = self.kv_cache.gather_layer(table, layer_idx)
                        missing = max_past - old_length
                        if missing:
                            key = torch.nn.functional.pad(key, (0, 0, missing, 0))
                            value = torch.nn.functional.pad(value, (0, 0, missing, 0))
                        keys.append(key)
                        values.append(value)
                    layer_caches.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
                past_key_values = self._cache_from_legacy(layer_caches)

            input_ids = torch.full(
                (batch_size, width), self.tokenizer.pad_token_id,
                dtype=torch.long, device=self.device,
            )
            attention_mask = None
            if prefill_backend == "hf":
                attention_mask = torch.zeros(
                    (batch_size, max_past + width), dtype=torch.long, device=self.device
                )
            position_ids = torch.zeros((batch_size, width), dtype=torch.long,
                                       device=self.device)
            for row, (chunk, chunk_length, old_length) in enumerate(
                zip(token_chunks, chunk_lengths, old_lengths)
            ):
                input_ids[row, :chunk_length] = torch.tensor(
                    chunk, dtype=torch.long, device=self.device
                )
                if old_length and prefill_backend == "hf":
                    attention_mask[row, max_past - old_length:max_past] = 1
                if prefill_backend == "hf":
                    attention_mask[row, max_past:max_past + chunk_length] = 1
                position_ids[row, :chunk_length] = torch.arange(
                    old_length, old_length + chunk_length,
                    dtype=torch.long, device=self.device,
                )

            try:
                with torch.inference_mode():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=(prefill_backend == "hf"),
                        return_dict=True,
                    )
            finally:
                self._paged_context = None
            logits = self._last_valid_logits(output.logits, chunk_lengths)
            if logits_capture is not None:
                logits_capture.extend(logits.detach().float().cpu().unbind(dim=0))
            next_tokens = logits.argmax(dim=-1).tolist()
            full_cache = (
                self._legacy_cache(output.past_key_values)
                if prefill_backend == "hf" else None
            )

            results = []
            for row, (chunk_length, old_length, token, final) in enumerate(
                zip(chunk_lengths, old_lengths, next_tokens, final_chunks)
            ):
                table = tables[row]
                if prefill_backend == "hf":
                    if table is None:
                        table = self.kv_cache.allocate(chunk_length)
                        created_tables.append(table)
                        tables[row] = table
                    else:
                        self.kv_cache.append(table, chunk_length)
                    for layer_idx, (key, value) in enumerate(full_cache):
                        key_tokens = key[
                            row, :, max_past:max_past + chunk_length, :
                        ].permute(1, 0, 2).contiguous()
                        value_tokens = value[
                            row, :, max_past:max_past + chunk_length, :
                        ].permute(1, 0, 2).contiguous()
                        self.kv_cache.write_layer(
                            table, layer_idx, old_length, key_tokens, value_tokens
                        )
                if table is None:
                    raise RuntimeError("paged prefill did not allocate a KV block table")
                results.append((int(token) if final else None, _SequenceState(table)))
            return results
        except Exception:
            self._paged_context = None
            released: set[int] = set()
            for table in match_owned_tables + created_tables:
                if not table._released and id(table) not in released:
                    self.kv_cache.release(table)
                    released.add(id(table))
            raise

    def prefill(self, prompts: Sequence[Sequence[int]], *,
                prefix_matches: Sequence[PrefixMatch] | None = None,
                logits_capture: list[Any] | None = None) -> list[tuple[int, object]]:
        if prefix_matches is not None:
            if len(prefix_matches) != len(prompts):
                raise ValueError("prefix match count must equal prompt batch size")
            if any(match.block_table is not None for match in prefix_matches):
                return self._prefill_with_prefixes(prompts, prefix_matches)
        if not prompts:
            return []
        torch = self.torch
        lengths = [len(prompt) for prompt in prompts]
        if any(length == 0 for length in lengths):
            raise ValueError("empty tokenized prompts are not supported by this backend")
        width = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=self.device)
            mask[row, :len(prompt)] = 1
        positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)
        with torch.inference_mode():
            output = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                use_cache=True, return_dict=True)
        logits = self._last_valid_logits(output.logits, lengths)
        if logits_capture is not None:
            logits_capture.append(logits.detach().float().cpu())
        next_tokens = logits.argmax(dim=-1).tolist()
        full_cache = self._legacy_cache(output.past_key_values)
        results = []
        allocated: list[BlockTable] = []
        try:
            for row, length in enumerate(lengths):
                table = self.kv_cache.allocate(length)
                allocated.append(table)
                for layer_idx, (key, value) in enumerate(full_cache):
                    key_tokens = key[row, :, :length, :].permute(1, 0, 2).contiguous()
                    value_tokens = value[row, :, :length, :].permute(1, 0, 2).contiguous()
                    self.kv_cache.write_layer(table, layer_idx, 0, key_tokens, value_tokens)
                results.append((int(next_tokens[row]), _SequenceState(table)))
        except Exception:
            for table in allocated:
                self.kv_cache.release(table)
            raise
        return results

    def _prefill_with_prefixes(
        self,
        suffixes: Sequence[Sequence[int]],
        prefix_matches: Sequence[PrefixMatch],
    ) -> list[tuple[int, object]]:
        """Prefill only suffix tokens, attending to cached prefix KV pages."""
        torch = self.torch
        if not suffixes:
            for match in prefix_matches:
                if match.block_table is not None and not match.block_table._released:
                    self.kv_cache.release(match.block_table)
            return []
        suffix_lengths = [len(suffix) for suffix in suffixes]
        prefix_lengths = [match.matched_tokens for match in prefix_matches]
        prefix_tables = [match.block_table for match in prefix_matches]
        batch_size = len(suffixes)
        tables: list[BlockTable] = [table for table in prefix_tables if table is not None]
        created_tables: list[BlockTable] = []

        try:
            if len(prefix_matches) != batch_size:
                raise ValueError("prefix match count must equal prompt batch size")
            if any(length == 0 for length in suffix_lengths):
                raise ValueError("prefix cache must leave at least one token for prefill")
            if any((table is None) != (length == 0)
                   for table, length in zip(prefix_tables, prefix_lengths)):
                raise ValueError("prefix match table and token count are inconsistent")
            if any(table is not None and table.token_count != length
                   for table, length in zip(prefix_tables, prefix_lengths)):
                raise ValueError("prefix match table length does not match its token count")
            max_prefix = max(prefix_lengths, default=0)
            width = max(suffix_lengths)
            past_key_values = None
            if max_prefix:
                layer_caches = []
                for layer_idx in range(self.kv_cache.num_layers):
                    keys, values = [], []
                    for table, prefix_len in zip(prefix_tables, prefix_lengths):
                        if table is None:
                            empty_shape = (1, self.kv_cache.num_kv_heads, 0,
                                           self.kv_cache.head_dim)
                            key = torch.empty(empty_shape, dtype=self.model.dtype,
                                              device=self.device)
                            value = torch.empty_like(key)
                        else:
                            key, value = self.kv_cache.gather_layer(table, layer_idx)
                        missing = max_prefix - prefix_len
                        if missing:
                            # Left-pad cached rows so each row's valid prefix
                            # stays adjacent to its suffix in the attention mask.
                            key = torch.nn.functional.pad(key, (0, 0, missing, 0))
                            value = torch.nn.functional.pad(value, (0, 0, missing, 0))
                        keys.append(key)
                        values.append(value)
                    layer_caches.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
                past_key_values = self._cache_from_legacy(layer_caches)

            ids = torch.full((batch_size, width), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=self.device)
            attention_mask = torch.zeros(
                (batch_size, max_prefix + width), dtype=torch.long, device=self.device
            )
            position_ids = torch.zeros((batch_size, width), dtype=torch.long,
                                       device=self.device)
            for row, (suffix, suffix_len, prefix_len) in enumerate(
                zip(suffixes, suffix_lengths, prefix_lengths)
            ):
                ids[row, :suffix_len] = torch.tensor(
                    suffix, dtype=torch.long, device=self.device
                )
                if prefix_len:
                    attention_mask[row, max_prefix - prefix_len:max_prefix] = 1
                attention_mask[row, max_prefix:max_prefix + suffix_len] = 1
                position_ids[row, :suffix_len] = torch.arange(
                    prefix_len, prefix_len + suffix_len,
                    dtype=torch.long, device=self.device,
                )

            with torch.inference_mode():
                output = self.model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            logits = self._last_valid_logits(output.logits, suffix_lengths)
            next_tokens = logits.argmax(dim=-1).tolist()
            full_cache = self._legacy_cache(output.past_key_values)

            results = []
            for row, (suffix_len, prefix_len, token) in enumerate(
                zip(suffix_lengths, prefix_lengths, next_tokens)
            ):
                table = prefix_tables[row]
                if table is None:
                    table = self.kv_cache.allocate(suffix_len)
                    created_tables.append(table)
                    write_start = 0
                else:
                    write_start = prefix_len
                    self.kv_cache.append(table, suffix_len)
                for layer_idx, (key, value) in enumerate(full_cache):
                    key_tokens = key[
                        row, :, max_prefix:max_prefix + suffix_len, :
                    ].permute(1, 0, 2).contiguous()
                    value_tokens = value[
                        row, :, max_prefix:max_prefix + suffix_len, :
                    ].permute(1, 0, 2).contiguous()
                    self.kv_cache.write_layer(
                        table, layer_idx, write_start, key_tokens, value_tokens
                    )
                results.append((int(token), _SequenceState(table)))
            return results
        except Exception:
            released: set[int] = set()
            for table in tables + created_tables:
                if not table._released and id(table) not in released:
                    self.kv_cache.release(table)
                    released.add(id(table))
            raise

    def decode_step(self, tokens: Sequence[int], states: Sequence[object]) -> list[tuple[int, object, bool]]:
        if not tokens:
            return []
        torch = self.torch
        seq_states = [state for state in states]
        lengths = [state.token_count for state in seq_states]
        if self._paged_decode_enabled:
            return self._decode_step_paged(tokens, seq_states, lengths)
        width = max(lengths)
        batched_layers = []
        for layer_index in range(self.kv_cache.num_layers):
            keys, values = [], []
            for state in seq_states:
                key, value = self.kv_cache.gather_layer(state.block_table, layer_index)
                missing = width - state.token_count
                if missing:
                    key = torch.nn.functional.pad(key, (0, 0, 0, missing))
                    value = torch.nn.functional.pad(value, (0, 0, 0, missing))
                keys.append(key)
                values.append(value)
            batched_layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        # Right-pad shorter KV histories. Mask padding positions and assign each
        # new token its logical position, independent of the padded batch width.
        mask = torch.zeros((len(tokens), width + 1), dtype=torch.long, device=self.device)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
            mask[row, width] = 1
        positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)[:, -1:]
        past_key_values = self._cache_from_legacy(batched_layers)
        with torch.inference_mode():
            output = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                past_key_values=past_key_values, use_cache=True, return_dict=True)
        next_tokens = output.logits[:, -1, :].argmax(dim=-1).tolist()
        cache = self._legacy_cache(output.past_key_values)
        results = []
        for row, (token, state) in enumerate(zip(next_tokens, seq_states)):
            old_length = state.token_count
            self.kv_cache.append(state.block_table, 1)
            for layer_idx, (key, value) in enumerate(cache):
                key_token = key[row, :, width:width + 1, :].permute(1, 0, 2).contiguous()
                value_token = value[row, :, width:width + 1, :].permute(1, 0, 2).contiguous()
                self.kv_cache.write_layer(state.block_table, layer_idx, old_length,
                                          key_token, value_token)
            token = int(token)
            results.append((token, state,
                            self.eos_token_id is not None and token == self.eos_token_id))
        return results

    def _decode_step_paged(self, tokens: Sequence[int], states: Sequence[_SequenceState],
                           old_lengths: list[int]) -> list[tuple[int, object, bool]]:
        """Run adapter-backed decode over paged KV without dense KV gathering."""
        torch = self.torch
        for state in states:
            self.kv_cache.append(state.block_table, 1)

        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        positions = torch.tensor(old_lengths, dtype=torch.long, device=self.device).unsqueeze(1)
        self._paged_context = {
            "tables": [state.block_table for state in states],
            "old_lengths": old_lengths,
        }
        try:
            self._paged_context["page_table"], self._paged_context["lengths"] = (
                self.kv_cache.device_page_table(self._paged_context["tables"])
            )
            with torch.inference_mode():
                output = self.model(
                    input_ids=ids,
                    attention_mask=None,
                    position_ids=positions,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            self._paged_context = None

        next_tokens = output.logits[:, -1, :].argmax(dim=-1).tolist()
        return [
            (int(token), state, self.eos_token_id is not None and int(token) == self.eos_token_id)
            for token, state in zip(next_tokens, states)
        ]
