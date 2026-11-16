"""A small Hugging Face causal-LM backend with batched prefill and decode.

This is a correctness-first backend for learning and experimentation. It uses
greedy sampling, the selected Hugging Face attention backend for prefill, and a
Triton kernel that reads Qwen3 decode KV directly through the page table. It is
not optimized like vLLM.
Install the dependencies listed in ``requirements-gpu.txt`` in a CUDA-enabled
PyTorch environment before running it on a GPU machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

from ..kv_cache import BlockTable, PagedKVTensorStore


@dataclass
class _SequenceState:
    block_table: BlockTable

    @property
    def token_count(self) -> int:
        return self.block_table.token_count


class HuggingFaceBackend:
    def __init__(self, model_name_or_path: str, *, dtype: str = "auto", device: str = "cuda",
                 attention_backend: str = "sdpa", max_num_seqs: int = 8,
                 max_model_len: int = 8192, kv_block_size: int = 16):
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
        self._attention_calls: dict[str, int] = {}
        self._attention_layers: list[tuple[str, str]] = []
        self._attention_modules: list[tuple[str, Any]] = []
        for name, module in self.model.named_modules():
            if not name.endswith(".self_attn") or not hasattr(module, "config"):
                continue
            configured_backend = getattr(module.config, "_attn_implementation", None)
            if configured_backend is None:
                continue
            self._attention_layers.append((name, configured_backend))
            self._attention_modules.append((name, module))
            self._attention_calls[name] = 0

            def count_attention_call(_module: Any, _inputs: Any, _output: Any,
                                     layer_name: str = name) -> None:
                self._attention_calls[layer_name] += 1

            module.register_forward_hook(count_attention_call)

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
        config = getattr(self.model.config, "text_config", None) or self.model.config
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
        self._paged_decode_enabled = (
            bool(self._attention_modules)
            and all(module.__class__.__name__ == "Qwen3Attention"
                    for _, module in self._attention_modules)
        )
        if self._paged_decode_enabled:
            self._paged_attention_calls = [0] * len(self._attention_modules)
            self._install_qwen3_paged_attention()
        else:
            self._paged_attention_calls = []
        if getattr(config, "model_type", None) == "qwen3" and not self._paged_decode_enabled:
            raise RuntimeError("This Transformers Qwen3 implementation cannot be patched for paged decode")

    def attention_summary(self) -> str | None:
        """Summarize configured backends and layers actually called so far."""
        if not self._attention_layers:
            return None
        total = len(self._attention_layers)
        invoked = sum(self._attention_calls[name] > 0 for name, _ in self._attention_layers)
        if self._paged_decode_enabled:
            paged_layers = sum(count > 0 for count in self._paged_attention_calls)
            decode_mode = f"Triton paged decode {paged_layers}/{len(self._paged_attention_calls)} layers"
        else:
            decode_mode = "HF dense cache"
        return (f"Attention layers executed: {invoked}/{total}; "
                f"prefill backend: {self.attention_backend}; decode: {decode_mode}")

    def kv_cache_summary(self) -> str:
        stats = self.kv_cache.manager.snapshot()
        return (f"Paged KV storage: block_size={self.kv_block_size}; "
                f"active_blocks={self.kv_cache.materialized_blocks}/"
                f"{stats['total_blocks']}")

    def release_state(self, state: object) -> None:
        if isinstance(state, _SequenceState):
            self.kv_cache.release(state.block_table)

    def _install_qwen3_paged_attention(self) -> None:
        for _, module in self._attention_modules:
            original_forward = module.forward

            def paged_forward(attention_module: Any, *args: Any,
                              _original: Any = original_forward, **kwargs: Any) -> Any:
                context = self._paged_context
                if context is None:
                    return _original(*args, **kwargs)
                return self._qwen3_paged_attention_forward(attention_module, context,
                                                           *args, **kwargs)

            module.forward = MethodType(paged_forward, module)

    def _qwen3_paged_attention_forward(self, module: Any, context: dict[str, Any],
                                       hidden_states: Any, **kwargs: Any) -> tuple[Any, None]:
        """Qwen3 single-token attention over the request's physical KV pages."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        torch = self.torch
        batch_size, query_length, _ = hidden_states.shape
        if query_length != 1:
            raise RuntimeError("paged decode currently supports one query token per sequence")
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("Qwen3 did not provide rotary position embeddings")
        hidden_shape = (batch_size, query_length, -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_shape = (batch_size, query_length, -1, module.head_dim)
        key = module.k_norm(module.k_proj(hidden_states).view(key_shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(key_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)

        layer_idx = int(module.layer_idx)
        for row, (table, old_length) in enumerate(zip(context["tables"], context["old_lengths"])):
            self.kv_cache.write_layer(
                table, layer_idx, old_length,
                key[row, :, 0, :].unsqueeze(0), value[row, :, 0, :].unsqueeze(0),
            )
        attention_output = self.kv_cache.paged_attention(
            layer_idx, query[:, :, 0, :], context["page_table"],
            context["lengths"], float(module.scaling),
        )
        self._paged_attention_calls[layer_idx] += 1
        attention_output = attention_output.unsqueeze(1).contiguous()
        attention_output = attention_output.reshape(batch_size, query_length, -1).contiguous()
        return module.o_proj(attention_output), None

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

    def prefill(self, prompts: Sequence[Sequence[int]]) -> list[tuple[int, object]]:
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
        """Run Qwen3 decode with direct block-table attention, without dense KV gather."""
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
