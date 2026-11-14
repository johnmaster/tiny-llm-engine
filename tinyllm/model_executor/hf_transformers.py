"""A small Hugging Face causal-LM backend with batched prefill and decode.

This is a correctness-first backend for learning and experimentation. It uses
greedy sampling and padded KV batches; it is not yet optimized like vLLM.
Install the dependencies listed in ``requirements-gpu.txt`` in a CUDA-enabled
PyTorch environment before running it on a GPU machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class _SequenceState:
    # Legacy HF cache: ((key, value), ...), each tensor shaped [1, heads, seq, dim].
    past_key_values: tuple[tuple[Any, Any], ...]
    token_count: int


class HuggingFaceBackend:
    def __init__(self, model_name_or_path: str, *, dtype: str = "auto", device: str = "cuda"):
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
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

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
        for row, length in enumerate(lengths):
            seq_cache = tuple((key[row:row + 1, ..., :length, :].contiguous(),
                               value[row:row + 1, ..., :length, :].contiguous())
                              for key, value in full_cache)
            results.append((int(next_tokens[row]), _SequenceState(seq_cache, length)))
        return results

    def decode_step(self, tokens: Sequence[int], states: Sequence[object]) -> list[tuple[int, object, bool]]:
        if not tokens:
            return []
        torch = self.torch
        seq_states = [state for state in states]
        lengths = [state.token_count for state in seq_states]
        width = max(lengths)
        batched_layers = []
        for layer_index in range(len(seq_states[0].past_key_values)):
            keys, values = [], []
            for state in seq_states:
                key, value = state.past_key_values[layer_index]
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
            new_length = state.token_count + 1
            # The current token is appended after the batch's padded KV width;
            # for a shorter sequence, slicing :new_length would select padding
            # and lose this token's KV entry.
            seq_cache = tuple((torch.cat((old_key, key[row:row + 1, ..., width:width + 1, :]), dim=-2).contiguous(),
                               torch.cat((old_value, value[row:row + 1, ..., width:width + 1, :]), dim=-2).contiguous())
                              for (old_key, old_value), (key, value) in zip(state.past_key_values, cache))
            token = int(token)
            results.append((token, _SequenceState(seq_cache, new_length),
                            self.eos_token_id is not None and token == self.eos_token_id))
        return results
