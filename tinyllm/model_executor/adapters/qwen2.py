"""Transformers Qwen2/Qwen2.5 attention adapter for paged KV execution."""
from __future__ import annotations

from typing import Any

from .base import ModelAdapter


class Qwen2Adapter(ModelAdapter):
    architectures = ("Qwen2ForCausalLM",)
    model_types = ("qwen2",)
    name = "qwen2"

    def find_attention_modules(self, model: Any) -> list[tuple[str, Any]]:
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(".self_attn")
            and module.__class__.__name__ == "Qwen2Attention"
            and hasattr(module, "config")
        ]
        if not modules:
            raise RuntimeError(
                "Qwen2 config was detected, but no compatible Qwen2Attention layers were found"
            )
        return modules

    def project_qkv(self, backend: Any, module: Any, hidden_states: Any,
                    kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        batch_size, query_length, _ = hidden_states.shape
        hidden_shape = (batch_size, query_length, -1, module.head_dim)
        query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            position_ids = kwargs.get("position_ids")
            rotary_emb = getattr(module, "rotary_emb", None)
            if position_ids is None or rotary_emb is None:
                raise RuntimeError(
                    "Qwen2 requires position_embeddings or compatible position_ids/rotary_emb"
                )
            position_embeddings = rotary_emb(value, position_ids)
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise RuntimeError("Qwen2 position_embeddings must contain (cos, sin)")
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        return query, key, value

    def layer_index(self, module: Any) -> int:
        return int(module.layer_idx)

    def attention_scale(self, module: Any) -> float:
        return float(module.scaling)

    def project_output(self, module: Any, attention_output: Any) -> Any:
        return module.o_proj(attention_output)

    def format_attention_output(self, module: Any, attention_output: Any) -> Any:
        return attention_output, None
