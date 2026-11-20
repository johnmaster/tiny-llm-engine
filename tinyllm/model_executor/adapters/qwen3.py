"""Transformers Qwen3 attention implementation for paged KV execution."""
from __future__ import annotations

from typing import Any

from .base import ModelAdapter


class Qwen3Adapter(ModelAdapter):
    architectures = ("Qwen3ForCausalLM",)
    model_types = ("qwen3",)
    name = "qwen3"

    def find_attention_modules(self, model: Any) -> list[tuple[str, Any]]:
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(".self_attn")
            and module.__class__.__name__ == "Qwen3Attention"
            and hasattr(module, "config")
        ]
        if not modules:
            raise RuntimeError(
                "Qwen3 config was detected, but no compatible Qwen3Attention layers were found"
            )
        return modules

    def project_qkv(self, backend: Any, module: Any, hidden_states: Any,
                    kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("Qwen3 did not provide rotary position embeddings")

        batch_size, query_length, _ = hidden_states.shape
        hidden_shape = (batch_size, query_length, -1, module.head_dim)
        query = module.q_norm(
            module.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key = module.k_norm(
            module.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
