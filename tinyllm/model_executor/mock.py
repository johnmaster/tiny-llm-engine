"""Deterministic backend for exercising the engine without a model or GPU."""
from __future__ import annotations
from typing import Sequence


class MockBackend:
    eos_token_id = None

    def encode(self, prompt: str) -> list[int]:
        return list(prompt.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def prefill(self, prompts: Sequence[Sequence[int]]) -> list[tuple[int, object]]:
        return [(ord("O"), 1) for _ in prompts]

    def decode_step(self, tokens: Sequence[int], states: Sequence[object]) -> list[tuple[int, object, bool]]:
        response = "K, this is the tiny-llm mock backend."
        results = []
        for _token, state in zip(tokens, states):
            index = int(state)
            if index >= len(response):
                results.append((0, index, True))
            else:
                results.append((ord(response[index]), index + 1, index + 1 >= len(response)))
        return results
