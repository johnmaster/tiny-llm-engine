"""FCFS continuous-batching scheduler with token-budgeted chunked prefill."""
from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Any

from ..config import EngineConfig
from ..core import Request
from ..kv_cache import ExactPrefixCache, KVCacheOutOfBlocks
from ..model_executor.backend import ModelBackend


class Scheduler:
    def __init__(self, backend: ModelBackend, config: EngineConfig | None = None):
        self.backend = backend
        self.config = config or EngineConfig()
        self.max_num_seqs = self.config.max_num_seqs
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._lock = RLock()
        self._prefill_rotation = 0
        manager = getattr(getattr(backend, "kv_cache", None), "manager", None)
        self.prefix_cache = (
            ExactPrefixCache(manager, self.config.max_prefix_cache_blocks)
            if self.config.enable_prefix_caching and manager is not None else None
        )

    def add_request(self, prompt: str, max_tokens: int = 32) -> Request:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        prompt_tokens = self.backend.encode(prompt)
        if not prompt_tokens:
            raise ValueError("prompt must tokenize to at least one token")
        if len(prompt_tokens) + max_tokens > self.config.max_model_len:
            raise ValueError("prompt plus max_tokens exceeds max_model_len")
        req = Request(prompt, max_tokens, prompt_tokens=prompt_tokens)
        with self._lock:
            self._waiting.append(req)
        return req

    def step(self) -> None:
        """Decode active sequences, then advance a token-budgeted prefill batch."""
        with self._lock:
            cancelled = [r for r in self._running if r.status == "cancelled"]
            for req in cancelled:
                self._release_request_state(req)
                req.ready.set()
            while self._waiting and len(self._running) < self.max_num_seqs:
                req = self._waiting.popleft()
                req.status = "running"
                self._running.append(req)
            decoding = [
                r for r in self._running
                if r.status == "running"
                and r.state is not None
                and r.prefill_cursor >= len(r.prompt_tokens)
                and r.output_tokens
            ]
            prefilling = [
                r for r in self._running
                if r.status == "running" and r.prefill_cursor < len(r.prompt_tokens)
            ]

        # A match owns block references until the first chunk consumes it.
        if self.prefix_cache is not None:
            try:
                for req in prefilling:
                    if not req.prefix_match_checked:
                        match = self.prefix_cache.match(req.prompt_tokens)
                        req.prefix_match = match
                        req.prefix_cached_tokens = match.matched_tokens
                        req.prefill_cursor = match.matched_tokens
                        req.prefix_match_checked = True
            except Exception as exc:
                self._fail_requests(prefilling, exc)

        # Decode receives priority so long prompts cannot pause active output.
        if decoding:
            try:
                outputs = self.backend.decode_step(
                    [r.output_tokens[-1] for r in decoding], [r.state for r in decoding]
                )
                if len(outputs) != len(decoding):
                    raise RuntimeError("backend decode returned the wrong batch size")
                for req, (token, state, backend_done) in zip(decoding, outputs):
                    with self._lock:
                        req.state = state
                        if req.status == "cancelled":
                            self._release_request_state(req)
                            req.ready.set()
                        else:
                            req.output_tokens.append(token)
                            if backend_done or self._done(req):
                                self._finish(req)
            except Exception as exc:
                self._fail_requests(decoding, exc)

        remaining_tokens = max(
            0, self.config.max_num_batched_tokens - len(decoding)
        )
        chunks = self._select_prefill_chunks(prefilling, remaining_tokens)
        if chunks:
            requests = [item[0] for item in chunks]
            token_chunks = [item[1] for item in chunks]
            final_chunks = [item[2] for item in chunks]
            states = [req.state for req in requests]
            matches = [req.prefix_match for req in requests]
            try:
                outputs = self.backend.prefill_chunk(
                    token_chunks, states, matches, final_chunks,
                    prefill_backend=self.config.prefill_backend,
                )
                if len(outputs) != len(requests):
                    raise RuntimeError("backend prefill_chunk returned the wrong batch size")
                for req, chunk, (token, state) in zip(requests, token_chunks, outputs):
                    with self._lock:
                        req.state = state
                        req.prefill_cursor += len(chunk)
                        req.prefill_chunks += 1
                        req.prefix_match = None
                        if req.status == "cancelled":
                            self._release_request_state(req)
                            req.ready.set()
                        elif token is not None:
                            if self.prefix_cache is not None:
                                self.prefix_cache.insert(req.prompt_tokens, state.block_table)
                            req.output_tokens.append(token)
                            if self._done(req):
                                self._finish(req)
            except KVCacheOutOfBlocks as exc:
                if self.prefix_cache is not None and self.prefix_cache.snapshot()["cached_blocks"]:
                    self.prefix_cache.clear()
                    # Drop cache-only match references, then retry unfinished
                    # prompts from their beginning on later scheduler steps.
                    for req in prefilling:
                        if req.state is None:
                            self._release_prefix_match(req)
                            req.prefix_cached_tokens = 0
                            req.prefill_cursor = 0
                            req.prefix_match_checked = True
                else:
                    self._fail_requests(requests, exc)
            except Exception as exc:
                self._fail_requests(requests, exc)

        with self._lock:
            self._running = [r for r in self._running if r.status == "running"]

    def _select_prefill_chunks(
        self, requests: list[Request], token_budget: int
    ) -> list[tuple[Request, list[int], bool]]:
        requests = [r for r in requests
                    if r.status == "running" and r.prefill_cursor < len(r.prompt_tokens)]
        if not requests or token_budget <= 0:
            return []
        start = self._prefill_rotation % len(requests)
        ordered = requests[start:] + requests[:start]
        selected: list[tuple[Request, list[int], bool]] = []
        remaining = token_budget
        for req in ordered:
            pending = len(req.prompt_tokens) - req.prefill_cursor
            count = min(self.config.prefill_chunk_size, pending, remaining)
            if count <= 0:
                break
            end = req.prefill_cursor + count
            selected.append((req, req.prompt_tokens[req.prefill_cursor:end],
                             end == len(req.prompt_tokens)))
            remaining -= count
        if selected:
            self._prefill_rotation = (start + 1) % len(requests)
        return selected

    def _done(self, req: Request) -> bool:
        eos = getattr(self.backend, "eos_token_id", None)
        return len(req.output_tokens) >= req.max_tokens or (
            eos is not None and bool(req.output_tokens) and req.output_tokens[-1] == eos
        )

    def _finish(self, req: Request) -> None:
        self._release_request_state(req)
        req.status = "finished"
        req.ready.set()

    def _release_prefix_match(self, req: Request) -> None:
        match = req.prefix_match
        table = getattr(match, "block_table", None)
        if table is not None and not table._released:
            cache = getattr(getattr(self.backend, "kv_cache", None), "manager", None)
            if cache is not None:
                cache.release(table)
        req.prefix_match = None

    def _release_request_state(self, req: Request) -> None:
        state_table = getattr(req.state, "block_table", None)
        match_table = getattr(req.prefix_match, "block_table", None)
        if match_table is not None and match_table is not state_table:
            self._release_prefix_match(req)
        else:
            req.prefix_match = None
        if req.state is None:
            return
        if state_table is not None:
            req.released_block_ids = list(state_table.block_ids)
        release_state = getattr(self.backend, "release_state", None)
        if release_state is not None:
            release_state(req.state)
        req.state = None

    def _fail_requests(self, requests: list[Request], error: Exception) -> None:
        with self._lock:
            for req in requests:
                if req.status not in ("running", "cancelled"):
                    continue
                try:
                    self._release_request_state(req)
                except Exception as cleanup_exc:
                    req.error = f"{error}; KV cache cleanup failed: {cleanup_exc}"
                else:
                    req.error = str(error)
                if req.status == "running":
                    req.status = "failed"
                req.ready.set()

    def cancel(self, req: Request) -> None:
        """Cancel a request; running model work is cleaned up by the next step."""
        with self._lock:
            if req.status in ("finished", "failed", "cancelled"):
                return
            if req.status == "waiting":
                try:
                    self._waiting.remove(req)
                except ValueError:
                    pass
                req.status = "cancelled"
                req.ready.set()
                return
            req.status = "cancelled"
            req.ready.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            stats = {
                "waiting": len(self._waiting),
                "running": len(self._running),
                "max_num_seqs": self.max_num_seqs,
                "max_model_len": self.config.max_model_len,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "prefill_chunk_size": self.config.prefill_chunk_size,
                "prefill_backend": self.config.prefill_backend,
            }
            if self.prefix_cache is not None:
                stats["prefix_cache"] = self.prefix_cache.snapshot()
            return stats
