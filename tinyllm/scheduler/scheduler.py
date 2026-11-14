"""FCFS continuous-batching scheduler; model execution stays in backends."""
from __future__ import annotations
from collections import deque
from threading import RLock
from typing import Any

from ..core import Request
from ..config import EngineConfig
from ..model_executor.backend import ModelBackend


class Scheduler:
    def __init__(self, backend: ModelBackend, config: EngineConfig | None = None):
        self.backend = backend
        self.config = config or EngineConfig()
        self.max_num_seqs = self.config.max_num_seqs
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._lock = RLock()

    def add_request(self, prompt: str, max_tokens: int = 32) -> Request:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        prompt_tokens = self.backend.encode(prompt)
        if len(prompt_tokens) + max_tokens > self.config.max_model_len:
            raise ValueError("prompt plus max_tokens exceeds max_model_len")
        req = Request(prompt, max_tokens, prompt_tokens=prompt_tokens)
        with self._lock:
            self._waiting.append(req)
        return req

    def step(self) -> None:
        """Admit waiting sequences, prefill newcomers, and decode active sequences."""
        with self._lock:
            cancelled = [r for r in self._running if r.status == "cancelled"]
            for req in cancelled:
                self._release_request_state(req)
                req.ready.set()
            while self._waiting and len(self._running) < self.max_num_seqs:
                req = self._waiting.popleft()
                req.status = "running"
                self._running.append(req)
            fresh = [r for r in self._running if r.status == "running" and r.state is None]
            active = [r for r in self._running if r.status == "running" and r.state is not None]

        try:
            # New requests can join the batch while older requests are decoding.
            if fresh:
                outputs = self.backend.prefill([r.prompt_tokens for r in fresh])
                if len(outputs) != len(fresh):
                    raise RuntimeError("backend prefill returned the wrong batch size")
                for req, (token, state) in zip(fresh, outputs):
                    with self._lock:
                        req.state = state
                        if req.status == "cancelled":
                            self._release_request_state(req)
                            req.ready.set()
                        else:
                            req.output_tokens.append(token)
                            if self._done(req):
                                self._finish(req)
            if active:
                outputs = self.backend.decode_step(
                    [r.output_tokens[-1] for r in active], [r.state for r in active]
                )
                if len(outputs) != len(active):
                    raise RuntimeError("backend decode returned the wrong batch size")
                for req, (token, state, backend_done) in zip(active, outputs):
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
            with self._lock:
                for req in fresh + active:
                    if req.status in ("running", "cancelled"):
                        try:
                            self._release_request_state(req)
                            if req.status == "running":
                                req.error = str(exc)
                        except Exception as cleanup_exc:
                            req.error = f"{exc}; KV cache cleanup failed: {cleanup_exc}"
                        if req.status == "running":
                            req.status = "failed"
                        req.ready.set()

        with self._lock:
            self._running = [r for r in self._running if r.status == "running"]

    def _done(self, req: Request) -> bool:
        eos = getattr(self.backend, "eos_token_id", None)
        return len(req.output_tokens) >= req.max_tokens or (
            eos is not None and bool(req.output_tokens) and req.output_tokens[-1] == eos
        )

    def _finish(self, req: Request) -> None:
        self._release_request_state(req)
        req.status = "finished"
        req.ready.set()

    def _release_request_state(self, req: Request) -> None:
        if req.state is None:
            return
        release_state = getattr(self.backend, "release_state", None)
        if release_state is not None:
            release_state(req.state)
        req.state = None

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
            return {"waiting": len(self._waiting), "running": len(self._running),
                    "max_num_seqs": self.max_num_seqs,
                    "max_model_len": self.config.max_model_len}
