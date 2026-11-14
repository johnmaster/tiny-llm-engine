"""Public engine API and scheduler worker lifecycle."""
from __future__ import annotations
from threading import Event, Thread

from ..core import Request
from ..config import EngineConfig
from ..model_executor.backend import ModelBackend
from ..scheduler import Scheduler


class LLMEngine:
    def __init__(self, backend: ModelBackend, config: EngineConfig | None = None, *, max_num_seqs: int | None = None):
        self.backend = backend
        if config is not None and max_num_seqs is not None:
            raise ValueError("pass either config or max_num_seqs, not both")
        if config is None:
            config = EngineConfig(max_num_seqs=8 if max_num_seqs is None else max_num_seqs)
        self.config = config
        self.scheduler = Scheduler(backend, config=config)
        self._stop = Event()
        self._thread: Thread | None = None

    def submit(self, prompt: str, max_tokens: int = 32) -> Request:
        return self.scheduler.add_request(prompt, max_tokens)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="tinyllm-scheduler")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.scheduler.step()
            self._stop.wait(0.001)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def generate(self, prompt: str, max_tokens: int = 32, timeout: float = 300) -> Request:
        req = self.submit(prompt, max_tokens)
        self.start()
        if not req.ready.wait(timeout):
            self.scheduler.cancel(req)
            raise TimeoutError(f"request {req.request_id} timed out")
        if req.status == "failed":
            raise RuntimeError(req.error)
        return req


# Backwards-friendly short name for the original public API.
Engine = LLMEngine
