"""Public engine API and scheduler worker lifecycle."""
from __future__ import annotations
from threading import Event, RLock, Thread, current_thread

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
        self._scheduler = Scheduler(backend, config=config)
        self._stop = Event()
        self._shutdown_complete = Event()
        self._lifecycle_lock = RLock()
        self._closed = False
        self._thread: Thread | None = None

    def submit(self, prompt: str, max_tokens: int = 32) -> Request:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("cannot submit a request after engine shutdown")
            req = self._scheduler.add_request(prompt, max_tokens)
            self.start()
            return req

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("cannot start an engine after shutdown")
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(target=self._loop, daemon=True, name="tinyllm-scheduler")
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._scheduler.step()
            self._stop.wait(0.001)

    def wait(self, req: Request, timeout: float | None = None) -> bool:
        """Wait until a submitted request reaches a terminal state."""
        return req.ready.wait(timeout)

    def cancel(self, req: Request) -> None:
        """Cancel a request submitted to this engine."""
        self._scheduler.cancel(req)

    def stats(self) -> dict[str, object]:
        """Return a snapshot of scheduler and cache counters."""
        return self._scheduler.stats()

    def clear_prefix_cache(self) -> None:
        """Clear cached KV prefixes after the current scheduler step finishes."""
        self._scheduler.clear_prefix_cache()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is current_thread():
                raise RuntimeError("shutdown cannot be called from the scheduler worker thread")
            first_shutdown = not self._closed
            if first_shutdown:
                self._closed = True
                self._scheduler.begin_shutdown()
                self._stop.set()

        # Wait for in-flight model calls before releasing their KV tables.
        if thread is not None:
            thread.join()
        if not first_shutdown:
            self._shutdown_complete.wait()
            return
        try:
            self._scheduler.finish_shutdown()
        finally:
            try:
                self._scheduler.clear_prefix_cache()
            finally:
                try:
                    release_cache = getattr(
                        self.backend, "release_kv_cache_storage", None
                    )
                    if release_cache is not None:
                        release_cache()
                finally:
                    self._shutdown_complete.set()

    def generate(self, prompt: str, max_tokens: int = 32, timeout: float = 300) -> Request:
        req = self.submit(prompt, max_tokens)
        if not self.wait(req, timeout):
            self.cancel(req)
            raise TimeoutError(f"request {req.request_id} timed out")
        if req.status == "failed":
            raise RuntimeError(req.error)
        if req.status == "cancelled":
            raise RuntimeError(req.error or f"request {req.request_id} was cancelled")
        return req


Engine = LLMEngine
