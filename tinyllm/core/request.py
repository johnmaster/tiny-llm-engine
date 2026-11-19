"""Request data shared by the engine and scheduler."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from threading import Event
from typing import Any

_ids = count(1)


@dataclass
class Request:
    prompt: str
    max_tokens: int
    request_id: str = field(default_factory=lambda: f"req-{next(_ids)}")
    prompt_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    prefix_cached_tokens: int = 0
    prefix_match_checked: bool = False
    prefill_cursor: int = 0
    prefill_chunks: int = 0
    released_block_ids: list[int] = field(default_factory=list)
    prefix_match: Any = field(default=None, repr=False)
    state: Any = None
    status: str = "waiting"
    error: str | None = None
    ready: Event = field(default_factory=Event, repr=False)
