"""Control-plane contract for a future prefill/decode KV handoff."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class KVTransferMetadata:
    request_id: str
    source_worker: str
    destination_worker: str
    token_count: int
    block_ids: tuple[int, ...]


class KVConnector(Protocol):
    """Transport KV blocks between a prefill worker and a decode worker."""
    def send(self, metadata: KVTransferMetadata, blocks: Sequence[object]) -> None: ...
    def receive(self, metadata: KVTransferMetadata) -> Sequence[object]: ...
