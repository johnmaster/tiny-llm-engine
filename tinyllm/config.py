"""Configuration for currently supported engine and scheduler settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 8
    max_model_len: int = 8192

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be at least 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be at least 1")
