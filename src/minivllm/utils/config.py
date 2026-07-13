import os
from dataclasses import dataclass

@dataclass(slots=True)
class Config:
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    max_num_sequences: int = 512
    max_model_length: int = 4096
    cache_block_size: int = 256
    max_cache_blocks: int = -1
    enforce_eager: bool = False
    eos_token_id: int = -1
    world_size: int = 1

    model_name_or_path: str = ""
    custom_model_config: dict | None = None

    def __post_init__(self):
        assert 1 <= self.world_size <= 8
        assert self.cache_block_size % 256 == 0
        if "max_position_embeddings" in self.custom_model_config:
            self.max_model_length = min(self.max_model_length, self.custom_model_config["max_position_embeddings"])
