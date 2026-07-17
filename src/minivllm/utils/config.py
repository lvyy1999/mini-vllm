from dataclasses import dataclass, field


@dataclass(slots=True)
class Config:
    # engine config
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    max_num_sequences: int = 512
    max_model_length: int = 4096
    cache_block_size: int = 256
    max_cache_blocks: int = -1
    kv_cache_dtype: str = 'auto'
    enforce_eager: bool = False
    eos_token_id: int = -1
    world_size: int = 1

    # custom model config
    model_name_or_path: str = ""
    # the config dict will be read in ModelRunner and
    custom_model_config: dict = field(default_factory=dict)

    def __post_init__(self):
        assert 1 <= self.world_size <= 8
        assert self.kv_cache_dtype in ('auto', 'int8')
        assert (
                self.cache_block_size > 0
                and (self.cache_block_size & (self.cache_block_size - 1)) == 0  # must be power of 2
        )
        if "max_position_embeddings" in self.custom_model_config:
            self.max_model_length = min(
                self.max_model_length,
                self.custom_model_config["max_position_embeddings"],
            )
