import pytest

from minivllm.utils.config import Config


class TestConfigDefaults:

    def test_engine_defaults(self):
        config = Config()

        assert config.max_num_batched_tokens == 16384
        assert config.gpu_memory_utilization == 0.9
        assert config.max_num_sequences == 512
        assert config.max_model_length == 4096
        assert config.cache_block_size == 256
        assert config.max_cache_blocks == -1
        assert config.enforce_eager is False
        assert config.eos_token_id == -1
        assert config.world_size == 1
        assert config.kv_cache_dtype == "auto"

    def test_custom_model_configs_are_not_shared(self):
        first = Config()
        second = Config()

        first.custom_model_config["hidden_size"] = 128

        assert second.custom_model_config == {}


class TestWorldSizeValidation:

    @pytest.mark.parametrize("world_size", [1, 2, 8])
    def test_accepts_supported_world_sizes(self, world_size):
        assert Config(world_size=world_size).world_size == world_size

    @pytest.mark.parametrize("world_size", [0, 9, -1])
    def test_rejects_unsupported_world_sizes(self, world_size):
        with pytest.raises(AssertionError):
            Config(world_size=world_size)


class TestCacheBlockSizeValidation:

    @pytest.mark.parametrize("block_size", [1, 2, 4, 16, 256])
    def test_accepts_positive_powers_of_two(self, block_size):
        assert Config(cache_block_size=block_size).cache_block_size == block_size

    @pytest.mark.parametrize("block_size", [-8, 0, 3, 6, 24])
    def test_rejects_non_positive_or_non_power_of_two_sizes(self, block_size):
        with pytest.raises(AssertionError):
            Config(cache_block_size=block_size)


class TestKVCacheDtypeValidation:

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "int8"])
    def test_accepts_supported_kv_cache_dtypes(self, kv_cache_dtype):
        config = Config(kv_cache_dtype=kv_cache_dtype)

        assert config.kv_cache_dtype == kv_cache_dtype

    @pytest.mark.parametrize("kv_cache_dtype", ["float16", "int4", "", None])
    def test_rejects_unsupported_kv_cache_dtypes(self, kv_cache_dtype):
        with pytest.raises(AssertionError):
            Config(kv_cache_dtype=kv_cache_dtype)


class TestModelLengthResolution:

    def test_clamps_requested_length_to_model_limit(self):
        config = Config(
            max_model_length=8192,
            custom_model_config={"max_position_embeddings": 2048},
        )

        assert config.max_model_length == 2048

    def test_keeps_smaller_requested_length(self):
        config = Config(
            max_model_length=1024,
            custom_model_config={"max_position_embeddings": 2048},
        )

        assert config.max_model_length == 1024

    def test_keeps_requested_length_when_model_limit_is_missing(self):
        config = Config(max_model_length=3072, custom_model_config={})

        assert config.max_model_length == 3072
