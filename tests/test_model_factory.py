from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("transformers")

import minivllm.models.model_factory as factory
from minivllm.models.model_factory import RuntimeModelInfo, create_model


class CapturingModel:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestRuntimeModelInfo:

    def test_is_immutable(self):
        info = RuntimeModelInfo(2, 4, 8, 32)

        with pytest.raises(FrozenInstanceError):
            info.num_layers = 3


class TestQwen3Builder:

    def test_uses_expected_defaults(self, monkeypatch):
        monkeypatch.setattr(factory, "Qwen3ForCausalLM", CapturingModel)

        model, info = factory._build_qwen3({})

        assert info == RuntimeModelInfo(
            num_layers=28,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=1024,
        )
        assert model.kwargs == {
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_heads": 16,
            "head_dim": 128,
            "num_kv_heads": 8,
            "max_position": 40960,
            "rms_norm_eps": 1e-6,
            "intermediate_size": 3072,
            "qkv_bias": False,
            "base": 1000000.0,
            "num_layers": 28,
            "tie_word_embeddings": True,
        }

    def test_maps_hugging_face_config_fields(self, monkeypatch):
        monkeypatch.setattr(factory, "Qwen3ForCausalLM", CapturingModel)
        config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "head_dim": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "intermediate_size": 64,
            "attention_bias": True,
            "rope_theta": 500000.0,
            "num_hidden_layers": 3,
            "tie_word_embeddings": False,
        }

        model, info = factory._build_qwen3(config)

        assert info == RuntimeModelInfo(3, 2, 8, 32)
        assert model.kwargs["vocab_size"] == 100
        assert model.kwargs["num_heads"] == 4
        assert model.kwargs["qkv_bias"] is True
        assert model.kwargs["tie_word_embeddings"] is False


class TestLlamaBuilder:

    def test_uses_expected_defaults(self, monkeypatch):
        monkeypatch.setattr(factory, "LlamaForCausalLM", CapturingModel)

        model, info = factory._build_llama({})

        assert info == RuntimeModelInfo(
            num_layers=16,
            num_kv_heads=8,
            head_dim=64,
            hidden_size=2048,
        )
        assert model.kwargs == {
            "vocab_size": 128256,
            "hidden_size": 2048,
            "num_heads": 32,
            "head_dim": 64,
            "num_kv_heads": 8,
            "max_position": 131072,
            "rms_norm_eps": 1e-5,
            "intermediate_size": 8192,
            "qkv_bias": False,
            "ffn_bias": False,
            "base": 500000.0,
            "num_layers": 16,
            "tie_word_embeddings": True,
        }

    def test_maps_llama_specific_bias_fields(self, monkeypatch):
        monkeypatch.setattr(factory, "LlamaForCausalLM", CapturingModel)
        config = {
            "attention_bias": True,
            "mlp_bias": True,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "hidden_size": 64,
        }

        model, info = factory._build_llama(config)

        assert info == RuntimeModelInfo(2, 1, 16, 64)
        assert model.kwargs["qkv_bias"] is True
        assert model.kwargs["ffn_bias"] is True


class TestCreateModel:

    def test_dispatches_by_final_path_component(self, monkeypatch):
        expected = (object(), RuntimeModelInfo(1, 1, 4, 8))
        builder = MagicMock(return_value=expected)
        monkeypatch.setitem(factory._MODEL_BUILDERS, "Qwen3-0.6B", builder)
        config = {"hidden_size": 8}

        result = create_model("/models/Qwen3-0.6B", config)

        assert result is expected
        builder.assert_called_once_with(config)

    def test_rejects_unsupported_model_and_lists_supported_names(self):
        with pytest.raises(ValueError) as exc_info:
            create_model("org/unsupported-model", {})

        message = str(exc_info.value)
        assert "org/unsupported-model" in message
        assert "Qwen3-0.6B" in message
        assert "Llama-3.2-1B-Instruct" in message
