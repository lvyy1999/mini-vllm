from dataclasses import dataclass
from pathlib import Path

from minivllm.models.llama import LlamaForCausalLM
from minivllm.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True, slots=True)
class RuntimeModelInfo:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int


def _build_qwen3(config: dict):
    info = RuntimeModelInfo(
        num_layers=config.get("num_hidden_layers", 28),
        num_kv_heads=config.get("num_key_value_heads", 8),
        head_dim=config.get("head_dim", 128),
        hidden_size=config.get("hidden_size", 1024),
    )

    model = Qwen3ForCausalLM(
        vocab_size=config.get("vocab_size", 151936),
        hidden_size=info.hidden_size,
        num_heads=config.get("num_attention_heads", 16),
        head_dim=info.head_dim,
        num_kv_heads=info.num_kv_heads,
        max_position=config.get("max_position_embeddings", 40960),
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        intermediate_size=config.get("intermediate_size", 3072),
        qkv_bias=config.get("attention_bias", False),
        base=config.get("rope_theta", 1000000.0),
        num_layers=info.num_layers,
        tie_word_embeddings=config.get("tie_word_embeddings", True),
    )

    return model, info


def _build_llama(config: dict):
    info = RuntimeModelInfo(
        num_layers=config.get("num_hidden_layers", 16),
        num_kv_heads=config.get("num_key_value_heads", 8),
        head_dim=config.get("head_dim", 64),
        hidden_size=config.get("hidden_size", 2048),
    )

    model = LlamaForCausalLM(
        vocab_size=config.get("vocab_size", 128256),
        hidden_size=info.hidden_size,
        num_heads=config.get("num_attention_heads", 32),
        head_dim=info.head_dim,
        num_kv_heads=info.num_kv_heads,
        max_position=config.get("max_position_embeddings", 131072),
        rms_norm_eps=config.get("rms_norm_eps", 1e-5),
        intermediate_size=config.get("intermediate_size", 8192),
        qkv_bias=config.get("attention_bias", False),
        ffn_bias=config.get("mlp_bias", False),
        base=config.get("rope_theta", 500000.0),
        num_layers=info.num_layers,
        tie_word_embeddings=config.get("tie_word_embeddings", True),
    )

    return model, info


_MODEL_BUILDERS = {
    "Qwen3-0.6B": _build_qwen3,
    "Llama-3.2-1B-Instruct": _build_llama,
}


def create_model(
    model_name_or_path: str,
    custom_model_config: dict,
):
    model_name = Path(model_name_or_path).name
    if model_name in _MODEL_BUILDERS:
        return _MODEL_BUILDERS[model_name](custom_model_config)

    supported = ", ".join(sorted(_MODEL_BUILDERS))
    raise ValueError(
        f"Unsupported model: {model_name_or_path}, " f"supported models: {supported}"
    )
