import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

from minivllm.layers import *

class LlamaAttn(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_attn_bias: bool = False,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_qo_heads
        self.num_heads = num_qo_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_qo_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_qo_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads

        self.qkv_proj = QKVColumnParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            bias=has_attn_bias,
        )
        
        # Llama 3.2 does not have q_norm or k_norm

        self.rotary_emb = RotaryEmbedding(
            base=rope_base,
            head_dim=head_dim,
            max_seq_len=max_position_embeddings,
            is_llama3=True
        )

        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            self.scale,
            self.num_kv_heads,
            self.block_size
        )

        self.o_proj = RowParallelLinear(
            input_size= self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attention(q, k, v)
        o = o.flatten(start_dim=1)
        return self.o_proj(o)

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiLUAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.activation(self.gate_up_proj(x)))
        return x

class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-06,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_epsilon)
        self.self_attn = LlamaAttn(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_attn_bias=has_attn_bias,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            block_size=block_size,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_epsilon)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x  # Save BEFORE normalization
            x = self.input_layernorm(x)
        x = self.self_attn(x, positions=positions)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual
    
# LlamaModel
# embedding
# layers stack
# final layer norm
class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-6,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        num_layers: int = 16,
        block_size: int = 256,
    ):
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                has_attn_bias=has_attn_bias,
                rms_norm_epsilon=rms_norm_epsilon,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_epsilon)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, positions, residual)
        x, _ = self.norm(x, residual)
        return x


class LlamaForCausalLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ('qkv_proj', 'q'),
        "k_proj": ('qkv_proj', 'k'),
        "v_proj": ('qkv_proj', 'v'),
        "gate_proj": ('gate_up_proj', 0),
        "up_proj": ('gate_up_proj', 1),
    }
    def __init__(self, config: dict, block_size: int = 256):
        """
        Args:
            config (dict): config of Llama
            block_size (int): size of kv cache block, used for flash paged attention
        """
        super().__init__()
        vocab_size = config.get("vocab_size", 128256)
        hidden_size = config.get("hidden_size", 2048)
        num_qo_heads = config.get("num_qo_heads", 32)
        head_dim = config["head_dim"] if "head_dim" in config else hidden_size // num_qo_heads
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=config.get("num_kv_heads", 8),
            has_attn_bias=config.get("has_attn_bias", False),
            rms_norm_epsilon=config.get("rms_norm_epsilon", 1e-5),
            rope_base=config.get("rope_base", 500000),
            max_position_embeddings=config.get("max_position_embeddings", 32768),
            intermediate_size=config.get("intermediate_size", 8192),
            ffn_bias=config.get("ffn_bias", False),
            num_layers=config.get("num_layers", 16),
            block_size=block_size,
        )
        tie_word_embeddings = config.get("tie_word_embeddings", True)
        self.lm_head = ParallelLMHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)
