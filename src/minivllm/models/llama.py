from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from minivllm.layers import *


class LlamaAttn(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 131072,
        qkv_bias: bool = False,
        base: float = 500000.0,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_heads
        )
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads

        self.qkv_proj = QKVColumnParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=qkv_bias,
        )

        # Llama 3.2 does not have q_norm or k_norm

        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim, max_position=max_position, base=base, is_llama3=True
        )

        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            self.scale,
            self.num_kv_heads,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
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
        num_heads: int = 32,
        num_kv_heads: int = 8,
        max_position: int = 131072,
        rms_norm_eps: float = 1e-5,
        intermediate_size: int = 8192,
        qkv_bias: bool = False,
        ffn_bias: bool = False,
        base: float = 500000.0,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = LlamaAttn(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            qkv_bias=qkv_bias,
            base=base,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size, bias=ffn_bias
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
# embedding ->
# layers stack ->
# final layer norm
class LlamaModel(nn.Module):

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        max_position: int = 131072,
        rms_norm_eps: float = 1e-5,
        intermediate_size: int = 8192,
        qkv_bias: bool = False,
        ffn_bias: bool = False,
        base: float = 500000.0,
        num_layers: int = 16,
    ):
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    max_position=max_position,
                    rms_norm_eps=rms_norm_eps,
                    intermediate_size=intermediate_size,
                    qkv_bias=qkv_bias,
                    ffn_bias=ffn_bias,
                    base=base,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

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
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        # the default values of followed params are the same as meta-llama/Llama-3.2-1B-Instruct
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        max_position: int = 131072,
        rms_norm_eps: float = 1e-5,
        intermediate_size: int = 8192,
        qkv_bias: bool = False,
        ffn_bias: bool = False,
        base: float = 500000.0,
        num_layers: int = 16,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            base=base,
            num_layers=num_layers,
        )
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
