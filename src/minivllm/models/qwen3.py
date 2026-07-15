import torch
import torch.nn as nn
import torch.distributed as dist

from minivllm.layers import *


# Qwen3Attention: 
# qkv projection ->
# rms_norm if not qkv_bias ->
# apply rope to q, k ->
# attention ->
# output projection.
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        max_position: int = 40960,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        base: float = 1000000.0,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        # split heads into different gpus
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0, "num_heads must be divisible by tensor parallel size."
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.total_num_kv_heads % self.tp_size == 0, "num_kv_heads must be divisible by tensor parallel size."
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = self.head_dim ** -0.5 # attention scale factor
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVColumnParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias
        )

        # Q and K norms used in Qwen3 after projection
        if not self.qkv_bias:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            base=base,
            head_dim=head_dim,
            max_position=max_position
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
        # ===== QKV Projection (Column Parallel) =====
        # Output shape per GPU: (total_tokens, head_dim * (num_heads + 2*num_kv_heads))
        # where num_heads = total_num_heads/tp_size
        #       num_kv_heads = total_num_kv_heads/tp_size
        qkv = self.qkv_proj(x)

        # ===== Split QKV on per GPU =====
        # q_size = head_dim * num_heads
        # kv_size = head_dim * num_kv_heads
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # ===== Apply Q and K norms =====
        # these are used in Qwen3 to stabilize attention
        # applied to q and k because they participate in attention_weight computation
        # removes possibility of large numbers that cause softmax instability
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ===== Apply RoPE =====
        q, k = self.rotary_emb(positions, q, k)

        # ===== Attention =====
        o = self.attention(q, k, v)
        # o shape: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
        o = o.flatten(start_dim=1)

        # ===== Output Projection (Row Parallel) =====
        # Input: (total_tokens, num_heads * head_dim) sharded across GPUs
        # Output: (total_tokens, hidden_size) REPLICATED on all GPUs (after all_reduce)
        o = self.o_proj(o)

        return o


# Qwen3MLP
# gate/up projection ->
# activation(SiLU) ->
# down projection.
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
        )
        self.activation = SiLUAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.activation(self.gate_up_proj(x)))
        return x


# Qwen3DecoderLayer
# input layernorm with residual ->
# self_attn ->
# post attention layernorm ->
# mlp.
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        max_position: int = 40960,
        rms_norm_eps: float = 1e-6,
        intermediate_size: int = 3072,
        qkv_bias: bool = False,
        base: float = 1000000.0,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            base=base,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x
            x = self.input_layernorm(x)
        x = self.self_attn(x, positions=positions)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


# Qwen3Model
# token embedding ->
# decoder layers stack ->
# final layer norm.
class Qwen3Model(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_heads: int,
            head_dim: int | None = None,
            num_kv_heads: int | None = None,
            max_position: int = 40960,
            rms_norm_eps: float = 1e-6,
            intermediate_size: int = 3072,
            qkv_bias: bool = False,
            base: float = 1000000.0,
            num_layers: int = 28,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                max_position=max_position,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                qkv_bias=qkv_bias,
                base=base,
            ) for _ in range(num_layers)
        ])
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


class Qwen3ForCausalLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ('qkv_proj', 'q'),
        "k_proj": ('qkv_proj', 'k'),
        "v_proj": ('qkv_proj', 'v'),
        "gate_proj": ('gate_up_proj', 0),
        "up_proj": ('gate_up_proj', 1),
    }

    def __init__(
            self,
            # the default values of followed params are the same as Qwen/Qwen3-0.6B
            vocab_size: int = 151936,
            hidden_size: int = 1024,
            num_heads: int = 16,
            head_dim: int = 128,
            num_kv_heads: int = 8,
            max_position: int = 40960,
            rms_norm_eps: float = 1e-6,
            intermediate_size: int = 3072,
            qkv_bias: bool = False,
            base: float = 1000000.0,
            num_layers: int = 28,
            tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            qkv_bias=qkv_bias,
            base=base,
            num_layers=num_layers
        )
        self.lm_head = ParallelLMHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size
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
