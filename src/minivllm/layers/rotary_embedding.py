import torch.nn as nn
import torch


# apply rope between two adjacent elements
def apply_rope_adjacent(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    if x.dim() == 3: # (total_tokens, num_heads, head_dim)
        # cos, sin shape: (seq_len, head_dim/2) -> (seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else: # (B, seq_len, num_heads, head_dim)
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    # Split x into two halves by interleaved mode
    # x0, x2, ...
    x_even = x[..., ::2]
    # x1, x3, ...
    x_odd = x[..., 1::2]
    # y0 = x0 * cos - x1 * sin
    # y1 = x1 * cos + x0 * sin
    # ......
    y_even = x_even * cos - x_odd * sin
    y_odd  = x_odd * cos + x_even * sin
    y = torch.zeros_like(x)
    y[:, ::2] = y_even
    y[:, 1::2] = y_odd
    return y


# apply rope between two elements at a distance of head_dim / 2
def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    if x.dim() == 3: # (total_tokens, num_heads, head_dim)
        # cos, sin shape: (seq_len, head_dim/2) -> (seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else: # (B, seq_len, num_heads, head_dim)
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    # Split x into two halves along the head dimension
    x1, x2 = x.chunk(2, dim=-1)
    # Apply rotary embedding with proper broadcasting
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float = 10000.0,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()

        inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2) / head_dim)
        ) # shape(head_dim/2, )

        if is_llama3:
            # specifically for llama3.2
            import math
            # no smooth if low_freq_factor == high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq

        positions = torch.arange(max_position).float() # shape(max_seq_len, )
        angles = torch.outer(positions, inv_freq) # shape(max_seq_len, head_dim/2)
        cos = torch.cos(angles) # shape(max_seq_len, head_dim/2)
        sin = torch.sin(angles) # shape(max_seq_len, head_dim/2)
        cos_sin_cache = torch.cat([cos, sin], dim=-1) # shape(max_seq_len, head_dim)
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )
