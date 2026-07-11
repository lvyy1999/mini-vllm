import triton 
import triton.language as tl
from minivllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Input layout: (num_tokens, num_kv_heads, head_dim)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # each thread responsible for one head of one token
    token_idx = tl.program_id(0) # program_id(0) = which token
    head_idx = tl.program_id(1) # program_id(1) = which head

    # which slot to store this token
    # slot_idx = block_idx * block_size + block_offset
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # calculate this head's offsets in input
    input_offsets = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                     head_idx * head_dim + # skip previous heads
                     head_offsets)
    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # and slot_idx = block_idx * block_size + block_offset
    # calculate this head's offsets in cache
    cache_offsets = (slot_idx * num_kv_heads * head_dim + # skip previous slots
                     head_idx * head_dim + # skip previous heads
                     head_offsets)
    # load key and value floats from the pointers' memory
    key = tl.load(key_ptr + input_offsets)
    value = tl.load(value_ptr + input_offsets)
    # store into cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    # each thread responsible for one head of one token
    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key, # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

@triton.jit
def flash_attention_varlen_kernel(
    q_ptr, # (total_tokens, num_heads, head_dim)
    k_ptr, # (total_tokens, num_kv_heads, head_dim)
    v_ptr, # (total_tokens, num_kv_heads, head_dim)
    o_ptr, # (total_tokens, num_heads, head_dim)
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    """
    # Each thread responsible for one head of one block of queries of one sequence
    block_idx = tl.program_id(0) # block index
    head_idx = tl.program_id(1) # head index of query
    seq_idx = tl.program_id(2) # sequence index in the continuous batch

    # Determine which KV head to use
    # For general MHA, num_heads == num_kv_heads -> kv_head_idx == head_idx
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load sequence boundaries of queries
    seq_start_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end_q = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len_q = seq_end_q - seq_start_q

    # Load sequence boundaries of key / value
    seq_start_kv = tl.load(cu_seqlens_k_ptr + seq_idx)
    seq_end_kv = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seq_len_kv = seq_end_kv - seq_start_kv
    
    # Early exit if this block is beyond sequence length
    if block_idx * BLOCK_M >= seq_len_q:
        return
    
    # Offset for this block of queries
    head_offsets = tl.arange(0, head_dim)
    block_offsets_q = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Q: (total_tokens, num_heads, head_dim)
    # each thread responsible for BLOCK_M * head_dim elements in Q
    # calculate the offsets of them
    q_offsets = ((seq_start_q + block_offsets_q[:, None]) * num_heads * head_dim +
                 head_idx * head_dim +
                 head_offsets[None, :])
    # Load Qi block - shape (BLOCK_M, head_dim)
    q_mask = block_offsets_q < seq_len_q
    q_i = tl.load(q_ptr + q_offsets, mask=q_mask[:, None], other=0.0)

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) # init unscaled sum of row
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10 # init max value of each row as -inf
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32) # init unscaled accumulate output

    # Loop process K, V blocks
    for kv_start in tl.range(0, seq_len_kv, BLOCK_N):
        # K/V: (total_tokens, num_kv_heads, head_dim)
        # each loop responsible for BLOCK_N * head_dim elements in K/V
        # calculate the offsets of them
        block_offsets_kv = kv_start + tl.arange(0, BLOCK_N)
        kv_offsets = ((seq_start_kv + block_offsets_kv[:, None]) * num_kv_heads * head_dim +
                      kv_head_idx * head_dim +
                      head_offsets[None, :])
        # Load Kj / Vj block - shape (BLOCK_N, head_dim)
        kv_mask = block_offsets_kv < seq_len_kv
        k_j = tl.load(k_ptr + kv_offsets, mask=kv_mask[:, None], other=0.0)
        v_j = tl.load(v_ptr + kv_offsets, mask=kv_mask[:, None], other=0.0)
        
        # Compute QK^T / sqrt(d) - shape (BLOCK_M, BLOCK_N)
        # s_ij = tl.dot(q_i, k_j, trans_b=True)
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        # Apply causal q_mask: only attend to positions <= current position
        mask_causal = (seq_start_q + block_offsets_q[:, None]) >= (seq_start_kv + block_offsets_kv[None, :])
        s_ij = tl.where(mask_causal & q_mask[:, None], s_ij, -1e10)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p_ij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p_ij, v_j)
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    # Store into O: (total_tokens, num_heads, head_dim)
    # each thread responsible for BLOCK_M * head_dim elements in O
    # the offsets of them in O is the same as A
    tl.store(o_ptr + q_offsets, acc, mask=q_mask[:, None])


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: cumulative sequence lengths of queries
        cu_seqlens_k: cumulative sequence lengths of key and value
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # Allocate memory for output
    output = torch.empty_like(q)
    
    # Choose block sizes to avoid OOM on shared memory.
    # Shared memory usage ~
    #     (2 * BLOCK_M + 2 * BLOCK_N) * head_dim * sizeof(dtype)
    #     + BLOCK_M * BLOCK_N * sizeof(dtype)
    #     + 2 * BLOCK_M * sizeof(dtype)
    BLOCK_M = 64 if head_dim <= 64 else 32 if head_dim <= 128 else 16
    BLOCK_N = 64 if head_dim <= 64 else 32 if head_dim <= 128 else 16

    # Calculate grid dimensions
    num_seqs = cu_seqlens_q.shape[0] - 1
    num_blocks = triton.cdiv(max_seqlen_q, BLOCK_M)
    grid = (num_blocks, num_heads, num_seqs)
    # launch num_blocks x num_heads × num_seqs threads
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output


@triton.jit
def flash_attention_with_kvcache_kernel(
    q_ptr, # (batch_size, num_heads, head_dim)
    k_cache_ptr, # (max_num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr, # (max_num_blocks, block_size, num_kv_heads, head_dim)
    output_ptr,
    block_tables_ptr,
    cache_seqlens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel with kv-cache for decoding.
    """
    # Each thread responsible for one head of one batch
    batch_idx = tl.program_id(0) # which batch
    head_idx = tl.program_id(1) # which head
    # Determine which KV head this query head uses
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load sequence cache length
    cache_seq_len = tl.load(cache_seqlens_ptr + batch_idx)
    
    # Load query
    head_offsets = tl.arange(0, head_dim)
    q_offsets = batch_idx * num_heads * head_dim + head_idx * head_dim + head_offsets
    q = tl.load(q_ptr + q_offsets)

    l_i = 0.0  # init unscaled sum of row
    m_i = -1e10  # init max value of row as -inf
    acc = tl.zeros([head_dim,], dtype=tl.float32) - 1e10  # init unscaled accumulate output

    # Loop process K, V caches
    # each loop responsible for BLOCK_N K/V caches
    for kv_start in tl.range(0, cache_seq_len, BLOCK_N):
        # Calculate the offsets of kv caches in this loop
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offsets < cache_seq_len

        # Compute attention scores: qK^T / sqrt(d) - shape (1, BLOCK_N)
        s = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Load k for each valid position and compute scores
        for j in tl.range(BLOCK_N):
            token_idx = kv_start + j
            if token_idx < cache_seq_len:
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                if block_idx < max_num_blocks:
                    # Look up physical block
                    block_table_offset = batch_idx * max_num_blocks + block_idx
                    physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                    if physical_block_idx != -1:
                        # cache: (max_num_blocks, block_size, num_kv_heads, head_dim)
                        k_offsets = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                     block_offset * num_kv_heads * head_dim +
                                     kv_head_idx * head_dim +
                                     head_offsets)
                        # Load k
                        k = tl.load(k_cache_ptr + k_offsets)
                        # Compute score for this token
                        score = tl.sum(q * k) * scale
                        # Update s[i] by tl.where
                        mask_i = tl.arange(0, BLOCK_N) == j
                        s = tl.where(mask_i, score, s)

        # Apply mask to invalid positions
        s = tl.where(kv_mask, s, -1e10)
        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        m_i = m_new
        p = tl.exp(s - m_new)
        # Rescale accumulator
        acc = acc * alpha
        l_i = l_i * alpha

        # Load v and accumulate
        for j in range(BLOCK_N):
            token_idx = kv_start + j
            if token_idx < cache_seq_len:
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                if block_idx < max_num_blocks:
                    # Look up physical block
                    block_table_offset = batch_idx * max_num_blocks + block_idx
                    physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                    if physical_block_idx != -1:
                        # cache: (max_num_blocks, block_size, num_kv_heads, head_dim)
                        v_offsets = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                     block_offset * num_kv_heads * head_dim +
                                     kv_head_idx * head_dim + head_offsets)
                        # Load v
                        v = tl.load(v_cache_ptr + v_offsets)
                        # Extract p[j]
                        mask_i = tl.arange(0, BLOCK_N) == j
                        p_j = tl.sum(tl.where(mask_i, p, 0.0))
                        # Accumulate
                        l_i = l_i + p_j
                        acc = acc + p_j * v

    # Final normalization
    acc = acc / l_i
    # Store output
    tl.store(output_ptr + q_offsets, acc)


def flash_attention_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cache_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        q: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        cache_seqlens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = q.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Make q contiguous
    q = q.contiguous()
    # Allocate memory for output
    output = torch.empty_like(q)
    
    # Choose chunk size for processing K/V caches
    BLOCK_N = 64 if head_dim <= 128 else 32

    # each thread responsible for one head of one batch
    grid = (batch_size, num_heads)
    # launch batch_size x num_heads threads
    flash_attention_with_kvcache_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        block_tables,
        cache_seqlens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill: # Prefill: use flash attention varlen mode
            o = flash_attention_varlen(
                q, k, v,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                context.max_seqlen_q,
                context.max_seqlen_k,
                self.scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim
            )
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else: # Decode
            o = flash_attention_with_kvcache(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)

