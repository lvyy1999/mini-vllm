import torch
import torch.nn as nn
import triton
import triton.language as tl

from minivllm.utils import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_cache_int8: tl.constexpr,
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Input layout: (num_tokens, num_kv_heads, head_dim)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # each thread responsible for one head of one token
    token_idx = tl.program_id(0)  # program_id(0) = which token
    head_idx = tl.program_id(1)  # program_id(1) = which head

    # which slot to store this token
    # slot_idx = block_idx * block_size + block_offset
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return  # when run cuda graph, maybe padding with -1

    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # calculate this head's offsets in input
    input_offsets = (
        token_idx * num_kv_heads * head_dim  # skip previous tokens
        + head_idx * head_dim  # skip previous heads
        + head_offsets
    )
    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # and slot_idx = block_idx * block_size + block_offset
    # calculate this head's offsets in cache
    cache_offsets = (
        slot_idx * num_kv_heads * head_dim  # skip previous slots
        + head_idx * head_dim  # skip previous heads
        + head_offsets
    )
    # load key and value from the pointers' memory
    key = tl.load(key_ptr + input_offsets)
    value = tl.load(value_ptr + input_offsets)

    if kv_cache_int8:  # int8 quantization for k/v cache if needed
        key_f32 = key.to(tl.float32)
        value_f32 = value.to(tl.float32)
        # calculate scale factor
        key_abs_max = tl.max(tl.abs(key_f32), axis=0)
        value_abs_max = tl.max(tl.abs(value_f32), axis=0)
        key_scale = tl.where(key_abs_max > 0.0, key_abs_max / 127.0, 1.0)
        value_scale = tl.where(value_abs_max > 0.0, value_abs_max / 127.0, 1.0)
        # clamp to [-127, 127]
        key_scaled = tl.maximum(tl.minimum(key_f32 / key_scale, 127.0), -127.0)
        value_scaled = tl.maximum(tl.minimum(value_f32 / value_scale, 127.0), -127.0)
        # round to the nearest integer and convert to int8
        key_quantized = tl.where(
            key_scaled >= 0.0,
            tl.floor(key_scaled + 0.5),
            tl.ceil(key_scaled - 0.5),
        ).to(tl.int8)
        value_quantized = tl.where(
            value_scaled >= 0.0,
            tl.floor(value_scaled + 0.5),
            tl.ceil(value_scaled - 0.5),
        ).to(tl.int8)
        # calculate scale offset and store
        scale_offset = slot_idx * num_kv_heads + head_idx
        tl.store(k_cache_ptr + cache_offsets, key_quantized)
        tl.store(v_cache_ptr + cache_offsets, value_quantized)
        tl.store(k_scale_cache_ptr + scale_offset, key_scale)
        tl.store(v_scale_cache_ptr + scale_offset, value_scale)
    else:  # store into cache directly
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
):
    """
    Store key-value pairs into paged cache.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        k_scale_cache: (num_blocks, block_size, num_kv_heads)
        v_scale_cache: (num_blocks, block_size, num_kv_heads)
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert (
        slot_mapping.numel() == num_tokens
    ), "Slot mapping size must match number of tokens"

    kv_cache_int8 = k_cache.dtype == torch.int8
    if kv_cache_int8:
        assert k_scale_cache is not None and v_scale_cache is not None
        assert k_scale_cache.shape == v_scale_cache.shape
        assert k_scale_cache.shape == k_cache.shape[:-1]
    else:  # avoid to pass null pointer to kernel
        k_scale_cache = k_cache
        v_scale_cache = v_cache

    # each thread responsible for one head of one token
    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key,  # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_int8=kv_cache_int8,
    )


@triton.jit
def flash_attention_prefill_with_cache_kernel(
    q_ptr,  # (total_q, num_heads, head_dim)
    k_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads)
    v_scale_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads)
    o_ptr,  # (total_q, num_heads, head_dim)
    scale,
    block_tables_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    kv_cache_int8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for chunked prefill with K/V cache.
    """
    # Each thread responsible for one head of one block of queries of one batch of sequences
    head_idx = tl.program_id(0)  # head index
    block_idx = tl.program_id(1)  # block index
    batch_idx = tl.program_id(2)  # batch index

    # Determine which KV head to use
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Load boundaries of key / value in the continuous batched sequence
    batch_start_kv = tl.load(cu_seqlens_k_ptr + batch_idx)
    batch_end_kv = tl.load(cu_seqlens_k_ptr + batch_idx + 1)
    full_seq_len = (
        batch_end_kv - batch_start_kv
    )  # actual length of full sequence  with cache

    # Load boundaries of queries in the continuous batched sequence
    batch_start_q = tl.load(cu_seqlens_q_ptr + batch_idx)
    batch_end_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    batch_len_q = (
        batch_end_q - batch_start_q
    )  # sequence length of queries in this batch

    # Early exit if this block is beyond sequence length
    if block_idx * BLOCK_M >= batch_len_q:
        return

    # Calculate the actual start position of queries in itself's sequence,
    # use for calculate causal mask
    actual_start_q = full_seq_len - batch_len_q

    # Q: (total_tokens, num_heads, head_dim)
    # each thread responsible for BLOCK_M * head_dim elements in Q
    # calculate the offsets of them
    head_offset = tl.arange(0, head_dim)
    batch_offset_q = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_offset = (
        (batch_start_q + batch_offset_q[:, None]) * num_heads * head_dim
        + head_idx * head_dim
        + head_offset[None, :]
    )
    # Load Qi block - shape (BLOCK_M, head_dim)
    q_mask = batch_offset_q < batch_len_q
    q_i = tl.load(q_ptr + q_offset, mask=q_mask[:, None], other=0.0)

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # init unscaled sum of row
    m_i = (
        tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    )  # init max value of each row as -inf
    acc = tl.zeros(
        [BLOCK_M, head_dim], dtype=tl.float32
    )  # init unscaled accumulate output

    # Loop process K, V blocks
    for kv_start in tl.range(0, full_seq_len, BLOCK_N):
        # K/V cache: (max_num_blocks, block_size, num_kv_heads, head_dim)
        # each loop responsible for BLOCK_N * head_dim elements in K/V
        # calculate the offsets of them
        kv_offset = kv_start + tl.arange(0, BLOCK_N)  # (BLOCK_N, )
        kv_mask = kv_offset < full_seq_len  # (BLOCK_N, )
        logical_block_idx = kv_offset // block_size  # (BLOCK_N, )
        logical_block_offset = kv_offset % block_size  # (BLOCK_N, )
        safe_logical_block_idx = tl.minimum(logical_block_idx, max_num_blocks - 1)
        physical_block_idx = tl.load(
            block_tables_ptr + batch_idx * max_num_blocks + safe_logical_block_idx,
        )  # (BLOCK_N, )
        # avoid to calculate negative addr, use kv_mask when really load k/v cache
        physical_block_idx = tl.maximum(physical_block_idx, 0)
        kv_cache_offset = (
            physical_block_idx[:, None] * block_size * num_kv_heads * head_dim
            + logical_block_offset[:, None] * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + head_offset[None, :]
        )  # (BLOCK_N, head_dim)
        kv_scale_offset = (
            physical_block_idx * block_size * num_kv_heads
            + logical_block_offset * num_kv_heads
            + kv_head_idx
        )  # (BLOCK_N, )

        # Load Kj block and compute S = QK^T / sqrt(d)
        k_j = tl.load(
            k_cache_ptr + kv_cache_offset, mask=kv_mask[:, None], other=0.0
        )  # shape (BLOCK_N, head_dim)
        if kv_cache_int8:  # dequantization
            k_scale = tl.load(
                k_scale_cache_ptr + kv_scale_offset, mask=kv_mask, other=0.0
            )  # shape(BLOCK_N, )
            k_j = (k_j.to(tl.float32) * k_scale[:, None]).to(q_i.dtype)
        # s_ij = tl.dot(q_i, k_j, trans_b=True)
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale  # shape (BLOCK_M, BLOCK_N)

        # Apply causal mask: only attend to positions <= current position
        causal_mask = (actual_start_q + batch_offset_q[:, None]) >= kv_offset[None, :]
        s_ij = tl.where(causal_mask & q_mask[:, None], s_ij, -1e10)

        # Online softmax update, P = Softmax(S)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])  # shape (BLOCK_M, BLOCK_N)
        m_i = m_new
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)

        # Load Vj block and update O = PV
        v_j = tl.load(
            v_cache_ptr + kv_cache_offset, mask=kv_mask[:, None], other=0.0
        )  # shape (BLOCK_N, head_dim)
        if kv_cache_int8:  # dequantization
            v_scale = tl.load(
                v_scale_cache_ptr + kv_scale_offset, mask=kv_mask, other=0.0
            )  # shape(BLOCK_N, )
            v_j = (v_j.to(tl.float32) * v_scale[:, None]).to(q_i.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(v_j.dtype), v_j)

    # Final normalization and store output
    tl.store(o_ptr + q_offset, acc / l_i[:, None], mask=q_mask[:, None])


@triton.jit
def flash_attention_prefill_without_cache_kernel(
    q_ptr,  # (total_tokens, num_heads, head_dim)
    k_ptr,  # (total_tokens, num_kv_heads, head_dim)
    v_ptr,  # (total_tokens, num_kv_heads, head_dim)
    o_ptr,  # (total_tokens, num_heads, head_dim)
    scale,
    cu_seqlens_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    """
    # Each thread responsible for one head of one block of queries of one batch of sequence
    head_idx = tl.program_id(0)  # head index
    block_idx = tl.program_id(1)  # block index
    batch_idx = tl.program_id(2)  # batch index

    # Determine which KV head to use
    # For general MHA, num_heads == num_kv_heads -> kv_head_idx == head_idx
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Load boundaries of sequence in the continuous batched sequence
    batch_start = tl.load(cu_seqlens_ptr + batch_idx)
    batch_end = tl.load(cu_seqlens_ptr + batch_idx + 1)
    batch_len = batch_end - batch_start

    # Early exit if this block is beyond sequence length
    if block_idx * BLOCK_M >= batch_len:
        return

    # Q: (total_tokens, num_heads, head_dim)
    # each thread responsible for BLOCK_M * head_dim elements in Q
    # calculate the offsets of them
    head_offset = tl.arange(0, head_dim)
    batch_offset_q = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_offset = (
        (batch_start + batch_offset_q[:, None]) * num_heads * head_dim
        + head_idx * head_dim
        + head_offset[None, :]
    )
    # Load Qi block - shape (BLOCK_M, head_dim)
    q_mask = batch_offset_q < batch_len
    q_i = tl.load(q_ptr + q_offset, mask=q_mask[:, None], other=0.0)

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # init unscaled sum of row
    m_i = (
        tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    )  # init max value of each row as -inf
    acc = tl.zeros(
        [BLOCK_M, head_dim], dtype=tl.float32
    )  # init unscaled accumulate output

    # Loop process K, V blocks
    for kv_start in tl.range(0, batch_len, BLOCK_N):
        # K/V: (total_tokens, num_kv_heads, head_dim)
        # each loop responsible for BLOCK_N * head_dim elements in K/V
        # calculate the offsets of them
        batch_offset_kv = kv_start + tl.arange(0, BLOCK_N)
        kv_offset = (
            (batch_start + batch_offset_kv[:, None]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + head_offset[None, :]
        )  # shape (BLOCK_N, head_dim)
        kv_mask = batch_offset_kv < batch_len

        # Load Kj block and compute S = QK^T / sqrt(d)
        k_j = tl.load(
            k_ptr + kv_offset, mask=kv_mask[:, None], other=0.0
        )  # shape (BLOCK_N, head_dim)
        # s_ij = tl.dot(q_i, k_j, trans_b=True)
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale  # shape (BLOCK_M, BLOCK_N)

        # Apply causal mask: only attend to positions <= current position
        causal_mask = batch_offset_q[:, None] >= batch_offset_kv[None, :]
        s_ij = tl.where(causal_mask & q_mask[:, None], s_ij, -1e10)

        # Online softmax update, P = Softmax(S)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])  # shape (BLOCK_M, BLOCK_N)
        m_i = m_new
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)

        # Load Vj block and update O = PV
        v_j = tl.load(
            v_ptr + kv_offset, mask=kv_mask[:, None], other=0.0
        )  # shape (BLOCK_N, head_dim)
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(v_j.dtype), v_j)

    # Final normalization and store output
    tl.store(o_ptr + q_offset, acc / l_i[:, None], mask=q_mask[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    block_tables: torch.Tensor | None = None,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Flash Attention for prefill with variable-length sequences.
    If block_tables is None, exec normal prefill (without cache);
    otherwise, exec chunked prefill (with cache), input k should be k_cache, v should be v_cache.

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim) if block_tables is None else (num_blocks, block_size, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim) if block_tables is None else (num_blocks, block_size, num_kv_heads, head_dim)
        scale: attention scale factor
        cu_seqlens_q: cumulative sequence lengths of queries
        cu_seqlens_k: cumulative sequence lengths of key and value
        max_seqlen_q: maximum sequence length of queries
        max_seqlen_k: maximum sequence length of key/value
        block_tables: (batch_size, max_num_blocks) or None

    Returns:
        output: (total_tokens, num_heads, head_dim)

    """
    # make queries contiguous
    q = q.contiguous()
    # allocate memory for output
    output = torch.empty_like(q)

    # get needed params
    _, num_heads, head_dim = q.shape

    # choose block sizes to avoid OOM on shared memory.
    # shared memory usage ~
    #     (2 * BLOCK_M + 2 * BLOCK_N) * head_dim * sizeof(dtype)
    #     + BLOCK_M * BLOCK_N * sizeof(dtype)
    #     + 2 * BLOCK_M * sizeof(dtype)
    BLOCK_M = 64 if head_dim <= 64 else 32 if head_dim <= 128 else 16
    BLOCK_N = 64 if head_dim <= 64 else 32 if head_dim <= 128 else 16

    # calculate grid dimensions
    num_batches = cu_seqlens_q.shape[0] - 1
    max_seqlen_q = (
        max_seqlen_q
        if max_seqlen_q > 0
        else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    )
    num_blocks = triton.cdiv(max_seqlen_q, BLOCK_M)
    grid = (num_heads, num_blocks, num_batches)
    # launch num_heads × num_blocks x num_batches threads
    if block_tables is not None:  # chunked prefill, k/v should be k_cache/v_cache
        k_cache, v_cache = k, v  # (num_blocks, block_size, num_kv_heads, head_dim)
        kv_cache_int8 = k_cache.dtype == torch.int8
        if kv_cache_int8:
            assert k_scale_cache is not None and v_scale_cache is not None
            assert k_scale_cache.shape == v_scale_cache.shape
            assert k_scale_cache.shape == k_cache.shape[:-1]
        else:  # avoid to pass null pointer to kernel
            k_scale_cache = k_cache
            v_scale_cache = v_cache
        block_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        max_num_blocks = block_tables.shape[1]
        cu_seqlens_k = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
        flash_attention_prefill_with_cache_kernel[grid](
            q, k_cache, v_cache,
            k_scale_cache,
            v_scale_cache,
            output,
            scale,
            block_tables,
            cu_seqlens_q,
            cu_seqlens_k,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            kv_cache_int8=kv_cache_int8,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:  # normal prefill
        k = k.contiguous()  # (total_tokens, num_kv_heads, head_dim)
        v = v.contiguous()
        num_kv_heads = k.shape[1]
        flash_attention_prefill_without_cache_kernel[grid](
            q, k, v,
            output,
            scale,
            cu_seqlens_q,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return output


@triton.jit
def flash_attention_decode_with_cache_kernel(
    q_ptr,  # (batch_size, num_heads, head_dim)
    k_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads)
    v_scale_cache_ptr,  # (max_num_blocks, block_size, num_kv_heads)
    output_ptr,
    scale,
    block_tables_ptr,
    cache_seqlens_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    kv_cache_int8: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel with kv-cache for decoding.
    """
    # Each thread responsible for one head of one batch
    head_idx = tl.program_id(0)  # which head
    batch_idx = tl.program_id(1)  # which batch

    # Determine which KV head this query head uses
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Load sequence cache length
    cache_seq_len = tl.load(cache_seqlens_ptr + batch_idx)

    # Load query
    head_offset = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + head_offset
    q = tl.load(q_ptr + q_offset)

    l_i = 0.0  # init unscaled sum of row
    m_i = -1e10  # init max value of row as -inf
    acc = tl.zeros(
        [
            head_dim,
        ],
        dtype=tl.float32,
    )  # init unscaled accumulate output

    # Loop process K, V caches
    # each loop responsible for BLOCK_N K/V caches
    for kv_start in tl.range(0, cache_seq_len, BLOCK_N):
        # K/V cache: (max_num_blocks, block_size, num_kv_heads, head_dim)
        # each loop responsible for BLOCK_N * head_dim elements in K/V
        # calculate the offsets of them
        kv_offset = kv_start + tl.arange(0, BLOCK_N)  # (BLOCK_N, )
        kv_mask = kv_offset < cache_seq_len  # (BLOCK_N, )
        logical_block_idx = kv_offset // block_size  # (BLOCK_N, )
        logical_block_offset = kv_offset % block_size  # (BLOCK_N, )
        physical_block_idx = tl.load(
            block_tables_ptr + batch_idx * max_num_blocks + logical_block_idx,
            mask=kv_mask,
            other=-1,
        )  # (BLOCK_N, )
        kv_cache_offset = (
            physical_block_idx[:, None] * block_size * num_kv_heads * head_dim
            + logical_block_offset[:, None] * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + head_offset[None, :]
        )  # (BLOCK_N, head_dim)
        kv_scale_offset = (
            physical_block_idx * block_size * num_kv_heads
            + logical_block_offset * num_kv_heads
            + kv_head_idx
        )  # (BLOCK_N, )

        # Load Kj block and compute attention scores: S = qK^T / sqrt(d)
        k_j = tl.load(
            k_cache_ptr + kv_cache_offset, mask=kv_mask[:, None], other=0.0
        )  # shape(BLOCK_N, head_dim)
        if kv_cache_int8:  # dequantization
            k_scale = tl.load(
                k_scale_cache_ptr + kv_scale_offset, mask=kv_mask, other=0.0
            )  # shape(BLOCK_N, )
            k_j = (k_j.to(tl.float32) * k_scale[:, None]).to(q.dtype)
        s = tl.sum(k_j * q[None, :], axis=1) * scale  # shape(BLOCK_N, )
        s = tl.where(kv_mask, s, -1e10)

        # Online softmax, P = Softmax(S)
        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        m_i = m_new
        p = tl.exp(s - m_new)  # shape(BLOCK_N, )
        l_i = l_i * alpha + tl.sum(p)

        # Load Vj block and update O = PV
        v_j = tl.load(
            v_cache_ptr + kv_cache_offset, mask=kv_mask[:, None], other=0.0
        )  # shape (BLOCK_N, head_dim)
        if kv_cache_int8:  # dequantization
            v_scale = tl.load(
                v_scale_cache_ptr + kv_scale_offset, mask=kv_mask, other=0.0
            )  # shape(BLOCK_N, )
            v_j = (v_j.to(tl.float32) * v_scale[:, None]).to(q.dtype)
        acc = acc * alpha + tl.sum(
            p[:, None].to(v_j.dtype) * v_j, axis=0
        )  # shape(head_dim, )

    # Final normalization and store output
    tl.store(output_ptr + q_offset, acc / l_i)


def flash_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.

    Args:
        q: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        scale: attention scale factor
        block_tables: (batch_size, max_num_blocks)
        cache_seqlens: (batch_size, 1)

    Returns:
        output: (batch_size, num_heads, head_dim)

    """
    # make q contiguous
    q = q.contiguous()
    # allocate memory for output
    output = torch.empty_like(q)

    # get needed params
    batch_size, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    max_num_blocks = block_tables.shape[1]

    # choose chunk size for processing K/V caches
    BLOCK_N = 64 if head_dim <= 128 else 32

    kv_cache_int8 = k_cache.dtype == torch.int8
    if kv_cache_int8:
        assert k_scale_cache is not None and v_scale_cache is not None
        assert k_scale_cache.shape == v_scale_cache.shape
        assert k_scale_cache.shape == k_cache.shape[:-1]
    else:  # avoid to pass null pointer to kernel
        k_scale_cache = k_cache
        v_scale_cache = v_cache

    # each thread responsible for one head of one batch
    grid = (num_heads, batch_size)
    # launch batch_size x num_heads threads
    flash_attention_decode_with_cache_kernel[grid](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        output,
        scale,
        block_tables,
        cache_seqlens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        kv_cache_int8=kv_cache_int8,
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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale_cache = self.v_scale_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        kv_cache_int8 = k_cache.dtype == torch.int8  # whether to use int8 quantization for k/v cache
        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if kv_cache_int8:  # use int8 quantization
                store_kvcache(
                    k, v, k_cache, v_cache,
                    context.slot_mapping,
                    self.k_scale_cache,
                    self.v_scale_cache
                )
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:  # Prefill
            if context.block_tables is not None:  # chunked prefill
                k, v = k_cache, v_cache
            return flash_attention_prefill(
                q, k, v,
                scale=self.scale,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                block_tables=context.block_tables,
                k_scale_cache=self.k_scale_cache if kv_cache_int8 else None,
                v_scale_cache=self.v_scale_cache if kv_cache_int8 else None,
            )
        else:  # Decode
            return flash_attention_decode(
                q, k_cache, v_cache,
                scale=self.scale,
                cache_seqlens=context.context_lens,
                block_tables=context.block_tables,
                k_scale_cache=self.k_scale_cache if kv_cache_int8 else None,
                v_scale_cache=self.v_scale_cache if kv_cache_int8 else None,
            )
