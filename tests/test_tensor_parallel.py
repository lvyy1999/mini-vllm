import socket
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from minivllm.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding
from minivllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from minivllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from minivllm.utils.context import reset_context, set_context


WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT_SECONDS = 90

requires_two_cuda_devices = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < WORLD_SIZE
    or not dist.is_nccl_available(),
    reason="requires two CUDA devices and the NCCL backend",
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank: int, world_size: int, port: int) -> torch.device:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    return torch.device("cuda", rank)


def _values(
    *shape: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scale: float = 0.01,
    offset: int = 0,
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(offset, offset + count, device=device, dtype=torch.float32)
    values = ((values % 29) - 14) * scale
    return values.reshape(shape).to(dtype)


def _cleanup_process_group() -> None:
    reset_context()
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


def _tp_layers_worker(rank: int, world_size: int, port: int) -> None:
    device = _init_process_group(rank, world_size, port)
    try:
        inputs = _values(3, 6, device=device)
        full_weight = _values(8, 6, device=device, offset=7)
        full_bias = _values(8, device=device, offset=19)
        column = ColumnParallelLinear(6, 8, bias=True).to(device)
        column.weight_loader(column.weight, full_weight)
        column.weight_loader(column.bias, full_bias)

        local_output = column(inputs)
        output_shards = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(output_shards, local_output)
        column_output = torch.cat(output_shards, dim=-1)
        torch.testing.assert_close(
            column_output,
            F.linear(inputs, full_weight, full_bias),
            rtol=1e-5,
            atol=1e-5,
        )

        row_inputs = _values(3, 8, device=device, offset=31)
        row_weight = _values(5, 8, device=device, offset=47)
        row_bias = _values(5, device=device, offset=73)
        row = RowParallelLinear(8, 5, bias=True).to(device)
        row.weight_loader(row.weight, row_weight)
        row.bias.data.copy_(row_bias)

        local_row_inputs = row_inputs.chunk(world_size, dim=-1)[rank].contiguous()
        row_output = row(local_row_inputs)
        torch.testing.assert_close(
            row_output,
            F.linear(row_inputs, row_weight, row_bias),
            rtol=1e-5,
            atol=1e-5,
        )

        hidden_size = 8
        intermediate_size = 12
        mlp_inputs = _values(4, hidden_size, device=device, offset=89)
        gate_weight = _values(
            intermediate_size, hidden_size, device=device, offset=101
        )
        up_weight = _values(
            intermediate_size, hidden_size, device=device, offset=149
        )
        down_weight = _values(
            hidden_size, intermediate_size, device=device, offset=197
        )
        mlp = Qwen3MLP(hidden_size, intermediate_size).to(device)
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, gate_weight, 0
        )
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, up_weight, 1
        )
        mlp.down_proj.weight_loader(mlp.down_proj.weight, down_weight)

        mlp_output = mlp(mlp_inputs)
        mlp_reference = F.linear(
            F.silu(F.linear(mlp_inputs, gate_weight))
            * F.linear(mlp_inputs, up_weight),
            down_weight,
        )
        torch.testing.assert_close(
            mlp_output, mlp_reference, rtol=1e-5, atol=1e-5
        )

        vocab_size = 7
        embedding_size = 6
        embedding_weight = _values(
            vocab_size, embedding_size, device=device, offset=251
        )
        token_ids = torch.tensor([0, 3, 4, 6, 1], device=device)
        embedding = VocabParallelEmbedding(vocab_size, embedding_size).to(device)
        embedding.weight_loader(embedding.weight, embedding_weight)

        embedding_output = embedding(token_ids)
        torch.testing.assert_close(
            embedding_output,
            F.embedding(token_ids, embedding_weight),
            rtol=1e-5,
            atol=1e-5,
        )
        if rank == world_size - 1:
            torch.testing.assert_close(
                embedding.weight[-1], torch.zeros_like(embedding.weight[-1])
            )

        lm_head = ParallelLMHead(vocab_size, embedding_size).to(device)
        lm_head.weight_loader(lm_head.weight, embedding_weight)
        hidden_states = _values(2, embedding_size, device=device, offset=293)
        set_context(is_prefill=False)
        decode_logits = lm_head(hidden_states)
        if rank == 0:
            torch.testing.assert_close(
                decode_logits,
                F.linear(hidden_states, embedding_weight),
                rtol=1e-5,
                atol=1e-5,
            )
        else:
            assert decode_logits.shape == (2, lm_head.shard_vocab_size)

        prefill_states = _values(5, embedding_size, device=device, offset=317)
        cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q)
        prefill_logits = lm_head(prefill_states)
        if rank == 0:
            torch.testing.assert_close(
                prefill_logits,
                F.linear(prefill_states[[1, 4]], embedding_weight),
                rtol=1e-5,
                atol=1e-5,
            )

        torch.cuda.synchronize(device)
    finally:
        _cleanup_process_group()


def _rms_norm_reference(
    inputs: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    variance = inputs.pow(2).mean(dim=-1, keepdim=True)
    inputs = inputs * torch.rsqrt(variance + eps)
    return inputs.to(origin_dtype) * weight


def _causal_gqa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    repeats = query.size(1) // key.size(1)
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    causal_mask = torch.triu(
        torch.ones(
            query.size(0), key.size(0), dtype=torch.bool, device=query.device
        ),
        diagonal=1,
    )
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value.float())
    return output.to(query.dtype)


def _tp_attention_worker(rank: int, world_size: int, port: int) -> None:
    device = _init_process_group(rank, world_size, port)
    try:
        dtype = torch.float16
        sequence_length = 5
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16

        attention = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_position=16,
            qkv_bias=False,
        ).to(device=device, dtype=dtype)
        attention.rotary_emb.cos_sin_cache = (
            attention.rotary_emb.cos_sin_cache.float()
        )

        query_weight = _values(
            num_heads * head_dim,
            hidden_size,
            device=device,
            dtype=dtype,
            scale=0.002,
            offset=11,
        )
        key_weight = _values(
            num_kv_heads * head_dim,
            hidden_size,
            device=device,
            dtype=dtype,
            scale=0.002,
            offset=37,
        )
        value_weight = _values(
            num_kv_heads * head_dim,
            hidden_size,
            device=device,
            dtype=dtype,
            scale=0.002,
            offset=71,
        )
        query_norm_weight = _values(
            head_dim,
            device=device,
            dtype=dtype,
            scale=0.01,
            offset=97,
        ) + 1.0
        key_norm_weight = _values(
            head_dim,
            device=device,
            dtype=dtype,
            scale=0.01,
            offset=113,
        ) + 1.0
        output_weight = _values(
            hidden_size,
            num_heads * head_dim,
            device=device,
            dtype=dtype,
            scale=0.002,
            offset=163,
        )
        inputs = _values(
            sequence_length,
            hidden_size,
            device=device,
            dtype=dtype,
            scale=0.01,
            offset=211,
        )
        positions = torch.arange(sequence_length, device=device)

        attention.qkv_proj.weight_loader(
            attention.qkv_proj.weight, query_weight, "q"
        )
        attention.qkv_proj.weight_loader(
            attention.qkv_proj.weight, key_weight, "k"
        )
        attention.qkv_proj.weight_loader(
            attention.qkv_proj.weight, value_weight, "v"
        )
        attention.q_norm.weight.data.copy_(query_norm_weight)
        attention.k_norm.weight.data.copy_(key_norm_weight)
        attention.o_proj.weight_loader(attention.o_proj.weight, output_weight)

        cu_seqlens = torch.tensor(
            [0, sequence_length], dtype=torch.int32, device=device
        )
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=sequence_length,
            max_seqlen_k=sequence_length,
        )
        actual = attention(inputs, positions)

        query = F.linear(inputs, query_weight).view(
            sequence_length, num_heads, head_dim
        )
        key = F.linear(inputs, key_weight).view(
            sequence_length, num_kv_heads, head_dim
        )
        value = F.linear(inputs, value_weight).view(
            sequence_length, num_kv_heads, head_dim
        )
        query = _rms_norm_reference(query, query_norm_weight, attention.q_norm.eps)
        key = _rms_norm_reference(key, key_norm_weight, attention.k_norm.eps)
        query, key = attention.rotary_emb(positions, query, key)
        reference_attention = _causal_gqa_reference(
            query, key, value, attention.scale
        )
        reference = F.linear(reference_attention.flatten(1), output_weight)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
        torch.cuda.synchronize(device)
    finally:
        _cleanup_process_group()


def _run_two_gpu_test(worker) -> None:
    mp.spawn(
        worker,
        args=(WORLD_SIZE, _find_free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )


class TestTwoGpuTensorParallel:

    @requires_two_cuda_devices
    def test_tp_layers_match_dense_references(self):
        _run_two_gpu_test(_tp_layers_worker)

    @requires_two_cuda_devices
    def test_qwen_attention_matches_dense_reference(self):
        _run_two_gpu_test(_tp_attention_worker)
