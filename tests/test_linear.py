from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.distributed as dist
import torch.nn.functional as F

from minivllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def set_tensor_parallel(monkeypatch, rank: int, world_size: int) -> None:
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)


class TestReplicatedLinear:

    def test_loads_full_parameters_and_matches_linear(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = ReplicatedLinear(3, 2, bias=True)
        weight = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
        bias = torch.tensor([0.25, -0.5])
        inputs = torch.tensor([[2.0, -1.0, 0.5]])

        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        output = layer(inputs)

        torch.testing.assert_close(output, F.linear(inputs, weight, bias))
        assert layer.tp_rank == 0
        assert layer.tp_size == 1

    def test_bias_can_be_disabled(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)

        layer = ReplicatedLinear(3, 2, bias=False)

        assert layer.bias is None


class TestColumnParallelLinear:

    def test_rank_loads_its_output_shard(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        layer = ColumnParallelLinear(3, 4, bias=True)
        full_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        full_bias = torch.arange(4, dtype=torch.float32)

        layer.weight.weight_loader(layer.weight, full_weight)
        layer.bias.weight_loader(layer.bias, full_bias)

        torch.testing.assert_close(layer.weight, full_weight[2:4])
        torch.testing.assert_close(layer.bias, full_bias[2:4])
        assert layer.weight.shape == (2, 3)
        assert layer.tp_dim == 0

    def test_forward_returns_local_output_shard(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=2)
        layer = ColumnParallelLinear(2, 4, bias=False)
        full_weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        )
        inputs = torch.tensor([[3.0, 4.0]])
        layer.weight.weight_loader(layer.weight, full_weight)

        output = layer(inputs)

        torch.testing.assert_close(output, F.linear(inputs, full_weight[:2]))

    def test_requires_divisible_output_size(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=2)

        with pytest.raises(AssertionError, match="Output size"):
            ColumnParallelLinear(3, 5)


class TestMergedColumnParallelLinear:

    def test_loads_each_merged_matrix_into_its_local_offset(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        layer = MergedColumnParallelLinear(2, [4, 2], bias=False)
        first = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        second = torch.arange(100, 104, dtype=torch.float32).reshape(2, 2)

        layer.weight.weight_loader(layer.weight, first, 0)
        layer.weight.weight_loader(layer.weight, second, 1)

        expected = torch.cat([first[2:4], second[1:2]], dim=0)
        torch.testing.assert_close(layer.weight, expected)


class TestQKVColumnParallelLinear:

    def test_loads_q_k_v_shards_into_expected_ranges(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        layer = QKVColumnParallelLinear(
            hidden_size=2,
            head_dim=2,
            num_heads=4,
            num_kv_heads=2,
        )
        query = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        key = torch.arange(100, 108, dtype=torch.float32).reshape(4, 2)
        value = torch.arange(200, 208, dtype=torch.float32).reshape(4, 2)

        layer.weight.weight_loader(layer.weight, query, "q")
        layer.weight.weight_loader(layer.weight, key, "k")
        layer.weight.weight_loader(layer.weight, value, "v")

        expected = torch.cat([query[4:8], key[2:4], value[2:4]], dim=0)
        torch.testing.assert_close(layer.weight, expected)
        assert layer.num_heads == 2
        assert layer.num_kv_heads == 1

    def test_rejects_unknown_qkv_identifier(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = QKVColumnParallelLinear(2, 2, 2, 2)

        with pytest.raises(AssertionError, match="load_weight_id"):
            layer.weight.weight_loader(layer.weight, torch.empty(4, 2), "x")


class TestRowParallelLinear:

    def test_rank_loads_its_input_shard(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        layer = RowParallelLinear(4, 2, bias=False)
        full_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        layer.weight.weight_loader(layer.weight, full_weight)

        torch.testing.assert_close(layer.weight, full_weight[:, 2:4])
        assert layer.weight.shape == (2, 2)
        assert layer.tp_dim == 1

    def test_forward_reduces_partial_outputs_and_omits_nonzero_rank_bias(
        self, monkeypatch
    ):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        all_reduce = MagicMock()
        monkeypatch.setattr(dist, "all_reduce", all_reduce)
        layer = RowParallelLinear(4, 2, bias=True)
        layer.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.bias.data.copy_(torch.tensor([100.0, 100.0]))
        inputs = torch.tensor([[2.0, 3.0]])

        output = layer(inputs)

        torch.testing.assert_close(output, F.linear(inputs, layer.weight, None))
        all_reduce.assert_called_once_with(output, op=dist.ReduceOp.SUM)

    def test_rank_zero_applies_bias(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = RowParallelLinear(2, 2, bias=True)
        layer.weight.data.copy_(torch.eye(2))
        layer.bias.data.copy_(torch.tensor([1.0, -1.0]))
        inputs = torch.tensor([[2.0, 3.0]])

        output = layer(inputs)

        torch.testing.assert_close(output, torch.tensor([[3.0, 2.0]]))

    def test_requires_divisible_input_size(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=2)

        with pytest.raises(AssertionError, match="Input size"):
            RowParallelLinear(5, 3)
