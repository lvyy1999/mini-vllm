from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.distributed as dist
import torch.nn.functional as F

from minivllm.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding
from minivllm.utils.context import reset_context, set_context


def set_tensor_parallel(monkeypatch, rank: int, world_size: int) -> None:
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)


@pytest.fixture(autouse=True)
def clean_context():
    reset_context()
    yield
    reset_context()


class TestVocabParallelEmbedding:

    def test_single_rank_matches_embedding_reference(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = VocabParallelEmbedding(vocab_size=4, hidden_size=2)
        weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [-1.0, 4.0]]
        )
        token_ids = torch.tensor([3, 0, 2])
        layer.weight.weight_loader(layer.weight, weight)

        output = layer(token_ids)

        torch.testing.assert_close(output, F.embedding(token_ids, weight))

    def test_last_rank_loads_real_rows_and_zeroes_padding(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        layer = VocabParallelEmbedding(vocab_size=5, hidden_size=2)
        full_weight = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        layer.weight.weight_loader(layer.weight, full_weight)

        expected = torch.cat([full_weight[3:5], torch.zeros(1, 2)], dim=0)
        torch.testing.assert_close(layer.weight, expected)
        assert layer.padded_vocab_size == 6
        assert layer.shard_vocab_size == 3

    def test_masks_tokens_owned_by_other_ranks(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=1, world_size=2)
        all_reduce = MagicMock()
        monkeypatch.setattr(dist, "all_reduce", all_reduce)
        layer = VocabParallelEmbedding(vocab_size=5, hidden_size=2)
        full_weight = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        layer.weight.weight_loader(layer.weight, full_weight)
        token_ids = torch.tensor([0, 3, 4, 5])

        output = layer(token_ids)

        expected = torch.stack(
            [torch.zeros(2), full_weight[3], full_weight[4], torch.zeros(2)]
        )
        torch.testing.assert_close(output, expected)
        all_reduce.assert_called_once_with(output, op=dist.ReduceOp.SUM)


class TestParallelLMHead:

    def test_decode_computes_logits_for_every_input_row(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = ParallelLMHead(vocab_size=3, hidden_size=2)
        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        layer.weight.data.copy_(weight)
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        set_context(is_prefill=False)

        logits = layer(hidden_states)

        torch.testing.assert_close(logits, F.linear(hidden_states, weight))

    def test_prefill_selects_each_sequences_last_hidden_state(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=1)
        layer = ParallelLMHead(vocab_size=3, hidden_size=2)
        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        layer.weight.data.copy_(weight)
        hidden_states = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        )
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        )

        logits = layer(hidden_states)

        expected_hidden_states = hidden_states[[1, 4]]
        torch.testing.assert_close(logits, F.linear(expected_hidden_states, weight))

    def test_rank_zero_gathers_and_trims_padded_vocabulary(self, monkeypatch):
        set_tensor_parallel(monkeypatch, rank=0, world_size=2)

        def fake_gather(local_logits, gather_list, dst):
            assert dst == 0
            gather_list[0].copy_(local_logits)
            gather_list[1].copy_(local_logits + 10)

        gather = MagicMock(side_effect=fake_gather)
        monkeypatch.setattr(dist, "gather", gather)
        layer = ParallelLMHead(vocab_size=5, hidden_size=2)
        layer.weight.data.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        hidden_states = torch.tensor([[1.0, 2.0]])
        set_context(is_prefill=False)

        logits = layer(hidden_states)

        torch.testing.assert_close(
            logits, torch.tensor([[1.0, 2.0, 3.0, 11.0, 12.0]])
        )
        assert logits.shape == (1, 5)
        gather.assert_called_once()
