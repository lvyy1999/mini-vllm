from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import minivllm.layers.attention as attention_module
from minivllm.layers.attention import Attention
from minivllm.utils.context import reset_context, set_context


@pytest.fixture(autouse=True)
def clean_context():
    reset_context()
    yield
    reset_context()


def make_qkv():
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn(3, 1, 4)
    return query, key, value


class TestAttentionPrefillDispatch:

    def test_regular_prefill_uses_current_key_and_value(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1)
        query, key, value = make_qkv()
        expected = torch.randn_like(query)
        prefill = MagicMock(return_value=expected)
        monkeypatch.setattr(attention_module, "flash_attention_prefill", prefill)
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
        )

        output = layer(query, key, value)

        assert output is expected
        args, kwargs = prefill.call_args
        assert args[0] is query
        assert args[1] is key
        assert args[2] is value
        assert kwargs["scale"] == 0.5
        assert kwargs["block_tables"] is None

    def test_prefill_stores_kv_when_cache_and_slot_mapping_exist(
        self, monkeypatch
    ):
        layer = Attention(num_heads=2, head_dim=4, num_kv_heads=1)
        layer.k_cache = torch.zeros(2, 4, 1, 4)
        layer.v_cache = torch.zeros(2, 4, 1, 4)
        query, key, value = make_qkv()
        slots = torch.tensor([0, 1, 2], dtype=torch.int32)
        store = MagicMock()
        monkeypatch.setattr(attention_module, "store_kvcache", store)
        monkeypatch.setattr(
            attention_module,
            "flash_attention_prefill",
            MagicMock(return_value=query),
        )
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=slots,
        )

        layer(query, key, value)

        store.assert_called_once_with(
            key, value, layer.k_cache, layer.v_cache, slots
        )

    def test_chunked_prefill_reads_key_and_value_from_cache(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, scale=0.25, num_kv_heads=1)
        layer.k_cache = torch.randn(3, 4, 1, 4)
        layer.v_cache = torch.randn(3, 4, 1, 4)
        query, key, value = make_qkv()
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
        expected = torch.randn_like(query)
        prefill = MagicMock(return_value=expected)
        monkeypatch.setattr(attention_module, "flash_attention_prefill", prefill)
        monkeypatch.setattr(attention_module, "store_kvcache", MagicMock())
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 6], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=6,
            slot_mapping=torch.tensor([4, 5, 6], dtype=torch.int32),
            block_tables=block_tables,
        )

        output = layer(query, key, value)

        assert output is expected
        args, kwargs = prefill.call_args
        assert args[0] is query
        assert args[1] is layer.k_cache
        assert args[2] is layer.v_cache
        assert kwargs["block_tables"] is block_tables


class TestAttentionDecodeDispatch:

    def test_decode_reads_paged_cache_metadata(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1)
        layer.k_cache = torch.randn(3, 4, 1, 4)
        layer.v_cache = torch.randn(3, 4, 1, 4)
        query, key, value = make_qkv()
        query = query[:2]
        key = key[:2]
        value = value[:2]
        context_lens = torch.tensor([5, 7], dtype=torch.int32)
        block_tables = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
        slots = torch.tensor([5, 10], dtype=torch.int32)
        expected = torch.randn_like(query)
        decode = MagicMock(return_value=expected)
        store = MagicMock()
        monkeypatch.setattr(attention_module, "flash_attention_decode", decode)
        monkeypatch.setattr(attention_module, "store_kvcache", store)
        set_context(
            is_prefill=False,
            slot_mapping=slots,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        output = layer(query, key, value)

        assert output is expected
        store.assert_called_once_with(
            key, value, layer.k_cache, layer.v_cache, slots
        )
        args, kwargs = decode.call_args
        assert args[0] is query
        assert args[1] is layer.k_cache
        assert args[2] is layer.v_cache
        assert kwargs["scale"] == 0.5
        assert kwargs["cache_seqlens"] is context_lens
        assert kwargs["block_tables"] is block_tables

    def test_empty_cache_skips_store(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, num_kv_heads=1)
        query, key, value = make_qkv()
        store = MagicMock()
        monkeypatch.setattr(attention_module, "store_kvcache", store)
        monkeypatch.setattr(
            attention_module,
            "flash_attention_decode",
            MagicMock(return_value=query),
        )
        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
            context_lens=torch.tensor([3], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
        )

        layer(query, key, value)

        store.assert_not_called()
