from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import minivllm.layers.attention as attention_module
from minivllm.layers.attention import (
    Attention,
    flash_attention_decode,
    flash_attention_prefill,
    store_kvcache,
)
from minivllm.utils.context import reset_context, set_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


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


def symmetric_quantize_reference(tensor):
    tensor_f32 = tensor.float()
    abs_max = tensor_f32.abs().amax(dim=-1)
    scale = torch.where(
        abs_max > 0,
        abs_max / 127.0,
        torch.ones_like(abs_max),
    )
    scaled = torch.clamp(tensor_f32 / scale.unsqueeze(-1), -127.0, 127.0)
    quantized = torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        torch.ceil(scaled - 0.5),
    ).to(torch.int8)
    return quantized, scale


def make_quantized_cache():
    torch.manual_seed(0)
    shape = (4, 4, 2, 16)
    k_cache = torch.randint(
        -127, 128, shape, device="cuda", dtype=torch.int16
    ).to(torch.int8)
    v_cache = torch.randint(
        -127, 128, shape, device="cuda", dtype=torch.int16
    ).to(torch.int8)
    k_scale_cache = torch.rand(shape[:-1], device="cuda", dtype=torch.float32)
    v_scale_cache = torch.rand(shape[:-1], device="cuda", dtype=torch.float32)
    k_scale_cache = k_scale_cache * 0.04 + 0.01
    v_scale_cache = v_scale_cache * 0.04 + 0.01
    k_reference = (k_cache.float() * k_scale_cache.unsqueeze(-1)).half()
    v_reference = (v_cache.float() * v_scale_cache.unsqueeze(-1)).half()
    return (
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        k_reference,
        v_reference,
    )


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


class TestAttentionInt8Dispatch:

    def test_chunked_prefill_passes_int8_scale_caches(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, scale=0.25, num_kv_heads=1)
        layer.k_cache = torch.zeros(3, 4, 1, 4, dtype=torch.int8)
        layer.v_cache = torch.zeros(3, 4, 1, 4, dtype=torch.int8)
        layer.k_scale_cache = torch.ones(3, 4, 1, dtype=torch.float32)
        layer.v_scale_cache = torch.ones(3, 4, 1, dtype=torch.float32)
        query, key, value = make_qkv()
        slots = torch.tensor([4, 5, 6], dtype=torch.int32)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
        expected = torch.randn_like(query)
        store = MagicMock()
        prefill = MagicMock(return_value=expected)
        monkeypatch.setattr(attention_module, "store_kvcache", store)
        monkeypatch.setattr(attention_module, "flash_attention_prefill", prefill)
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 6], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=6,
            slot_mapping=slots,
            block_tables=block_tables,
        )

        output = layer(query, key, value)

        assert output is expected
        store.assert_called_once_with(
            key,
            value,
            layer.k_cache,
            layer.v_cache,
            slots,
            layer.k_scale_cache,
            layer.v_scale_cache,
        )
        args, kwargs = prefill.call_args
        assert args[1] is layer.k_cache
        assert args[2] is layer.v_cache
        assert kwargs["k_scale_cache"] is layer.k_scale_cache
        assert kwargs["v_scale_cache"] is layer.v_scale_cache

    def test_decode_passes_int8_scale_caches(self, monkeypatch):
        layer = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1)
        layer.k_cache = torch.zeros(3, 4, 1, 4, dtype=torch.int8)
        layer.v_cache = torch.zeros(3, 4, 1, 4, dtype=torch.int8)
        layer.k_scale_cache = torch.ones(3, 4, 1, dtype=torch.float32)
        layer.v_scale_cache = torch.ones(3, 4, 1, dtype=torch.float32)
        query, key, value = make_qkv()
        query, key, value = query[:2], key[:2], value[:2]
        slots = torch.tensor([5, 10], dtype=torch.int32)
        context_lens = torch.tensor([5, 7], dtype=torch.int32)
        block_tables = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
        expected = torch.randn_like(query)
        store = MagicMock()
        decode = MagicMock(return_value=expected)
        monkeypatch.setattr(attention_module, "store_kvcache", store)
        monkeypatch.setattr(attention_module, "flash_attention_decode", decode)
        set_context(
            is_prefill=False,
            slot_mapping=slots,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        output = layer(query, key, value)

        assert output is expected
        store.assert_called_once_with(
            key,
            value,
            layer.k_cache,
            layer.v_cache,
            slots,
            layer.k_scale_cache,
            layer.v_scale_cache,
        )
        args, kwargs = decode.call_args
        assert args[1] is layer.k_cache
        assert args[2] is layer.v_cache
        assert kwargs["k_scale_cache"] is layer.k_scale_cache
        assert kwargs["v_scale_cache"] is layer.v_scale_cache


@requires_cuda
class TestInt8KVCacheKernels:

    def test_store_quantizes_each_token_and_kv_head(self):
        num_tokens, num_kv_heads, head_dim = 3, 2, 16
        source = torch.arange(
            num_tokens * num_kv_heads * head_dim * 2,
            device="cuda",
            dtype=torch.float16,
        ).reshape(num_tokens, num_kv_heads, head_dim * 2)
        key = (source - 80.0)[..., ::2]
        value = (40.0 - source)[..., 1::2]
        key[0, 0].zero_()
        value[1, 1].zero_()
        assert not key.is_contiguous()
        assert not value.is_contiguous()

        k_cache = torch.zeros(
            3,
            4,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.int8,
        )
        v_cache = torch.zeros_like(k_cache)
        k_scale_cache = torch.zeros(3, 4, num_kv_heads, device="cuda")
        v_scale_cache = torch.zeros_like(k_scale_cache)
        slot_mapping = torch.tensor([1, 7, 8], device="cuda", dtype=torch.int32)

        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale_cache,
            v_scale_cache,
        )
        torch.cuda.synchronize()

        expected_key, expected_k_scale = symmetric_quantize_reference(key)
        expected_value, expected_v_scale = symmetric_quantize_reference(value)
        slots = slot_mapping.long()
        flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
        flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
        flat_k_scale = k_scale_cache.view(-1, num_kv_heads)
        flat_v_scale = v_scale_cache.view(-1, num_kv_heads)

        assert torch.equal(flat_k_cache[slots], expected_key)
        assert torch.equal(flat_v_cache[slots], expected_value)
        torch.testing.assert_close(
            flat_k_scale[slots], expected_k_scale, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            flat_v_scale[slots], expected_v_scale, rtol=1e-6, atol=1e-7
        )
        assert torch.count_nonzero(flat_k_cache[0]) == 0
        assert torch.count_nonzero(flat_v_cache[0]) == 0
        assert torch.count_nonzero(flat_k_scale[0]) == 0
        assert torch.count_nonzero(flat_v_scale[0]) == 0

    def test_decode_matches_explicitly_dequantized_cache(self):
        (
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            k_reference,
            v_reference,
        ) = make_quantized_cache()
        query = torch.randn(2, 4, 16, device="cuda", dtype=torch.float16)
        cache_seqlens = torch.tensor([6, 7], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor(
            [[2, 0], [3, 1]], device="cuda", dtype=torch.int32
        )
        scale = 16**-0.5

        expected = flash_attention_decode(
            query,
            k_reference,
            v_reference,
            scale,
            cache_seqlens,
            block_tables,
        )
        actual = flash_attention_decode(
            query,
            k_cache,
            v_cache,
            scale,
            cache_seqlens,
            block_tables,
            k_scale_cache,
            v_scale_cache,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_chunked_prefill_matches_explicitly_dequantized_cache(self):
        (
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            k_reference,
            v_reference,
        ) = make_quantized_cache()
        query = torch.randn(5, 4, 16, device="cuda", dtype=torch.float16)
        cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 6, 13], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor(
            [[2, 0], [3, 1]], device="cuda", dtype=torch.int32
        )
        scale = 16**-0.5

        expected = flash_attention_prefill(
            query,
            k_reference,
            v_reference,
            scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=3,
            max_seqlen_k=7,
            block_tables=block_tables,
        )
        actual = flash_attention_prefill(
            query,
            k_cache,
            v_cache,
            scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=3,
            max_seqlen_k=7,
            block_tables=block_tables,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
