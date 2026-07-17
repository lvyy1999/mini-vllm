import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F

from minivllm.layers.activation import SiLUAndMul
from minivllm.layers.rmsnorm import RMSNorm
from minivllm.layers.rotary_embedding import (
    RotaryEmbedding,
    apply_rope_adjacent,
    apply_rotary_pos_emb,
)
from minivllm.layers.sampler import SamplerLayer


class TestSiLUAndMul:

    def test_matches_reference_formula(self):
        inputs = torch.tensor(
            [[-1.0, 0.5, 2.0, 3.0], [1.5, -2.0, -1.0, 4.0]]
        )
        gate, values = inputs.chunk(2, dim=-1)

        output = SiLUAndMul()(inputs)

        torch.testing.assert_close(output, F.silu(gate) * values)
        assert output.shape == (2, 2)


class TestRMSNorm:

    def test_matches_float32_reference(self):
        layer = RMSNorm(hidden_size=4, eps=1e-6)
        layer.weight.data.copy_(torch.tensor([1.0, 0.5, 2.0, -1.0]))
        inputs = torch.tensor([[1.0, 2.0, -3.0, 4.0]])
        expected = inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected * layer.weight

        output = layer(inputs.clone())

        torch.testing.assert_close(output, expected)

    def test_preserves_low_precision_dtype(self):
        layer = RMSNorm(hidden_size=4)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float16)

        output = layer(inputs)

        assert output.dtype == torch.float16

    def test_residual_path_returns_sum_and_normalized_output(self):
        layer = RMSNorm(hidden_size=4, eps=1e-6)
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
        residual = torch.tensor([[0.5, -1.0, 2.0, 0.0]], dtype=torch.float16)
        expected_residual = (inputs.float() + residual.float()).to(torch.float16)
        normalized = expected_residual.float()
        expected_output = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + layer.eps
        )
        expected_output = (expected_output * layer.weight).to(torch.float16)

        output, updated_residual = layer(inputs, residual)

        torch.testing.assert_close(updated_residual, expected_residual)
        torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-3)


class TestRotaryFunctions:

    def test_adjacent_rotation_uses_even_odd_pairs(self):
        inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        cos = torch.zeros(1, 1, 2)
        sin = torch.ones(1, 1, 2)

        output = apply_rope_adjacent(inputs, cos, sin)

        expected = torch.tensor([[[-2.0, 1.0, -4.0, 3.0]]])
        torch.testing.assert_close(output, expected)

    def test_half_rotation_matches_reference(self):
        inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        cos = torch.zeros(1, 1, 2)
        sin = torch.ones(1, 1, 2)

        output = apply_rotary_pos_emb(inputs, cos, sin)

        expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])
        torch.testing.assert_close(output, expected)

    def test_rotary_embedding_position_zero_is_identity(self):
        layer = RotaryEmbedding(head_dim=4, max_position=8)
        positions = torch.tensor([0, 0])
        query = torch.randn(2, 3, 4)
        key = torch.randn(2, 1, 4)

        rotated_query, rotated_key = layer(positions, query, key)

        torch.testing.assert_close(rotated_query, query)
        torch.testing.assert_close(rotated_key, key)
        assert layer.cos_sin_cache.shape == (8, 1, 4)

    def test_rotary_embedding_matches_cached_reference(self):
        layer = RotaryEmbedding(head_dim=4, max_position=8, base=100.0)
        positions = torch.tensor([1, 3])
        query = torch.randn(2, 2, 4)
        key = torch.randn(2, 1, 4)
        cos, sin = layer.cos_sin_cache[positions].chunk(2, dim=-1)

        rotated_query, rotated_key = layer(positions, query, key)

        torch.testing.assert_close(
            rotated_query, apply_rotary_pos_emb(query, cos, sin)
        )
        torch.testing.assert_close(rotated_key, apply_rotary_pos_emb(key, cos, sin))

    def test_llama3_scaled_cache_is_finite(self):
        layer = RotaryEmbedding(
            head_dim=8,
            max_position=16,
            is_llama3=True,
            llama3_rope_factor=8.0,
            llama3_rope_high_freq_factor=4.0,
            llama3_rope_low_freq_factor=1.0,
            llama3_rope_original_max_position_embeddings=8,
        )

        assert torch.isfinite(layer.cos_sin_cache).all()
        torch.testing.assert_close(
            layer.cos_sin_cache[0], torch.tensor([[1.0] * 4 + [0.0] * 4])
        )


class TestSamplerLayer:

    def test_returns_one_token_per_row(self):
        logits = torch.tensor(
            [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0]]
        )
        temperatures = torch.tensor([0.5, 2.0])

        output = SamplerLayer()(logits, temperatures)

        assert output.dtype == torch.int64
        assert output.tolist() == [0, 1]

    def test_sampled_tokens_stay_inside_vocabulary(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 11)
        temperatures = torch.linspace(0.5, 1.5, 32)

        output = SamplerLayer()(logits, temperatures)

        assert output.shape == (32,)
        assert torch.all(output >= 0)
        assert torch.all(output < 11)
