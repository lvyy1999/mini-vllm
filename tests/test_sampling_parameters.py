import pytest

from minivllm.sampling_parameters import SamplingParams


class TestSamplingParams:

    def test_defaults(self):
        params = SamplingParams()

        assert params.temperature == 1.0
        assert params.max_tokens == 64
        assert params.ignore_eos is False

    def test_custom_values(self):
        params = SamplingParams(temperature=0.7, max_tokens=128, ignore_eos=True)

        assert params.temperature == 0.7
        assert params.max_tokens == 128
        assert params.ignore_eos is True

    @pytest.mark.parametrize("temperature", [0.1, 1.0, 2.0, 1.1e-10])
    def test_accepts_positive_sampling_temperatures(self, temperature):
        assert SamplingParams(temperature=temperature).temperature == temperature

    @pytest.mark.parametrize("temperature", [-1.0, 0.0, 1e-10])
    def test_rejects_greedy_or_invalid_temperatures(self, temperature):
        with pytest.raises(AssertionError, match="greedy sampling"):
            SamplingParams(temperature=temperature)
