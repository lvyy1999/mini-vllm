from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

import torch.nn as nn

import minivllm.utils.loader as loader_module
from minivllm.utils.loader import default_weight_loader, load_weights_from_checkpoint


class FakeSafeFile:

    def __init__(self, weights, get_tensor_calls):
        self.weights = weights
        self.get_tensor_calls = get_tensor_calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return self.weights.keys()

    def get_tensor(self, name):
        self.get_tensor_calls.append(name)
        return self.weights[name]


def install_fake_safe_open(monkeypatch, weights_by_file):
    opened_files = []
    get_tensor_calls = []

    def fake_safe_open(file_path, framework, device):
        assert framework == "pt"
        assert device == "cpu"
        filename = Path(file_path).name
        opened_files.append(filename)
        return FakeSafeFile(weights_by_file[filename], get_tensor_calls)

    monkeypatch.setattr(loader_module, "safe_open", fake_safe_open)
    return opened_files, get_tensor_calls


class TestDefaultWeightLoader:

    def test_copies_matching_tensor(self):
        parameter = nn.Parameter(torch.zeros(2, 3))
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        default_weight_loader(parameter, weight)

        torch.testing.assert_close(parameter, weight)

    def test_rejects_shape_mismatch_without_modifying_parameter(self):
        parameter = nn.Parameter(torch.ones(2, 3))
        original = parameter.detach().clone()

        with pytest.raises(ValueError, match="Shape mismatch"):
            default_weight_loader(parameter, torch.zeros(3, 2))

        torch.testing.assert_close(parameter, original)


class TestCheckpointResolution:

    def test_local_directory_without_safetensors_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="No .safetensors files"):
            load_weights_from_checkpoint(nn.Linear(2, 2), str(tmp_path))

    def test_download_error_contains_model_name_and_original_reason(self, monkeypatch):
        huggingface_hub = pytest.importorskip("huggingface_hub")
        download = MagicMock(side_effect=RuntimeError("network unavailable"))
        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

        with pytest.raises(ValueError) as exc_info:
            load_weights_from_checkpoint(nn.Linear(2, 2), "org/missing-model")

        message = str(exc_info.value)
        assert "org/missing-model" in message
        assert "network unavailable" in message


class TestCheckpointLoading:

    def test_loads_files_in_sorted_order_and_reads_each_tensor_once(
        self, monkeypatch, tmp_path
    ):
        class TwoParameterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Parameter(torch.zeros(2))
                self.second = nn.Parameter(torch.zeros(2))

        (tmp_path / "b.safetensors").touch()
        (tmp_path / "a.safetensors").touch()
        first_weight = torch.tensor([1.0, 2.0])
        second_weight = torch.tensor([3.0, 4.0])
        opened_files, get_tensor_calls = install_fake_safe_open(
            monkeypatch,
            {
                "a.safetensors": {"first": first_weight},
                "b.safetensors": {"second": second_weight},
            },
        )
        model = TwoParameterModel()

        load_weights_from_checkpoint(model, str(tmp_path))

        assert opened_files == ["a.safetensors", "b.safetensors"]
        assert get_tensor_calls == ["first", "second"]
        torch.testing.assert_close(model.first, first_weight)
        torch.testing.assert_close(model.second, second_weight)

    def test_routes_packed_weights_to_parameter_loader(self, monkeypatch, tmp_path):
        class PackedModel(nn.Module):
            packed_module_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
            }

            def __init__(self):
                super().__init__()
                self.qkv_proj = nn.Parameter(torch.zeros(4, 2))
                self.loader_calls = []

                def packed_loader(param, weight, weight_id):
                    self.loader_calls.append((weight_id, weight.detach().clone()))
                    offset = 0 if weight_id == "q" else 2
                    param.data[offset : offset + 2].copy_(weight)

                self.qkv_proj.weight_loader = packed_loader

        (tmp_path / "model.safetensors").touch()
        query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        key = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        _, get_tensor_calls = install_fake_safe_open(
            monkeypatch,
            {"model.safetensors": {"q_proj": query, "k_proj": key}},
        )
        model = PackedModel()

        load_weights_from_checkpoint(model, str(tmp_path))

        assert [weight_id for weight_id, _ in model.loader_calls] == ["q", "k"]
        assert get_tensor_calls == ["q_proj", "k_proj"]
        torch.testing.assert_close(model.qkv_proj, torch.cat([query, key], dim=0))

    def test_reports_unloaded_model_parameters(self, monkeypatch, tmp_path, capsys):
        class PartiallyLoadedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = nn.Parameter(torch.zeros(1))
                self.missing = nn.Parameter(torch.ones(1))

        (tmp_path / "model.safetensors").touch()
        install_fake_safe_open(
            monkeypatch,
            {"model.safetensors": {"loaded": torch.tensor([2.0])}},
        )

        load_weights_from_checkpoint(PartiallyLoadedModel(), str(tmp_path))
        output = capsys.readouterr().out

        assert "Successfully load 1 model weights from checkpoint." in output
        assert "missing" in output
