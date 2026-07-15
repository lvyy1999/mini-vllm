import os

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig

# os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def default_weight_loader(param: nn.Parameter, weight: torch.Tensor):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(
            f"Shape mismatch: param {param.shape} vs weight {weight.shape}"
        )
    param.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up_proj weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None
    # First, try local paths
    if model_name_or_path.startswith("~"):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        from huggingface_hub import snapshot_download

        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=[
                    "*.msgpack",
                    "*.h5",
                    "*.bin",
                ],  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [
        f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
    ]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Load weights from HF model into our model
    loaded_params = set()
    packed_modules_mapping = getattr(model, "packed_module_mapping", {})
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                weight = f.get_tensor(weight_name)
                for k in packed_modules_mapping.keys():
                    if k in weight_name:
                        v, id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        loaded_params.add(param_name)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, weight, id)
                        break
                else:
                    loaded_params.add(weight_name)
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))

    # Check for model parameters that weren't loaded
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'=' * 80}")
    print(f"Weight Loading Summary:")
    print(f"{'=' * 80}")
    print(f"Successfully loaded: {len(loaded_params)} parameter groups")

    if unloaded_params:
        print(
            f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:"
        )
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f"  ... and {len(unloaded_params) - 15} more")
    print(f"{'=' * 80}")
