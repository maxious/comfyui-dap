import sys
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock

# 1. Add plugin and dap_core to path
plugin_dir = Path(__file__).parent.parent
sys.path.insert(0, str(plugin_dir))
sys.path.insert(0, str(plugin_dir / "dap_core"))

# 2. Mock ComfyUI modules
mock_folder_paths = MagicMock()
mock_folder_paths.models_dir = str(plugin_dir / "models_mock")
mock_folder_paths.get_full_path.return_value = None
sys.modules["folder_paths"] = mock_folder_paths

mock_mm = MagicMock()
mock_mm.get_torch_device.return_value = torch.device("cpu")
mock_mm.unet_offload_device.return_value = torch.device("cpu")
sys.modules["comfy.model_management"] = mock_mm

mock_comfy = MagicMock()
mock_comfy.model_management = mock_mm
sys.modules["comfy"] = mock_comfy

mock_utils = MagicMock()
sys.modules["comfy.utils"] = mock_utils

# Mock huggingface_hub
mock_hf = MagicMock()
mock_hf.hf_hub_url.return_value = "http://mock-url"
sys.modules["huggingface_hub"] = mock_hf

# Mock requests
mock_requests = MagicMock()
sys.modules["requests"] = mock_requests


# 3. Fixtures
@pytest.fixture
def dummy_image():
    """Create a dummy 2:1 panorama-style image tensor [1, 512, 1024, 3]"""
    return torch.rand((1, 512, 1024, 3))


@pytest.fixture
def mock_dap_model():
    inner_model = MagicMock(spec=torch.nn.Module)

    def side_effect(x):
        B, C, H, W = x.shape
        return {
            "pred_depth": torch.ones((B, 1, H, W)),
            "pred_mask": torch.ones((B, 1, H, W)),
        }

    inner_model.side_effect = side_effect
    inner_model.to.return_value = inner_model
    inner_model.half.return_value = inner_model
    inner_model.eval.return_value = inner_model

    patcher = MagicMock()
    patcher.model = inner_model
    patcher.model_patches_models.return_value = []

    return {"model": patcher, "precision": "fp32"}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: tests that require downloading weights"
    )


def pytest_ignore_collect(collection_path, config):
    """Ignore __init__.py files during collection to avoid relative import errors"""
    if collection_path.name == "__init__.py":
        return True
    return False
