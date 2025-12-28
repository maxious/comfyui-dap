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
# We use simple objects instead of MagicMock for the base modules to avoid weird recursion
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = str(plugin_dir / "models_mock")
mock_folder_paths.get_full_path = lambda *args, **kwargs: None
mock_folder_paths.get_filename_list = lambda *args, **kwargs: []
sys.modules["folder_paths"] = mock_folder_paths

mock_mm = type("model_management", (), {})()
mock_mm.get_torch_device = lambda: torch.device("cpu")
mock_mm.unet_offload_device = lambda: torch.device("cpu")
mock_mm.load_model_gpu = lambda x: None
mock_mm.soft_empty_cache = lambda: None
mock_mm.is_device_mps = lambda x: False
mock_mm.get_autocast_device = lambda x: "cpu"

mock_comfy = type("comfy", (), {})()
mock_comfy.model_management = mock_mm
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = mock_mm

# 3. Fixtures


@pytest.fixture
def dummy_image():
    """Create a dummy 2:1 panorama-style image tensor [1, 512, 1024, 3]"""
    return torch.rand((1, 512, 1024, 3))


@pytest.fixture
def mock_dap_model():
    """Create a mock model object that behaves like what DAP_Loader returns"""
    model = MagicMock(spec=torch.nn.Module)

    # Mock the forward pass to return a dict with tensors
    def side_effect(x):
        B, C, H, W = x.shape
        return {
            "pred_depth": torch.ones((B, 1, H, W)),
            "pred_mask": torch.ones((B, 1, H, W)),
        }

    model.side_effect = side_effect
    model.to.return_value = model
    model.half.return_value = model
    model.eval.return_value = model
    return {"model": model, "precision": "fp32"}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: tests that require downloading weights"
    )


def pytest_ignore_collect(collection_path, config):
    """Ignore __init__.py files during collection to avoid relative import errors"""
    if collection_path.name == "__init__.py":
        return True
    return False
