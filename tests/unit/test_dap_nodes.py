import pytest
import torch
from dap_modules.dap_loader import DAP_Loader
from dap_modules.dap_inference import DAP_Inference


def test_dap_loader_init():
    """Verify DAP_Loader can be instantiated and has correct INPUT_TYPES"""
    loader = DAP_Loader()
    input_types = loader.INPUT_TYPES()
    assert "model_size" in input_types["required"]
    assert "precision" in input_types["required"]


def test_dap_inference_init():
    """Verify DAP_Inference can be instantiated and has correct INPUT_TYPES"""
    inference = DAP_Inference()
    input_types = inference.INPUT_TYPES()
    assert "dap_model" in input_types["required"]
    assert "image" in input_types["required"]


def test_dap_inference_process_mock(mock_dap_model, dummy_image):
    """Test the inference processing logic using a mock model"""
    inference = DAP_Inference()

    # Run process
    # INPUTS: dap_model, image, invert_output, resize_input
    out_images, out_masks = inference.process(
        dap_model=mock_dap_model,
        image=dummy_image,
        invert_output=True,
        resize_input=True,
    )

    # Check output types and shapes
    assert torch.is_tensor(out_images)
    assert torch.is_tensor(out_masks)

    # Input dummy_image was [1, 512, 1024, 3]
    # Output should be [1, 512, 1024, 3] for depth image
    # and [1, 512, 1024] for mask
    assert out_images.shape == (1, 512, 1024, 3)
    assert out_masks.shape == (1, 512, 1024)

    # Verify values are in 0..1 range
    assert out_images.min() >= 0.0
    assert out_images.max() <= 1.0
