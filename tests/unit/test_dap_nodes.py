import pytest
import torch
from dap_modules.dap_loader import DAP_Loader
from dap_modules.dap_inference import DAP_Inference
from dap_modules.dap_geometry import DAP_Panoramic_Mesh


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
    out_images, out_masks = inference.process(
        dap_model=mock_dap_model,
        image=dummy_image,
        invert_output=True,
        resize_input=True,
    )

    # Check output types and shapes
    assert torch.is_tensor(out_images)
    assert torch.is_tensor(out_masks)
    assert out_images.shape == (1, 512, 1024, 3)
    assert out_masks.shape == (1, 512, 1024)


def test_dap_geometry_init():
    """Verify DAP_Panoramic_Mesh can be instantiated"""
    mesh_gen = DAP_Panoramic_Mesh()
    input_types = mesh_gen.INPUT_TYPES()
    assert "depth" in input_types["required"]
    assert "mesh_scale" in input_types["required"]


def test_dap_geometry_process(dummy_image):
    """Test the mesh generation logic"""
    mesh_gen = DAP_Panoramic_Mesh()

    # Run process
    # depth is [1, 512, 1024, 3] from dummy_image fixture
    result = mesh_gen.generate_mesh(depth=dummy_image, mesh_scale=1.0, downsample=4)

    mesh = result[0]
    assert hasattr(mesh, "vertices")
    assert hasattr(mesh, "faces")
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
