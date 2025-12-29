import pytest
import torch
from dap_modules.dap_loader import DAP_Loader
from dap_modules.dap_inference import DAP_Inference
from dap_modules.dap_geometry import DAP_Panoramic_Mesh
from dap_modules.dap_conversions import DAP_ERP_to_Cubemap
from dap_modules.dap_normal import DAP_Normal_Map


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


def test_dap_conversion_init():
    """Verify DAP_ERP_to_Cubemap can be instantiated"""
    conv = DAP_ERP_to_Cubemap()
    input_types = conv.INPUT_TYPES()
    assert "image" in input_types["required"]
    assert "face_size" in input_types["required"]


def test_dap_conversion_process(dummy_image):
    """Test the ERP to Cubemap conversion logic"""
    conv = DAP_ERP_to_Cubemap()

    # Test six_faces output
    result = conv.convert(
        image=dummy_image, face_size=256, layout="six_faces", interpolation="linear"
    )
    # Output should be batch of 6
    assert result[0].shape == (6, 256, 256, 3)

    # Test cross layout output
    result_cross = conv.convert(
        image=dummy_image, face_size=256, layout="cross", interpolation="linear"
    )
    # Output should be single image 3x4 grid
    assert result_cross[0].shape == (1, 3 * 256, 4 * 256, 3)


def test_dap_normal_init():
    """Verify DAP_Normal_Map can be instantiated"""
    node = DAP_Normal_Map()
    input_types = node.INPUT_TYPES()
    assert "depth" in input_types["required"]
    assert "normal_standard" in input_types["required"]


def test_dap_normal_process(dummy_image):
    """Test normal map generation logic"""
    node = DAP_Normal_Map()

    # We use a downsampled version of dummy_image for speed if possible
    # but the node doesn't have a downsample param yet.
    # We'll just run it on the 512x1024 image.
    result = node.generate_normal(depth=dummy_image, normal_standard="ComfyUI")

    assert torch.is_tensor(result[0])
    assert result[0].shape == (1, 512, 1024, 3)
    # Check normalization (values should be around 0.5 for flat areas)
    assert result[0].min() >= 0.0
    assert result[0].max() <= 1.0
