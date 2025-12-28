import pytest
import torch
import os
from dap_modules.dap_loader import DAP_Loader
from dap_modules.dap_inference import DAP_Inference


@pytest.mark.real_model
def test_dap_vitl_real_inference(dummy_image):
    """
    Integration test using real weights for DAP-vitl.
    """
    loader = DAP_Loader()
    inference = DAP_Inference()

    # 1. Load Real Model
    print("\nLoading real DAP-vitl model...")
    result = loader.load_model(model_size="vitl", precision="fp32")
    dap_model = result[0]

    assert "model" in dap_model
    assert isinstance(dap_model["model"], torch.nn.Module)

    # 2. Run Inference
    print("Running inference on dummy panorama...")
    out_images, out_masks = inference.process(
        dap_model=dap_model, image=dummy_image, invert_output=True, resize_input=True
    )

    # 3. Verify Output
    assert out_images.shape == (1, 512, 1024, 3)
    assert out_images.max() > out_images.min(), (
        "Output depth map is flat (zero/constant)"
    )
    print(
        f"Inference successful. Depth range: {out_images.min():.4f} - {out_images.max():.4f}"
    )
