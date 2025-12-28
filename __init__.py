import os

# Check if we're running under pytest
RUNNING_PYTEST = "PYTEST_CURRENT_TEST" in os.environ

if not RUNNING_PYTEST:
    from .dap_modules.dap_loader import DAP_Loader
    from .dap_modules.dap_inference import DAP_Inference
else:
    # Set dummy values for pytest collection
    DAP_Loader = None
    DAP_Inference = None

NODE_CLASS_MAPPINGS = {
    "DAP_Loader": DAP_Loader,
    "DAP_Inference": DAP_Inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAP_Loader": "DAP Loader",
    "DAP_Inference": "DAP Inference",
}
