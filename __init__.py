import os

# Check if we're running under pytest
RUNNING_PYTEST = "PYTEST_CURRENT_TEST" in os.environ

if not RUNNING_PYTEST:
    from .dap_modules.dap_loader import DAP_Loader
    from .dap_modules.dap_inference import DAP_Inference
    from .dap_modules.dap_geometry import DAP_Panoramic_Mesh
    from .dap_modules.dap_conversions import DAP_ERP_to_Cubemap
    from .dap_modules.dap_normal import DAP_Normal_Map
else:
    # Set dummy values for pytest collection
    DAP_Loader = None
    DAP_Inference = None
    DAP_Panoramic_Mesh = None
    DAP_ERP_to_Cubemap = None
    DAP_Normal_Map = None

NODE_CLASS_MAPPINGS = {
    "DAP_Loader": DAP_Loader,
    "DAP_Inference": DAP_Inference,
    "DAP_Panoramic_Mesh": DAP_Panoramic_Mesh,
    "DAP_ERP_to_Cubemap": DAP_ERP_to_Cubemap,
    "DAP_Normal_Map": DAP_Normal_Map,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAP_Loader": "DAP Loader",
    "DAP_Inference": "DAP Inference",
    "DAP_Panoramic_Mesh": "DAP Panoramic Mesh",
    "DAP_ERP_to_Cubemap": "DAP ERP to Cubemap",
    "DAP_Normal_Map": "DAP Normal Map",
}
