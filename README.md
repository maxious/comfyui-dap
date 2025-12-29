# ComfyUI-DAP

A ComfyUI extension for the [Insta360 DAP](https://github.com/Insta360-Research-Team/DAP) (Depth Any Panorama) model.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` folder.
2.  Install the required dependencies in your ComfyUI virtual environment:

```bash
# Example for Windows with the default venv
"S:/ComfyUI/venv/Scripts/python.exe" -m pip install -r requirements.txt

# IMPORTANT: You need xformers for optimal performance and compatibility
"S:/ComfyUI/venv/Scripts/python.exe" -m pip install torch torchvision xformers -U --index-url https://download.pytorch.org/whl/cu130
```

## Nodes

### DAP Loader
*   **model_size**: Currently only `vitl` is available.
*   **precision**: Choose between `fp16`, `fp32`, and `bf16`.
*   **Downloads**: Automatically downloads the model from HuggingFace on first use.

### DAP Inference
*   **image**: Input panoramic image.
*   **invert_output**: Invert the depth map (standard for ComfyUI depth).
*   **resize_input**: Resizes to the model's preferred patch size (multiples of 14).

### DAP Panoramic Mesh
*   **depth**: The depth map from DAP.
*   **mesh_scale**: Factor to scale the 3D geometry.
*   **image**: (Optional) Use for vertex coloring.
*   **downsample**: Reduce mesh density for performance.
*   **Compatibility**: This node outputs a `TRIMESH` type, making it fully compatible with [ComfyUI-GeometryPack](https://github.com/idm-lab/ComfyUI-GeometryPack) for advanced mesh editing and saving.

## Testing
This plugin includes a `pytest` suite for headless verification.

```bash
"S:/ComfyUI/venv/Scripts/python.exe" -m pytest
```
