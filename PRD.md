# Product Requirements Document (PRD): ComfyUI-DAP

## 1. Project Overview
**Goal**: Create a native ComfyUI extension for the [Insta360 DAP](https://github.com/Insta360-Research-Team/DAP) model (Depth Any Panorama).
**Repository**: `https://github.com/Insta360-Research-Team/DAP`
**Model Weights**: `https://huggingface.co/Insta360-Research/DAP-weights`

## 2. User Stories
- **US-01**: As a user, I want to use the DAP model without manually downloading weights or setting up complex python paths.
- **US-02**: As a user, I want to input a panoramic image and receive a high-quality depth map mask.
- **US-03**: As a user, I want to select different model sizes (Small/Base/Large) to balance VRAM usage and quality.
- **US-04**: As a user, I want to convert between Equirectangular (ERP) and Cubemap formats for specialized workflows.

## 3. Node Specifications

### Node A: `DAP_Loader`
- **Category**: `DAP/Loaders`
- **Inputs**:
  - `model_size` (Enum): `["vitl"]` (Default: `vitl`)
  - `precision` (Enum): `["fp16", "fp32", "bf16"]` (Default: `fp16`)
- **Outputs**:
  - `DAP_MODEL`: A custom object containing the loaded PyTorch model and config.
- **Logic**:
  1. Check `ComfyUI/models/dap` for weights.
  2. If missing, download from HF `Insta360-Research/DAP-weights`.
  3. Temporarily inject `dap_core` into `sys.path`.
  4. Instantiate `DAP` class (with path patch).
  5. Load state dict.

### Node B: `DAP_Inference`
- **Category**: `DAP/Inference`
- **Inputs**:
  - `dap_model`: (`DAP_MODEL`)
  - `image`: (`IMAGE`)
  - `invert_output`: (`BOOLEAN`, Default: `True`)
  - `resize_input`: (`BOOLEAN`, Default: `True`)
- **Outputs**:
  - `depth`: (`IMAGE`) - The normalized depth map.
  - `mask`: (`MASK`) - Cleanup mask for non-panorama content.
- **Logic**:
  1. Move model to GPU.
  2. Resize input image to nearest multiple of 14 (patch size).
  3. Run inference.
  4. Use `pred_mask` to clean up depth boundaries.
  5. Resize output back to original dimensions.
  6. Normalize values for visualization.

### Node C: `DAP_Panoramic_Mesh`
- **Category**: `DAP/Geometry`
- **Inputs**:
  - `depth`: (`IMAGE`) - Depth map from DAP.
  - `image`: (`IMAGE`, Optional) - For vertex coloring.
  - `mesh_scale`: (`FLOAT`, Default: `1.0`)
  - `downsample`: (`INT`, Default: `1`)
- **Outputs**:
  - `TRIMESH`: (`TRIMESH`) - Trimesh object compatible with `ComfyUI-GeometryPack`.
- **Logic**:
  1. Use DAP spherical unprojection math (`spherical_uv_to_directions`) to handle equirectangular distortion.
  2. Build a mesh using `trimesh`.
  3. Map pixel colors to vertices.

### Node D: `DAP_ERP_to_Cubemap` (New)
- **Category**: `DAP/Conversions`
- **Inputs**:
  - `image`: (`IMAGE`) - ERP image.
  - `face_size`: (`INT`, Default: `1024`)
  - `layout`: (`Enum`: `["six_faces", "cross"]`)
  - `interpolation`: (`Enum`: `["lanczos", "cubic", "linear", "nearest"]`)
- **Outputs**:
  - `image`: (`IMAGE`) - Batch of 6 face images or a single cross-layout image.
- **Logic**:
  1. Perform coordinate remapping from ERP to Cube faces using OpenCV.

## 4. Technical Constraints & Risks
- **Interoperability**: We output `TRIMESH` types for `ComfyUI-GeometryPack` compatibility.
- **Hardcoded Paths**: The `DAP` class in `networks/dap.py` has a hardcoded relative path: `dinov3_repo_dir="./depth_anything_v2_metric/..."`.
    - **Solution**: We use a `cwd` context manager during loader initialization.
- **Imports**: The codebase assumes it is the root module.
    - **Solution**: Use `sys.path.append(os.path.join(current_dir, "dap_core"))`.
- **Dependencies**: `open3d` is in `requirements.txt`.
    - **Status**: Essential for point cloud logic, confirmed installed.

## 5. Development Backlog

| ID | Task | Priority | Status |
|----|------|----------|--------|
| **BL-01** | **Environment Verification**: Create `verify_dap.py` and pytest. | P0 | Completed |
| **BL-02** | **Project Skeleton**: Create `__init__.py`, `dap_modules/`, and Git init. | P0 | Completed |
| **BL-03** | **DAP Loader Implementation**: Implement auto-download and model loading. | P1 | Completed |
| **BL-04** | **DAP Inference Implementation**: Implement image tensor conversion and inference loop. | P1 | Completed |
| **BL-05** | **Optimization**: Implement model offloading (CPU/GPU switching). | P2 | Completed |
| **BL-06** | **Panoramic Mesh**: Implement `DAP_Panoramic_Mesh` node (GeometryPack compatible). | P2 | Completed |
| **BL-07** | **Conversions**: Implement `DAP_ERP_to_Cubemap` node. | P2 | Completed |
| **BL-08** | **Normal Map Node**: Implement `DAP_Normal_Map` node. | P3 | Pending |
| **BL-09** | **UI/Docs**: Add "invert" option and `README.md`. | P3 | Completed |
