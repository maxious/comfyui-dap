# Product Requirements Document (PRD): ComfyUI-DAP

## 1. Project Overview
**Goal**: Create a native ComfyUI extension for the [Insta360 DAP](https://github.com/Insta360-Research-Team/DAP) model (Depth Any Panorama).
**Repository**: `https://github.com/Insta360-Research-Team/DAP`
**Model Weights**: `https://huggingface.co/Insta360-Research/DAP-weights`

## 2. User Stories
- **US-01**: As a user, I want to use the DAP model without manually downloading weights or setting up complex python paths.
- **US-02**: As a user, I want to input a panoramic image and receive a high-quality depth map mask.
- **US-03**: As a user, I want to select different model sizes (Small/Base/Large) to balance VRAM usage and quality.

## 3. Node Specifications

### Node A: `DAP_Loader`
- **Category**: `DAP/Loaders`
- **Inputs**:
  - `model_size` (Enum): `["vitl", "vitb", "vits", "vitg"]` (Default: `vitl`)
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
- **Outputs**:
  - `depth`: (`IMAGE`) - The normalized depth map.
  - `mask`: (`MASK`) - Raw 1-channel output.
- **Logic**:
  1. Move model to GPU.
  2. Resize input image to nearest multiple of 14 (patch size).
  3. Run inference.
  4. Resize output back to original dimensions.
  5. Normalize values for visualization.

## 4. Technical Constraints & Risks
- **Hardcoded Paths**: The `DAP` class in `networks/dap.py` has a hardcoded relative path: `dinov3_repo_dir="./depth_anything_v2_metric/..."`.
    - **Solution**: We must either modify the file on first run or monkey-patch `DAP.__init__` to accept a dynamic path.
- **Imports**: The codebase assumes it is the root module.
    - **Solution**: Use `sys.path.append(os.path.join(current_dir, "dap_core"))` inside the node initialization.
- **Dependencies**: `open3d` is in `requirements.txt`. It is heavy and often causes conflicts.
    - **Solution**: Check if we can run inference without importing `open3d`. If it's only for visualization, we can mock it or remove the import.

## 5. Development Backlog

| ID | Task | Priority | Status |
|----|------|----------|--------|
| **BL-01** | **Environment Verification**: Create `verify_dap.py` to test imports and weight loading. | P0 | Pending |
| **BL-02** | **Project Skeleton**: Create `__init__.py`, `nodes/`, and basic project structure. | P0 | Pending |
| **BL-03** | **DAP Loader Implementation**: Implement auto-download and model loading with path patching. | P1 | Pending |
| **BL-04** | **DAP Inference Implementation**: Implement image tensor conversion and inference loop. | P1 | Pending |
| **BL-05** | **Optimization**: Implement model offloading (CPU/GPU switching). | P2 | Pending |
| **BL-06** | **UI/Docs**: Add "invert" option and `README.md`. | P3 | Pending |
