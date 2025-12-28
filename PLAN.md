# ComfyUI-DAP Development Plan

## Phase 1: Validation & Analysis (Current Status: Completed)
- [x] **Repository Setup**: Clone `Insta360-Research-Team/DAP` into `custom_nodes/comfyui-dap/dap_core`.
- [x] **Dependency Check**: Verified `requirements.txt` and installed core deps in venv.
- [x] **Proof-of-Concept**: Created `verify_nodes.py` and a `pytest` suite to confirm logic works without GUI.
    - **Result**: Core node logic verified with real weights (`model.pth`). Relative import and path patching issues resolved.

## Phase 2: Implementation Strategy (Current Status: Completed)
1.  **Skeleton Creation**: (Done)
    - Set up `__init__.py` with `RUNNING_PYTEST` guard.
    - Created `dap_modules/` directory structure.
2.  **DAP Loader Node**: (Done)
    - Implement `huggingface_hub` integration for auto-downloading weights.
    - **Status**: Verified with real downloads.
3.  **DAP Inference Node**: (Done)
    - Implement ComfyUI `IMAGE` tensor handling.
    - **Status**: Verified with real inference pass.

## Phase 3: Optimization & Polish (Next Steps)
- [ ] **Model Selection**: Monitor HF for other model sizes (vits/vitb/vitg) and update `DAP_Loader` if released.
- [ ] **Tiling Support**: Implement tiling for ultra-high res panoramas.
- [ ] **Mask Output**: Extract and refine the `pred_mask` from DAP for better compositing.

## Phase 3: Optimization & Polish
- [ ] **Memory Management**: Ensure models offload to CPU when not in use (standard ComfyUI practice).
- [ ] **Tiling Support**: Adapt the "Tiled Processing" logic from `DepthAnythingV3` if applicable.
- [ ] **UI Tweaks**: Add `invert_depth` and `normalization` toggles.
