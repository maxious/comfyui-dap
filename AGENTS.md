# Developer Notes & Learnings

## Environment Setup
1.  **Virtual Environment**: Always use the specific ComfyUI virtual environment for python commands to ensure access to torch, numpy, etc.
    *   Path: `S:\ComfyUI\venv`
    *   Python Executable: `S:\ComfyUI\venv\Scripts\python.exe`

2.  **Windows Bash Path Handling**:
    *   The environment uses a bash shell on Windows.
    *   Standard Windows paths (`S:\ComfyUI`) may be interpreted or converted by the shell to Unix-style paths (`/s/comfyui`) in some contexts.
    *   **Best Practice**: Use forward slashes for paths in bash commands (e.g., `"S:/ComfyUI/venv/Scripts/python.exe"`), or ensure proper quoting if using backslashes to avoid escape character issues.

## ComfyUI Development
*   **Imports**: Custom nodes often rely on `comfy` and `folder_paths` modules which reside in the ComfyUI root.
    *   When testing scripts locally inside `custom_nodes/my_node/`, you must explicitly add the ComfyUI root directory to `sys.path` to simulate the runtime environment.
*   **Headless Testing**:
    *   Use `pytest` with a `conftest.py` that mocks `comfy` and `folder_paths` by injecting them into `sys.modules`.
    *   Set `import-mode=importlib` in `pytest.ini` to handle relative imports in custom nodes correctly.
    *   Use `if not 'PYTEST_CURRENT_TEST' in os.environ:` in root `__init__.py` to skip ComfyUI-specific initialization during test collection.
*   **HuggingFace Exploration**:
    *   Use the `huggingface-cli` to list files in a repository without needing a browser.
    *   Command: `huggingface-cli ls Insta360-Research/DAP-weights`
*   **Optimal Performance**:
    *   Users should install `xformers` to ensure compatibility and optimal performance with the underlying backbones.
    *   Command: `pip install torch torchvision xformers -U --index-url https://download.pytorch.org/whl/cu130`
