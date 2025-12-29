import os
import sys
import torch
import requests
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from huggingface_hub import hf_hub_download, hf_hub_url
from argparse import Namespace

# Add dap_core to sys.path so imports work
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dap_core_path = os.path.join(current_dir, "dap_core")
if dap_core_path not in sys.path:
    sys.path.insert(0, dap_core_path)

# Import DAP class after sys.path update
try:
    from networks.dap import DAP
except ImportError:
    # Fallback/Safety check
    print("DAP import failed. Ensure dap_core is in path.")
    DAP = None


def download_with_progress(url, dest_path, filename):
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    pbar = ProgressBar(total_size)

    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Download with progress updates
    with open(dest_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update_absolute(downloaded, total_size)
    print(f"Finished downloading {filename}")


class DAP_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_size": (["vitl"], {"default": "vitl"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("DAP_MODEL",)
    RETURN_NAMES = ("dap_model",)
    FUNCTION = "load_model"
    CATEGORY = "DAP"

    def load_model(self, model_size, precision):
        device = mm.get_torch_device()

        # 1. Define Model Files
        model_filename = "model.pth"
        repo_id = "Insta360-Research/DAP-weights"

        # 2. Download/Locate Model
        ckpt_path = folder_paths.get_full_path("checkpoints", model_filename)

        if not ckpt_path:
            dap_models_dir = os.path.join(folder_paths.models_dir, "dap")
            ckpt_path = os.path.join(dap_models_dir, model_filename)

            if not os.path.exists(ckpt_path):
                url = hf_hub_url(repo_id=repo_id, filename=model_filename)
                try:
                    download_with_progress(url, ckpt_path, model_filename)
                except Exception as e:
                    # Fallback to standard hf_hub_download if manual fails
                    print(
                        f"Manual download failed: {e}. Falling back to hf_hub_download..."
                    )
                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_filename,
                        local_dir=dap_models_dir,
                    )

        # 3. Initialize Model with Patching
        args = Namespace()
        args.midas_model_type = model_size
        args.fine_tune_type = "none"
        args.min_depth = 0.001
        args.max_depth = 1.0
        args.train_decoder = True

        original_cwd = os.getcwd()
        try:
            os.chdir(dap_core_path)
            model = DAP(args)

            state_dict = torch.load(ckpt_path, map_location="cpu")
            if state_dict and list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            model.load_state_dict(state_dict, strict=False)

        finally:
            os.chdir(original_cwd)

        # 4. Quantization / Precision
        if precision == "fp16":
            model = model.half()
        elif precision == "bf16":
            model = model.bfloat16()

        model.eval()

        # 5. Wrap in ModelPatcher for memory management
        import comfy.model_patcher

        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=device, offload_device=mm.unet_offload_device()
        )

        return ({"model": patcher, "precision": precision},)
