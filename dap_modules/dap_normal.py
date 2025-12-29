import torch
import numpy as np
import os
import sys

# Add dap_core to sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dap_core_path = os.path.join(current_dir, "dap_core")
if dap_core_path not in sys.path:
    sys.path.insert(0, dap_core_path)


class DAP_Normal_Map:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE",),
                "normal_standard": (["ComfyUI", "DAP"], {"default": "ComfyUI"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "generate_normal"
    CATEGORY = "DAP/Geometry"

    def generate_normal(self, depth, normal_standard, mask=None):
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError(
                "Open3D is required for normal map generation. Install with: pip install open3d"
            )

        # 1. Prepare depth (B, H, W)
        d_tensor = depth[0]
        H, W, C = d_tensor.shape
        d_np = d_tensor.cpu().numpy()
        if C > 1:
            d_np = np.mean(d_np, axis=2)
        else:
            d_np = d_np.squeeze()

        # 2. Unproject to points
        y, x = np.mgrid[0:H, 0:W]
        u = x / (W - 1)
        v = y / (H - 1)

        # Equirectangular math
        theta = (1.0 - u) * (2.0 * np.pi)
        phi = v * np.pi

        dx = np.sin(phi) * np.cos(theta)
        dy = np.sin(phi) * np.sin(theta)
        dz = np.cos(phi)

        dirs = np.stack([dx, dy, dz], axis=-1)
        points = (dirs * d_np[..., None]).reshape(-1, 3)

        # 3. Open3D Normal Estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Standard estimation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # Orient towards center (spherical)
        pcd.orient_normals_towards_camera_location([0, 0, 0])

        normals = np.asarray(pcd.normals).reshape(H, W, 3)

        # 4. Standards
        if normal_standard == "DAP":
            # DAP reorders and flips
            normals = normals * np.array([-1, -1, 1])
            # Reorder to [X, Z, Y] as seen in depth2normal.py
            normals = np.stack(
                [normals[..., 0], normals[..., 2], normals[..., 1]], axis=-1
            )

        # Convert [-1, 1] to [0, 1]
        normal_rgb = (normals + 1.0) * 0.5

        # 5. Masking
        if mask is not None:
            m_np = mask[0].cpu().numpy()
            normal_rgb = normal_rgb * m_np[..., None]

        res = torch.from_numpy(normal_rgb.astype(np.float32)).unsqueeze(0)
        return (res,)
