import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

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
                "downsample": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "generate_normal"
    CATEGORY = "DAP/Geometry"

    def generate_normal(self, depth, normal_standard, downsample=1, mask=None):
        device = depth.device
        d_tensor = depth[0]
        H, W, C = d_tensor.shape

        if downsample > 1:
            d_small = d_tensor.permute(2, 0, 1).unsqueeze(0)
            d_small = F.interpolate(
                d_small, size=(H // downsample, W // downsample), mode="bilinear"
            )
            d_tensor = d_small.squeeze(0).permute(1, 2, 0)
            curr_H, curr_W = d_tensor.shape[0], d_tensor.shape[1]
        else:
            curr_H, curr_W = H, W

        d_np = d_tensor.cpu().numpy()
        if C > 1:
            d_np = np.mean(d_np, axis=2)
        else:
            d_np = d_np.squeeze()

        y, x = np.mgrid[0:curr_H, 0:curr_W]
        u = x / (curr_W - 1)
        v = y / (curr_H - 1)

        theta = (1.0 - u) * (2.0 * np.pi)
        phi = v * np.pi

        dx = np.sin(phi) * np.cos(theta)
        dy = np.sin(phi) * np.sin(theta)
        dz = np.cos(phi)

        dirs = np.stack([dx, dy, dz], axis=-1)
        points = dirs * d_np[..., None]

        p_tensor = torch.from_numpy(points.astype(np.float32)).to(device)

        p_pad = (
            F.pad(
                p_tensor.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        v_right = p_pad[1:-1, 2:, :] - p_tensor
        v_left = p_tensor - p_pad[1:-1, :-2, :]
        v_down = p_pad[2:, 1:-1, :] - p_tensor
        v_up = p_tensor - p_pad[:-2, 1:-1, :]

        n1 = torch.cross(v_right, v_down, dim=-1)
        n2 = torch.cross(v_down, v_left, dim=-1)
        n3 = torch.cross(v_left, v_up, dim=-1)
        n4 = torch.cross(v_up, v_right, dim=-1)

        normals = (n1 + n2 + n3 + n4) / 4.0

        norm = torch.norm(normals, dim=-1, keepdim=True)
        normals = normals / (norm + 1e-6)

        dot_product = torch.sum(normals * p_tensor, dim=-1, keepdim=True)
        normals = torch.where(dot_product > 0, -normals, normals)

        if normal_standard == "DAP":
            normals = normals * torch.tensor([-1.0, -1.0, 1.0], device=device)
            normals = torch.stack(
                [normals[..., 0], normals[..., 2], normals[..., 1]], dim=-1
            )

        if downsample > 1:
            normals = normals.permute(2, 0, 1).unsqueeze(0)
            normals = F.interpolate(normals, size=(H, W), mode="bilinear")
            normals = normals.squeeze(0).permute(1, 2, 0)
            norm = torch.norm(normals, dim=-1, keepdim=True)
            normals = normals / (norm + 1e-6)

        normal_rgb = (normals + 1.0) * 0.5

        if mask is not None:
            m_tensor = mask[0].to(device).unsqueeze(-1)
            normal_rgb = normal_rgb * m_tensor

        res = normal_rgb.unsqueeze(0).cpu()
        return (res,)
