import torch
import numpy as np
import trimesh
import os
import sys

# Add dap_core to sys.path so imports work
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dap_core_path = os.path.join(current_dir, "dap_core")
if dap_core_path not in sys.path:
    sys.path.insert(0, dap_core_path)


def spherical_uv_to_directions_np(uv: np.ndarray):
    """
    Equirectangular UV to Unit Sphere Directions
    uv: [H, W, 2] in range [0, 1]
    """
    theta = (1.0 - uv[..., 0]) * (2.0 * np.pi)
    phi = uv[..., 1] * np.pi
    directions = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1
    )
    return directions


class DAP_Panoramic_Mesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE",),
                "mesh_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "downsample": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate_mesh"
    CATEGORY = "DAP/Geometry"

    def generate_mesh(self, depth, mesh_scale, image=None, mask=None, downsample=1):
        # ComfyUI Image is [B, H, W, C]
        # We take the first image in batch
        d_tensor = depth[0]
        H, W, C = d_tensor.shape

        # 1. Grayscale depth
        d_np = d_tensor.cpu().numpy()
        if C > 1:
            d_np = np.mean(d_np, axis=2)
        else:
            d_np = d_np.squeeze()

        # Prevent vertices from collapsing to 0,0,0 by clamping min depth
        d_np = np.clip(d_np, 0.01, 1.0)

        # 2. Downsample for performance if requested

        if downsample > 1:
            d_np = d_np[::downsample, ::downsample]
            h_new, w_new = d_np.shape
        else:
            h_new, w_new = H, W

        # 3. Create UV grid
        y, x = np.mgrid[0:h_new, 0:w_new]
        u = x / (w_new - 1)
        v = y / (h_new - 1)
        uv = np.stack([u, v], axis=-1)

        # 4. Unproject to 3D directions
        dirs = spherical_uv_to_directions_np(uv)

        # 5. Scale by depth
        # Assume depth is 0..1, we multiply by scale
        vertices = (dirs * d_np[..., None] * mesh_scale).reshape(-1, 3)

        # 6. Generate faces for a grid
        # vertices are ordered row-major
        # [0, 1, ..., w-1]
        # [w, w+1, ..., 2w-1]
        indices = np.arange(h_new * w_new).reshape(h_new, w_new)

        # Two triangles per grid cell
        f1 = np.stack([indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1]], axis=-1)
        f2 = np.stack([indices[:-1, 1:], indices[1:, 1:], indices[1:, :-1]], axis=-1)
        faces = np.vstack([f1.reshape(-1, 3), f2.reshape(-1, 3)])

        # 7. Apply mask if provided
        if mask is not None:
            m_np = mask[0].cpu().numpy()
            if downsample > 1:
                m_np = m_np[::downsample, ::downsample]

            valid_verts = m_np.flatten() > 0.5
            # We keep all vertices for indexing simplicity but only valid faces
            # Actually, standard trimesh cleanup is better
            face_mask = m_np[:-1, :-1] > 0.5
            # This is complex for indexed grids.
            # Simpler: just create mesh and remove invalid vertices/faces
            pass

        # 8. Colors
        vertex_colors = None
        if image is not None:
            i_np = image[0].cpu().numpy()
            if downsample > 1:
                # Basic nearest neighbor downsampling for colors
                i_np = i_np[::downsample, ::downsample]

            # ComfyUI image is 0..1, trimesh expects 0..255 or 0..1?
            # Trimesh usually takes 0..255 uint8 for visual.
            vertex_colors = (i_np.reshape(-1, 3) * 255).astype(np.uint8)

        # 9. Build Mesh
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False
        )

        # Clean up if a mask was used or to remove long edge artifacts at the equirectangular seam
        # TODO: Seam stitching

        return (mesh,)
