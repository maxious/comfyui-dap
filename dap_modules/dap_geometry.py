import torch
import numpy as np
import trimesh
import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dap_core_path = os.path.join(current_dir, "dap_core")
if dap_core_path not in sys.path:
    sys.path.insert(0, dap_core_path)


def spherical_uv_to_directions_np(uv: np.ndarray):
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
                "downsample": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "remove_long_edges": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "stitch_seam": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate_mesh"
    CATEGORY = "DAP/Geometry"

    def generate_mesh(
        self,
        depth,
        mesh_scale,
        downsample,
        image=None,
        mask=None,
        remove_long_edges=0.5,
        stitch_seam=True,
    ):
        d_tensor = depth[0]
        H, W, C = d_tensor.shape
        d_np = d_tensor.cpu().numpy()
        if C > 1:
            d_np = np.mean(d_np, axis=2)
        else:
            d_np = d_np.squeeze()

        d_np = np.clip(d_np, 0.01, 1.0)

        if downsample > 1:
            d_np = d_np[::downsample, ::downsample]

        h_new, w_new = d_np.shape

        # Pole Snapping: Average depth at poles
        d_np[0, :] = np.mean(d_np[0, :])
        d_np[-1, :] = np.mean(d_np[-1, :])

        y, x = np.mgrid[0:h_new, 0:w_new]
        u = x / (w_new - 1)
        v = y / (h_new - 1)
        uv = np.stack([u, v], axis=-1)
        dirs = spherical_uv_to_directions_np(uv)
        vertices = (dirs * d_np[..., None] * mesh_scale).reshape(-1, 3)

        indices = np.arange(h_new * w_new).reshape(h_new, w_new)

        # Grid faces
        f1 = np.stack([indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1]], axis=-1)
        f2 = np.stack([indices[:-1, 1:], indices[1:, 1:], indices[1:, :-1]], axis=-1)
        faces_list = [f1.reshape(-1, 3), f2.reshape(-1, 3)]

        # Seam Stitching (Right column back to Left column)
        if stitch_seam:
            s_f1 = np.stack(
                [indices[:-1, -1], indices[:-1, 0], indices[1:, -1]], axis=-1
            )
            s_f2 = np.stack([indices[:-1, 0], indices[1:, 0], indices[1:, -1]], axis=-1)
            faces_list.append(s_f1.reshape(-1, 3))
            faces_list.append(s_f2.reshape(-1, 3))

        faces = np.vstack(faces_list)

        vertex_colors = None
        if image is not None:
            i_np = image[0].cpu().numpy()
            if downsample > 1:
                i_np = i_np[::downsample, ::downsample]
            vertex_colors = (i_np.reshape(-1, 3) * 255).astype(np.uint8)

        # Build initial mesh
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False
        )

        # Filter by Mask
        if mask is not None:
            m_np = mask[0].cpu().numpy()
            if downsample > 1:
                m_np = m_np[::downsample, ::downsample]
            m_flat = m_np.flatten()

            # Remove faces where ANY vertex is masked (value < 0.5)
            face_mask = (m_flat[mesh.faces] > 0.5).all(axis=1)
            mesh.update_faces(face_mask)
            mesh.remove_unreferenced_vertices()

        # Remove Long Edges (Streaks)
        if remove_long_edges > 0:
            edges = mesh.vertices[mesh.edges_unique]
            edge_lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
            # Find vertices involved in long edges and remove faces?
            # Better: find faces with long edges
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]

            e1 = np.linalg.norm(v0 - v1, axis=1)
            e2 = np.linalg.norm(v1 - v2, axis=1)
            e3 = np.linalg.norm(v2 - v0, axis=1)

            # Threshold relative to average edge or fixed
            # 0.5 is huge for a 1.0 radius sphere
            bad_faces = (
                (e1 > remove_long_edges)
                | (e2 > remove_long_edges)
                | (e3 > remove_long_edges)
            )
            mesh.update_faces(~bad_faces)
            mesh.remove_unreferenced_vertices()

        return (mesh,)
