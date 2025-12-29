import torch
import numpy as np
import cv2
import math


class DAP_ERP_to_Cubemap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_size": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 64},
                ),
                "layout": (["six_faces", "cross"], {"default": "six_faces"}),
                "interpolation": (
                    ["lanczos", "cubic", "linear", "nearest"],
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "DAP/Conversions"

    def convert(self, image, face_size, layout, interpolation):
        # ComfyUI image is [B, H, W, C], 0..1
        B, H, W, C = image.shape

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_map[interpolation]

        faces_order = ["right", "left", "top", "bottom", "front", "back"]

        all_results = []

        for b in range(B):
            erp_img = (image[b].cpu().numpy() * 255).astype(np.uint8)

            faces = {}
            for face_name in faces_order:
                map_x_norm, map_y_norm = self.build_face_map(face_size, face_name)

                # Apply map
                map_x = map_x_norm * (W - 1)
                map_y = map_y_norm * (H - 1)

                face_img = cv2.remap(
                    erp_img,
                    map_x,
                    map_y,
                    interpolation=interp,
                    borderMode=cv2.BORDER_WRAP,
                )
                faces[face_name] = face_img

            if layout == "six_faces":
                # Return as a batch of 6 images
                for name in faces_order:
                    res = torch.from_numpy(faces[name].astype(np.float32) / 255.0)
                    all_results.append(res)
            else:
                # Return as a single cross layout image (3x4 grid)
                cross = self.make_cross_layout(faces, face_size, C)
                res = torch.from_numpy(cross.astype(np.float32) / 255.0)
                all_results.append(res)

        final_image = torch.stack(all_results)
        return (final_image,)

    def build_face_map(self, face_size, face):
        s = face_size
        jj, ii = np.meshgrid(
            np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32)
        )
        a = 2.0 * (jj + 0.5) / s - 1.0
        b = 2.0 * (ii + 0.5) / s - 1.0

        if face == "right":  # +X
            dx, dy, dz = np.ones_like(a), -b, -a
        elif face == "left":  # -X
            dx, dy, dz = -np.ones_like(a), -b, a
        elif face == "top":  # +Y
            dx, dy, dz = a, np.ones_like(a), b
        elif face == "bottom":  # -Y
            dx, dy, dz = a, -np.ones_like(a), -b
        elif face == "front":  # +Z
            dx, dy, dz = a, -b, np.ones_like(a)
        elif face == "back":  # -Z
            dx, dy, dz = -a, -b, -np.ones_like(a)
        else:
            raise ValueError(f"Unknown face: {face}")

        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= norm
        dy /= norm
        dz /= norm

        theta = np.arctan2(dz, dx)
        phi = np.arcsin(dy)

        map_x_norm = (theta + math.pi) / (2.0 * math.pi)
        map_y_norm = (math.pi / 2 - phi) / math.pi

        return map_x_norm.astype(np.float32), map_y_norm.astype(np.float32)

    def make_cross_layout(self, faces, face_size, channels):
        S = face_size
        canvas = np.zeros((3 * S, 4 * S, channels), dtype=np.uint8)

        def put(name, row, col):
            canvas[row * S : (row + 1) * S, col * S : (col + 1) * S] = faces[name]

        put("top", 0, 1)
        put("left", 1, 0)
        put("front", 1, 1)
        put("right", 1, 2)
        put("back", 1, 3)
        put("bottom", 2, 1)
        return canvas
