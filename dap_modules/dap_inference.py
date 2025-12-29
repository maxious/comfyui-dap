import torch
import torch.nn.functional as F
import comfy.model_management as mm
import numpy as np


class DAP_Inference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dap_model": ("DAP_MODEL",),
                "image": ("IMAGE",),
                "invert_output": ("BOOLEAN", {"default": True}),
                "resize_input": (
                    "BOOLEAN",
                    {"default": True, "label": "Resize to Patch Size (14px)"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth", "mask")
    FUNCTION = "process"
    CATEGORY = "DAP"

    def process(self, dap_model, image, invert_output, resize_input):
        patcher = dap_model["model"]
        precision = dap_model["precision"]
        device = mm.get_torch_device()

        # 1. Prepare Model
        mm.load_model_gpu(patcher)
        model = patcher.model

        B, H, W, C = image.shape

        results = []
        masks = []

        # Process batch
        for i in range(B):
            img = image[i]

            # 1. Resize
            if resize_input:
                target_h = int(H // 14) * 14
                target_w = int(W // 14) * 14
                if target_h < H:
                    target_h += 14
                if target_w < W:
                    target_w += 14

                img_t = img.permute(2, 0, 1).unsqueeze(0)
                img_resized = F.interpolate(
                    img_t,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
                img_final = img_resized.squeeze(0)
            else:
                img_final = img.permute(2, 0, 1)

            # 2. Normalize
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

            img_final = img_final.to(device)
            if precision == "fp16":
                img_final = img_final.half()
                mean = mean.half()
                std = std.half()
            elif precision == "bf16":
                img_final = img_final.bfloat16()
                mean = mean.bfloat16()

            img_norm = (img_final - mean) / std
            img_input = img_norm.unsqueeze(0)

            print(
                f"DAP Debug: Input Tensor Min={img_input.min().item():.4f}, Max={img_input.max().item():.4f}, NaNs={torch.isnan(img_input).sum().item()}"
            )

            # 3. Inference

            with torch.no_grad():
                outputs = model(img_input)
                pred_depth = outputs["pred_depth"]  # [1, 1, H, W]
                pred_mask = outputs.get("pred_mask")  # [1, 1, H, W]

                print(
                    f"DAP Debug: Raw Output Min={pred_depth.min().item():.4f}, Max={pred_depth.max().item():.4f}, Mean={pred_depth.mean().item():.4f}, NaNs={torch.isnan(pred_depth).sum().item()}"
                )

                # Cleanup depth using mask if available

                if pred_mask is not None:
                    # Original logic: mask = 1 - pred_mask; mask = mask > 0.5; depth[~mask] = 1
                    # This removes non-panorama content
                    mask_bool = (1.0 - pred_mask) > 0.5
                    # We fill masked areas with the max depth value found in the valid area
                    # or just a very large value that will be normalized to 'far'.
                    # For now, let's use the max of the current prediction.
                    valid_max = pred_depth[mask_bool].max() if mask_bool.any() else 1.0
                    pred_depth[~mask_bool] = valid_max

            # 4. Resize back
            pred_depth = F.interpolate(
                pred_depth, size=(H, W), mode="bilinear", align_corners=True
            )
            if pred_mask is not None:
                pred_mask = F.interpolate(
                    pred_mask, size=(H, W), mode="bilinear", align_corners=True
                )

            # 5. Normalize
            d_min, d_max = pred_depth.min(), pred_depth.max()
            depth_norm = (pred_depth - d_min) / (d_max - d_min + 1e-8)

            if invert_output:
                depth_norm = 1.0 - depth_norm

            # 6. Format Outputs
            depth_out = depth_norm.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3)
            results.append(depth_out.cpu())

            if pred_mask is not None:
                # pred_mask in DAP seems to be 0 for valid, 1 for invalid?
                # Let's return it as a standard mask (1 = valid/selected)
                mask_out = 1.0 - pred_mask.squeeze().cpu()
            else:
                mask_out = torch.ones((H, W))
            masks.append(mask_out)

        final_depth = torch.stack(results)
        final_mask = torch.stack(masks)

        return (final_depth, final_mask)
