"""
PR-IQA inference: Generate dense quality maps.

Pipeline:
  generated_image + reference_image → FeatureMetric (partial map) → PR-IQA → quality_map

Usage:
    python inference.py \
        --generated gen.jpg \
        --reference ref.jpg \
        --checkpoint checkpoints/priqa_base.pt \
        --output quality_map.png
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from pr_iqa.model import build_priqa


# ── Constants ──
IMG_SIZE = 256
OUTPUT_SIZE = (294, 518)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ── Transforms ──
RGB_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

GREY_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

OUTPUT_TRANSFORM = T.Compose([
    T.Resize(OUTPUT_SIZE),
])


# ── Utilities ──

def minmax01(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1]."""
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=np.float32)
    vmin = float(a[finite].min())
    vmax = float(a[finite].max())
    if vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    a_clean = a.copy()
    a_clean[~finite] = vmin
    return np.clip((a_clean - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)


def build_mask_pyramid(mask_whole: torch.Tensor):
    """(1, 1, H, W) → 4-scale pyramid."""
    return (
        mask_whole,
        F.interpolate(mask_whole, scale_factor=0.5, mode="nearest"),
        F.interpolate(mask_whole, scale_factor=0.25, mode="nearest"),
        F.interpolate(mask_whole, scale_factor=0.125, mode="nearest"),
    )


def generate_partial_map(
    generated_path: str,
    reference_path: str,
    device: str = "cuda",
    use_loftup: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate partial map and overlap mask using FeatureMetric.

    Returns:
        partial_map: (1, 1, S, S) tensor in [0, 1]
        mask: (1, 1, S, S) tensor in [0, 1]
    """
    from pr_iqa.partial_map import FeatureMetric

    metric = FeatureMetric(img_size=IMG_SIZE, use_vggt=True, use_loftup=use_loftup)
    metric = metric.to(device).eval()

    gen_img = np.array(Image.open(generated_path).convert("RGB"))
    ref_img = np.array(Image.open(reference_path).convert("RGB"))

    gen_t = torch.from_numpy(gen_img).permute(2, 0, 1).float() / 255.0
    ref_t = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0

    gen_t = F.interpolate(gen_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    ref_t = F.interpolate(ref_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)

    images = torch.stack([gen_t, ref_t], dim=0).to(device)

    score, mask, score_map, _ = metric(
        device=device,
        images=images,
        return_overlap_mask=True,
        return_score_map=True,
        partial_generation=True,
    )

    if score_map is not None:
        smap = score_map[0].detach().float().cpu().numpy()
        if smap.ndim == 1:
            side = int(np.sqrt(smap.shape[0]))
            smap = smap.reshape(side, side)
        smap = np.clip(smap, 0, 1)
        pmap_t = torch.from_numpy(smap).unsqueeze(0).unsqueeze(0)
        pmap_t = F.interpolate(pmap_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    else:
        pmap_t = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)

    if mask is not None:
        msk = mask[0].detach().float().cpu().numpy()
        if msk.ndim == 3:
            msk = msk[0]
        msk = np.clip(msk, 0, 1)
        mask_t = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(IMG_SIZE, IMG_SIZE), mode="nearest")
    else:
        mask_t = torch.ones(1, 1, IMG_SIZE, IMG_SIZE)

    print(f"[FeatureMetric] score={score:.4f}")
    return pmap_t, mask_t


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load PR-IQA model from checkpoint."""
    model = build_priqa(out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[PR-IQA] Loaded checkpoint: {checkpoint_path}")
    return model


# ── Core Inference ──

@torch.no_grad()
def predict_quality_map(
    model: torch.nn.Module,
    partial_map: torch.Tensor,     # (1, 1, S, S) [0, 1]
    generated_img: torch.Tensor,   # (1, 3, S, S) ImageNet-normalized
    reference_img: torch.Tensor,   # (1, 3, S, S) ImageNet-normalized
    mask: torch.Tensor,            # (1, 1, S, S) [0, 1]
    device: str = "cuda",
) -> np.ndarray:
    """Run PR-IQA inference and return quality map as (H, W) float32 [0, 1]."""
    tgt_img = partial_map.repeat(1, 3, 1, 1).to(device)
    dif_img = generated_img.to(device)
    ref_img = reference_img.to(device)
    tgt_mask = mask.to(device)

    tgt_masks = build_mask_pyramid(tgt_mask)
    ones = torch.ones_like(tgt_mask)
    full_masks = build_mask_pyramid(ones)

    use_amp = device.startswith("cuda")
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                tgt_img, dif_img, ref_img,
                *tgt_masks, *full_masks, *full_masks,
            )
    else:
        output = model(
            tgt_img, dif_img, ref_img,
            *tgt_masks, *full_masks, *full_masks,
        )

    out = output.detach().squeeze().to(torch.float32).cpu().numpy()
    return minmax01(out)


def run_single(
    model, device,
    generated_path: str,
    reference_path: str,
    output_path: str = "quality_map.png",
    use_loftup: bool = True,
):
    """Run inference on a single sample (end-to-end)."""
    print("[PR-IQA] Generating partial map...")
    pmap, mask = generate_partial_map(generated_path, reference_path, device, use_loftup)

    gen = RGB_TRANSFORM(Image.open(generated_path).convert("RGB")).unsqueeze(0)
    ref = RGB_TRANSFORM(Image.open(reference_path).convert("RGB")).unsqueeze(0)

    qmap = predict_quality_map(model, pmap, gen, ref, mask, device)

    # Save
    pil = Image.fromarray((np.clip(qmap, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
    pil = OUTPUT_TRANSFORM(pil)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    print(f"[PR-IQA] Saved: {output_path} (mean={qmap.mean():.4f})")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="PR-IQA Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="PR-IQA checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--generated", type=str, required=True, help="Query / generated image path")
    parser.add_argument("--reference", type=str, required=True, help="Reference image path")
    parser.add_argument("--output", type=str, default="quality_map.png")
    parser.add_argument("--use_loftup", action="store_true", default=True,
                        help="Use LoftUp upsampling in FeatureMetric")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    run_single(model, device, args.generated, args.reference, args.output, args.use_loftup)


if __name__ == "__main__":
    main()
