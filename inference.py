"""
PR-IQA inference: Generate dense quality maps.

Pipeline:
  partial_map + generated_image + reference_image → PR-IQA → quality_map

Usage:
    # Single image inference
    python inference.py \
        --partial_map partial.png \
        --generated gen.jpg \
        --reference ref.jpg \
        --checkpoint checkpoints/epoch003.pt \
        --output quality_map.png

    # Batch inference on a directory
    python inference.py \
        --input_dir /path/to/interval \
        --checkpoint checkpoints/epoch003.pt
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

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
    partial_map_path: str,
    generated_path: str,
    reference_path: str,
    mask_path: str = None,
    output_path: str = "quality_map.png",
):
    """Run inference on a single sample."""
    pmap = GREY_TRANSFORM(Image.open(partial_map_path).convert("L")).unsqueeze(0)
    gen = RGB_TRANSFORM(Image.open(generated_path).convert("RGB")).unsqueeze(0)
    ref = RGB_TRANSFORM(Image.open(reference_path).convert("RGB")).unsqueeze(0)

    if mask_path and Path(mask_path).exists():
        mask = GREY_TRANSFORM(Image.open(mask_path).convert("L")).unsqueeze(0)
    else:
        mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE)

    qmap = predict_quality_map(model, pmap, gen, ref, mask, device)

    # Save
    pil = Image.fromarray((np.clip(qmap, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
    pil = OUTPUT_TRANSFORM(pil)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    print(f"[PR-IQA] Saved: {output_path} (mean={qmap.mean():.4f})")


def run_batch(model, device, input_dir: str, output_dirname: str = "pr_iqa_output"):
    """Run inference on all images in a directory structure.

    Expects input_dir to contain:
      - diffusion/  (generated images)
      - partial_map/ (partial quality maps)
      - partial_mask/ (overlap masks)
      - gt/ (reference images)
    """
    _num = re.compile(r"(\d+)")
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _num.split(s)]

    input_dir = Path(input_dir)
    diff_dir = input_dir / "diffusion"
    pmap_dir = input_dir / "partial_map"
    mask_dir = input_dir / "partial_mask"
    gt_dir = input_dir / "gt"
    out_dir = input_dir / output_dirname

    diff_files = sorted(
        [p for p in diff_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTS],
        key=lambda p: natural_key(p.name),
    )

    gt_files = sorted(
        [p for p in gt_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTS],
        key=lambda p: natural_key(p.name),
    ) if gt_dir.exists() else []

    print(f"[PR-IQA] Processing {len(diff_files)} images from {input_dir}")

    for i, dif_path in enumerate(diff_files):
        pmap_path = pmap_dir / (dif_path.stem + ".png")
        mask_path = mask_dir / (dif_path.stem + ".png")

        # Select nearest reference
        if gt_files:
            ref_path = gt_files[0] if i < len(diff_files) // 2 else gt_files[-1]
        else:
            ref_path = None

        if not pmap_path.exists():
            pmap_img = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        else:
            pmap_img = GREY_TRANSFORM(Image.open(pmap_path).convert("L")).unsqueeze(0)

        gen_img = RGB_TRANSFORM(Image.open(dif_path).convert("RGB")).unsqueeze(0)

        if ref_path and ref_path.exists():
            ref_img = RGB_TRANSFORM(Image.open(ref_path).convert("RGB")).unsqueeze(0)
        else:
            ref_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

        if mask_path.exists():
            mask = GREY_TRANSFORM(Image.open(mask_path).convert("L")).unsqueeze(0)
        else:
            mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE)

        qmap = predict_quality_map(model, pmap_img, gen_img, ref_img, mask, device)

        save_path = out_dir / (dif_path.stem + ".png")
        pil = Image.fromarray((np.clip(qmap, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
        pil = OUTPUT_TRANSFORM(pil)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(save_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(diff_files):
            print(f"  [{i + 1}/{len(diff_files)}] mean_score={qmap.mean():.4f}")

    print(f"[PR-IQA] Done. Output: {out_dir}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="PR-IQA Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="PR-IQA checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")

    # Single image mode
    parser.add_argument("--partial_map", type=str, default=None)
    parser.add_argument("--generated", type=str, default=None)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output", type=str, default="quality_map.png")

    # Batch mode
    parser.add_argument("--input_dir", type=str, default=None, help="Directory for batch inference")
    parser.add_argument("--output_dirname", type=str, default="pr_iqa_output")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    if args.input_dir:
        run_batch(model, device, args.input_dir, args.output_dirname)
    elif args.partial_map and args.generated and args.reference:
        run_single(model, device, args.partial_map, args.generated,
                   args.reference, args.mask, args.output)
    else:
        parser.print_help()
        print("\nProvide either --input_dir for batch or --partial_map/--generated/--reference for single.")


if __name__ == "__main__":
    main()
