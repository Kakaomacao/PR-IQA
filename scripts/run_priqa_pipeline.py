"""
End-to-end quality assessment pipeline:
  FeatureMetric (partial map generation) → PR-IQA (dense quality map)

Given a set of images (generated + references), this script:
  1. Loads FeatureMetric (DINOv2 + VGGT) to compute partial maps and overlap masks
  2. Loads PR-IQA model to produce dense per-pixel quality maps
  3. Saves partial maps, overlap masks, and final quality maps

Usage:
    # Minimal: generated images + reference images
    python scripts/run_quality_pipeline.py \
        --generated_dir /path/to/generated \
        --reference_dir /path/to/references \
        --checkpoint checkpoints/priqa_base.pt \
        --output_dir /path/to/output

    # With explicit partial maps (skip FeatureMetric, run PR-IQA only)
    python scripts/run_quality_pipeline.py \
        --generated_dir /path/to/generated \
        --reference_dir /path/to/references \
        --partial_map_dir /path/to/partial_maps \
        --mask_dir /path/to/masks \
        --checkpoint checkpoints/priqa_base.pt \
        --output_dir /path/to/output \
        --skip_feature_metric

Dependencies:
    - Core: torch, torchvision, einops, xformers
    - FeatureMetric (optional): vggt, loftup, pytorch3d (submodules)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

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

OUTPUT_TRANSFORM = T.Compose([T.Resize(OUTPUT_SIZE)])

_NUM_RE = re.compile(r"(\d+)")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(s)]


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS],
        key=lambda p: natural_key(p.name),
    )


def minmax01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=np.float32)
    vmin, vmax = float(a[finite].min()), float(a[finite].max())
    if vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    a[~finite] = vmin
    return np.clip((a - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)


def build_mask_pyramid(mask: torch.Tensor):
    return (
        mask,
        F.interpolate(mask, scale_factor=0.5, mode="nearest"),
        F.interpolate(mask, scale_factor=0.25, mode="nearest"),
        F.interpolate(mask, scale_factor=0.125, mode="nearest"),
    )


def save_grey(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
    pil.save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: FeatureMetric → partial maps + overlap masks
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_metric(
    generated_paths: List[Path],
    reference_paths: List[Path],
    output_dir: Path,
    device: str,
    use_loftup: bool = False,
) -> Tuple[Path, Path]:
    """Generate partial maps and overlap masks using FeatureMetric."""
    from pr_iqa.partial_map import FeatureMetric

    pmap_dir = output_dir / "partial_map"
    mask_dir = output_dir / "partial_mask"

    print("[Step 1] Loading FeatureMetric (DINOv2 + VGGT)...")
    metric = FeatureMetric(
        img_size=IMG_SIZE,
        use_vggt=True,
        use_loftup=use_loftup,
    ).to(device).eval()

    n_gen = len(generated_paths)
    n_ref = len(reference_paths)
    print(f"  Generated: {n_gen} images")
    print(f"  Reference: {n_ref} images")

    for i, gen_path in enumerate(generated_paths):
        # Select nearest reference (first half → first ref, second half → last ref)
        if n_ref == 1:
            ref_path = reference_paths[0]
        elif i < n_gen // 2:
            ref_path = reference_paths[0]
        else:
            ref_path = reference_paths[-1]

        # Load images as (2, 3, H, W) float [0, 1]
        gen_img = np.array(Image.open(gen_path).convert("RGB"))
        ref_img = np.array(Image.open(ref_path).convert("RGB"))

        gen_t = torch.from_numpy(gen_img).permute(2, 0, 1).float() / 255.0
        ref_t = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0

        # Resize to square for FeatureMetric
        gen_t = F.interpolate(gen_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        ref_t = F.interpolate(ref_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)

        images = torch.stack([gen_t, ref_t], dim=0).to(device)

        try:
            score, mask, score_map, _ = metric(
                device=device,
                images=images,
                return_overlap_mask=True,
                return_score_map=True,
                partial_generation=True,
            )
        except Exception as e:
            print(f"  [WARN] FeatureMetric failed for {gen_path.name}: {e}")
            # Save zeros as fallback
            save_grey(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32), pmap_dir / gen_path.name)
            save_grey(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32), mask_dir / gen_path.name)
            continue

        # Save partial map
        if score_map is not None:
            smap = score_map[0].cpu().numpy()
            if smap.ndim == 1:
                side = int(np.sqrt(smap.shape[0]))
                smap = smap.reshape(side, side)
            save_grey(np.clip(smap, 0, 1), pmap_dir / (gen_path.stem + ".png"))
        else:
            save_grey(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32), pmap_dir / (gen_path.stem + ".png"))

        # Save overlap mask
        if mask is not None:
            msk = mask[0].cpu().numpy().astype(np.float32)
            if msk.ndim == 3:
                msk = msk[0]
            save_grey(np.clip(msk, 0, 1), mask_dir / (gen_path.stem + ".png"))
        else:
            save_grey(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32), mask_dir / (gen_path.stem + ".png"))

        if (i + 1) % 10 == 0 or (i + 1) == n_gen:
            print(f"  [{i + 1}/{n_gen}] score={score:.4f}")

    print(f"[Step 1] Done. Saved to {pmap_dir} and {mask_dir}")
    return pmap_dir, mask_dir


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: PR-IQA → dense quality maps
# ──────────────────────────────────────────────────────────────────────────────

def run_priqa_inference(
    generated_paths: List[Path],
    reference_paths: List[Path],
    pmap_dir: Path,
    mask_dir: Path,
    checkpoint: str,
    output_dir: Path,
    device: str,
):
    """Run PR-IQA model on partial maps to produce dense quality maps."""
    from pr_iqa.model import build_priqa

    qmap_dir = output_dir / "quality_map"
    qmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 2] Loading PR-IQA model from {checkpoint}...")
    model = build_priqa(out_channels=1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    n_gen = len(generated_paths)
    n_ref = len(reference_paths)
    use_amp = device.startswith("cuda")

    scores = []
    for i, gen_path in enumerate(generated_paths):
        stem = gen_path.stem

        # Load generated image
        gen_img = RGB_TRANSFORM(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)

        # Load reference image
        if n_ref == 1:
            ref_path = reference_paths[0]
        elif i < n_gen // 2:
            ref_path = reference_paths[0]
        else:
            ref_path = reference_paths[-1]
        ref_img = RGB_TRANSFORM(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)

        # Load partial map
        pmap_path = pmap_dir / (stem + ".png")
        if pmap_path.exists():
            pmap = GREY_TRANSFORM(Image.open(pmap_path).convert("L")).unsqueeze(0).to(device)
        else:
            pmap = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)

        # Load mask
        mask_path = mask_dir / (stem + ".png")
        if mask_path.exists():
            mask = GREY_TRANSFORM(Image.open(mask_path).convert("L")).unsqueeze(0).to(device)
        else:
            mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)

        # Build inputs
        tgt_img = pmap.repeat(1, 3, 1, 1)
        tgt_masks = build_mask_pyramid(mask)
        ones = torch.ones_like(mask)
        full_masks = build_mask_pyramid(ones)

        # Inference
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        tgt_img, gen_img, ref_img,
                        *tgt_masks, *full_masks, *full_masks,
                    )
            else:
                output = model(
                    tgt_img, gen_img, ref_img,
                    *tgt_masks, *full_masks, *full_masks,
                )

        out = output.detach().squeeze().to(torch.float32).cpu().numpy()
        qmap = minmax01(out)
        mean_score = float(qmap.mean())
        scores.append(mean_score)

        # Save quality map
        pil = Image.fromarray((np.clip(qmap, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
        pil = OUTPUT_TRANSFORM(pil)
        save_path = qmap_dir / (stem + ".png")
        pil.save(save_path)

        if (i + 1) % 10 == 0 or (i + 1) == n_gen:
            print(f"  [{i + 1}/{n_gen}] {stem}: score={mean_score:.4f}")

    avg = np.mean(scores) if scores else 0.0
    print(f"[Step 2] Done. Average quality: {avg:.4f}")
    print(f"  Quality maps saved to {qmap_dir}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PR-IQA: End-to-end quality assessment pipeline"
    )
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory with generated / distorted images")
    parser.add_argument("--reference_dir", type=str, required=True,
                        help="Directory with reference images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/priqa_base.pt",
                        help="PR-IQA model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for all results")
    parser.add_argument("--device", type=str, default="cuda")

    # Optional: skip FeatureMetric if partial maps already exist
    parser.add_argument("--skip_feature_metric", action="store_true",
                        help="Skip partial map generation (use existing maps)")
    parser.add_argument("--partial_map_dir", type=str, default=None,
                        help="Existing partial map directory (with --skip_feature_metric)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Existing mask directory (with --skip_feature_metric)")

    # FeatureMetric options
    parser.add_argument("--use_loftup", action="store_true",
                        help="Use LoftUp for feature upsampling")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = list_images(Path(args.generated_dir))
    reference_paths = list_images(Path(args.reference_dir))

    if not generated_paths:
        print(f"[ERROR] No images found in {args.generated_dir}")
        return
    if not reference_paths:
        print(f"[ERROR] No images found in {args.reference_dir}")
        return

    print(f"{'=' * 60}")
    print(f"PR-IQA Quality Assessment Pipeline")
    print(f"  Generated: {len(generated_paths)} images")
    print(f"  Reference: {len(reference_paths)} images")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Step 1: Partial map generation
    if args.skip_feature_metric:
        pmap_dir = Path(args.partial_map_dir) if args.partial_map_dir else output_dir / "partial_map"
        mask_dir = Path(args.mask_dir) if args.mask_dir else output_dir / "partial_mask"
        print(f"[Step 1] Skipped (using existing maps from {pmap_dir})")
    else:
        pmap_dir, mask_dir = run_feature_metric(
            generated_paths, reference_paths, output_dir, device,
            use_loftup=args.use_loftup,
        )

    # Step 2: PR-IQA inference
    scores = run_priqa_inference(
        generated_paths, reference_paths,
        pmap_dir, mask_dir,
        args.checkpoint, output_dir, device,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  Images processed: {len(scores)}")
    if scores:
        print(f"  Mean quality:     {np.mean(scores):.4f}")
        print(f"  Min quality:      {np.min(scores):.4f}")
        print(f"  Max quality:      {np.max(scores):.4f}")
    print(f"  Output dir:       {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
