"""
Generate partial maps and total maps from input images using FeatureMetric.

This script creates the training data for PR-IQA by:
  1. Loading images from a scene directory
  2. Computing DINOv2 features + VGGT depth/pose
  3. Generating partial_map (3D rendered feature similarity) and total_map (direct cosine sim)
  4. Saving overlap masks

Usage:
    python scripts/generate_partial_maps.py \
        --scene_dir /path/to/scene \
        --output_dir /path/to/output

Dependencies: Level 1 (VGGT + LoftUp + PyTorch3D)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pr_iqa.partial_map import FeatureMetric


def main():
    parser = argparse.ArgumentParser(description="Generate partial maps for PR-IQA training")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Scene directory with total/ and tgt_diffusion/ subdirs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as scene_dir)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ref_deltas", type=int, nargs="+", default=[-20, -10, 10, 20],
                        help="Reference frame offsets")

    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scene_dir
    device = args.device if torch.cuda.is_available() else "cpu"

    total_dir = scene_dir / "total"
    diff_dir = scene_dir / "tgt_diffusion"

    if not total_dir.exists():
        print(f"[ERROR] total/ not found in {scene_dir}")
        return

    total_images = sorted(total_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    print(f"[INFO] Found {len(total_images)} total images")

    # Load FeatureMetric
    print("[INFO] Loading FeatureMetric (DINOv2 + VGGT)...")
    metric = FeatureMetric(
        img_size=256,
        use_vggt=True,
        use_loftup=False,
    )
    metric = metric.to(device).eval()

    for tgt_idx, tgt_path in enumerate(total_images):
        tgt_stem = tgt_path.stem

        # Find diffusion images for this target
        tgt_diff_dir = diff_dir / tgt_stem
        if not tgt_diff_dir.exists():
            continue

        diff_images = sorted(tgt_diff_dir.glob("*_diff_*.jpg"))
        if not diff_images:
            continue

        for diff_path in diff_images:
            diff_stem = diff_path.stem

            for delta in args.ref_deltas:
                ref_idx = (tgt_idx + delta) % len(total_images)
                ref_path = total_images[ref_idx]
                ref_stem = ref_path.stem

                # Load images: target (diffusion) and reference
                tgt_img = np.array(Image.open(diff_path).convert("RGB"))
                ref_img = np.array(Image.open(ref_path).convert("RGB"))

                # Stack as (2, 3, H, W) float32 [0, 1]
                tgt_t = torch.from_numpy(tgt_img).permute(2, 0, 1).float() / 255.0
                ref_t = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
                images = torch.stack([tgt_t, ref_t], dim=0).to(device)

                # Compute partial map (3D mode)
                try:
                    score, mask, score_map, _ = metric(
                        device=device,
                        images=images,
                        return_overlap_mask=True,
                        return_score_map=True,
                        partial_generation=True,
                    )
                except Exception as e:
                    print(f"  [WARN] Failed for {diff_stem} ref{delta:+d}: {e}")
                    continue

                if score_map is None:
                    continue

                # Save partial_map
                smap = score_map[0].cpu().numpy()
                smap_uint8 = (np.clip(smap, 0, 1) * 255).astype(np.uint8)
                pmap_path = output_dir / "partial_map" / tgt_stem / f"{diff_stem}_ref{delta:+d}_{ref_stem}.png"
                pmap_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(smap_uint8, mode="L").save(pmap_path)

                # Save partial_mask
                if mask is not None:
                    msk = mask[0].cpu().numpy().astype(np.float32)
                    msk_uint8 = (np.clip(msk, 0, 1) * 255).astype(np.uint8)
                    mask_path = output_dir / "partial_mask" / tgt_stem / f"{diff_stem}_ref{delta:+d}_{ref_stem}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(msk_uint8, mode="L").save(mask_path)

        print(f"  [OK] {tgt_stem}: {len(diff_images)} diffusion images processed")

    print("[DONE] Partial map generation complete")


if __name__ == "__main__":
    main()
