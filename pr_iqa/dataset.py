"""
Dataset for PR-IQA training.

Expected directory structure per scene::

    s000/
    ├── total/                    # Original keyframe images (RGB)
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    ├── tgt_diffusion/            # Generated images per target
    │   └── 0005/
    │       ├── 0005_diff_0.jpg
    │       └── ...
    ├── total_map/                # Full quality maps (GT, grayscale)
    │   └── 0005/
    │       ├── 0005_diff_0.png
    │       └── ...
    ├── partial_map/              # Partial quality maps (from FeatureMetric)
    │   └── 0005/
    │       ├── 0005_diff_0_ref+10_0015.png
    │       └── ...
    └── partial_mask/             # Overlap masks
        └── 0005/
            ├── 0005_diff_0_ref+10_0015.png
            └── ...

Each sample is a tuple: (tgt, tgt_diff, full_map, partial_map, partial_mask, current_ref).
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SceneDataset(Dataset):
    """Dataset that enumerates all valid (tgt, diff, ref, partial_map, mask) combinations."""

    def __init__(self, root_dir, rgb_transform=None, grayscale_transform=None, training=True):
        self.root_dir = Path(root_dir)
        self.rgb_transform = rgb_transform
        self.grayscale_transform = grayscale_transform
        self.samples = []
        self.ref_deltas = [-20, -10, 10, 20]
        self.training = training

        for scene_path in sorted(self.root_dir.glob("s*")):
            if not scene_path.is_dir():
                continue
            total_dir = scene_path / "total"
            if not total_dir.is_dir():
                continue

            total_images = sorted(total_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            num_total = len(total_images)
            if num_total == 0:
                continue

            for i, tgt_path in enumerate(total_images):
                tgt_stem = tgt_path.stem

                # Find reference images at fixed offsets
                ref_info_list = []
                complete = True
                for d in self.ref_deltas:
                    ref_idx = (i + d) % num_total
                    ref_path = total_images[ref_idx]
                    if not ref_path.exists():
                        complete = False
                        break
                    ref_info_list.append({"path": ref_path, "offset": d})

                if not complete:
                    continue

                tgt_diff_dir = scene_path / "tgt_diffusion" / tgt_stem
                total_map_dir = scene_path / "total_map" / tgt_stem

                for tgt_diff_path in sorted(tgt_diff_dir.glob("*_diff_*.jpg")):
                    full_map_path = total_map_dir / f"{tgt_diff_path.stem}.png"
                    if not full_map_path.exists():
                        continue

                    tgt_diff_stem = tgt_diff_path.stem

                    for ref_info in ref_info_list:
                        ref_path = ref_info["path"]
                        ref_stem = ref_path.stem
                        d = ref_info["offset"]

                        mask_path = (
                            scene_path / "partial_mask" / tgt_stem
                            / f"{tgt_diff_stem}_ref{d:+d}_{ref_stem}.png"
                        )
                        map_path = (
                            scene_path / "partial_map" / tgt_stem
                            / f"{tgt_diff_stem}_ref{d:+d}_{ref_stem}.png"
                        )

                        if mask_path.exists() and map_path.exists():
                            self.samples.append({
                                "tgt": tgt_path,
                                "tgt_diff": tgt_diff_path,
                                "full_map": full_map_path,
                                "partial_mask": mask_path,
                                "partial_map": map_path,
                                "current_ref": ref_path,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]

        tgt_img = Image.open(paths["tgt"]).convert("RGB")
        tgt_diff_img = Image.open(paths["tgt_diff"]).convert("RGB")
        full_map_img = Image.open(paths["full_map"]).convert("L")
        partial_mask_img = Image.open(paths["partial_mask"]).convert("L")
        partial_map_img = Image.open(paths["partial_map"]).convert("L")
        cur_ref_img = Image.open(paths["current_ref"]).convert("RGB")

        # -- Augmentation (training only) --
        if self.training:
            if random.random() > 0.5:
                tgt_img = TF.hflip(tgt_img)
                tgt_diff_img = TF.hflip(tgt_diff_img)
                cur_ref_img = TF.hflip(cur_ref_img)
                full_map_img = TF.hflip(full_map_img)
                partial_mask_img = TF.hflip(partial_mask_img)
                partial_map_img = TF.hflip(partial_map_img)

            if random.random() > 0.7:
                tgt_img = TF.vflip(tgt_img)
                tgt_diff_img = TF.vflip(tgt_diff_img)
                cur_ref_img = TF.vflip(cur_ref_img)
                full_map_img = TF.vflip(full_map_img)
                partial_mask_img = TF.vflip(partial_mask_img)
                partial_map_img = TF.vflip(partial_map_img)

            if random.random() > 0.5:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                saturation = random.uniform(0.9, 1.1)
                for fn in [TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation]:
                    val = brightness if fn == TF.adjust_brightness else (
                        contrast if fn == TF.adjust_contrast else saturation
                    )
                    tgt_img = fn(tgt_img, val)
                    tgt_diff_img = fn(tgt_diff_img, val)
                    cur_ref_img = fn(cur_ref_img, val)

        if self.rgb_transform:
            tgt_img, tgt_diff_img, cur_ref_img = map(
                self.rgb_transform, [tgt_img, tgt_diff_img, cur_ref_img]
            )
        if self.grayscale_transform:
            full_map_img, partial_mask_img, partial_map_img = map(
                self.grayscale_transform, [full_map_img, partial_mask_img, partial_map_img]
            )

        return {
            "tgt": tgt_img,
            "tgt_diff": tgt_diff_img,
            "partial_mask": partial_mask_img,
            "partial_map": partial_map_img,
            "full_map": full_map_img,
            "current_ref": cur_ref_img,
        }
