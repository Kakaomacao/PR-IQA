"""
FeatureMetric: DINOv2 + LoftUp feature-based quality metric.

Generates partial quality maps by:
  1. Extracting DINOv2 features (upsampled via LoftUp) from input images
  2. Using VGGT for monocular depth and pose estimation
  3. Constructing a colored 3D point cloud with features
  4. Rendering the point cloud from the target viewpoint via PyTorch3D
  5. Computing cosine similarity between rendered features and target features

Two modes:
  - partial_generation=True: Full 3D pipeline → partial map + overlap mask
  - partial_generation=False: Direct cosine similarity → total quality map

Dependencies (Level 1):
  - VGGT (facebook/VGGT-1B)
  - LoftUp (andrehuang/loftup)
  - PyTorch3D
"""

import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
from einops import rearrange

# VGGT
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

# LoftUp
from loftup.featurizers import get_featurizer
from loftup.upsamplers import norm

# PyTorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection


class FeatureMetric(Module):
    """DINOv2 + LoftUp + VGGT → partial / total quality map.

    Args:
        img_size: Inference image size (controls rasterizer resolution).
        feature_backbone: Name of the feature backbone (default: ``"dinov2"``).
        loftup_torch_hub: Torch Hub repository for LoftUp.
        loftup_model_name: LoftUp model name.
        vggt_weights: HuggingFace model ID for VGGT.
        use_vggt: Load VGGT for depth/pose estimation.
        use_loftup: Load LoftUp for feature upsampling.
    """

    def __init__(
        self,
        img_size: int = 256,
        feature_backbone: str = "dinov2",
        loftup_torch_hub: Union[str, Path] = "andrehuang/loftup",
        loftup_model_name: Union[str, Path] = "loftup_dinov2s",
        vggt_weights: Union[str, Path] = "facebook/VGGT-1B",
        use_vggt: bool = True,
        use_loftup: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.feature_backbone, self.patch_size, self.dim = get_featurizer(feature_backbone)

        self.upsampler = (
            torch.hub.load(loftup_torch_hub, loftup_model_name, pretrained=True)
            if use_loftup else None
        )
        self.use_loftup = use_loftup

        if use_vggt:
            self.vggt = VGGT.from_pretrained(vggt_weights)

        self.compositor = AlphaCompositor()

    def _render(self, point_clouds: Pointclouds, **kwargs):
        """Render point cloud features to images."""
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)

        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        images = images.permute(0, 2, 3, 1)
        return images, fragments.zbuf

    @torch.no_grad()
    def forward(
        self,
        device: str,
        images: Tensor,                    # (K, 3, H, W)
        return_overlap_mask: bool = False,
        return_score_map: bool = False,
        return_projections: bool = False,
        partial_generation: bool = False,
        use_filtering: bool = False,
    ) -> Tuple[float, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Compute quality score map.

        Args:
            device: Torch device string.
            images: (K, 3, H, W) input images. First image is the target.
            partial_generation: If True, use full 3D pipeline for partial map.

        Returns:
            (score_scalar, overlap_mask, score_map, projections)
        """
        k, c, h, w = images.shape

        # Setup rasterizer
        raster_settings = PointsRasterizationSettings(
            image_size=(h, w), radius=0.01, points_per_pixel=10, bin_size=0,
        )
        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        # Extract features
        images_norm = norm(images)
        hr_feats = []
        for i in range(k):
            img = images_norm[i:i + 1]
            lr_feat = self.feature_backbone(img)
            if self.use_loftup and self.upsampler is not None:
                hr_feat = self.upsampler(lr_feat, img)
            else:
                hr_feat = lr_feat
            hr_feat = rearrange(hr_feat, "b c h w -> b (h w) c")
            hr_feats.append(hr_feat)
        hr_feats = torch.cat(hr_feats, dim=0)

        if not partial_generation:
            # Fast cosine similarity mode
            dot = (hr_feats[0] * hr_feats[1]).sum(dim=1)
            tgt_norm = torch.linalg.norm(hr_feats[0], dim=1)
            ref_norm = torch.linalg.norm(hr_feats[1], dim=1)
            cosine_sim = dot / (tgt_norm * ref_norm + 1e-8)
            score_map = torch.clamp(cosine_sim, min=0.0, max=1.0)

            H_out, W_out = 518, 294
            score_map = score_map.reshape(W_out, H_out).unsqueeze(0)
            return score_map.mean().item(), None, score_map if return_score_map else None, None

        # Full 3D partial map generation
        preds = self.vggt(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        depth, depth_conf = preds["depth"], preds["depth_conf"]

        point_map = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0),
        )
        cols = images.cpu().numpy().transpose(0, 2, 3, 1)
        cols = cols / cols.max()
        pts_flatten = torch.from_numpy(
            rearrange(point_map, "k h w c -> k (h w) c")
        ).float().to(device)

        if use_filtering:
            percent = 20
            quantile = torch.quantile(depth_conf, percent / 100.0)
            mask_flat = rearrange((depth_conf > quantile).squeeze(0), "k h w -> k (h w)")
            points_list, features_list = [], []
            for i in range(k):
                valid = mask_flat[i]
                points_list.append(pts_flatten[i][valid])
                features_list.append(hr_feats[i][valid])
            point_clouds = Pointclouds(points=points_list, features=features_list)
        else:
            point_clouds = Pointclouds(points=pts_flatten, features=hr_feats)

        # Render from target viewpoint
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        E, K = extrinsic.squeeze(0), intrinsic.squeeze(0)
        R0, T0, K0 = E[0, :3, :3], E[0, :3, 3], K[0]
        B = pts_flatten.shape[0]

        R_repeat = R0.unsqueeze(0).repeat(B, 1, 1)
        T_repeat = T0.unsqueeze(0).repeat(B, 1)
        K_repeat = K0.unsqueeze(0).repeat(B, 1, 1)
        im_size = torch.tensor([[h, w]]).repeat(B, 1).to(device)

        cameras_p3d = _cameras_from_opencv_projection(R_repeat, T_repeat, K_repeat, im_size)

        with torch.autocast("cuda", enabled=False):
            bg_color = torch.tensor(
                [-10000] * hr_feats[0].shape[-1], dtype=torch.float32, device=device,
            )
            rendering, zbuf = self._render(point_clouds, cameras=cameras_p3d, background_color=bg_color)
        rendering = rearrange(rendering, "k h w c -> k c h w")

        # Cosine similarity score map
        target = rendering[0:1]
        reference = rendering[1:]
        dot = (reference * target).sum(dim=1)
        tgt_norm = torch.linalg.norm(target, dim=1)
        ref_norm = torch.linalg.norm(reference, dim=1)
        cosine_sim = dot / (tgt_norm * ref_norm + 1e-8)
        score_map = torch.clamp(cosine_sim, min=0.0, max=1.0)

        # Mask true background
        target_mask = zbuf[0, ..., 0] >= 0
        reference_mask = zbuf[1:, ..., 0] >= 0
        true_bg = ~target_mask & ~torch.any(reference_mask, dim=0)
        score_map[:, true_bg] = 0.0

        overlap_mask = zbuf[1:, ..., 0] >= 0

        return (
            score_map.mean().item(),
            overlap_mask if return_overlap_mask else None,
            score_map if return_score_map else None,
            rendering if return_projections else None,
        )
