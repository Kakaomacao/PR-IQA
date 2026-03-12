"""
Data transforms and batch preparation utilities for PR-IQA training.

ImageNet normalization is applied to RGB inputs.
Grayscale inputs (partial maps, masks) are kept in [0, 1].
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_rgb_transform(img_size: int = 256) -> T.Compose:
    """Transform for RGB images: resize → tensor → ImageNet normalize."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_grey_transform(img_size: int = 256) -> T.Compose:
    """Transform for grayscale images (maps/masks): resize → tensor [0,1]."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])


def make_pyramid_masks(mask_whole: torch.Tensor):
    """Build 3 downscaled masks from (B, 1, H, W) → half, quarter, tiny."""
    mask_half = F.interpolate(mask_whole, scale_factor=0.5, mode="nearest")
    mask_quarter = F.interpolate(mask_whole, scale_factor=0.25, mode="nearest")
    mask_tiny = F.interpolate(mask_whole, scale_factor=0.125, mode="nearest")
    return mask_half, mask_quarter, mask_tiny


def prepare_batch(batch: dict, device: torch.device):
    """Prepare a training batch for the PR-IQA model.

    Takes a dataset batch dict and returns (model_args, gt) where
    model_args is a tuple of 15 tensors matching PRIQA.forward() signature.

    Returns:
        model_args: (tgt_img, dif_img, ref_img, + 12 mask tensors)
        gt: (B, 1, H, W) ground truth quality map
    """
    dtype = torch.bfloat16

    dif_img = batch["tgt_diff"].to(device, dtype=dtype, non_blocking=True,
                                   memory_format=torch.channels_last)
    tgt_mask_whole = batch["partial_mask"].to(device, dtype=dtype, non_blocking=True,
                                              memory_format=torch.channels_last)
    tgt_img_1ch = batch["partial_map"].to(device, dtype=dtype, non_blocking=True,
                                          memory_format=torch.channels_last)
    tgt_img = tgt_img_1ch.repeat(1, 3, 1, 1)
    ref_img = batch["current_ref"].to(device, dtype=dtype, non_blocking=True,
                                      memory_format=torch.channels_last)
    gt = batch["full_map"].to(device, dtype=dtype, non_blocking=True,
                              memory_format=torch.channels_last)

    tgt_mask_half, tgt_mask_quarter, tgt_mask_tiny = make_pyramid_masks(tgt_mask_whole)

    ones = torch.ones_like
    dif_mask_whole = ones(tgt_mask_whole)
    dif_mask_half = ones(tgt_mask_half)
    dif_mask_quarter = ones(tgt_mask_quarter)
    dif_mask_tiny = ones(tgt_mask_tiny)

    ref_mask_whole = ones(tgt_mask_whole)
    ref_mask_half = ones(tgt_mask_half)
    ref_mask_quarter = ones(tgt_mask_quarter)
    ref_mask_tiny = ones(tgt_mask_tiny)

    model_args = (
        tgt_img, dif_img, ref_img,
        tgt_mask_whole, tgt_mask_half, tgt_mask_quarter, tgt_mask_tiny,
        dif_mask_whole, dif_mask_half, dif_mask_quarter, dif_mask_tiny,
        ref_mask_whole, ref_mask_half, ref_mask_quarter, ref_mask_tiny,
    )
    return model_args, gt
