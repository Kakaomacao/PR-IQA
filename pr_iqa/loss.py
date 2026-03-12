"""
Loss functions for PR-IQA training.

Core losses:
  - JSD (Jensen-Shannon Divergence): Distribution matching
  - Masked L1: Pixel-wise L1 on partial map regions
  - Pearson: Correlation-based structural loss

Additional losses (optional):
  - Ranking: Pairwise ranking consistency
  - Global mean/std: Statistics matching
"""

import torch
import torch.nn.functional as F


def loss_jsd(pred, target, tau=0.2, reduction="mean", eps=1e-6):
    """Jensen-Shannon Divergence loss.

    Converts pixel maps to probability distributions via softmax over logits,
    then computes symmetric KL divergence.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        p = pred.float().clamp(min=eps, max=1 - eps)
        y = target.float().clamp(min=eps, max=1 - eps)

        p_logit = torch.logit(p, eps=eps) / tau
        y_logit = torch.logit(y, eps=eps) / tau

        q_hat = torch.softmax(p_logit.flatten(start_dim=1), dim=1)
        q = torch.softmax(y_logit.flatten(start_dim=1), dim=1)

        m = 0.5 * (q + q_hat)

        def _kl(a, b):
            return torch.sum(a * (torch.log(a + eps) - torch.log(b + eps)), dim=1)

        jsd_per = 0.5 * (_kl(q, m) + _kl(q_hat, m))

        if reduction == "mean":
            return jsd_per.mean().to(pred.dtype)
        elif reduction == "sum":
            return jsd_per.sum().to(pred.dtype)
        return jsd_per.to(pred.dtype)


def loss_masked_l1(pred, target, mask, reduction="mean"):
    """L1 loss masked to partial map regions."""
    l = torch.abs(pred - target)
    masked = l * mask
    if reduction == "mean":
        return masked.sum() / (mask.sum() + 1e-8)
    elif reduction == "sum":
        return masked.sum()
    return masked


def loss_l1(pred, target, reduction="mean"):
    """Standard L1 loss."""
    l = (pred - target).abs()
    if reduction == "mean":
        return l.mean().to(pred.dtype)
    elif reduction == "sum":
        return l.sum().to(pred.dtype)
    return l.to(pred.dtype)


def loss_pearson(pred, target, reduction="mean", eps=1e-6):
    """1 - Pearson correlation coefficient."""
    x = pred.float().reshape(pred.shape[0], -1).contiguous()
    y = target.float().reshape(target.shape[0], -1).contiguous()

    mx = x.mean(dim=1)
    my = y.mean(dim=1)
    x = x - mx[:, None]
    y = y - my[:, None]

    xx = (x * x).sum(dim=1)
    yy = (y * y).sum(dim=1)
    denom = torch.sqrt(xx * yy + eps)
    rho = ((x * y).sum(dim=1) / denom).clamp(-1.0, 1.0)

    loss = 1.0 - rho
    if reduction == "mean":
        return loss.mean().to(pred.dtype)
    elif reduction == "sum":
        return loss.sum().to(pred.dtype)
    return loss.to(pred.dtype)


def loss_ranking(pred, gt, margin=0.1):
    """Pairwise ranking loss for relative quality ordering."""
    B, C, H, W = pred.shape
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)

    n = int(H * W * 0.5)
    idx1 = torch.randint(0, H * W, (B, n), device=pred.device)
    idx2 = torch.randint(0, H * W, (B, n), device=pred.device)

    pred1 = pred_flat.gather(1, idx1)
    pred2 = pred_flat.gather(1, idx2)
    gt1 = gt_flat.gather(1, idx1)
    gt2 = gt_flat.gather(1, idx2)

    target = torch.sign(gt1 - gt2)
    return F.margin_ranking_loss(pred1, pred2, target, margin=margin)
