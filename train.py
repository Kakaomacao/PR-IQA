"""
Train PR-IQA model.

Supports DDP multi-GPU training with bfloat16 AMP.

Usage:
    # Single GPU
    python train.py --root /path/to/data --ckpt_dir ./checkpoints

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train.py --root /path/to/data
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pr_iqa.model import build_priqa
from pr_iqa.loss import loss_jsd, loss_masked_l1, loss_pearson
from pr_iqa.transforms import prepare_batch
from pr_iqa.dataset import SceneDataset
from pr_iqa.transforms import build_rgb_transform, build_grey_transform

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_FOUND = False

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True


def _finite(*tensors):
    return all(torch.isfinite(t).all() for t in tensors)


def _unwrap(m):
    while hasattr(m, "module"):
        m = m.module
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def save_ckpt(path, model, optimizer, epoch, global_step, best_val=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
    }, path)


def load_ckpt(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("best_val")


# ── Training ──

def train_one_epoch(args, model, loader, optimizer, device, scaler, scheduler,
                    epoch, global_step, writer=None):
    model.train()
    total_sum = torch.zeros((), device=device)
    total_count = 0
    rank = int(os.environ.get("RANK", "0"))

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="train",
                dynamic_ncols=True, disable=(rank != 0))

    for it, batch in pbar:
        if args.max_steps and (it + 1) > args.max_steps:
            break

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.amp and device.type == "cuda")):
            model_args, gt = prepare_batch(batch, device)
            partial_mask = batch["partial_mask"].to(device)
            partial_map_gt = batch["partial_map"].to(device)

            pred = model(*model_args)

            jsd = loss_jsd(pred, gt)
            l1 = loss_masked_l1(pred, partial_map_gt, partial_mask)
            pear = loss_pearson(pred, gt)

            loss = args.lambda_jsd * jsd + args.lambda_l1 * l1 + args.lambda_pearson * pear

        if not _finite(loss, jsd, l1, pear):
            raise RuntimeError("Non-finite loss detected")

        if args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()
            global_step += 1

        if writer and rank == 0 and global_step % args.tb_every == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/jsd", jsd.item(), global_step)
            writer.add_scalar("train/l1", l1.item(), global_step)
            writer.add_scalar("train/pearson", pear.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        total_sum += loss.detach()
        total_count += 1

        if (it + 1) % args.log_interval == 0 and rank == 0:
            avg = (total_sum / total_count).item()
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3e}")

    return (total_sum / max(1, total_count)).item(), global_step


@torch.no_grad()
def eval_one_epoch(args, model, loader, device):
    model.eval()
    total_sum = torch.zeros((), device=device)
    total_count = 0
    rank = int(os.environ.get("RANK", "0"))

    pbar = tqdm(enumerate(loader), total=len(loader), desc="eval",
                dynamic_ncols=True, disable=(rank != 0))

    for it, batch in pbar:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.amp and device.type == "cuda")):
            model_args, gt = prepare_batch(batch, device)
            partial_mask = batch["partial_mask"].to(device)
            partial_map_gt = batch["partial_map"].to(device)
            pred = model(*model_args)
            jsd = loss_jsd(pred, gt)
            l1 = loss_masked_l1(pred, partial_map_gt, partial_mask)
            pear = loss_pearson(pred, gt)
            loss = args.lambda_jsd * jsd + args.lambda_l1 * l1 + args.lambda_pearson * pear

        total_sum += loss.detach()
        total_count += 1

    return (total_sum / max(1, total_count)).item()


# ── Main ──

def main():
    args = parse_args()

    # DDP init
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        is_distributed = True
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(args.seed)

    # TensorBoard
    writer = None
    if TENSORBOARD_FOUND and rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    # Dataset
    rgb_t = build_rgb_transform(args.img_size)
    grey_t = build_grey_transform(args.img_size)
    dataset = SceneDataset(args.root, rgb_transform=rgb_t, grayscale_transform=grey_t, training=True)
    if rank == 0:
        print(f"[Dataset] {len(dataset)} samples from {args.root}")

    # Model
    model = build_priqa(
        out_channels=1, dim=args.dim,
        num_blocks=tuple(args.num_blocks), heads=tuple(args.heads),
        ffn_expansion_factor=args.ffn_expansion, bias=args.bias,
        layernorm_type=args.ln_type,
    )
    model = model.to(device, memory_format=torch.channels_last)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] {model.__class__.__name__}: {n_params:,} params "
              f"({n_params * 4 / 1024**2:.1f} MB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    start_epoch, global_step = 0, 0
    if args.resume and Path(args.resume).is_file():
        start_epoch, global_step, _ = load_ckpt(Path(args.resume), model, optimizer)
        if rank == 0:
            print(f"[Resume] epoch={start_epoch}, step={global_step}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, world_size) / args.batch_size))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-6)

    # Eval loader
    eval_loader = None
    if args.eval_root and rank == 0:
        eval_ds = SceneDataset(args.eval_root, rgb_transform=rgb_t, grayscale_transform=grey_t, training=False)
        if len(eval_ds) > 0:
            eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
            print(f"[Eval] {len(eval_ds)} samples")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=args.num_workers,
            pin_memory=True, drop_last=True, persistent_workers=(args.num_workers > 0),
        )

        loss, global_step = train_one_epoch(
            args, model, loader, optimizer, device, scaler,
            scheduler, epoch + 1, global_step, writer,
        )

        if rank == 0:
            print(f"[Epoch {epoch + 1}] train_loss={loss:.6f}, lr={optimizer.param_groups[0]['lr']:.3e}")

        if eval_loader and rank == 0:
            val_loss = eval_one_epoch(args, model, eval_loader, device)
            print(f"[Epoch {epoch + 1}] val_loss={val_loss:.6f}")
            if writer:
                writer.add_scalar("eval/loss", val_loss, epoch + 1)

        if (epoch + 1) % args.save_every == 0 and rank == 0:
            out = Path(args.ckpt_dir) / f"epoch{epoch + 1:03d}.pt"
            save_ckpt(out, model, optimizer, epoch + 1, global_step)
            print(f"[ckpt] {out}")

    # Final save
    if rank == 0:
        save_ckpt(Path(args.ckpt_dir) / "last.pt", model, optimizer, args.epochs, global_step)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    if writer:
        writer.close()


def parse_args():
    p = argparse.ArgumentParser(description="Train PR-IQA")

    # Data
    p.add_argument("--root", type=str, required=True, help="Training data root")
    p.add_argument("--eval_root", type=str, default=None, help="Evaluation data root")
    p.add_argument("--img_size", type=int, default=224)

    # Model
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_blocks", type=int, nargs=4, default=[2, 3, 3, 4])
    p.add_argument("--heads", type=int, nargs=4, default=[1, 2, 4, 8])
    p.add_argument("--ffn_expansion", type=float, default=2.66)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--ln_type", type=str, default="WithBias", choices=["WithBias", "BiasFree"])

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--max_steps", type=int, default=0, help="Max steps per epoch (0=unlimited)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Loss weights
    p.add_argument("--lambda_jsd", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=0.5)
    p.add_argument("--lambda_pearson", type=float, default=0.25)

    # Checkpoints
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--resume", type=str, default=None)

    # Logging
    p.add_argument("--log_dir", type=str, default="./runs/pr_iqa")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--tb_every", type=int, default=20)

    return p.parse_args()


if __name__ == "__main__":
    main()
