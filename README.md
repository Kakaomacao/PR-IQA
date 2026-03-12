# <img src="assets/pr-iqa.png" width="32"> PR-IQA: Partial-Reference Image Quality Assessment

Dense per-pixel quality assessment for generated images using partial 3D references.

**Input**: partial quality map + generated image + reference image
**Output**: dense quality map (per-pixel score in [0, 1])

## Architecture

PR-IQA is a 3-input U-Net encoder-decoder with cross-attention (59.2M parameters).

```
                  ref_img ──► img_encoder (shared) ──► kv
                                                        │
partial_map ──► map_encoder ──────────────────────► cross-attn ──┐
                                                                  ├─► qfuse ──► decoder ──► quality_map
generated_img ──► img_encoder (shared) ──────────► cross-attn ──┘
```

- **Encoder**: 4 levels (48 → 96 → 192 → 384), TransformerLikeBlocks with ChannelGate + xformers Attention + FFN
- **Decoder**: 3 levels with skip connections from the generated image encoder
- **Output**: sigmoid-activated per-pixel quality map

Each input is accompanied by a 4-scale mask pyramid (whole → half → quarter → tiny) for mask-aware processing.

## Setup

```bash
git clone https://github.com/Kakaomacao/PR-IQA.git
cd PR-IQA

# Create environment
conda create -n priqa python=3.10 -y
conda activate priqa

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install PR-IQA package
pip install -e .
```

## Quick Start

### Inference (single image)

```bash
python inference.py \
    --checkpoint checkpoints/priqa_base.pt \
    --partial_map examples/partial_map.png \
    --generated examples/generated.png \
    --reference examples/reference.png \
    --output output/quality_map.png
```

### Inference (batch)

Expects a directory with subdirectories: `diffusion/`, `partial_map/`, `partial_mask/`, `gt/`.

```bash
python inference.py \
    --checkpoint checkpoints/priqa_base.pt \
    --input_dir /path/to/interval
```

### Python API

```python
from pr_iqa import build_priqa
from inference import load_model, predict_quality_map

model = load_model("checkpoints/priqa_base.pt", device="cuda")

# quality_map: np.ndarray (H, W), values in [0, 1]
quality_map = predict_quality_map(
    model, partial_map, generated_img, reference_img, mask, device="cuda"
)
```

### End-to-end pipeline (FeatureMetric + PR-IQA)

```bash
# Full pipeline: partial map generation → dense quality map
python scripts/run_quality_pipeline.py \
    --generated_dir /path/to/generated_images \
    --reference_dir /path/to/reference_images \
    --checkpoint checkpoints/priqa_base.pt \
    --output_dir output/

# PR-IQA only (skip FeatureMetric, use existing partial maps)
python scripts/run_quality_pipeline.py \
    --generated_dir /path/to/generated_images \
    --reference_dir /path/to/reference_images \
    --partial_map_dir /path/to/partial_maps \
    --mask_dir /path/to/masks \
    --checkpoint checkpoints/priqa_base.pt \
    --output_dir output/ \
    --skip_feature_metric
```

Output structure:
```
output/
├── partial_map/    # FeatureMetric partial quality maps
├── partial_mask/   # Overlap masks
└── quality_map/    # Final dense quality maps (from PR-IQA)
```

## Training

```bash
# Single GPU
python train.py \
    --root /path/to/training_data \
    --ckpt_dir checkpoints/ \
    --epochs 3 --batch_size 8 --lr 1e-4

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py \
    --root /path/to/training_data \
    --ckpt_dir checkpoints/ \
    --epochs 3 --batch_size 8 --lr 1e-4
```

### Training Data Structure

```
training_data/
├── scene_001/
│   ├── diffusion/         # Generated / distorted images (RGB)
│   ├── gt/                # Ground-truth reference images (RGB)
│   ├── partial_map/       # Partial quality maps (grayscale)
│   └── partial_mask/      # Overlap masks (grayscale)
├── scene_002/
│   └── ...
```

### Loss Function

| Loss | Weight | Description |
|------|--------|-------------|
| JSD  | 1.0    | Jensen-Shannon divergence between predicted and target distributions |
| Masked L1 | 0.5 | L1 loss weighted by the overlap mask |
| Pearson | 0.25 | Pearson correlation coefficient (structural similarity) |

## Partial Map Generation (Optional)

To generate training data (partial quality maps) from scratch, you need additional dependencies:

```bash
# Install submodules
git submodule update --init --recursive

# Additional dependencies
pip install pytorch3d einops
```

```bash
python scripts/generate_partial_maps.py \
    --image_dir /path/to/images \
    --output_dir /path/to/output \
    --vggt_weights facebook/VGGT-1B
```

This step requires: `vggt` (submodule), `loftup` (submodule), `pytorch3d`.
The core model (training and inference) does **not** require these.

## Project Structure

```
PR-IQA/
├── pr_iqa/
│   ├── model/
│   │   ├── layers.py          # PartialConv2d, TransformerLikeBlock, Attention, ...
│   │   └── priqa.py           # PRIQA model (59.2M params)
│   ├── partial_map/
│   │   └── feature_metric.py  # DINOv2 + LoftUp + VGGT → partial quality map
│   ├── loss.py                # JSD, masked L1, Pearson, ranking losses
│   ├── transforms.py          # ImageNet normalization, mask pyramids
│   └── dataset.py             # SceneDataset with augmentation
├── train.py                   # DDP training script
├── inference.py               # Single / batch inference
├── scripts/
│   └── generate_partial_maps.py
├── configs/
│   └── default.yaml
├── checkpoints/
│   └── priqa_base.pt
└── submodules/
    ├── loftup/
    └── vggt/
```

## Configuration

See [`configs/default.yaml`](configs/default.yaml) for all hyperparameters.

Key defaults:

| Parameter | Value |
|-----------|-------|
| Base dim  | 48    |
| Encoder blocks | [2, 3, 3, 4] |
| Attention heads | [1, 2, 4, 8] |
| Learning rate | 1e-4 |
| AMP dtype | bfloat16 |
| Input size (train) | 224 |
| Input size (inference) | 256 |

## Requirements

**Core** (model + training + inference):
- PyTorch >= 2.0
- torchvision >= 0.15
- einops >= 0.7.0
- xformers >= 0.0.22

**Partial map generation** (optional):
- pytorch3d >= 0.7.0
- vggt (submodule)
- loftup (submodule)

## License

TBD
