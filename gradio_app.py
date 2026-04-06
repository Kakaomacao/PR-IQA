from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    if exc.name in {"gradio", "gradio_client"}:
        missing = exc.name
        raise ModuleNotFoundError(
            f"Missing dependency: {missing}\n"
            f"Current Python: {sys.executable}\n"
            "Install the UI deps in this same environment with:\n"
            f"  {sys.executable} -m pip install gradio gradio_client\n"
            "Or reinstall all repo deps with:\n"
            f"  {sys.executable} -m pip install -r requirements.txt"
        ) from exc
    raise


REPO_ROOT = Path(__file__).resolve().parent
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RGB_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
FEATURE_METRIC_CACHE: Dict[Tuple[str, bool], object] = {}


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def to_display_image(arr: np.ndarray) -> np.ndarray:
    grey = (normalize_map(arr) * 255.0 + 0.5).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


def to_heatmap_image(arr: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    colored = colormaps[cmap_name](normalize_map(arr))[..., :3]
    return (colored * 255.0 + 0.5).astype(np.uint8)


def ensure_spatial_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        side = int(np.sqrt(arr.shape[0]))
        arr = arr.reshape(side, side)
    return arr


def resize_map(arr: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    interpolate_kwargs = {"size": size, "mode": mode}
    if mode != "nearest":
        interpolate_kwargs["align_corners"] = False
    resized = F.interpolate(tensor, **interpolate_kwargs)
    return resized[0, 0].cpu().numpy()


def image_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def image_to_model_tensor(image: np.ndarray) -> torch.Tensor:
    return RGB_TRANSFORM(image_to_pil(image)).unsqueeze(0)


def map_to_model_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalize_map(arr)).unsqueeze(0).unsqueeze(0)


def get_runtime_status(checkpoint_path: str) -> dict:
    checkpoint_path = (checkpoint_path or "").strip()
    checkpoint_exists = bool(checkpoint_path) and Path(checkpoint_path).expanduser().exists()

    xformers_ok = module_available("xformers")
    pytorch3d_ok = module_available("pytorch3d")
    vggt_dir = REPO_ROOT / "submodules" / "vggt"
    loftup_dir = REPO_ROOT / "submodules" / "loftup"
    vggt_ok = vggt_dir.exists() or module_available("vggt")
    loftup_ok = loftup_dir.exists() or module_available("featurizers")

    return {
        "checkpoint_exists": checkpoint_exists,
        "xformers_ok": xformers_ok,
        "pytorch3d_ok": pytorch3d_ok,
        "vggt_ok": vggt_ok,
        "loftup_ok": loftup_ok,
        "partial_ready": pytorch3d_ok and vggt_ok and loftup_ok,
        "inference_ready": checkpoint_exists and xformers_ok,
    }


def render_diagnostics(checkpoint_path: str) -> str:
    status = get_runtime_status(checkpoint_path)
    checkpoint_display = (checkpoint_path or "").strip() or "<not set>"

    lines = [
        "### Runtime Checks",
        f"- device auto-detect: `{'cuda' if torch.cuda.is_available() else 'cpu'}`",
        f"- checkpoint: `{checkpoint_display}`",
        f"- checkpoint exists: `{'yes' if status['checkpoint_exists'] else 'no'}`",
        f"- xformers: `{'ready' if status['xformers_ok'] else 'missing'}`",
        f"- pytorch3d: `{'ready' if status['pytorch3d_ok'] else 'missing'}`",
        f"- vggt backend: `{'ready' if status['vggt_ok'] else 'missing'}`",
        f"- loftup backend: `{'ready' if status['loftup_ok'] else 'missing'}`",
        f"- partial map stage: `{'ready' if status['partial_ready'] else 'blocked'}`",
        f"- PR-IQA inference stage: `{'ready' if status['inference_ready'] else 'blocked'}`",
    ]

    if not status["partial_ready"]:
        lines.append("")
        lines.append("To enable partial map generation:")
        lines.append("`git submodule update --init --recursive`")
        lines.append("`pip install \"git+https://github.com/facebookresearch/pytorch3d.git\" --no-build-isolation`")

    if not status["checkpoint_exists"]:
        lines.append("")
        lines.append("Set a checkpoint path with `--checkpoint` or the `PRIQA_CHECKPOINT` environment variable.")

    return "\n".join(lines)


def get_feature_metric(device: str, use_loftup: bool):
    key = (device, use_loftup)
    if key not in FEATURE_METRIC_CACHE:
        from pr_iqa.partial_map import FeatureMetric

        metric = FeatureMetric(
            img_size=IMG_SIZE,
            use_vggt=True,
            use_loftup=use_loftup,
        )
        FEATURE_METRIC_CACHE[key] = metric.to(device).eval()
    return FEATURE_METRIC_CACHE[key]


def get_priqa_model(checkpoint_path: str, device: str):
    resolved = str(Path(checkpoint_path).expanduser().resolve())
    key = (resolved, device)
    if key not in MODEL_CACHE:
        from inference import load_model

        MODEL_CACHE[key] = load_model(resolved, device=device)
    return MODEL_CACHE[key]


def generate_partial_map(reference_image: np.ndarray, query_image: np.ndarray, device: str, use_loftup: bool):
    metric = get_feature_metric(device, use_loftup)

    gen_arr = np.asarray(image_to_pil(query_image), dtype=np.float32) / 255.0
    ref_arr = np.asarray(image_to_pil(reference_image), dtype=np.float32) / 255.0

    gen_t = torch.from_numpy(gen_arr).permute(2, 0, 1).float()
    ref_t = torch.from_numpy(ref_arr).permute(2, 0, 1).float()

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

    if score_map is None:
        partial_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        partial_map = ensure_spatial_map(score_map[0].detach().float().cpu().numpy())
        partial_map = resize_map(normalize_map(partial_map), size=(IMG_SIZE, IMG_SIZE), mode="bilinear")

    if mask is None:
        overlap_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        overlap_mask = ensure_spatial_map(mask[0].detach().float().cpu().numpy())
        overlap_mask = resize_map(normalize_map(overlap_mask), size=(IMG_SIZE, IMG_SIZE), mode="nearest")

    return normalize_map(partial_map), normalize_map(overlap_mask), float(score)


def run_priqa_inference(
    reference_image: np.ndarray,
    query_image: np.ndarray,
    partial_map: np.ndarray,
    overlap_mask: np.ndarray,
    checkpoint_path: str,
    device: str,
):
    from inference import predict_quality_map

    model = get_priqa_model(checkpoint_path, device)

    gen_t = image_to_model_tensor(query_image)
    ref_t = image_to_model_tensor(reference_image)
    pmap_t = map_to_model_tensor(partial_map)
    mask_t = map_to_model_tensor(overlap_mask)

    quality_map = predict_quality_map(
        model,
        pmap_t,
        gen_t,
        ref_t,
        mask_t,
        device=device,
    )
    return normalize_map(quality_map), float(np.mean(quality_map))


def run_demo(
    query_image: np.ndarray,
    reference_image: np.ndarray,
    checkpoint_path: str,
    device_choice: str,
    use_loftup: bool,
    progress=gr.Progress(track_tqdm=False),
):
    if query_image is None or reference_image is None:
        return (
            "Both `query` and `reference` images are required.",
            None,
            None,
            None,
            None,
        )

    device = resolve_device(device_choice)
    status = get_runtime_status(checkpoint_path)

    if not status["partial_ready"]:
        return (
            "Partial map generation is blocked. Install the required dependencies and submodules first.",
            None,
            None,
            None,
            None,
        )

    progress(0.1, desc="Loading")
    partial_map, overlap_mask, partial_score = generate_partial_map(
        reference_image=reference_image,
        query_image=query_image,
        device=device,
        use_loftup=use_loftup,
    )

    display_size = (int(query_image.shape[0]), int(query_image.shape[1]))
    partial_display = to_heatmap_image(resize_map(partial_map, size=display_size, mode="bilinear"))
    overlap_display = to_display_image(resize_map(overlap_mask, size=display_size, mode="nearest"))

    quality_display = None
    mean_quality = None

    if status["inference_ready"]:
        progress(0.65, desc="Running PR-IQA")
        quality_map, mean_quality = run_priqa_inference(
            reference_image=reference_image,
            query_image=query_image,
            partial_map=partial_map,
            overlap_mask=overlap_mask,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        quality_display = to_heatmap_image(resize_map(quality_map, size=display_size, mode="bilinear"))

    progress(1.0, desc="Done")

    result_lines = [
        "### Result",
        f"- Feature score: `{partial_score:.4f}`",
        f"- Mean quality score: `{mean_quality:.4f}`" if mean_quality is not None else "- Mean quality score: `<skipped>`",
    ]
    result = "\n".join(result_lines)

    return (
        result,
        partial_display,
        overlap_display,
        quality_display,
        mean_quality,
    )


def build_demo(default_checkpoint: str, default_device: str, default_use_loftup: bool):
    example_ref = REPO_ROOT / "examples" / "case1" / "reference.jpg"
    example_query = REPO_ROOT / "examples" / "case1" / "query.png"

    with gr.Blocks(title="PR-IQA Gradio Demo") as demo:
        gr.Markdown(
            """
            # PR-IQA Demo
            Upload a `query` image and a `reference` image to run:

            1. FeatureMetric partial map generation
            2. PR-IQA map inference
            """
        )

        with gr.Row():
            query_input = gr.Image(label="Query image", type="numpy")
            reference_input = gr.Image(label="Reference image", type="numpy")

        with gr.Accordion("Advanced", open=False):
            checkpoint_input = gr.Textbox(
                label="Checkpoint path",
                value=default_checkpoint,
                placeholder="checkpoints/priqa_base.pt",
            )
            with gr.Row():
                device_input = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda", "cpu"],
                    value=default_device,
                )
                loftup_input = gr.Checkbox(
                    label="Use LoftUp upsampler",
                    value=default_use_loftup,
                )

        with gr.Row():
            run_button = gr.Button("Generate partial map and infer", variant="primary")
            clear_button = gr.ClearButton(
                [query_input, reference_input],
                value="Clear images",
            )

        result_output = gr.Markdown()

        with gr.Row():
            partial_map_output = gr.Image(label="Partial map", type="numpy")
            overlap_mask_output = gr.Image(label="Overlap mask", type="numpy")
            quality_map_output = gr.Image(label="PR-IQA map", type="numpy")

        mean_score_output = gr.Number(label="Mean quality score")

        run_button.click(
            fn=run_demo,
            inputs=[query_input, reference_input, checkpoint_input, device_input, loftup_input],
            outputs=[result_output, partial_map_output, overlap_mask_output, quality_map_output, mean_score_output],
        )

        if example_ref.exists() and example_query.exists():
            gr.Examples(
                examples=[[str(example_query), str(example_ref), default_checkpoint, default_device, default_use_loftup]],
                inputs=[query_input, reference_input, checkpoint_input, device_input, loftup_input],
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the PR-IQA Gradio demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default=os.environ.get("PRIQA_CHECKPOINT", ""),
        default="priqa_base.pt",
        help="Path to the PR-IQA checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--use-loftup",
        action="store_true",
        default=True,
        help="Use LoftUp upsampling inside FeatureMetric",
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo(args.checkpoint, args.device, args.use_loftup)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
