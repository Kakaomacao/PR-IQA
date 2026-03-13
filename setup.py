from setuptools import setup, find_packages

setup(
    name="pr-iqa",
    version="0.1.0",
    description="PR-IQA: Partial-Reference Image Quality Assessment",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core
        "torch>=2.0",
        "torchvision>=0.15",
        "einops>=0.7.0",
        "xformers>=0.0.22",
        "tqdm",
        "pillow",
        "numpy",
        "jaxtyping",
        # Submodule deps (vggt + loftup)
        "huggingface_hub",
        "safetensors",
        "timm",
        "matplotlib",
        # pytorch3d: must be installed separately before pip install -e .
        #   pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
    ],
    extras_require={
        "train": [
            "tensorboard",
            "pyyaml",
        ],
    },
)
