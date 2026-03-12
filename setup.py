from setuptools import setup, find_packages

setup(
    name="pr-iqa",
    version="0.1.0",
    description="PR-IQA: Partial-Reference Image Quality Assessment",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "einops>=0.7.0",
        "xformers>=0.0.22",
        "tqdm",
        "pillow",
    ],
    extras_require={
        "train": [
            "tensorboard",
            "pyyaml",
        ],
        "partial_map": [
            "pytorch3d>=0.7.0",
            "jaxtyping",
        ],
    },
)
