from .priqa import PRIQA
from .layers import (
    PartialConv2d,
    GatedPartialEmb,
    GatedEmb,
    FeedForward,
    ChannelGate,
    Attention,
    TransformerLikeBlock,
    SandwichBlock,
    Downsample,
    Upsample,
    Pos2d,
    DropPath,
    LayerNorm,
)


def build_priqa(
    out_channels: int = 1,
    dim: int = 48,
    num_blocks: tuple = (2, 3, 3, 4),
    heads: tuple = (1, 2, 4, 8),
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    layernorm_type: str = "WithBias",
    use_partial_conv: bool = True,
) -> PRIQA:
    """Build a PR-IQA model with default or custom hyperparameters."""
    return PRIQA(
        inp_channels=4,
        out_channels=out_channels,
        dim=dim,
        num_blocks=list(num_blocks),
        heads=list(heads),
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=layernorm_type,
        use_partial_conv=use_partial_conv,
    )


__all__ = ["PRIQA", "build_priqa"]
