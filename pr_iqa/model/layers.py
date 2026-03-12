"""
Building blocks for the PR-IQA architecture.

Includes:
  - PartialConv2d: Mask-aware convolution for inpainting
  - GatedPartialEmb / GatedEmb: Gated patch embeddings
  - FeedForward (FFN): Gated depth-wise separable FFN
  - ChannelGate: SE/CBAM-style channel attention
  - Attention: Spatial attention with xformers memory-efficient attention
  - TransformerLikeBlock: Channel gate → Spatial attn → FFN with residuals
  - SandwichBlock: FFN → Channel gate → Spatial attn → FFN
  - Downsample / Upsample: Strided conv / PixelShuffle
  - Pos2d: 2D sinusoidal positional encoding
  - DropPath: Stochastic depth
  - LayerNorm: Bias-free or with-bias layer normalization
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from xformers import ops


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Partial Convolution
# ---------------------------------------------------------------------------

class PartialConv2d(nn.Module):
    """Mask-aware convolution for inpainting.

    Given input ``x`` and binary mask ``mask`` (1 = valid), the output is
    normalized by the number of valid pixels in each receptive field.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(1, out_ch, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x, mask):
        with torch.no_grad():
            mask_sum = self.mask_conv(mask).clamp(min=1e-8)
            new_mask = (mask_sum > 0).float()

        output = self.conv(x * mask) / mask_sum
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        output = output * new_mask
        return output, new_mask[:, 0:1]


# ---------------------------------------------------------------------------
# Gated Embeddings
# ---------------------------------------------------------------------------

class GatedPartialEmb(nn.Module):
    """Gated patch embedding using PartialConv2d (for masked inputs)."""

    def __init__(self, in_c=4, embed_dim=48, bias=False):
        super().__init__()
        self.pconv = PartialConv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x_with_mask, mask):
        """
        Args:
            x_with_mask: (B, in_c, H, W) — e.g. RGB(3) + mask(1) concatenated.
            mask: (B, 1, H, W) — binary mask for partial conv.
        """
        x, mask_out = self.pconv(x_with_mask, mask)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return x, mask_out


class GatedEmb(nn.Module):
    """Gated patch embedding (standard, no partial conv)."""

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)
        return F.gelu(x1) * x2


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Gated depth-wise separable FFN."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Channel Attention
# ---------------------------------------------------------------------------

class ChannelGate(nn.Module):
    """SE/CBAM-style channel gate."""

    def __init__(self, dim, reduction=16, use_max=True, bias=True):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=bias),
        )
        self.use_max = use_max

    def _pooled(self, t):
        avg = F.adaptive_avg_pool2d(t, 1)
        if self.use_max:
            mx = F.adaptive_max_pool2d(t, 1)
            pooled = avg + mx
        else:
            pooled = avg
        return self.mlp(pooled)

    def forward(self, x, kv=None):
        gate_logits = self._pooled(x) if kv is None else (self._pooled(x) + self._pooled(kv))
        gate = torch.sigmoid(gate_logits)
        x_gated = x * gate
        kv_gated = kv * gate if kv is not None else None
        return x_gated, kv_gated


# ---------------------------------------------------------------------------
# Spatial Attention (xformers)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Spatial attention with xformers memory-efficient attention.

    Supports both self-attention (kv=None) and cross-attention (kv provided).
    Includes a spatial gating branch.
    """

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads

        # Self-attention projections
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )

        # Cross-attention projections
        self.q_proj_qonly = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw_qonly = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_proj_cross = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv_cross = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias,
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Spatial gating
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample_to = lambda t, size: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, "WithBias"),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, "WithBias"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, kv=None):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        if kv is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
        else:
            q = self.q_dw_qonly(self.q_proj_qonly(x))
            kv_feat = self.kv_dwconv_cross(self.kv_proj_cross(kv))
            k, v = kv_feat.chunk(2, dim=1)

        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 3, 1, 2).contiguous()
        k = k.view(b, self.num_heads, head_dim, -1).permute(0, 3, 1, 2).contiguous()
        v = v.view(b, self.num_heads, head_dim, -1).permute(0, 3, 1, 2).contiguous()

        out = ops.memory_efficient_attention(q, k, v)
        out = out.permute(0, 2, 3, 1).reshape(b, c, h, w)

        # Spatial gating
        spatial_weight = self.avg_pool(x)
        spatial_weight = self.conv(spatial_weight)
        spatial_weight = self.upsample_to(spatial_weight, (h, w))
        out = out * spatial_weight

        return self.project_out(out)


# ---------------------------------------------------------------------------
# Drop Path (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


# ---------------------------------------------------------------------------
# Transformer-like Block
# ---------------------------------------------------------------------------

class TransformerLikeBlock(nn.Module):
    """Channel gate → Spatial attention → FFN with layer scale and residuals."""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 drop_path=0.0, layerscale_init=1e-2):
        super().__init__()
        self.norm_c = LayerNorm(dim, LayerNorm_type)
        self.chan = ChannelGate(dim, reduction=16, use_max=True, bias=bias)
        self.norm_s = LayerNorm(dim, LayerNorm_type)
        self.sattn = Attention(dim, num_heads, bias)
        self.norm_f = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.gamma_c = nn.Parameter(torch.ones(1, dim, 1, 1) * layerscale_init)
        self.gamma_s = nn.Parameter(torch.ones(1, dim, 1, 1) * layerscale_init)
        self.gamma_f = nn.Parameter(torch.ones(1, dim, 1, 1) * layerscale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, kv=None):
        xc = self.norm_c(x)
        xc_gated, kv_gated = self.chan(xc, kv)
        x = x + self.drop_path(self.gamma_c * xc_gated)

        xs = self.norm_s(x)
        xs = self.sattn(xs, kv_gated if kv is not None else None)
        x = x + self.drop_path(self.gamma_s * xs)

        xf = self.norm_f(x)
        xf = self.ffn(xf)
        x = x + self.drop_path(self.gamma_f * xf)
        return x


# ---------------------------------------------------------------------------
# Sandwich Block
# ---------------------------------------------------------------------------

class SandwichBlock(nn.Module):
    """FFN → Channel gate → Spatial attn → FFN."""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_c = LayerNorm(dim, LayerNorm_type)
        self.chan = ChannelGate(dim, reduction=16, use_max=True, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, kv=None):
        x = x + self.ffn1(self.norm1_1(x))
        xc = self.norm_c(x)
        xc_gated, kv_gated = self.chan(xc, kv)
        x = x + xc_gated
        x = x + self.attn(self.norm1(x), kv_gated if kv is not None else None)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x, mask=None):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x, mask=None):
        return self.body(x)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class Pos2d(nn.Module):
    """2D sinusoidal positional encoding."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(4, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        pe4 = torch.stack([xx, yy, torch.sin(xx * 3.14159), torch.cos(yy * 3.14159)], dim=0)
        pe = self.proj(pe4.unsqueeze(0)).repeat(B, 1, 1, 1)
        return x + pe
