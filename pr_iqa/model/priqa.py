"""
PR-IQA: Partial-Reference Image Quality Assessment model.

3-input U-Net encoder-decoder with cross-attention:
  - tgt_img: partial quality map (from FeatureMetric) replicated to 3ch
  - dif_img: generated / distorted image
  - ref_img: reference image

Each input comes with a 4-scale mask pyramid (whole, half, quarter, tiny).

Architecture:
  Encoder: 4 levels (dim → 2*dim → 4*dim → 8*dim)
    - img_encoder: shared for ref_img and dif_img (self-attention)
    - map_encoder: for tgt_img (cross-attention with ref features)
    - qfuse: fuses dif and tgt encoder outputs at each level
  Decoder: 3 levels with skip connections from the dif encoder
  Output: tanh-activated quality map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    GatedPartialEmb,
    GatedEmb,
    TransformerLikeBlock,
    Downsample,
    Upsample,
    Pos2d,
)


class PRIQA(nn.Module):
    """Partial-Reference Image Quality Assessment model.

    Args:
        inp_channels: Input channels per image (typically 4 = RGB + mask).
        out_channels: Output channels (1 for quality map, 3 for RGB).
        dim: Base feature dimension (doubles at each encoder level).
        num_blocks: Number of TransformerLikeBlocks at each level.
        heads: Number of attention heads at each level.
        ffn_expansion_factor: FFN hidden dim multiplier.
        bias: Use bias in convolutions.
        LayerNorm_type: ``"WithBias"`` or ``"BiasFree"``.
        use_partial_conv: Use PartialConv2d in patch embedding.
    """

    def __init__(
        self,
        inp_channels=4,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        use_partial_conv=True,
    ):
        super().__init__()
        self.use_partial_conv = use_partial_conv

        # -- Patch embedding --
        if use_partial_conv:
            self.patch_embed = GatedPartialEmb(inp_channels, dim, bias)
        else:
            self.patch_embed = GatedEmb(inp_channels, dim, bias)

        # -- Quality fusion (dif + tgt) at each level --
        self.qfuse_l1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.qfuse_l2 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.qfuse_l3 = nn.Conv2d(int(dim * 2 ** 2) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.qfuse_l4 = nn.Conv2d(int(dim * 2 ** 3) * 2, int(dim * 2 ** 3), kernel_size=1, bias=bias)

        # -- Downsampler --
        self.down1_2 = Downsample(dim)
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # -- Positional Encoding --
        self.pos_l1 = Pos2d(dim)
        self.pos_l2 = Pos2d(int(dim * 2 ** 1))
        self.pos_l3 = Pos2d(int(dim * 2 ** 2))
        self.pos_l4 = Pos2d(int(dim * 2 ** 3))
        self.pos_d3 = Pos2d(int(dim * 2 ** 2))
        self.pos_d2 = Pos2d(int(dim * 2 ** 1))
        self.pos_d1 = Pos2d(int(dim * 2 ** 1))

        # -- Encoder (img: shared for ref & dif) --
        def _make_encoder(level_dim, n_blocks, n_heads):
            return nn.ModuleList([
                TransformerLikeBlock(
                    dim=level_dim, num_heads=n_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias, LayerNorm_type=LayerNorm_type,
                )
                for _ in range(n_blocks)
            ])

        self.img_encoder_level1 = _make_encoder(dim, num_blocks[0], heads[0])
        self.img_encoder_level2 = _make_encoder(int(dim * 2 ** 1), num_blocks[1], heads[1])
        self.img_encoder_level3 = _make_encoder(int(dim * 2 ** 2), num_blocks[2], heads[2])
        self.img_latent = _make_encoder(int(dim * 2 ** 3), num_blocks[3], heads[3])

        # -- Encoder (map: for tgt, cross-attention with ref) --
        self.map_encoder_level1 = _make_encoder(dim, num_blocks[0], heads[0])
        self.map_encoder_level2 = _make_encoder(int(dim * 2 ** 1), num_blocks[1], heads[1])
        self.map_encoder_level3 = _make_encoder(int(dim * 2 ** 2), num_blocks[2], heads[2])
        self.map_latent = _make_encoder(int(dim * 2 ** 3), num_blocks[3], heads[3])

        # -- Decoder --
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = _make_encoder(int(dim * 2 ** 2), num_blocks[2], heads[2])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = _make_encoder(int(dim * 2 ** 1), num_blocks[1], heads[1])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = _make_encoder(int(dim * 2 ** 1), num_blocks[0], heads[0])

        # -- Output --
        self.output = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(
        self,
        tgt_img, dif_img, ref_img,
        tgt_mask_whole, tgt_mask_half, tgt_mask_quarter, tgt_mask_tiny,
        dif_mask_whole, dif_mask_half, dif_mask_quarter, dif_mask_tiny,
        ref_mask_whole, ref_mask_half, ref_mask_quarter, ref_mask_tiny,
    ):
        """
        Args:
            tgt_img: (B, 3, H, W) — partial quality map replicated to 3ch.
            dif_img: (B, 3, H, W) — generated / distorted image.
            ref_img: (B, 3, H, W) — reference image.
            *_mask_*: (B, 1, H/s, W/s) — mask pyramids at 4 scales.

        Returns:
            (B, out_channels, H, W) quality map (tanh activated).
        """
        # -- Patch embedding --
        if self.use_partial_conv:
            tgt_enc_level1, _ = self.patch_embed(
                torch.cat((tgt_img, tgt_mask_whole), dim=1), tgt_mask_whole,
            )
            dif_enc_level1, _ = self.patch_embed(
                torch.cat((dif_img, dif_mask_whole), dim=1), dif_mask_whole,
            )
            ref_enc_level1, _ = self.patch_embed(
                torch.cat((ref_img, ref_mask_whole), dim=1), ref_mask_whole,
            )
        else:
            tgt_enc_level1 = self.patch_embed(torch.cat((tgt_img, tgt_mask_whole), dim=1))
            dif_enc_level1 = self.patch_embed(torch.cat((dif_img, dif_mask_whole), dim=1))
            ref_enc_level1 = self.patch_embed(torch.cat((ref_img, ref_mask_whole), dim=1))

        tgt_enc_level1 = self.pos_l1(tgt_enc_level1)
        dif_enc_level1 = self.pos_l1(dif_enc_level1)
        ref_enc_level1 = self.pos_l1(ref_enc_level1)

        # ── ENCODER Level 1 ──
        out_ref_enc_level1 = ref_enc_level1
        for block in self.img_encoder_level1:
            out_ref_enc_level1 = block(out_ref_enc_level1)
        kv_level1 = out_ref_enc_level1

        out_tgt_enc_level1 = tgt_enc_level1
        for block in self.map_encoder_level1:
            out_tgt_enc_level1 = block(out_tgt_enc_level1, kv_level1)

        out_dif_enc_level1 = dif_enc_level1
        for block in self.img_encoder_level1:
            out_dif_enc_level1 = block(out_dif_enc_level1, kv_level1)

        out_dif_enc_level1 = self.qfuse_l1(torch.cat([out_dif_enc_level1, out_tgt_enc_level1], dim=1))

        # ── ENCODER Level 2 ──
        inp_tgt_enc_level2 = self.pos_l2(self.down1_2(out_tgt_enc_level1, tgt_mask_whole))
        inp_dif_enc_level2 = self.pos_l2(self.down1_2(out_dif_enc_level1, dif_mask_whole))
        inp_ref_enc_level2 = self.pos_l2(self.down1_2(out_ref_enc_level1, ref_mask_whole))

        out_ref_enc_level2 = inp_ref_enc_level2
        for block in self.img_encoder_level2:
            out_ref_enc_level2 = block(out_ref_enc_level2)
        kv_level2 = out_ref_enc_level2

        out_tgt_enc_level2 = inp_tgt_enc_level2
        for block in self.map_encoder_level2:
            out_tgt_enc_level2 = block(out_tgt_enc_level2, kv_level2)

        out_dif_enc_level2 = inp_dif_enc_level2
        for block in self.img_encoder_level2:
            out_dif_enc_level2 = block(out_dif_enc_level2, kv_level2)

        out_dif_enc_level2 = self.qfuse_l2(torch.cat([out_dif_enc_level2, out_tgt_enc_level2], dim=1))

        # ── ENCODER Level 3 ──
        inp_tgt_enc_level3 = self.pos_l3(self.down2_3(out_tgt_enc_level2, tgt_mask_half))
        inp_dif_enc_level3 = self.pos_l3(self.down2_3(out_dif_enc_level2, dif_mask_half))
        inp_ref_enc_level3 = self.pos_l3(self.down2_3(out_ref_enc_level2, ref_mask_half))

        out_ref_enc_level3 = inp_ref_enc_level3
        for block in self.img_encoder_level3:
            out_ref_enc_level3 = block(out_ref_enc_level3)
        kv_level3 = out_ref_enc_level3

        out_tgt_enc_level3 = inp_tgt_enc_level3
        for block in self.map_encoder_level3:
            out_tgt_enc_level3 = block(out_tgt_enc_level3, kv_level3)

        out_dif_enc_level3 = inp_dif_enc_level3
        for block in self.img_encoder_level3:
            out_dif_enc_level3 = block(out_dif_enc_level3, kv_level3)

        out_dif_enc_level3 = self.qfuse_l3(torch.cat([out_dif_enc_level3, out_tgt_enc_level3], dim=1))

        # ── ENCODER Level 4 (Latent) ──
        inp_tgt_enc_level4 = self.pos_l4(self.down3_4(out_tgt_enc_level3, tgt_mask_quarter))
        inp_dif_enc_level4 = self.pos_l4(self.down3_4(out_dif_enc_level3, dif_mask_quarter))
        inp_ref_enc_level4 = self.pos_l4(self.down3_4(out_ref_enc_level3, ref_mask_quarter))

        ref_latent_out = inp_ref_enc_level4
        for block in self.img_latent:
            ref_latent_out = block(ref_latent_out)
        kv_level4 = ref_latent_out

        tgt_latent_out = inp_tgt_enc_level4
        for block in self.map_latent:
            tgt_latent_out = block(tgt_latent_out, kv_level4)

        dif_latent_out = inp_dif_enc_level4
        for block in self.img_latent:
            dif_latent_out = block(dif_latent_out, kv_level4)

        latent_out = self.qfuse_l4(torch.cat([dif_latent_out, tgt_latent_out], dim=1))

        # ── DECODER ──
        inp_dec_level3 = self.up4_3(latent_out, dif_mask_tiny)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_dif_enc_level3], 1)
        inp_dec_level3 = self.pos_d3(self.reduce_chan_level3(inp_dec_level3))
        out_dec_level3 = inp_dec_level3
        for block in self.decoder_level3:
            out_dec_level3 = block(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, dif_mask_quarter)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_dif_enc_level2], 1)
        inp_dec_level2 = self.pos_d2(self.reduce_chan_level2(inp_dec_level2))
        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level2:
            out_dec_level2 = block(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, dif_mask_half)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_dif_enc_level1], 1)
        inp_dec_level1 = self.pos_d1(inp_dec_level1)
        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level1:
            out_dec_level1 = block(out_dec_level1)

        return torch.tanh(self.output(out_dec_level1))
