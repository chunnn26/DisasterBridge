
# -*- coding: utf-8 -*-
"""Conditioned U-Net for DisasterBridge residual prediction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    emb = math.log(10000) / max(1, (half - 1))
    emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def norm_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class AdaGroupNorm(nn.Module):
    """
    GroupNorm + spatially-adaptive (gamma,beta) predicted from a condition feature map.
    This is a SPADE-like mechanism for discrete masks.
    """
    def __init__(self, num_channels: int, cond_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(norm_groups(num_channels), num_channels, eps=1e-6, affine=False)
        self.to_gamma_beta = nn.Conv2d(cond_channels, 2 * num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.gn(x)
        if cond is None:
            return x
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="nearest")
        gb = self.to_gamma_beta(cond)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return x * (1.0 + gamma) + beta


class DisasterStyleAdaIN(nn.Module):
    """Residual AdaIN-style modulation from a global disaster embedding."""
    def __init__(self, num_channels: int, style_dim: int, eps: float = 1e-6):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_channels, affine=False, eps=eps)
        self.to_gamma_beta = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, 2 * num_channels),
        )
        # identity at init
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x: torch.Tensor, style: Optional[torch.Tensor]) -> torch.Tensor:
        if style is None:
            return x
        x_dtype = x.dtype
        gb = self.to_gamma_beta(style.float())
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = gamma.to(dtype=x_dtype)[:, :, None, None]
        beta = beta.to(dtype=x_dtype)[:, :, None, None]
        x_norm = self.inorm(x.float()).to(dtype=x_dtype)
        return x + gamma * x_norm + beta


class ResBlockSPADE(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        cond_ch: int,
        dropout: float = 0.0,
        style_dim: int = 0,
        use_style: bool = False,
    ):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch, cond_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = AdaGroupNorm(out_ch, cond_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.use_style = bool(use_style and style_dim > 0)
        if self.use_style:
            self.style1 = DisasterStyleAdaIN(out_ch, style_dim)
            self.style2 = DisasterStyleAdaIN(out_ch, style_dim)
        else:
            self.style1 = nn.Identity()
            self.style2 = nn.Identity()

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: Optional[torch.Tensor],
        style_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if self.use_style:
            h = self.style1(h, style_emb)
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        if self.use_style:
            h = self.style2(h, style_emb)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CrossAttention2d(nn.Module):
    """Return the projected GCA response ``P(softmax(QK^T/sqrt(d))V)``."""

    def __init__(self, channels: int, context_channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head

        self.to_q = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(context_channels, inner, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(context_channels, inner, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None:
            raise ValueError("GCA requires the corresponding optical feature")
        b, _, h, w = x.shape

        if context.shape[-2:] != (h, w):
            raise ValueError(
                "GCA query and optical feature must have the same spatial size, "
                f"got query={(h, w)} and optical={tuple(context.shape[-2:])}"
            )

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, self.heads, self.dim_head, h * w)
        k = k.view(b, self.heads, self.dim_head, h * w)
        v = v.view(b, self.heads, self.dim_head, h * w)

        q = q * (self.dim_head ** -0.5)
        attn = torch.einsum("bhde,bhdf->bhef", q, k)
        attn_dtype = attn.dtype
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=attn_dtype)

        out = torch.einsum("bhef,bhdf->bhde", attn, v)
        out = out.reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


class MaskPyramidEncoder(nn.Module):
    """
    Encode mask_onehot (B,C,H,W) into a pyramid of condition features for SPADE.

    - level 0: H
    - level i: H / 2^i

    The number of levels follows the U-Net hierarchy so that spatial damage
    features remain aligned at every resolution.
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 32, num_levels: int = 4):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base_ch = int(base_ch)
        self.num_levels = int(max(1, num_levels))

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.base_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.base_ch, self.base_ch, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Each down block halves resolution.
        downs = []
        for _ in range(self.num_levels - 1):
            downs.append(
                nn.Sequential(
                    nn.Conv2d(self.base_ch, self.base_ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
            )
        self.downs = nn.ModuleList(downs)

    def forward(self, mask_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.conv0(mask_onehot)
        out: Dict[str, torch.Tensor] = {"s0": h}
        for i, down in enumerate(self.downs, start=1):
            h = down(h)
            out[f"s{i}"] = h
        return out


@dataclass
class DenoisingUNetConfig:
    in_channels: int = 6
    out_channels: int = 3
    base_channels: int = 64
    channel_mult: Sequence[int] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0

    # mask conditioning (SPADE-like)
    use_spade: bool = True
    mask_in_channels: int = 4
    mask_base_channels: int = 32

    # optical condition for the three GCA layers
    cond_channels: int = 128

    # ---- disaster-type conditioning ----
    # n_disaster_types == 0 -> disable global disaster style entirely.
    n_disaster_types: int = 0
    # embedding dim before projection to time/style space (0 -> use base_channels)
    disaster_emb_dim: int = 0
    # global style modulation on feature maps (AdaIN-style residual branch)
    use_disaster_style: bool = False
    disaster_style_dim: int = 0
    disaster_style_on_down: bool = False
    disaster_style_on_mid: bool = True
    disaster_style_on_up: bool = True

    use_gca: bool = True
    gca_heads: int = 4
    gca_dim_head: int = 32
    # High- and medium-resolution GCA are placed at down levels 2 and 3;
    # the low-resolution GCA is placed at the bottleneck.
    gca_down_levels: Sequence[int] = (2, 3)
    gca_at_bottleneck: bool = True
    gca_rho0: float = 1.0
    gca_rho1: float = 1.0
    gca_rho2: float = 0.6
    gca_rho3: float = 0.15


class DenoisingUNet(nn.Module):
    def __init__(self, cfg: DenoisingUNetConfig):
        super().__init__()
        self.cfg = cfg
        base = int(cfg.base_channels)
        time_dim = base * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # ---- disaster-type conditioning ----
        self.n_disaster_types = int(getattr(cfg, "n_disaster_types", 0) or 0)
        emb_dim = int(getattr(cfg, "disaster_emb_dim", 0) or 0)
        if emb_dim <= 0:
            emb_dim = base
        self.use_disaster_style = bool(getattr(cfg, "use_disaster_style", False))
        style_dim = int(getattr(cfg, "disaster_style_dim", 0) or 0)
        if style_dim <= 0:
            style_dim = emb_dim
        self.disaster_style_dim = style_dim if self.use_disaster_style else 0
        if self.n_disaster_types > 0:
            self.disaster_embed = nn.Embedding(self.n_disaster_types, emb_dim)
            self.disaster_mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, time_dim))
            # zero-init so a missing/zeroed style embedding keeps the baseline path unchanged
            nn.init.zeros_(self.disaster_mlp[-1].weight)
            nn.init.zeros_(self.disaster_mlp[-1].bias)
        else:
            self.disaster_embed = None
            self.disaster_mlp = None

        self.in_conv = nn.Conv2d(cfg.in_channels, base, kernel_size=3, padding=1)

        self.mask_encoder = MaskPyramidEncoder(
            in_ch=int(cfg.mask_in_channels),
            base_ch=int(cfg.mask_base_channels),
            num_levels=len(cfg.channel_mult) + 1,
        ) if cfg.use_spade else None

        rhos = (cfg.gca_rho0, cfg.gca_rho1, cfg.gca_rho2, cfg.gca_rho3)
        if any(not 0.0 <= float(rho) <= 1.0 for rho in rhos):
            raise ValueError(f"All GCA coefficients must be in [0, 1], got {rhos}")

        # Down path
        self.downs = nn.ModuleList()
        ch = base
        self.skip_channels: List[int] = []
        for i, mult in enumerate(cfg.channel_mult):
            out_ch = base * int(mult)
            resblocks = nn.ModuleList()
            use_style_here = bool(cfg.use_disaster_style and cfg.disaster_style_on_down)
            for j in range(int(cfg.num_res_blocks)):
                resblocks.append(ResBlockSPADE(
                    ch if j == 0 else out_ch,
                    out_ch,
                    time_dim,
                    cond_ch=int(cfg.mask_base_channels),
                    dropout=float(cfg.dropout),
                    style_dim=int(self.disaster_style_dim),
                    use_style=use_style_here,
                ))

            gca = CrossAttention2d(
                out_ch,
                context_channels=int(cfg.cond_channels),
                heads=int(cfg.gca_heads),
                dim_head=int(cfg.gca_dim_head),
            ) if (cfg.use_gca and i in set(cfg.gca_down_levels)) else nn.Identity()
            downsample = Downsample(out_ch)

            self.downs.append(nn.ModuleDict({
                "resblocks": resblocks,
                "gca": gca,
                "downsample": downsample,
            }))
            self.skip_channels.append(out_ch)
            ch = out_ch

        # Mid
        self.mid1 = ResBlockSPADE(
            ch,
            ch,
            time_dim,
            cond_ch=int(cfg.mask_base_channels),
            dropout=float(cfg.dropout),
            style_dim=int(self.disaster_style_dim),
            use_style=bool(cfg.use_disaster_style and cfg.disaster_style_on_mid),
        )
        self.mid_gca = CrossAttention2d(
            ch,
            context_channels=int(cfg.cond_channels),
            heads=int(cfg.gca_heads),
            dim_head=int(cfg.gca_dim_head),
        ) if (cfg.use_gca and cfg.gca_at_bottleneck) else nn.Identity()
        self.mid2 = ResBlockSPADE(
            ch,
            ch,
            time_dim,
            cond_ch=int(cfg.mask_base_channels),
            dropout=float(cfg.dropout),
            style_dim=int(self.disaster_style_dim),
            use_style=bool(cfg.use_disaster_style and cfg.disaster_style_on_mid),
        )

        # Up path
        self.ups = nn.ModuleList()
        prev_ch = ch
        for out_ch in reversed(self.skip_channels):
            upsample = Upsample(prev_ch, out_ch)

            resblocks = nn.ModuleList()
            use_style_here = bool(cfg.use_disaster_style and cfg.disaster_style_on_up)
            resblocks.append(ResBlockSPADE(
                out_ch + out_ch,
                out_ch,
                time_dim,
                cond_ch=int(cfg.mask_base_channels),
                dropout=float(cfg.dropout),
                style_dim=int(self.disaster_style_dim),
                use_style=use_style_here,
            ))
            for _ in range(int(cfg.num_res_blocks) - 1):
                resblocks.append(ResBlockSPADE(
                    out_ch,
                    out_ch,
                    time_dim,
                    cond_ch=int(cfg.mask_base_channels),
                    dropout=float(cfg.dropout),
                    style_dim=int(self.disaster_style_dim),
                    use_style=use_style_here,
                ))

            self.ups.append(nn.ModuleDict({
                "upsample": upsample,
                "resblocks": resblocks,
            }))
            prev_ch = out_ch

        self.out_norm = nn.GroupNorm(norm_groups(prev_ch), prev_ch)
        self.out_conv = nn.Conv2d(prev_ch, cfg.out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _select_mask_cond(
        mask_pyr: Optional[Dict[str, torch.Tensor]],
        h: int,
    ) -> Optional[torch.Tensor]:
        if mask_pyr is None:
            return None
        # exact match first
        for k, v in mask_pyr.items():
            if v.shape[-1] == h:
                return v
        # otherwise, resize the highest-res condition to the required size
        v0 = mask_pyr.get("s0", next(iter(mask_pyr.values())))
        return F.interpolate(v0, size=(h, h), mode="bilinear", align_corners=False)

    @staticmethod
    def _select_opt_cond(opt_feats: Optional[Dict[str, torch.Tensor]], h: int) -> torch.Tensor:
        if opt_feats is None:
            raise ValueError("GCA requires three-scale optical features")
        for k in ("f1", "f2", "f3"):
            if opt_feats.get(k) is not None and opt_feats[k].shape[-1] == h:
                return opt_feats[k]
        available = {key: tuple(value.shape[-2:]) for key, value in opt_feats.items()}
        raise ValueError(f"No optical feature matches GCA size {(h, h)}; got {available}")

    def _make_gca_gate(self, mask_onehot: torch.Tensor, h: int) -> torch.Tensor:
        """Implement ``G_l=S_l(sum_k rho_k M_k)`` from the paper."""
        if mask_onehot is None or mask_onehot.shape[1] != 4:
            shape = None if mask_onehot is None else tuple(mask_onehot.shape)
            raise ValueError(f"GCA requires a four-channel one-hot damage map, got {shape}")
        gate = (
            float(self.cfg.gca_rho0) * mask_onehot[:, 0:1]
            + float(self.cfg.gca_rho1) * mask_onehot[:, 1:2]
            + float(self.cfg.gca_rho2) * mask_onehot[:, 2:3]
            + float(self.cfg.gca_rho3) * mask_onehot[:, 3:4]
        )
        if gate.shape[-2:] != (h, h):
            gate = F.interpolate(gate, size=(h, h), mode="nearest")
        return gate

    def _apply_gca(
        self,
        hidden: torch.Tensor,
        gca: nn.Module,
        opt_feats: Optional[Dict[str, torch.Tensor]],
        mask_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Implement ``H_out=H+G_l*P_l(C_l)`` from the paper."""
        optical = self._select_opt_cond(opt_feats, hidden.shape[-1])
        response = gca(hidden, optical)
        gate = self._make_gca_gate(mask_onehot, hidden.shape[-1])
        return hidden + gate * response

    def _encode_disaster_style(
        self,
        disaster_type: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.disaster_embed is None or disaster_type is None:
            return None
        dt = disaster_type.to(device=device, dtype=torch.long)
        if dt.dim() == 0:
            dt = dt.view(1).expand(batch_size)
        elif dt.dim() == 2 and dt.shape[1] == 1:
            dt = dt.view(-1)
        if dt.shape[0] != batch_size:
            raise ValueError(
                f"disaster_type batch mismatch: got {tuple(dt.shape)}, "
                f"batch={batch_size}"
            )

        valid = dt >= 0
        if not valid.any():
            return None

        dt_safe = dt.clamp(min=0, max=self.n_disaster_types - 1)
        style = self.disaster_embed(dt_safe)
        if (~valid).any():
            style = style.clone()
            style[~valid] = 0.0
        return style

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        opt_feats: Optional[Dict[str, torch.Tensor]] = None,
        mask_onehot: Optional[torch.Tensor] = None,
        disaster_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # time embedding
        t_emb = sinusoidal_time_embedding(t, self.cfg.base_channels)
        t_emb = self.time_mlp(t_emb)

        # global disaster style / type embedding
        style_emb = self._encode_disaster_style(
            disaster_type,
            batch_size=x.shape[0],
            device=x.device,
        )
        if style_emb is not None and self.disaster_mlp is not None:
            t_emb = t_emb + self.disaster_mlp(style_emb)

        mask_pyr = None
        if self.cfg.use_spade:
            if mask_onehot is None:
                raise ValueError("The denoising U-Net requires a damage condition")
            mask_pyr = self.mask_encoder(mask_onehot)

        h = self.in_conv(x)

        skips = []
        # down
        for block in self.downs:
            cond_mask = self._select_mask_cond(mask_pyr, h.shape[-1])
            for rb in block["resblocks"]:
                h = rb(h, t_emb, cond_mask, style_emb=style_emb)

            if not isinstance(block["gca"], nn.Identity):
                h = self._apply_gca(h, block["gca"], opt_feats, mask_onehot)

            skips.append(h)
            h = block["downsample"](h)

        # mid
        cond_mask = self._select_mask_cond(mask_pyr, h.shape[-1])
        h = self.mid1(h, t_emb, cond_mask, style_emb=style_emb)
        if not isinstance(self.mid_gca, nn.Identity):
            h = self._apply_gca(h, self.mid_gca, opt_feats, mask_onehot)
        h = self.mid2(h, t_emb, cond_mask, style_emb=style_emb)

        # up
        for block, skip in zip(self.ups, reversed(skips)):
            h = block["upsample"](h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")

            h = torch.cat([h, skip], dim=1)

            cond_mask = self._select_mask_cond(mask_pyr, h.shape[-1])
            for rb in block["resblocks"]:
                h = rb(h, t_emb, cond_mask, style_emb=style_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
