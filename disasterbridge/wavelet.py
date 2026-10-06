
# -*- coding: utf-8 -*-
"""
Haar Wavelet Packet Transform (WPT) in pure PyTorch.

Why WPT (not standard DWT)?
- For levels=2 on 1024x1024:
    WPT -> 16 subbands, each 256x256 (uniform resolution), good as diffusion latent.

This implementation is:
- differentiable
- perfectly invertible (up to floating point)
- no external deps
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def _check_even(x: torch.Tensor):
    h, w = x.shape[-2:]
    if (h % 2 != 0) or (w % 2 != 0):
        raise ValueError(f"WPT expects even H,W, got {h},{w}")


def dwt2d_haar_once(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-level 2D Haar DWT (orthonormal scaling).
    Input:  [B,C,H,W]
    Output: LL,LH,HL,HH each [B,C,H/2,W/2]
    """
    _check_even(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def idwt2d_haar_once(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of one-level Haar DWT.
    Inputs: LL,LH,HL,HH each [B,C,H,W]
    Output: x [B,C,H*2,W*2]
    """
    # Reconstruct 2x2 block
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5

    b, c, h, w = ll.shape
    out = torch.zeros((b, c, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


def wpt2d_haar(x: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """
    Wavelet Packet Transform:
        recursively apply DWT to ALL subbands each level.

    Input:
        x: [B,C,H,W]
    Output:
        z: [B, C*(4**levels), H/(2**levels), W/(2**levels)]
    """
    if levels < 1:
        return x
    bands: List[torch.Tensor] = [x]
    for _ in range(levels):
        new_bands: List[torch.Tensor] = []
        for b in bands:
            ll, lh, hl, hh = dwt2d_haar_once(b)
            new_bands.extend([ll, lh, hl, hh])
        bands = new_bands
    return torch.cat(bands, dim=1)


def iwpt2d_haar(z: torch.Tensor, levels: int = 2, in_channels: int = 1) -> torch.Tensor:
    """
    Inverse Wavelet Packet Transform.

    z: [B, C*(4**levels), H, W]
    return x: [B, C, H*(2**levels), W*(2**levels)]
    """
    if levels < 1:
        return z
    n_bands = 4 ** levels
    if z.shape[1] % n_bands != 0:
        raise ValueError(
            f"iwpt expects channels divisible by 4**levels={n_bands}, "
            f"got {z.shape[1]}"
        )
    c = z.shape[1] // n_bands
    if in_channels is not None and c != in_channels:
        raise ValueError(
            f"IWPT represents {c} input channels, expected {in_channels}"
        )

    bands: List[torch.Tensor] = list(torch.chunk(z, n_bands, dim=1))

    for _ in range(levels):
        new_bands: List[torch.Tensor] = []
        for i in range(0, len(bands), 4):
            parent = idwt2d_haar_once(bands[i], bands[i + 1], bands[i + 2], bands[i + 3])
            new_bands.append(parent)
        bands = new_bands

    if len(bands) != 1:
        raise RuntimeError("iwpt internal error: expected single reconstructed band")
    return bands[0]
