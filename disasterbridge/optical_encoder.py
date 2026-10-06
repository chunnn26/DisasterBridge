
# -*- coding: utf-8 -*-
"""ResNet18-FPN encoder for multiscale optical guidance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import resnet18
    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    resnet18 = None
    _HAS_TORCHVISION = False


@dataclass
class OpticalEncoderConfig:
    out_channels: int = 128
    pretrained: bool = False
    freeze_bn: bool = True


class OpticalEncoderResNetFPN(nn.Module):
    def __init__(self, cfg: OpticalEncoderConfig):
        super().__init__()
        self.cfg = cfg

        if not _HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required for OpticalEncoderResNetFPN, "
                "but it is not available"
            )

        backbone = None
        if cfg.pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
                backbone = resnet18(weights=weights)
            except Exception:
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # high spatial resolution
        self.layer2 = backbone.layer2  # medium spatial resolution
        self.layer3 = backbone.layer3  # low spatial resolution

        if cfg.freeze_bn:
            self._freeze_bn()

        oc = int(cfg.out_channels)

        # lateral 1x1 convs
        self.lat1 = nn.Conv2d(64, oc, kernel_size=1)
        self.lat2 = nn.Conv2d(128, oc, kernel_size=1)
        self.lat3 = nn.Conv2d(256, oc, kernel_size=1)

        # smooth 3x3 convs
        self.smooth1 = nn.Conv2d(oc, oc, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(oc, oc, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(oc, oc, kernel_size=3, padding=1)

        self.out_norm1 = nn.GroupNorm(num_groups=min(32, oc), num_channels=oc)
        self.out_norm2 = nn.GroupNorm(num_groups=min(32, oc), num_channels=oc)
        self.out_norm3 = nn.GroupNorm(num_groups=min(32, oc), num_channels=oc)

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = True

    @staticmethod
    def _imagenet_norm(x_01: torch.Tensor) -> torch.Tensor:
        # x_01 in [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x_01.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x_01.device)[None, :, None, None]
        return (x_01 - mean) / std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,H,W] in [-1,1]
        return:
            f1, f2, f3 in the low-, medium-, and high-resolution order used
            by the paper. With a 256x256 input, their sizes are 16, 32, 64.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"OpticalEncoderResNetFPN expects [B,3,H,W], got {tuple(x.shape)}")

        # Convert [-1,1] -> [0,1], then ImageNet normalize
        x_01 = (x + 1.0) * 0.5
        x_in = self._imagenet_norm(x_01)

        h = self.conv1(x_in)   # /2
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)    # /4

        c1 = self.layer1(h)    # /4
        c2 = self.layer2(c1)   # /8
        c3 = self.layer3(c2)   # /16

        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")

        p3 = self.out_norm3(F.silu(self.smooth3(p3)))
        p2 = self.out_norm2(F.silu(self.smooth2(p2)))
        p1 = self.out_norm1(F.silu(self.smooth1(p1)))

        return {"f1": p3, "f2": p2, "f3": p1}
