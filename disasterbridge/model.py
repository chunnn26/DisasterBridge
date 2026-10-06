"""DisasterBridge model for conditional post-disaster SAR synthesis."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import (
    ConditionalBrownianBridge,
    DiffusionConfig,
)
from .optical_encoder import OpticalEncoderConfig, OpticalEncoderResNetFPN
from .unet import DenoisingUNet, DenoisingUNetConfig
from .wavelet import iwpt2d_haar, wpt2d_haar


class DisasterBridge(nn.Module):
    """Damage-adaptive frequency-decoupled Brownian bridge generator.

    The model receives a pre-event RGB optical image, a four-class damage map
    encoded as condition channels, and a disaster-type identifier. The bridge
    is built in a two-level Haar wavelet-packet representation and generates a
    single-channel post-event SAR image.
    """

    def __init__(self, model_config: dict):
        super().__init__()
        params = model_config["bridge"]
        unet_params = model_config.get("denoiser", {})
        data_cfg = model_config.get("data", {})

        self.image_size = int(data_cfg.get("image_size", 1024))
        self.levels = int(data_cfg.get("wavelet_levels", 2))
        divisor = 2**self.levels
        if self.image_size % divisor:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"2**wavelet_levels={divisor}"
            )

        self.latent_size = self.image_size // divisor
        self.sar_channels = int(data_cfg.get("sar_channels", 1))
        self.n_bands = 4**self.levels
        latent_channels = self.sar_channels * self.n_bands
        mask_channels = int(data_cfg.get("mask_num_classes", 4))
        if mask_channels != 4:
            raise ValueError(
                "DDM requires exactly four one-hot damage channels, "
                f"got mask_num_classes={mask_channels}"
            )

        self.opt_endpoint_bandmix = nn.Conv2d(
            in_channels=3 * self.n_bands,
            out_channels=latent_channels,
            kernel_size=1,
            groups=self.n_bands,
            bias=True,
        )
        self._init_optical_band_projection()

        cond_channels = int(unet_params.get("cond_channels", 128))
        self.optical_encoder = OpticalEncoderResNetFPN(
            OpticalEncoderConfig(
                out_channels=cond_channels,
                pretrained=bool(unet_params.get("opt_pretrained", False)),
                freeze_bn=bool(unet_params.get("opt_freeze_bn", True)),
            )
        )

        unet_cfg = DenoisingUNetConfig(
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            base_channels=int(unet_params.get("model_channels", 64)),
            channel_mult=tuple(unet_params.get("channel_mult", [1, 2, 4, 8])),
            num_res_blocks=int(unet_params.get("num_res_blocks", 2)),
            dropout=float(unet_params.get("dropout", 0.0)),
            use_spade=bool(unet_params.get("use_spade", True)),
            mask_in_channels=mask_channels,
            mask_base_channels=int(unet_params.get("mask_base_channels", 32)),
            cond_channels=cond_channels,
            n_disaster_types=int(
                data_cfg.get(
                    "n_disaster_types",
                    unet_params.get("n_disaster_types", 0),
                )
                or 0
            ),
            disaster_emb_dim=int(unet_params.get("disaster_emb_dim", 0) or 0),
            use_disaster_style=bool(unet_params.get("use_disaster_style", False)),
            disaster_style_dim=int(unet_params.get("disaster_style_dim", 0) or 0),
            disaster_style_on_down=bool(
                unet_params.get("disaster_style_on_down", False)
            ),
            disaster_style_on_mid=bool(
                unet_params.get("disaster_style_on_mid", True)
            ),
            disaster_style_on_up=bool(
                unet_params.get("disaster_style_on_up", True)
            ),
            use_gca=bool(unet_params.get("use_gca", True)),
            gca_heads=int(unet_params.get("gca_heads", 4)),
            gca_dim_head=int(unet_params.get("gca_dim_head", 32)),
            gca_down_levels=tuple(unet_params.get("gca_down_levels", [2, 3])),
            gca_at_bottleneck=bool(unet_params.get("gca_at_bottleneck", True)),
            gca_rho0=float(unet_params.get("gca_rho0", 1.0)),
            gca_rho1=float(unet_params.get("gca_rho1", 1.0)),
            gca_rho2=float(unet_params.get("gca_rho2", 0.6)),
            gca_rho3=float(unet_params.get("gca_rho3", 0.15)),
        )

        diffusion_cfg = DiffusionConfig(
            num_timesteps=int(params.get("num_timesteps", 500)),
            mt_type=str(params.get("mt_type", "linear")),
            max_var=float(params.get("max_var", 1.0)),
            eta=float(params.get("eta", 1.0)),
            objective=str(params.get("objective", "grad")),
            skip_sample=bool(params.get("skip_sample", True)),
            sample_type=str(params.get("sample_type", "cosine")),
            sample_step=int(params.get("sample_step", 100)),
        )

        self.inner_model = ConditionalBrownianBridge(
            DenoisingUNet(unet_cfg),
            diffusion_cfg,
        )

    def _init_optical_band_projection(self) -> None:
        coefficients = torch.tensor([0.2989, 0.5870, 0.1140])
        with torch.no_grad():
            weights = self.opt_endpoint_bandmix.weight
            weights.zero_()
            for index in range(weights.shape[0]):
                weights[index, :, 0, 0] = coefficients.to(
                    device=weights.device,
                    dtype=weights.dtype,
                )
            self.opt_endpoint_bandmix.bias.zero_()

    def decode_sar(self, latent: torch.Tensor) -> torch.Tensor:
        return iwpt2d_haar(
            latent,
            levels=self.levels,
            in_channels=self.sar_channels,
        )

    def encode_optical_endpoint(self, optical: torch.Tensor) -> torch.Tensor:
        optical_wpt = wpt2d_haar(optical, levels=self.levels)
        return self.opt_endpoint_bandmix(optical_wpt)

    def encode_optical_features(self, optical: torch.Tensor) -> Dict[str, torch.Tensor]:
        resized = F.interpolate(
            optical,
            size=(self.latent_size, self.latent_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.optical_encoder(resized)

    @staticmethod
    def resize_condition(condition: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(condition, size=(size, size), mode="nearest")

    @torch.inference_mode()
    def sample(
        self,
        optical: torch.Tensor,
        damage_condition: torch.Tensor,
        disaster_type: torch.Tensor,
        return_intermediate: bool = False,
    ):
        damage_latent = self.resize_condition(
            damage_condition,
            optical.shape[-1] // (2**self.levels),
        )
        optical_endpoint = self.encode_optical_endpoint(optical)
        optical_features = self.encode_optical_features(optical)

        result = self.inner_model.sample(
            optical_endpoint,
            damage_latent,
            clip_denoised=False,
            return_intermediate=return_intermediate,
            opt_feats=optical_features,
            disaster_type=disaster_type,
        )
        if return_intermediate:
            latent, latent_steps = result
            image = self.decode_sar(latent).clamp(-1.0, 1.0)
            images = [self.decode_sar(step).clamp(-1.0, 1.0) for step in latent_steps]
            return image, images

        return self.decode_sar(result).clamp(-1.0, 1.0)
