"""Conditional Brownian bridge process used by DisasterBridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch = timesteps.shape[0]
    selected = values.gather(-1, timesteps)
    return selected.reshape(batch, *((1,) * (len(shape) - 1)))


@dataclass
class DiffusionConfig:
    num_timesteps: int = 500
    mt_type: str = "linear"
    max_var: float = 1.0
    eta: float = 1.0
    objective: str = "grad"
    skip_sample: bool = True
    sample_type: str = "cosine"
    sample_step: int = 100


class ConditionalBrownianBridge(nn.Module):
    """Damage-guided conditional Brownian bridge model."""

    def __init__(
        self,
        denoise_fn: nn.Module,
        diffusion_cfg: DiffusionConfig,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.diff = diffusion_cfg
        if diffusion_cfg.objective != "grad":
            raise ValueError("DisasterBridge requires objective='grad'")
        self.num_timesteps = int(diffusion_cfg.num_timesteps)
        self.register_schedule()
        self.register_sampling_steps()

    def register_schedule(self) -> None:
        total = self.num_timesteps
        if self.diff.mt_type != "linear":
            raise ValueError(
                "DisasterBridge requires a linear bridge schedule, "
                f"got {self.diff.mt_type}"
            )
        mean = np.linspace(0.0, 1.0, total, dtype=np.float64)

        mean_previous = np.append(0.0, mean[:-1])
        variance = 2.0 * (mean - mean**2) * float(self.diff.max_var)
        variance_previous = np.append(0.0, variance[:-1])
        transition_variance = variance - variance_previous * (
            (1.0 - mean) / (1.0 - mean_previous)
        ) ** 2
        posterior_variance = np.zeros_like(variance)
        np.divide(
            transition_variance * variance_previous,
            variance,
            out=posterior_variance,
            where=variance > 0.0,
        )
        # At the exact optical endpoint, the reverse conditional reduces to
        # the next bridge marginal around the estimated SAR endpoint.
        posterior_variance[-1] = variance_previous[-1]

        to_tensor = lambda value: torch.tensor(value, dtype=torch.float32)
        self.register_buffer("m_t", to_tensor(mean))
        self.register_buffer("m_tminus", to_tensor(mean_previous))
        self.register_buffer("variance_t", to_tensor(variance))
        self.register_buffer("variance_tminus", to_tensor(variance_previous))
        self.register_buffer("variance_t_tminus", to_tensor(transition_variance))
        self.register_buffer("posterior_variance_t", to_tensor(posterior_variance))

    @staticmethod
    def _strict_cosine_steps(total: int, count: int) -> torch.Tensor:
        phase = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
        target = (torch.cos(phase * torch.pi) + 1.0) * 0.5 * (total - 1)
        selected = []
        previous = total
        for index, value in enumerate(target):
            remaining = count - index - 1
            upper = total - 1 if index == 0 else previous - 1
            step = max(remaining, min(int(torch.round(value).item()), upper))
            selected.append(step)
            previous = step
        return torch.tensor(selected, dtype=torch.long)

    def register_sampling_steps(self) -> None:
        total = self.num_timesteps
        if not self.diff.skip_sample:
            steps = torch.arange(total - 1, -1, -1)
        else:
            count = int(self.diff.sample_step)
            if count < 2 or count > total:
                raise ValueError(
                    f"sample_step must be between 2 and {total}, got {count}"
                )
            if self.diff.sample_type == "linear":
                steps = torch.linspace(total - 1, 0, count).round().long()
            elif self.diff.sample_type == "cosine":
                steps = self._strict_cosine_steps(total, count)
            else:
                raise ValueError(f"Unknown sampling schedule: {self.diff.sample_type}")
            if len(torch.unique(steps)) != count or steps[-1].item() != 0:
                raise RuntimeError("Sampling schedule must contain distinct steps ending at zero")
        self.register_buffer("steps", steps)

    def _make_noise_like(self, value: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(value)

    def relax_endpoint(
        self,
        endpoint: torch.Tensor,
        mask_onehot: torch.Tensor,
        endpoint_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace damaged and destroyed endpoint regions with a noise packet."""
        if mask_onehot.shape[1] != 4:
            raise ValueError("Endpoint relaxation requires four damage classes")

        damaged = mask_onehot[:, 2:4].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        noise = endpoint_noise
        if noise is None:
            noise = self._make_noise_like(endpoint)
        if noise.shape != endpoint.shape:
            raise ValueError(
                f"endpoint noise shape {tuple(noise.shape)} does not match "
                f"endpoint shape {tuple(endpoint.shape)}"
            )
        return (1.0 - damaged) * endpoint + damaged * noise

    def predict_x0_from_objective(
        self,
        state: torch.Tensor,
        endpoint: torch.Tensor,
        timestep: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        del endpoint, timestep
        return state - prediction

    @torch.inference_mode()
    def p_sample(
        self,
        state: torch.Tensor,
        endpoint: torch.Tensor,
        mask_onehot: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: Optional[torch.Tensor],
        clip_denoised: bool = False,
        opt_feats: Optional[Dict[str, torch.Tensor]] = None,
        disaster_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.denoise_fn(
            torch.cat([state, endpoint], dim=1),
            timestep,
            opt_feats=opt_feats,
            mask_onehot=mask_onehot,
            disaster_type=disaster_type,
        )
        x0 = self.predict_x0_from_objective(
            state,
            endpoint,
            timestep,
            prediction,
        )
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        if next_timestep is None:
            return x0, x0

        mean = extract(self.m_t, timestep, state.shape)
        next_mean = extract(self.m_t, next_timestep, state.shape)
        variance = extract(self.variance_t, timestep, state.shape)
        next_variance = extract(self.variance_t, next_timestep, state.shape)

        epsilon = torch.finfo(variance.dtype).eps
        safe_variance = variance.clamp_min(epsilon)
        posterior_variance = (
            variance
            - next_variance
            * (1.0 - mean).square()
            / (1.0 - next_mean).square().clamp_min(epsilon)
        ) * next_variance / safe_variance
        posterior_variance = posterior_variance.clamp(min=0.0)
        posterior_mean = (
            (1.0 - next_mean) * x0
            + next_mean * endpoint
            + ((next_variance - posterior_variance) / safe_variance)
            .clamp(min=0.0)
            .sqrt()
            * (state - (1.0 - mean) * x0 - mean * endpoint)
        )

        at_optical_endpoint = mean >= 1.0 - epsilon
        endpoint_mean = (1.0 - next_mean) * x0 + next_mean * endpoint
        posterior_mean = torch.where(at_optical_endpoint, endpoint_mean, posterior_mean)
        posterior_variance = torch.where(
            at_optical_endpoint,
            next_variance,
            posterior_variance,
        )
        standard_deviation = posterior_variance.sqrt() * float(self.diff.eta)
        return posterior_mean + standard_deviation * torch.randn_like(state), x0

    @torch.inference_mode()
    def sample(
        self,
        endpoint: torch.Tensor,
        mask_onehot: torch.Tensor,
        clip_denoised: bool = False,
        return_intermediate: bool = False,
        opt_feats: Optional[Dict[str, torch.Tensor]] = None,
        disaster_type: Optional[torch.Tensor] = None,
    ):
        endpoint_noise = self._make_noise_like(endpoint)
        relaxed_endpoint = self.relax_endpoint(
            endpoint,
            mask_onehot,
            endpoint_noise=endpoint_noise,
        )
        state = relaxed_endpoint.clone()
        intermediates = []

        for index, step in enumerate(self.steps):
            timestep = step.repeat(endpoint.shape[0]).to(endpoint.device)
            next_timestep = None
            if index + 1 < len(self.steps):
                next_timestep = self.steps[index + 1].repeat(endpoint.shape[0]).to(
                    endpoint.device
                )
            state, reconstruction = self.p_sample(
                state,
                relaxed_endpoint,
                mask_onehot,
                timestep,
                next_timestep,
                clip_denoised=clip_denoised,
                opt_feats=opt_feats,
                disaster_type=disaster_type,
            )
            if return_intermediate:
                intermediates.append(reconstruction.detach().cpu())

        if return_intermediate:
            return state, intermediates
        return state
