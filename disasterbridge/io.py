"""Input, output, condition, and checkpoint utilities for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SCHEDULE_BUFFERS = {
    "inner_model.m_t",
    "inner_model.m_tminus",
    "inner_model.variance_t",
    "inner_model.variance_tminus",
    "inner_model.variance_t_tminus",
    "inner_model.posterior_variance_t",
    "inner_model.steps",
}


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(path))
    return np.asarray(Image.open(path))


def _to_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] <= 4 and array.shape[-1] > 4:
        return np.transpose(array, (1, 2, 0))
    return array


def load_optical(path: Path, image_size: int) -> torch.Tensor:
    array = _to_hwc(_read_image(path))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported optical image shape at {path}: {array.shape}")

    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Optical image contains non-finite values: {path}")
    maximum = float(array.max()) if array.size else 0.0
    if maximum <= 1.5:
        normalized = np.clip(array, 0.0, 1.0)
    else:
        normalized = np.clip(array / max(maximum, 1.0), 0.0, 1.0)

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return tensor.mul(2.0).sub(1.0)


def load_damage_map(path: Path, image_size: int) -> torch.Tensor:
    array = _to_hwc(_read_image(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported damage-map shape at {path}: {array.shape}")
    array = np.asarray(array, dtype=np.int64)
    array[array == 255] = 0
    labels = set(np.unique(array).tolist())
    if not labels.issubset({0, 1, 2, 3}):
        raise ValueError(
            f"Damage map {path} contains labels {sorted(labels)}; expected only 0, 1, 2, 3"
        )

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return tensor[:, 0].long()


def make_damage_condition(labels: torch.Tensor) -> torch.Tensor:
    """Return the four-channel class-indicator map defined by DDM."""
    return F.one_hot(labels, num_classes=4).permute(0, 3, 1, 2).float()


def save_sar(image: torch.Tensor, path: Path) -> None:
    array = image.detach().float().cpu().squeeze().clamp(-1.0, 1.0)
    array = ((array + 1.0) * 127.5).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        tifffile.imwrite(path, array)
    else:
        Image.fromarray(array, mode="L").save(path)


def image_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_pre_disaster", "_building_damage"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_images(directory: Path) -> Mapping[str, Path]:
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = image_id(path)
            if key in images:
                raise ValueError(f"Duplicate sample identifier '{key}' in {directory}")
            images[key] = path
    return images


def _load_checkpoint_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = _load_checkpoint_file(path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} does not contain a model state dictionary")

    state = dict(checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    for key in SCHEDULE_BUFFERS:
        state.pop(key, None)

    model_state = model.state_dict()
    shape_mismatches = {
        key: (tuple(model_state[key].shape), tuple(value.shape))
        for key, value in state.items()
        if key in model_state and model_state[key].shape != value.shape
    }
    if shape_mismatches:
        preview = list(shape_mismatches.items())[:5]
        raise RuntimeError(f"Checkpoint tensor shapes do not match the model: {preview}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    unresolved_missing = sorted(set(missing) - SCHEDULE_BUFFERS)
    if unresolved_missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match DisasterBridge: "
            f"missing={unresolved_missing[:10]}, unexpected={list(unexpected)[:10]}"
        )
