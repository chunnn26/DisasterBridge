"""Generate post-event SAR images with DisasterBridge."""

from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from disasterbridge import DisasterBridge
from disasterbridge.io import (
    image_id,
    list_images,
    load_checkpoint,
    load_damage_map,
    load_optical,
    make_damage_condition,
    save_sar,
)


DISASTER_TYPES = (
    "earthquake",
    "storm",
    "wildfire",
    "flood",
    "volcano",
    "explosion",
    "conflict",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Conditional post-disaster SAR synthesis with DisasterBridge"
    )
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disaster-type",
        required=True,
        choices=DISASTER_TYPES,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "disasterbridge.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def sample_seed(base_seed: int, identifier: str) -> int:
    token = f"{base_seed}:{identifier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31)


def input_pairs(optical: Path, mask: Path):
    if optical.is_file() and mask.is_file():
        return [(image_id(optical), optical, mask)]
    if optical.is_dir() and mask.is_dir():
        optical_images = list_images(optical)
        mask_images = list_images(mask)
        missing_masks = sorted(set(optical_images) - set(mask_images))
        missing_optical = sorted(set(mask_images) - set(optical_images))
        if missing_masks or missing_optical:
            raise ValueError(
                "Optical and mask directories do not contain the same sample identifiers: "
                f"missing_masks={missing_masks[:10]}, missing_optical={missing_optical[:10]}"
            )
        if not optical_images:
            raise ValueError("No supported images were found in the input directories")
        return [
            (identifier, optical_images[identifier], mask_images[identifier])
            for identifier in sorted(optical_images)
        ]
    raise ValueError("--optical and --mask must both be files or both be directories")


def output_path(output: Path, identifier: str, multiple: bool) -> Path:
    if multiple or output.suffix.lower() not in {".tif", ".tiff", ".png"}:
        return output / f"{identifier}_post_disaster.tif"
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    bridge_config = config["bridge"]
    if int(bridge_config.get("sample_step", 0)) != 100:
        raise ValueError("The release configuration requires exactly 100 sampling steps")
    if str(bridge_config.get("sample_type", "")).lower() != "cosine":
        raise ValueError("The release configuration requires cosine reverse sampling")

    device = resolve_device(args.device)
    model = DisasterBridge(config)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    if model.inner_model.steps.numel() != 100:
        raise RuntimeError("The reverse bridge schedule does not contain 100 steps")

    pairs = input_pairs(args.optical, args.mask)
    image_size = int(config["data"]["image_size"])
    disaster_id = DISASTER_TYPES.index(args.disaster_type)
    use_amp = device.type == "cuda" and not args.no_amp

    for index, (identifier, optical_path, mask_path) in enumerate(pairs, start=1):
        seed = sample_seed(args.seed, identifier)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        optical = load_optical(optical_path, image_size).to(device)
        labels = load_damage_map(mask_path, image_size).to(device)
        condition = make_damage_condition(labels).to(device)
        disaster_type = torch.tensor([disaster_id], device=device, dtype=torch.long)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            prediction = model.sample(
                optical=optical,
                damage_condition=condition,
                disaster_type=disaster_type,
            )

        destination = output_path(args.output, identifier, len(pairs) > 1)
        save_sar(prediction, destination)
        print(f"[{index}/{len(pairs)}] {destination}", flush=True)


if __name__ == "__main__":
    main()
