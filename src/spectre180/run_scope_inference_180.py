#!/usr/bin/env python3
"""Run six-state SCOPE inference with the paper's 600-pixel geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

from scope_model_180 import (
    NUM_OUTPUT_CHANNELS,
    NUM_POLARIZATIONS,
    build_scope_model_180,
    split_prediction,
)


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
HLP_STATE_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 180-channel six-state SCOPE inference."
    )
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mask",
        required=True,
        help="Measured 180-channel calibration tensor downloaded with the checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--top-left-size",
        type=int,
        default=600,
        help="Crop native input and calibration to top-left NxN; use 0 to retain input size.",
    )
    parser.add_argument(
        "--input-normalization",
        choices=("bitdepth", "none"),
        default="bitdepth",
        help="bitdepth divides integer input by its dtype maximum; no per-image min-max is performed.",
    )
    parser.add_argument("--hlp-training-scale", type=float, default=15.0)
    parser.add_argument("--skip-sigma", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument(
        "--save-uint16-mu",
        action="store_true",
        help="Also save clipped [0,1] mean predictions as uint16 TIFFs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported inputs under {path}")
    return files


def load_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        array = np.asarray(Image.open(path))
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{path}: expected grayscale image, got {array.shape}")
    return np.asarray(array)


def normalize_input(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        denominator = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / denominator
    result = array.astype(np.float32)
    low = float(np.nanmin(result))
    high = float(np.nanmax(result))
    if low < 0 or high > 1.0001:
        raise ValueError(
            f"Float input range [{low}, {high}] is not [0,1]; refusing implicit min-max normalization"
        )
    return result


def crop_top_left(array: np.ndarray, size: int, source: Path) -> np.ndarray:
    if size <= 0:
        return array
    if array.shape[0] < size or array.shape[1] < size:
        raise ValueError(f"{source}: cannot top-left crop {array.shape} to {size}")
    return array[:size, :size]


def prepare_mask(mask: torch.Tensor, height: int, width: int, crop_size: int) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4 or mask.shape[1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(
            f"Expected mask [1,{NUM_OUTPUT_CHANNELS},H,W], got {tuple(mask.shape)}"
        )
    if crop_size > 0:
        if mask.shape[-2] < crop_size or mask.shape[-1] < crop_size:
            raise ValueError(f"Calibration {tuple(mask.shape)} is smaller than crop {crop_size}")
        mask = mask[..., :crop_size, :crop_size]
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.float(), size=(height, width), mode="bilinear", align_corners=False)
    return mask.float()


def restore_hlp_scale(
    mu: torch.Tensor, sigma: torch.Tensor, training_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if training_scale <= 0:
        raise ValueError("--hlp-training-scale must be positive")
    hlp_indices = torch.arange(
        HLP_STATE_INDEX,
        NUM_OUTPUT_CHANNELS,
        NUM_POLARIZATIONS,
        device=mu.device,
    )
    mu[:, hlp_indices] /= training_scale
    sigma[:, hlp_indices] /= training_scale
    return mu, sigma


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    mask_path = Path(args.mask)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = build_scope_model_180().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    calibration = torch.load(mask_path, map_location="cpu", weights_only=False)
    if not isinstance(calibration, torch.Tensor):
        raise TypeError(f"{mask_path} did not contain a tensor")

    manifest_rows: list[dict] = []
    for file_path in collect_inputs(input_path):
        raw = load_grayscale(file_path)
        normalized = normalize_input(raw, args.input_normalization)
        normalized = crop_top_left(normalized, args.top_left_size, file_path)
        input_tensor = (
            torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)
        )
        current_mask = prepare_mask(
            calibration,
            input_tensor.shape[-2],
            input_tensor.shape[-1],
            args.top_left_size,
        ).to(device)

        start = time.time()
        with torch.no_grad():
            prediction = model(input_tensor, current_mask)
            mu, sigma = split_prediction(prediction)
            mu, sigma = restore_hlp_scale(mu, sigma, args.hlp_training_scale)
        elapsed = time.time() - start

        mu_np = mu.squeeze(0).cpu().numpy().astype(np.float32)
        sigma_np = sigma.squeeze(0).cpu().numpy().astype(np.float32)
        stem = file_path.stem
        mu_name = f"{stem}_mu_float32.tif"
        tifffile.imwrite(output_dir / mu_name, mu_np)

        sigma_name = None
        if not args.skip_sigma:
            sigma_name = f"{stem}_sigma_float32.tif"
            tifffile.imwrite(output_dir / sigma_name, sigma_np)
        uint16_name = None
        if args.save_uint16_mu:
            uint16_name = f"{stem}_mu_uint16.tif"
            mu_uint16 = np.rint(np.clip(mu_np, 0, 1) * 65535).astype(np.uint16)
            tifffile.imwrite(output_dir / uint16_name, mu_uint16)
        if args.save_npy:
            np.save(output_dir / f"{stem}_mu.npy", mu_np)
            if not args.skip_sigma:
                np.save(output_dir / f"{stem}_sigma.npy", sigma_np)

        manifest_rows.append(
            {
                "input": str(file_path),
                "input_shape": list(raw.shape),
                "inference_shape": list(mu_np.shape),
                "mu_float32_tif": mu_name,
                "sigma_float32_tif": sigma_name,
                "mu_uint16_tif": uint16_name,
                "seconds": elapsed,
            }
        )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "device": str(device),
        "top_left_size": args.top_left_size,
        "input_normalization": args.input_normalization,
        "hlp_training_scale_restored": args.hlp_training_scale,
        "analyzer_order": ["LCP", "RCP", "HLP", "VLP", "45LP", "135LP"],
        "files": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
