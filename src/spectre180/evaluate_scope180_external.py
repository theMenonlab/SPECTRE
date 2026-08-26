#!/usr/bin/env python3
"""Evaluate a frozen SCOPE-180 checkpoint on complete external categories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from paired_dataset import PairedScope150Dataset, collect_training_pairs
from scope_model_180 import NUM_OUTPUT_CHANNELS, build_scope_model_180
from train_scope_180_20260730 import build_hlp_scale_vector, choose_device, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--mask",
        required=True,
    )
    parser.add_argument(
        "--spatial-mask",
        required=True,
    )
    parser.add_argument("--top-left-size", type=int, default=600)
    parser.add_argument("--hlp-target-scale", type=float, default=15.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--per-state-metrics",
        action="store_true",
        help=(
            "Also report L1, PSNR, spectral correlation, and SAM separately "
            "for LCP, RCP, HLP, VLP, 45LP, and 135LP in the same pass."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Optionally evaluate only the deterministic manifest-compatible split.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    mask_path = Path(args.mask)
    spatial_mask_path = Path(args.spatial_mask)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = tuple(item.strip() for item in args.categories.split(",") if item.strip())
    samples = collect_training_pairs(
        categories,
        data_root=args.data_root,
        test_every=10,
        allow_mismatch=False,
    )
    if args.split != "all":
        samples = [sample for sample in samples if sample.split == args.split]
    if not samples:
        raise ValueError(
            f"No samples remain for categories={categories}, split={args.split}"
        )
    dataset = PairedScope150Dataset(
        samples,
        resize_to=args.top_left_size,
        crop_size=None,
        training=False,
        top_left_size=args.top_left_size,
        expected_channels=NUM_OUTPUT_CHANNELS,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = choose_device(args.device)
    model = build_scope_model_180().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)

    calibration = torch.load(mask_path, map_location="cpu", weights_only=False).float()
    if calibration.ndim == 3:
        calibration = calibration.unsqueeze(0)
    if calibration.shape[1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(f"Expected {NUM_OUTPUT_CHANNELS} calibration channels")
    if calibration.shape[-2] < args.top_left_size or calibration.shape[-1] < args.top_left_size:
        calibration = F.interpolate(
            calibration,
            size=(args.top_left_size, args.top_left_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        calibration = calibration[..., : args.top_left_size, : args.top_left_size]

    spatial = np.asarray(Image.open(spatial_mask_path).convert("L"), dtype=np.uint8)
    spatial_tensor = torch.from_numpy((spatial > 0).astype(np.float32))[None, None]
    if spatial_tensor.shape[-2] < args.top_left_size or spatial_tensor.shape[-1] < args.top_left_size:
        spatial_tensor = F.interpolate(
            spatial_tensor,
            size=(args.top_left_size, args.top_left_size),
            mode="nearest",
        )
    else:
        spatial_tensor = spatial_tensor[..., : args.top_left_size, : args.top_left_size]
    if spatial_tensor.sum().item() <= 0:
        raise ValueError(
            f"Spatial mask has no valid pixels after top-left crop to {args.top_left_size}"
        )

    metrics = evaluate(
        model,
        loader,
        calibration.to(device),
        spatial_tensor.to(device),
        device,
        build_hlp_scale_vector(args.hlp_target_scale, device),
        max_samples=args.max_samples,
        return_per_state=args.per_state_metrics,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "spatial_mask": str(spatial_mask_path),
        "categories": list(categories),
        "split": args.split,
        "available_sample_count": len(samples),
        "evaluated_sample_count": min(len(samples), args.max_samples)
        if args.max_samples is not None
        else len(samples),
        "top_left_size": args.top_left_size,
        "hlp_target_scale_restored_before_metrics": args.hlp_target_scale,
        "per_state_metrics_requested": args.per_state_metrics,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
