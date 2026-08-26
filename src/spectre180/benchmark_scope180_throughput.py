#!/usr/bin/env python3
"""Benchmark frozen SCOPE-180 forward-pass latency with CUDA synchronization."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scope_model_180 import NUM_OUTPUT_CHANNELS, build_scope_model_180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values), percent))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the submission throughput audit")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    mask_path = Path(args.mask)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_scope_model_180().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    raw = np.asarray(Image.open(input_path).convert("L"), dtype=np.float32) / 255.0
    if raw.shape[0] < args.size or raw.shape[1] < args.size:
        raise ValueError(f"Input {input_path} is smaller than {args.size}: {raw.shape}")
    image = torch.from_numpy(raw[: args.size, : args.size])[None, None].to(device)

    calibration = torch.load(mask_path, map_location="cpu", weights_only=False).float()
    if calibration.ndim == 3:
        calibration = calibration.unsqueeze(0)
    if calibration.shape[1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(f"Expected {NUM_OUTPUT_CHANNELS} calibration channels")
    if calibration.shape[-2] < args.size or calibration.shape[-1] < args.size:
        calibration = F.interpolate(
            calibration,
            size=(args.size, args.size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        calibration = calibration[..., : args.size, : args.size]
    calibration = calibration.to(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            prediction = model(image, calibration)
        torch.cuda.synchronize()

        elapsed: list[float] = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model(image, calibration)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

    mean_seconds = float(statistics.mean(elapsed))
    report = {
        "scope": "model forward pass only; excludes image decoding, TIFF writing, and movie rendering",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "input": str(input_path),
        "input_shape": list(image.shape),
        "output_shape": list(prediction.shape),
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "latency_seconds": {
            "mean": mean_seconds,
            "median": float(statistics.median(elapsed)),
            "p05": percentile(elapsed, 5),
            "p95": percentile(elapsed, 95),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "throughput_frames_per_second_batch1": 1.0 / mean_seconds,
        "camera_acquisition_frames_per_second": 60.0,
        "can_claim_online_60fps_from_this_benchmark": bool(mean_seconds <= 1.0 / 60.0),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
