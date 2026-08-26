#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paired_dataset import PairedScope150Dataset, collect_training_pairs, save_manifest
from scope_model_180 import (
    NUM_OUTPUT_CHANNELS,
    NUM_POLARIZATIONS,
    build_scope_model_180,
    split_prediction,
    warmstart_from_120_checkpoint,
)

SOURCE_60_POLARIZATIONS = ("HLP", "VLP")
TARGET_180_POLARIZATIONS = ("LCP", "RCP", "HLP", "VLP", "45LP", "135LP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 180-channel six-state SCOPE model for the 20260730 comparison."
    )
    parser.add_argument("--mask", required=True)
    parser.add_argument(
        "--spatial-loss-mask",
        required=True,
        help="Single-channel valid-FOV mask used for losses and metrics, not as model calibration input.",
    )
    parser.add_argument(
        "--initialization",
        choices=["from60", "from120", "random"],
        default="from60",
        help="Initialize from the two-state 60-channel model, the legacy four-state 120-channel model, or randomly.",
    )
    parser.add_argument(
        "--warmstart-checkpoint",
        default=None,
        help="Required for from60/from120 initialization; ignored for random initialization.",
    )
    parser.add_argument("--train-categories", default="20260717_fixed_c_elegans,20260721_c-elegans_fixed_older,20260723_tardigrades")
    parser.add_argument("--data-root", required=True, help="Root containing <category>/hamamatsu and <category>_mini_FC folders.")
    parser.add_argument(
        "--mini-category-map",
        default="",
        help="Comma-separated hamamatsu=miniscope base names, e.g. 20250504_4x_polystyrene=20260504_4x_polystyrene.",
    )
    parser.add_argument("--test-every", type=int, default=10, help="Use every Nth paired image as test and image N+1 as validation.")
    parser.add_argument("--allow-count-mismatch", action="store_true", help="Pair by sorted order up to the shorter mini/hamamatsu count.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Accumulate this many physical batches before each optimizer step.",
    )
    parser.add_argument("--crop-size", type=int, default=600)
    parser.add_argument("--resize", type=int, default=600)
    parser.add_argument(
        "--top-left-size",
        type=int,
        default=600,
        help="Use the native top-left NxN region for input and target, with no image interpolation. Overrides resize/crop spatial processing.",
    )
    parser.add_argument(
        "--calibration-native-size",
        type=int,
        default=600,
        help="Resize the calibration mask to this native size before applying the top-left crop.",
    )
    parser.add_argument(
        "--oversample-category",
        default=None,
        help="Repeat training samples from this category; validation and test samples are never repeated.",
    )
    parser.add_argument("--oversample-factor", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss-mode", choices=["laplace_l1", "laplace_spectral", "l1", "rmse", "mrae"], default="laplace_l1")
    parser.add_argument("--hlp-target-scale", type=float, default=15.0, help="Multiply HLP target channels during training; predictions are divided back for metrics.")
    parser.add_argument("--laplace-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=0.05)
    parser.add_argument("--mrae-eps", type=float, default=1e-3)
    parser.add_argument("--sam-weight", type=float, default=2.0)
    parser.add_argument("--spectral-grad-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=1.0,
        help="Initial CUDA GradScaler value. A conservative value prevents warm-start Laplace gradients from silently overflowing.",
    )
    parser.add_argument(
        "--fail-on-skipped-optimizer-step",
        action="store_true",
        help="Abort immediately if CUDA GradScaler suppresses an optimizer update.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--checkpoint-prefix", default="scope180")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _move_channel_axis_first(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    return tensor.transpose(0, channel_dim).contiguous()


def expand_60_output_to_180(
    tensor: torch.Tensor, channel_dim: int
) -> torch.Tensor:
    """Expand wavelength-major HLP/VLP outputs to neutral six-state outputs."""
    moved = _move_channel_axis_first(tensor, channel_dim)
    trailing_shape = moved.shape[1:]
    source = moved.reshape(30, 2, *trailing_shape)
    target = moved.new_zeros((30, 6, *trailing_shape))
    neutral = 0.5 * (source[:, 0] + source[:, 1])
    target[:, 0] = neutral
    target[:, 1] = neutral
    target[:, 2] = source[:, 0]
    target[:, 3] = source[:, 1]
    target[:, 4] = neutral
    target[:, 5] = neutral
    expanded = target.reshape(180, *trailing_shape)
    return expanded.transpose(0, channel_dim).contiguous()


def expand_60_calibration_input_to_180(
    tensor: torch.Tensor, channel_dim: int
) -> torch.Tensor:
    """Keep the source HLP/VLP response while new calibration states learn."""
    moved = _move_channel_axis_first(tensor, channel_dim)
    trailing_shape = moved.shape[1:]
    source = moved.reshape(30, 2, *trailing_shape)
    target = moved.new_zeros((30, 6, *trailing_shape))
    target[:, 2] = source[:, 0]
    target[:, 3] = source[:, 1]
    expanded = target.reshape(180, *trailing_shape)
    return expanded.transpose(0, channel_dim).contiguous()


def warmstart_from_60_checkpoint(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    else:
        source_state = checkpoint
    source_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    target_state = model.state_dict()
    loaded: list[str] = []
    expanded: list[str] = []
    skipped: list[str] = []

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            skipped.append(key)
            continue
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            loaded.append(key)
            continue

        mapped = None
        if source_value.shape[0] == 60 and target_value.shape[0] == 180:
            mapped = expand_60_output_to_180(source_value, 0)
        elif (
            source_value.ndim > 1
            and source_value.shape[1] == 60
            and target_value.shape[1] == 180
        ):
            mapped = expand_60_calibration_input_to_180(source_value, 1)
        elif source_value.shape[0] == 120 and target_value.shape[0] == 360:
            mu = expand_60_output_to_180(source_value[:60], 0)
            sigma = expand_60_output_to_180(source_value[60:], 0)
            mapped = torch.cat([mu, sigma], dim=0)

        if mapped is None or mapped.shape != target_value.shape:
            skipped.append(key)
            continue
        target_state[key] = mapped
        loaded.append(key)
        expanded.append(key)

    unexpected = sorted(set(source_state) - set(target_state))
    if skipped or unexpected:
        raise RuntimeError(
            "60-channel warm start did not map cleanly: "
            f"skipped={skipped}, unexpected={unexpected}"
        )
    model.load_state_dict(target_state, strict=True)
    return {
        "initialization": "from60",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "loaded_key_count": len(loaded),
        "expanded_keys": expanded,
        "skipped_keys": skipped,
        "unexpected_source_keys": unexpected,
        "source_order": list(SOURCE_60_POLARIZATIONS),
        "target_order": list(TARGET_180_POLARIZATIONS),
        "output_mapping": {
            "HLP": "source HLP",
            "VLP": "source VLP",
            "LCP": "mean(source HLP, source VLP)",
            "RCP": "mean(source HLP, source VLP)",
            "45LP": "mean(source HLP, source VLP)",
            "135LP": "mean(source HLP, source VLP)",
        },
        "calibration_adapter_mapping": {
            "HLP": "source HLP weight",
            "VLP": "source VLP weight",
            "new_state_weights": "zero initialized",
        },
    }


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_mini_category_map(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --mini-category-map item {item!r}; expected hamamatsu=miniscope")
        category, mini_category = item.split("=", 1)
        result[category.strip()] = mini_category.strip()
    return result


def masked_mean(values: torch.Tensor, spatial_mask: torch.Tensor | None) -> torch.Tensor:
    if spatial_mask is None:
        return values.mean()
    weights = spatial_mask.to(device=values.device, dtype=values.dtype).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def laplace_nll(
    target: torch.Tensor,
    prediction: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
    sigma_min: float = 1e-3,
) -> torch.Tensor:
    constant = math.log(2.0)
    n = target.shape[1]
    mu = prediction[:, :n, :, :]
    sigma = torch.clamp(F.softplus(prediction[:, n:, :, :]) + sigma_min, min=sigma_min)
    values = torch.abs((mu - target) / sigma) + torch.log(sigma) + constant
    return masked_mean(values, spatial_mask)


def spectral_angle_mapper_loss(mu: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu_flat = mu.permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    mu_norm = torch.linalg.norm(mu_flat, dim=1).clamp_min(eps)
    target_norm = torch.linalg.norm(target_flat, dim=1).clamp_min(eps)
    cosine = torch.sum(mu_flat * target_flat, dim=1) / (mu_norm * target_norm)
    return torch.acos(torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)).mean()


def spectral_gradient_loss(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(mu[:, 1:, :, :] - mu[:, :-1, :, :], target[:, 1:, :, :] - target[:, :-1, :, :])


def cosine_similarity_loss(mu: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu_flat = mu.permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    cosine = F.cosine_similarity(mu_flat, target_flat, dim=1, eps=eps)
    return (1.0 - cosine).mean()


def total_loss(
    targets: torch.Tensor,
    prediction: torch.Tensor,
    args: argparse.Namespace,
    spatial_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    # SCOPE forward runs under autocast, but the Laplace division can overflow
    # float16 gradients for a strong warm start even when the scalar loss is
    # finite. Compute all loss terms in float32 so GradScaler does not silently
    # skip the first optimizer steps.
    targets = targets.float()
    prediction = prediction.float()
    if spatial_mask is not None:
        spatial_mask = spatial_mask.float()
    mu, _sigma = split_prediction(prediction)
    nll = laplace_nll(targets, prediction, spatial_mask=spatial_mask)
    l1 = masked_mean(torch.abs(mu - targets), spatial_mask)
    rmse = torch.sqrt(masked_mean((mu - targets) ** 2, spatial_mask).clamp_min(1e-12))
    mrae = masked_mean(
        torch.abs(mu - targets) / torch.clamp(torch.abs(targets), min=args.mrae_eps),
        spatial_mask,
    )
    base_parts = {
        "nll": float(nll.detach().item()),
        "l1": float(l1.detach().item()),
        "rmse": float(rmse.detach().item()),
        "mrae": float(mrae.detach().item()),
        "sam": 0.0,
        "spectral_grad": 0.0,
        "cosine": 0.0,
    }
    if args.loss_mode == "l1":
        return l1, base_parts
    if args.loss_mode == "rmse":
        return rmse, base_parts
    if args.loss_mode == "mrae":
        return mrae, base_parts
    if args.loss_mode == "laplace_l1":
        loss = args.laplace_weight * nll + args.l1_weight * l1
        return loss, base_parts

    sam = spectral_angle_mapper_loss(mu, targets)
    grad = spectral_gradient_loss(mu, targets)
    cosine = cosine_similarity_loss(mu, targets)
    loss = (
        args.laplace_weight * nll
        + args.l1_weight * l1
        + args.sam_weight * sam
        + args.spectral_grad_weight * grad
        + args.cosine_weight * cosine
    )
    parts = {
        "nll": float(nll.detach().item()),
        "l1": float(l1.detach().item()),
        "rmse": float(rmse.detach().item()),
        "mrae": float(mrae.detach().item()),
        "sam": float(sam.detach().item()),
        "spectral_grad": float(grad.detach().item()),
        "cosine": float(cosine.detach().item()),
    }
    return loss, parts


def build_hlp_scale_vector(hlp_target_scale: float, device: torch.device | str = "cpu") -> torch.Tensor:
    scale = torch.ones(NUM_OUTPUT_CHANNELS, dtype=torch.float32, device=device)
    scale[2::NUM_POLARIZATIONS] = float(hlp_target_scale)
    return scale.view(1, NUM_OUTPUT_CHANNELS, 1, 1)


def mean_channel_psnr(
    target: np.ndarray, prediction: np.ndarray, spatial_mask: np.ndarray
) -> float:
    valid = spatial_mask.astype(bool)
    mse = ((target - prediction) ** 2)[:, valid].mean(axis=1)
    return float((10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))).mean())


def collapse_to_30(cube: np.ndarray) -> np.ndarray:
    return cube.reshape(30, NUM_POLARIZATIONS, cube.shape[1], cube.shape[2]).mean(axis=1)


def mean_spectrum_corr(
    target: np.ndarray, prediction: np.ndarray, spatial_mask: np.ndarray
) -> float:
    valid = spatial_mask.astype(bool)
    a = target[:, valid].mean(axis=1)
    b = prediction[:, valid].mean(axis=1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spectral_angle(
    target: np.ndarray, prediction: np.ndarray, spatial_mask: np.ndarray
) -> float:
    valid = spatial_mask.astype(bool)
    a = target[:, valid].mean(axis=1)
    b = prediction[:, valid].mean(axis=1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-8:
        return 0.0
    value = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def evaluate(
    model,
    loader,
    mask,
    spatial_mask,
    device,
    channel_scale,
    max_samples: int | None = None,
    return_per_state: bool = False,
) -> dict:
    model.eval()
    nll_values = []
    l1_values = []
    psnr150_values = []
    psnr30_values = []
    corr_values = []
    sam_values = []
    inference_times = []
    state_names = ("LCP", "RCP", "HLP", "VLP", "45LP", "135LP")
    state_values = {
        name: {"l1": [], "psnr": [], "mean_spectrum_corr": [], "sam_degrees": []}
        for name in state_names
    }

    with torch.no_grad():
        for sample_index, (inputs, targets) in enumerate(loader):
            if max_samples is not None and sample_index >= max_samples:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_mask = mask.expand(inputs.shape[0], -1, -1, -1)

            start = time.time()
            prediction = model(inputs, batch_mask)
            inference_times.append((time.time() - start) / max(inputs.shape[0], 1))

            mu, _sigma = split_prediction(prediction)
            targets_scaled = targets * channel_scale
            mu_original = mu / channel_scale
            batch_spatial_mask = spatial_mask.expand(inputs.shape[0], -1, -1, -1)
            nll_values.append(
                float(
                    laplace_nll(
                        targets_scaled,
                        prediction,
                        spatial_mask=batch_spatial_mask,
                    ).item()
                )
            )
            l1_values.append(
                float(masked_mean(torch.abs(mu_original - targets), batch_spatial_mask).item())
            )

            target_np = targets.detach().cpu().numpy()
            pred_np = np.clip(mu_original.detach().cpu().numpy(), 0.0, 1.0)
            spatial_mask_np = spatial_mask[0, 0].detach().cpu().numpy().astype(bool)
            for target_item, pred_item in zip(target_np, pred_np):
                psnr150_values.append(mean_channel_psnr(target_item, pred_item, spatial_mask_np))
                psnr30_values.append(
                    mean_channel_psnr(
                        collapse_to_30(target_item),
                        collapse_to_30(pred_item),
                        spatial_mask_np,
                    )
                )
                corr_values.append(
                    mean_spectrum_corr(target_item, pred_item, spatial_mask_np)
                )
                sam_values.append(
                    spectral_angle(target_item, pred_item, spatial_mask_np)
                )
                if return_per_state:
                    for state_index, state_name in enumerate(state_names):
                        target_state = target_item[
                            state_index::NUM_POLARIZATIONS
                        ]
                        pred_state = pred_item[state_index::NUM_POLARIZATIONS]
                        valid = spatial_mask_np.astype(bool)
                        state_values[state_name]["l1"].append(
                            float(
                                np.abs(
                                    target_state[:, valid]
                                    - pred_state[:, valid]
                                ).mean()
                            )
                        )
                        state_values[state_name]["psnr"].append(
                            mean_channel_psnr(
                                target_state, pred_state, spatial_mask_np
                            )
                        )
                        state_values[state_name]["mean_spectrum_corr"].append(
                            mean_spectrum_corr(
                                target_state, pred_state, spatial_mask_np
                            )
                        )
                        state_values[state_name]["sam_degrees"].append(
                            spectral_angle(
                                target_state, pred_state, spatial_mask_np
                            )
                        )

    result = {
        "nll": float(np.mean(nll_values)) if nll_values else None,
        "l1": float(np.mean(l1_values)) if l1_values else None,
        "psnr180": float(np.mean(psnr150_values)) if psnr150_values else None,
        "psnr30": float(np.mean(psnr30_values)) if psnr30_values else None,
        "mean_spectrum_corr": float(np.mean(corr_values)) if corr_values else None,
        "sam_degrees": float(np.mean(sam_values)) if sam_values else None,
        "inference_time_seconds": float(np.mean(inference_times)) if inference_times else None,
    }
    if return_per_state:
        result["per_state"] = {
            state_name: {
                metric_name: (
                    float(np.mean(values)) if values else None
                )
                for metric_name, values in metrics.items()
            }
            for state_name, metrics in state_values.items()
        }
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    set_seed(args.seed)

    train_categories = tuple([item.strip() for item in args.train_categories.split(",") if item.strip()])
    all_samples = collect_training_pairs(
        train_categories,
        data_root=args.data_root,
        mini_category_map=parse_mini_category_map(args.mini_category_map),
        test_every=args.test_every,
        allow_mismatch=args.allow_count_mismatch,
    )
    save_manifest(all_samples, output_dir / "paired_samples_manifest.json")
    train_samples = [row for row in all_samples if row.split == "train"]
    val_samples = [row for row in all_samples if row.split == "val"]
    test_samples = [row for row in all_samples if row.split == "test"]
    if args.oversample_factor < 1:
        raise ValueError("--oversample-factor must be at least 1")
    if args.oversample_category and args.oversample_factor > 1:
        selected = [row for row in train_samples if row.category == args.oversample_category]
        if not selected:
            raise ValueError(f"No training samples found for oversample category {args.oversample_category!r}")
        train_samples.extend(selected * (args.oversample_factor - 1))
    save_manifest(train_samples, output_dir / "effective_training_manifest.json")

    mask = torch.load(args.mask, map_location="cpu").float().unsqueeze(0)
    spatial_mask_array = np.asarray(
        Image.open(args.spatial_loss_mask).convert("L"), dtype=np.uint8
    )
    spatial_mask = torch.from_numpy((spatial_mask_array > 0).astype(np.float32))[None, None]
    if args.top_left_size is not None:
        if args.top_left_size <= 0:
            raise ValueError("--top-left-size must be positive")
        spatial_size = args.top_left_size
        calibration_native_size = args.calibration_native_size or spatial_size
        if calibration_native_size < spatial_size:
            raise ValueError("--calibration-native-size cannot be smaller than --top-left-size")
        native_mask = mask if mask.shape[-2:] == (calibration_native_size, calibration_native_size) else F.interpolate(
            mask, size=(calibration_native_size, calibration_native_size), mode="bilinear", align_corners=False
        )
        fixed_mask = native_mask[..., :spatial_size, :spatial_size]
        train_mask = fixed_mask
        eval_mask = fixed_mask
        if spatial_mask.shape[-2:] == (spatial_size, spatial_size):
            pass
        elif (
            spatial_mask.shape[-2] >= spatial_size
            and spatial_mask.shape[-1] >= spatial_size
        ):
            spatial_mask = spatial_mask[..., :spatial_size, :spatial_size]
        else:
            spatial_mask = F.interpolate(
                spatial_mask,
                size=(spatial_size, spatial_size),
                mode="nearest",
            )
    elif mask.shape[-2:] != (args.crop_size, args.crop_size):
        train_mask = F.interpolate(mask, size=(args.crop_size, args.crop_size), mode="bilinear", align_corners=False)
        eval_mask = mask if mask.shape[-2:] == (args.resize, args.resize) else F.interpolate(mask, size=(args.resize, args.resize), mode="bilinear", align_corners=False)
        spatial_mask = F.interpolate(
            spatial_mask, size=(args.resize, args.resize), mode="nearest"
        )
    else:
        train_mask = mask
        eval_mask = mask if mask.shape[-2:] == (args.resize, args.resize) else F.interpolate(mask, size=(args.resize, args.resize), mode="bilinear", align_corners=False)
        spatial_mask = (
            spatial_mask
            if spatial_mask.shape[-2:] == (args.resize, args.resize)
            else F.interpolate(spatial_mask, size=(args.resize, args.resize), mode="nearest")
        )

    if spatial_mask.sum().item() <= 0:
        raise ValueError("Spatial loss mask has no valid pixels")
    train_ds = PairedScope150Dataset(train_samples, resize_to=args.resize, crop_size=args.crop_size, training=True, top_left_size=args.top_left_size, expected_channels=NUM_OUTPUT_CHANNELS)
    val_ds = PairedScope150Dataset(val_samples, resize_to=args.resize, crop_size=None, training=False, top_left_size=args.top_left_size, expected_channels=NUM_OUTPUT_CHANNELS)
    test_ds = PairedScope150Dataset(test_samples, resize_to=args.resize, crop_size=None, training=False, top_left_size=args.top_left_size, expected_channels=NUM_OUTPUT_CHANNELS)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    model = build_scope_model_180().to(device)
    if args.initialization != "random" and not args.warmstart_checkpoint:
        raise ValueError("--warmstart-checkpoint is required unless --initialization random")
    if args.initialization == "from60":
        warmstart_info = warmstart_from_60_checkpoint(
            model, args.warmstart_checkpoint, device
        )
    elif args.initialization == "from120":
        warmstart_info = warmstart_from_120_checkpoint(
            model, args.warmstart_checkpoint, device
        )
        warmstart_info["initialization"] = "from120"
        warmstart_info["checkpoint_sha256"] = sha256_file(args.warmstart_checkpoint)
    else:
        warmstart_info = {
            "initialization": "random",
            "random_seed": args.seed,
            "checkpoint_path": None,
            "mapping": None,
        }
    with open(output_dir / "warmstart_info.json", "w") as handle:
        json.dump(warmstart_info, handle, indent=2)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda",
        init_scale=args.amp_init_scale,
    )
    channel_scale = build_hlp_scale_vector(args.hlp_target_scale, device)
    history: list[dict] = []
    best_psnr = -float("inf")
    best_nll = float("inf")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        train_loss_sum = 0.0
        train_steps = 0
        optimizer_steps_attempted = 0
        optimizer_steps_applied = 0
        optimizer_steps_skipped = 0
        train_part_sums = {"nll": 0.0, "l1": 0.0, "rmse": 0.0, "mrae": 0.0, "sam": 0.0, "spectral_grad": 0.0, "cosine": 0.0}
        if args.grad_accum_steps < 1:
            raise ValueError("--grad-accum-steps must be at least 1")
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_mask = train_mask.to(device, non_blocking=True).expand(inputs.shape[0], -1, -1, -1)
            batch_spatial_mask = spatial_mask.to(device, non_blocking=True).expand(inputs.shape[0], -1, -1, -1)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                prediction = model(inputs, batch_mask)
                loss, parts = total_loss(
                    targets * channel_scale,
                    prediction,
                    args,
                    spatial_mask=batch_spatial_mask,
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            reached_limit = args.max_batches is not None and batch_index + 1 >= args.max_batches
            should_step = (
                (batch_index + 1) % args.grad_accum_steps == 0
                or batch_index + 1 == len(train_loader)
                or reached_limit
            )
            if should_step:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps_attempted += 1
                if device.type == "cuda" and scaler.get_scale() < scale_before:
                    optimizer_steps_skipped += 1
                    if args.fail_on_skipped_optimizer_step:
                        raise RuntimeError(
                            "GradScaler skipped an optimizer update; aborting "
                            "instead of silently training fewer steps."
                        )
                else:
                    optimizer_steps_applied += 1
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += float(loss.item())
            for key, value in parts.items():
                train_part_sums[key] += value
            train_steps += 1
            if reached_limit:
                break

        val_metrics = evaluate(
            model,
            val_loader,
            eval_mask.to(device),
            spatial_mask.to(device),
            device,
            channel_scale,
            args.max_eval_samples,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_steps, 1),
            "train_nll": train_part_sums["nll"] / max(train_steps, 1),
            "train_l1": train_part_sums["l1"] / max(train_steps, 1),
            "train_rmse": train_part_sums["rmse"] / max(train_steps, 1),
            "train_mrae": train_part_sums["mrae"] / max(train_steps, 1),
            "train_sam": train_part_sums["sam"] / max(train_steps, 1),
            "train_spectral_grad": train_part_sums["spectral_grad"] / max(train_steps, 1),
            "train_cosine": train_part_sums["cosine"] / max(train_steps, 1),
            "seconds": time.time() - start,
            "optimizer_steps_attempted": optimizer_steps_attempted,
            "optimizer_steps_applied": optimizer_steps_applied,
            "optimizer_steps_skipped": optimizer_steps_skipped,
            "peak_cuda_memory_gb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**3))
                if device.type == "cuda"
                else None
            ),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)

        latest_path = checkpoints_dir / f"{args.checkpoint_prefix}_latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
                "hlp_channel_indices": list(range(2, NUM_OUTPUT_CHANNELS, NUM_POLARIZATIONS)),
                "spatial_loss_mask": args.spatial_loss_mask,
                "spatial_loss_valid_pixels": int(spatial_mask.sum().item()),
                "channel_scale": channel_scale.detach().cpu().view(-1).tolist(),
            },
            latest_path,
        )

        if val_metrics["psnr180"] is not None and val_metrics["psnr180"] > best_psnr:
            best_psnr = val_metrics["psnr180"]
            torch.save(torch.load(latest_path, map_location="cpu", weights_only=False), checkpoints_dir / f"{args.checkpoint_prefix}_best_psnr.pth")
        if val_metrics["nll"] is not None and val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            torch.save(torch.load(latest_path, map_location="cpu", weights_only=False), checkpoints_dir / f"{args.checkpoint_prefix}_best_nll.pth")

    with open(output_dir / "training_history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    with open(output_dir / "training_history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    best_checkpoint = checkpoints_dir / f"{args.checkpoint_prefix}_best_psnr.pth"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint, strict=True)
    test_metrics = evaluate(
        model,
        test_loader,
        eval_mask.to(device),
        spatial_mask.to(device),
        device,
        channel_scale,
        args.max_eval_samples,
    )
    with open(output_dir / "test_metrics.json", "w") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
