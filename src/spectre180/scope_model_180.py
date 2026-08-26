from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scope_model_150 import ScopeNet150


NUM_OUTPUT_CHANNELS = 180
NUM_POLARIZATIONS = 6
NUM_WAVELENGTHS = 30
SIGMA_MIN = 1e-3


class ScopeNet180(ScopeNet150):
    """The established SCOPE body with 180-channel calibration and dual output."""

    def __init__(self) -> None:
        super().__init__(
            out_channels=NUM_OUTPUT_CHANNELS * 2,
            mask_channels=NUM_OUTPUT_CHANNELS,
            width=32,
            output_head_rank=64,
        )


def split_prediction(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = prediction[:, :NUM_OUTPUT_CHANNELS]
    sigma_raw = prediction[:, NUM_OUTPUT_CHANNELS:]
    sigma = F.softplus(sigma_raw) + SIGMA_MIN
    return mu, sigma


def build_scope_model_180() -> ScopeNet180:
    return ScopeNet180()


def expand_120_to_180_channels(
    tensor: torch.Tensor, channel_dim: int
) -> torch.Tensor:
    tensor = tensor.transpose(channel_dim, 0)
    original_shape = tensor.shape
    source = tensor.reshape(120, -1).reshape(NUM_WAVELENGTHS, 4, -1)
    expanded = torch.zeros(
        (NUM_WAVELENGTHS, NUM_POLARIZATIONS, source.shape[-1]),
        dtype=tensor.dtype,
    )
    expanded[:, :4] = source
    expanded[:, 4] = source[:, 3]
    expanded[:, 5] = source[:, 2]
    expanded = expanded.reshape((NUM_OUTPUT_CHANNELS,) + original_shape[1:])
    return expanded.transpose(0, channel_dim).contiguous()


def warmstart_from_120_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, device: torch.device
) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    source_state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    target_state = model.state_dict()
    loaded: list[str] = []
    expanded_keys: list[str] = []
    skipped: list[str] = []

    for key, target_value in target_state.items():
        if key not in source_state:
            skipped.append(key)
            continue
        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            loaded.append(key)
            continue

        expanded = None
        if source_value.shape[0] == 120 and target_value.shape[0] == 180:
            expanded = expand_120_to_180_channels(source_value, 0)
        elif (
            source_value.ndim > 1
            and source_value.shape[1] == 120
            and target_value.shape[1] == 180
        ):
            expanded = expand_120_to_180_channels(source_value, 1)
        elif source_value.shape[0] == 240 and target_value.shape[0] == 360:
            mu = expand_120_to_180_channels(source_value[:120], 0)
            sigma = expand_120_to_180_channels(source_value[120:], 0)
            expanded = torch.cat([mu, sigma], dim=0)

        if expanded is not None and expanded.shape == target_value.shape:
            target_state[key] = expanded
            loaded.append(key)
            expanded_keys.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(target_state, strict=True)
    return {
        "loaded_keys": loaded,
        "expanded_keys": expanded_keys,
        "skipped_keys": skipped,
        "checkpoint_path": str(checkpoint_path),
        "mapping": {
            "source_order": ["LCP", "RCP", "HLP", "VLP"],
            "target_order": ["LCP", "RCP", "HLP", "VLP", "45LP", "135LP"],
            "45LP_initializer": "VLP",
            "135LP_initializer": "HLP",
        },
    }
