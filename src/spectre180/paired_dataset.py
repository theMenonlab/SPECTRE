from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairedSample150:
    category: str
    sample_index: int
    split: str
    miniscope_path: str
    hamamatsu_path: str


def split_from_index(index: int, test_every: int = 10) -> str:
    test_every = max(2, int(test_every))
    mod = index % test_every
    if mod == 0:
        return "test"
    if mod == 1:
        return "val"
    return "train"


def _load_miniscope(path: Path) -> np.ndarray:
    image = np.array(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
    return image


def _natural_stems(directory: Path, suffix: str) -> list[Path]:
    return natsorted([path for path in directory.glob(f"*{suffix}") if path.is_file()], key=lambda p: p.name)


def _pair_category(
    category: str,
    data_root: str | Path | None = None,
    mini_category: str | None = None,
    test_every: int = 10,
    allow_mismatch: bool = False,
) -> list[PairedSample150]:
    if data_root is None:
        raise ValueError("data_root is required")
    root = Path(data_root)
    mini_name = mini_category if mini_category is not None else category
    mini_dir = root / f"{mini_name}_mini_FC"
    ham_dir = root / category / "hamamatsu"

    mini_files = _natural_stems(mini_dir, ".png")
    ham_files = _natural_stems(ham_dir, ".tif")
    if len(mini_files) != len(ham_files):
        if not allow_mismatch:
            raise ValueError(f"Mismatched counts for {category}: {len(mini_files)} miniscope vs {len(ham_files)} hamamatsu")
        count = min(len(mini_files), len(ham_files))
        mini_files = mini_files[:count]
        ham_files = ham_files[:count]

    rows: list[PairedSample150] = []
    for index, (mini_path, ham_path) in enumerate(zip(mini_files, ham_files)):
        rows.append(
            PairedSample150(
                category=category,
                sample_index=index,
                split=split_from_index(index, test_every=test_every),
                miniscope_path=str(mini_path),
                hamamatsu_path=str(ham_path),
            )
        )
    return rows


def collect_training_pairs(
    categories: tuple[str, ...] | list[str] | None = None,
    data_root: str | Path | None = None,
    mini_category_map: dict[str, str] | None = None,
    test_every: int = 10,
    allow_mismatch: bool = False,
) -> list[PairedSample150]:
    if categories is None:
        raise ValueError("categories are required")
    mini_category_map = mini_category_map or {}
    rows: list[PairedSample150] = []
    for category in categories:
        rows.extend(
            _pair_category(
                category,
                data_root=data_root,
                mini_category=mini_category_map.get(category),
                test_every=test_every,
                allow_mismatch=allow_mismatch,
            )
        )
    return rows


def save_manifest(samples: list[PairedSample150], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump([asdict(sample) for sample in samples], handle, indent=2)


class PairedScope150Dataset(Dataset):
    def __init__(
        self,
        samples: list[PairedSample150],
        resize_to: int = 512,
        crop_size: int | None = None,
        training: bool = False,
        top_left_size: int | None = None,
        expected_channels: int = 150,
    ):
        self.samples = samples
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.training = training
        self.top_left_size = top_left_size
        self.expected_channels = int(expected_channels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        mini = _load_miniscope(Path(sample.miniscope_path))
        mini_tensor = torch.from_numpy(mini).unsqueeze(0).unsqueeze(0).float()
        if self.top_left_size is not None:
            size = self.top_left_size
            if mini_tensor.shape[-2] < size or mini_tensor.shape[-1] < size:
                raise ValueError(
                    f"Miniscope image smaller than top-left crop {size}: "
                    f"{sample.miniscope_path} has {tuple(mini_tensor.shape[-2:])}"
                )
            mini_tensor = mini_tensor[..., :size, :size]
        elif mini_tensor.shape[-2:] != (self.resize_to, self.resize_to):
            mini_tensor = F.interpolate(mini_tensor, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        mini_tensor = mini_tensor.squeeze(0)

        gt = tifffile.imread(sample.hamamatsu_path).astype(np.float32)
        if gt.ndim != 3 or gt.shape[0] != self.expected_channels:
            raise ValueError(
                f"Expected ({self.expected_channels}, H, W) from "
                f"{sample.hamamatsu_path}, got {gt.shape}"
            )
        gt = gt / 65535.0
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
        if self.top_left_size is not None:
            size = self.top_left_size
            if gt_tensor.shape[-2] < size or gt_tensor.shape[-1] < size:
                raise ValueError(
                    f"Hamamatsu stack smaller than top-left crop {size}: "
                    f"{sample.hamamatsu_path} has {tuple(gt_tensor.shape[-2:])}"
                )
            gt_tensor = gt_tensor[..., :size, :size]
        elif gt_tensor.shape[-2:] != (self.resize_to, self.resize_to):
            gt_tensor = F.interpolate(gt_tensor, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        gt_tensor = gt_tensor.squeeze(0)

        if self.top_left_size is None and self.crop_size and self.crop_size < self.resize_to:
            if self.training:
                y = random.randint(0, self.resize_to - self.crop_size)
                x = random.randint(0, self.resize_to - self.crop_size)
            else:
                y = (self.resize_to - self.crop_size) // 2
                x = (self.resize_to - self.crop_size) // 2
            mini_tensor = mini_tensor[:, y : y + self.crop_size, x : x + self.crop_size]
            gt_tensor = gt_tensor[:, y : y + self.crop_size, x : x + self.crop_size]

        return mini_tensor, gt_tensor
