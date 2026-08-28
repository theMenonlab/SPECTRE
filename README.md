# SPECTRE

SPECTRE (**S**pectro-**P**olarimetric **E**stimator with **C**alibrated, **T**rustworthy **R**econstruction **E**rrors) reconstructs a six-state, 30-wavelength hyperspectral-polarization stack from one PHAMscope Miniscope image and its measured calibration tensor. The released model predicts 180 mean channels and 180 per-pixel uncertainty channels.

This repository contains the runnable PyTorch model, inference, training, evaluation, and compact result files. Large checkpoints, calibration tensors, and example image pairs are hosted separately on Kaggle so that the Git history stays small.

## Download links

| Asset | Link | Contents |
|---|---|---|
| SPECTRE model weights | [Kaggle Models](https://www.kaggle.com/models/alingold/spectre-models) | Released SPECTRE model files and associated model assets |
| General 60-channel dataset (512 px) | [Kaggle Dataset](https://www.kaggle.com/datasets/alingold/spectre-general-60ch-dataset-512) | General 60-channel SPECTRE dataset at 512-pixel spatial resolution |
| SPECTRE 150-channel dataset | [Kaggle Dataset](https://www.kaggle.com/datasets/alingold/spectre-150ch) | SPECTRE 150-channel dataset |
| Source code | [GitHub](https://github.com/theMenonlab/SPECTRE) | Inference, training, evaluation, timing, and compact result provenance |

The prepared Kaggle filenames and hashes are listed in [`models/checkpoints_manifest.csv`](models/checkpoints_manifest.csv). Do not substitute checkpoints with similar names: Model A/B/C use one measured calibration tensor, while c42d uses its own acquisition-specific calibration.

## Released checkpoints

| Checkpoint | Intended role | Calibration |
|---|---|---|
| `model_A_best_psnr.pth` | Publication organism model, initialized from the established physical 60-channel model | `model_ABC_mask_180_600.pt` |
| `model_B_best_psnr.pth` | Inclusive model including optical targets | `model_ABC_mask_180_600.pt` |
| `model_C_best_psnr.pth` | Matched random-initialization control; not the recommended application model | `model_ABC_mask_180_600.pt` |
| `c42d_best_psnr.pth` | Checkpoint used for the released live-video reconstruction lineage | `c42d_mask_180_600.pt` |

These are best-PSNR checkpoints. Timing JSON files are measurements, not weights. The exact historical 60-channel checkpoint supporting the Table S2 architecture result is not in this release because the audited file was unavailable; see [`results/benchmark60/README.md`](results/benchmark60/README.md).

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/theMenonlab/SPECTRE.git
cd SPECTRE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build appropriate for your CPU/CUDA platform from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
pip install -r requirements.txt
```

The recorded research environment used PyTorch 2.0.1 with CUDA 11.8, NumPy 1.26.4, Pillow 11.1.0, and tifffile 2023.2.28. These are provenance versions, not a claim that newer compatible releases will fail.

### 2. Download and verify the Kaggle assets

Place the downloaded files under a local directory such as `release-assets/`, preserving the Kaggle tree. From that directory:

```bash
sha256sum -c manifests/SHA256SUMS.txt
```

### 3. Run one 180-channel reconstruction

```bash
python src/spectre180/run_scope_inference_180.py \
  --input release-assets/examples/scope180/pos0001_mini_fc.png \
  --checkpoint release-assets/checkpoints/publication/model_A_best_psnr.pth \
  --mask release-assets/calibration/model_ABC_mask_180_600.pt \
  --output-dir outputs/model_A_example \
  --device auto \
  --save-uint16-mu
```

The primary output is a float32 TIFF with shape `180 x H x W`. Channel order is wavelength-major; within each wavelength the six states are `LCP, RCP, HLP, VLP, 45LP, 135LP`. The uncertainty TIFF has the same layout. The inference manifest records input, checkpoint, calibration, shapes, timing, and hashes.

The published training contract scaled the HLP target by 15 during fitting. The inference script restores that scale by default. Do not change `--hlp-training-scale` unless using a checkpoint trained with a different value.

## Training and evaluation

The paired-data convention is:

```text
DATA_ROOT/
  category_name/hamamatsu/*.tif
  category_name_mini_FC/*.png
```

A minimal random-initialization training smoke test is:

```bash
python src/spectre180/train_scope_180_20260730.py \
  --data-root /path/to/paired/data \
  --train-categories category_name \
  --mask /path/to/model_ABC_mask_180_600.pt \
  --spatial-loss-mask /path/to/largest_safe_circle_mask_608.png \
  --initialization random \
  --output-dir outputs/training_smoke \
  --epochs 1 \
  --max-batches 1 \
  --max-eval-samples 1
```

For transfer initialization, use `--initialization from60` or `from120` and supply the matching `--warmstart-checkpoint`. The two historical 60-channel lineage checkpoints were deliberately not included in the initial Kaggle payload because they add about 909 MB and are not required for inference with the released 180-channel weights.

Evaluate a frozen checkpoint with:

```bash
python src/spectre180/evaluate_scope180_external.py \
  --checkpoint /path/to/model_A_best_psnr.pth \
  --data-root /path/to/paired/data \
  --categories category_name \
  --mask /path/to/model_ABC_mask_180_600.pt \
  --spatial-mask /path/to/largest_safe_circle_mask_608.png \
  --per-state-metrics \
  --output outputs/model_A_metrics.json
```

## Repository layout

- `src/spectre180/`: model, inference, training, evaluation, and throughput tools.
- `models/checkpoints_manifest.csv`: checkpoint/calibration identities and SHA-256 values.
- `results/scope180/`: compact frozen external-evaluation and throughput records.
- `results/benchmark60/`: compact historical architecture-benchmark results; no third-party benchmark source trees.
- `licenses/NAFNet_LICENSE.txt`: upstream NAFNet MIT terms and its included BasicSR Apache-2.0 notice.

## Architecture and third-party code

SPECTRE's backbone is derived from [NAFNet](https://github.com/megvii-research/NAFNet), which is released under MIT terms and incorporates BasicSR material under Apache-2.0. SPECTRE adds calibration-conditioned feature gates, the factorized mean/uncertainty output head, and six-state 180-channel handling. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the complete upstream notice in [`licenses/NAFNet_LICENSE.txt`](licenses/NAFNet_LICENSE.txt).

[SPECAT](https://github.com/THU-luvision/SPECAT) was evaluated as one comparator in the historical architecture benchmark. No SPECAT implementation is included or required by the SPECTRE model. Its official repository did not provide a software license when this release was prepared, so this repository links to it rather than redistributing it.

## License

Menon Lab SPECTRE contributions are released under the [MIT License](LICENSE). Third-party portions remain under their respective licenses and notices.
