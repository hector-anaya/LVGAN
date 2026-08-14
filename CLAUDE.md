# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StyleGAN-V (CVPR 2022) is a continuous video generator that extends StyleGAN2. It decomposes video generation into **content** (image quality, from a static latent `z`) and **motion** (temporal dynamics, from a Conv1d-based motion mapping network using Fourier positional embeddings). This enables unbounded-length, high-quality video synthesis.

## Environment Setup

```bash
# Standard GPUs (V100, CUDA 10.1)
conda env create -f environment.yaml -p env
conda activate ./env

# Ampere GPUs (A6000, A100, RTX-3090, CUDA 11.1)
conda env create -f environment-ampere.yaml -p env
conda activate ./env
```

If CUDA kernels (`bias_act`, `upfirdn2d`) fail to compile on older GPUs:
```bash
export TORCH_CUDA_ARCH_LIST="7.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

## Common Commands

### Training

```bash
# FaceForensics 256x256 (4 GPUs, ~2 days)
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs dataset.resolution=256 num_gpus=4

# SkyTimelapse 256x256 (slower motion — increase min_period_len)
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=sky_timelapse dataset.resolution=256 num_gpus=4 model.generator.time_enc.min_period_len=256

# Resume from checkpoint (git hash + exp_suffix must match)
python src/infra/launch.py ... training.resume=latest

# Submit via SLURM
python src/infra/launch.py ... slurm=true

# Suppress Hydra log directories
python src/infra/launch.py ... hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled

# MoCoGAN+SG2 baseline
python src/infra/launch.py hydra.run.dir=. +exp_suffix=my_exp env=local dataset=sky_timelapse dataset.resolution=256 num_gpus=4 model=mocogan sampling=uniform sampling.max_dist_between_frames=1
```

### Inference

```bash
# Generate videos from a checkpoint
python src/scripts/generate.py \
  --network_pkl /path/to/network-snapshot.pkl \
  --num_videos 25 --as_grids true --save_as_mp4 true \
  --fps 25 --video_len 128 --batch_size 25 \
  --outdir /path/to/output --truncation_psi 0.9

# Auto-select best checkpoint by FVD metric
python src/scripts/generate.py \
  --networks_dir PATH_TO_EXPERIMENT/output \
  --num_videos 25 --as_grids true --save_as_mp4 true \
  --fps 25 --video_len 128 --batch_size 25 \
  --outdir /path/to/output --truncation_psi 0.9

# Content/motion decomposition grid
python src/scripts/generate.py \
  --networks_dir PATH_TO_EXPERIMENT/output --moco_decomposition 1 \
  --num_videos 25 --as_grids true --save_as_mp4 true \
  --fps 25 --video_len 128 --batch_size 25 --outdir tmp --truncation_psi 0.8
```

### Evaluation

```bash
# FVD/FID between two datasets (zip archives or directories)
python src/scripts/calc_metrics_for_dataset.py \
  --real_data_path /path/to/dataset_a.zip \
  --fake_data_path /path/to/dataset_b.zip \
  --mirror 1 --gpus 4 --resolution 256 \
  --metrics fvd2048_16f,fvd2048_128f,fvd2048_128f_subsample8f,fid50k_full \
  --verbose 0 --use_cache 0

# FVD for a trained model
python src/scripts/calc_metrics.py ...
```

### Data Preprocessing

```bash
# Convert a video file to frame-wise dataset format
python src/scripts/convert_video_to_dataset.py \
  -s /path/to/video.mp4 -t /path/to/output_dir \
  --target_size 256 -sf 150 -cs 512

# Preprocess FaceForensics (face crop extraction)
python src/scripts/preprocess_ffs.py
```

### Tests

```bash
# Run the test suite (minimal — only data utilities)
python -m pytest tests/
```

## Architecture

### Config System (Hydra)

All configuration is managed via [Hydra](https://hydra.cc/). The launcher `src/infra/launch.py` assembles configs from `configs/` and spawns `src/train.py` via `torch.distributed`.

Key config groups:
- `configs/model/stylegan-v.yaml` — generator/discriminator architecture, `z_dim`, motion params, time encoding
- `configs/dataset/*.yaml` — dataset path, resolution, FPS per dataset
- `configs/training/base.yaml` — batch size, training duration (`kimg`), augmentation, metrics schedule
- `configs/sampling/*.yaml` — frame sampling strategy (random vs. uniform)
- `configs/infra.yaml` — number of GPUs, SLURM sbatch args, experiment naming
- `configs/env/*.yaml` — machine-specific paths (CUDA toolkit, dataset roots)

Experiment output directories are named by git hash + `exp_suffix`, which is how `training.resume=latest` locates prior runs.

### Core Model Components

- **`src/training/networks.py`** — `Generator` and `Discriminator`. The generator uses modulated Conv2d layers (from StyleGAN2). The discriminator concatenates frames at 16×16 resolution, dividing channel dim by the number of frames.
- **`src/training/motion.py`** — `MotionMappingNetwork`: Conv1d-based network that maps random noise + Fourier time embeddings to per-frame motion latents (`w_motion`). Replaces LSTMs to allow arbitrary-length generation.
- **`src/training/layers.py`** — StyleGAN building blocks: `SynthesisLayer`, `ToRGBLayer`, `MappingNetwork`, `FullyConnectedLayer`.
- **`src/training/dataset.py`** — `VideoFramesFolderDataset`: loads frame-wise zip archives or directories, returns sparse frame samples per video.
- **`src/training/training_loop.py`** — Main training loop: ADA augmentation scheduling, periodic metric computation, checkpoint saving.
- **`src/training/loss.py`** — Adversarial losses (R1 regularization, path length regularization).
- **`src/training/mocogan.py`** — MoCoGAN+SG2 baseline model.

### Metrics

Re-implemented in PyTorch (TorchScript), faster than originals:
- **FVD** (`src/metrics/frechet_video_distance.py`) — I3D model pretrained on Kinetics-400
- **FID** (`src/metrics/frechet_inception_distance.py`) — standard Inception-v3
- **Video IS** (`src/metrics/video_inception_score.py`) — C3D model finetuned on UCF101

### Dataset Format

```
dataset/
    video1/
        frame1.jpg
        frame2.jpg
        ...
    video2/
        ...
```

Packed as `.zip` by default (faster cluster I/O). Train from directory by removing `.zip` from `dataset.path` in the dataset config. Custom datasets: create `configs/dataset/my_dataset.yaml` following `configs/dataset/ffs.yaml`.

### Key Design Parameters

- `model.generator.time_enc.min_period_len` — controls temporal frequency of motion; increase for slower/smoother motions (e.g., 256 for SkyTimelapse vs. 64 for FaceForensics)
- `model.generator.motion.motion_z_distance` — companion to `min_period_len`; both together most influence motion quality
- `training.resume=latest` — resumes optimizer state, logging, and step count (requires matching git hash + `exp_suffix`)
- Pre-trained checkpoints: [Yandex.Disk](https://disk.yandex.ru/d/v7MS7zu4mmZxXw)

### Motion Generation Strategy (`model.generator.motion.gen_strategy`)

Four options, all implemented in [src/training/motion.py](src/training/motion.py):

| Strategy | Description |
|---|---|
| `conv` (default) | Two stacked `EqLRConv1d` layers, `padding=0`. Fast and stable but only sees a fixed local window (`kernel_size`). Needs `num_additional_codes = (kernel_size-1)*2` warmup inputs. |
| `autoregressive` | LSTM. Can diverge for large `t`. |
| `ltc` | **Liquid Time-Constant network.** Recurrent cell whose time constant τ adapts to the input, solving `dx/dt = -x/τ(x,u) + f(x,u)` with `ode_unfolds` Euler sub-steps. Hidden state persists across the full trajectory, eliminating window-boundary artefacts and naturally producing smooth motion. |
| `cfc` | **Closed-form Continuous-time network** (Hasani et al., Nature MI 2022). Same continuous-time inductive bias as LTC, but each step is a single analytic time-gated update — `x(t+Δt) = σ(-τ·Δt)⊙g + (1-σ(-τ·Δt))⊙h` — with no Euler `ode_unfolds` loop, so it is faster while keeping smooth, coherent motion. |

To train with LTC:

```bash
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=ltc \
  model.generator.motion.ltc_hidden_dim=512 \
  model.generator.motion.ltc_ode_unfolds=6
```

`ltc_ode_unfolds` controls how many Euler sub-steps are used per `motion_z_distance` interval; higher values are more accurate but slower. `ltc_hidden_dim` defaults to `motion.z_dim` if omitted.

To train with CfC:

```bash
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=cfc \
  model.generator.motion.cfc_hidden_dim=512
```

CfC needs no `ode_unfolds` (it is closed-form in Δt). `cfc_hidden_dim` defaults to `motion.z_dim` if omitted.
