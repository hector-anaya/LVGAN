# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StyleGAN-V (CVPR 2022) is a continuous video generator that extends StyleGAN2. It decomposes video generation into **content** (image quality, from a static latent `z`) and **motion** (temporal dynamics, from a Conv1d-based motion mapping network using Fourier positional embeddings). This enables unbounded-length, high-quality video synthesis.

## Environment Setup

### The pins that actually work — do not guess, do not "modernize"

Every completed run used this exact set. It is the reference for any new machine:

```
python 3.9   torch 1.9.0+cu111   torchvision 0.10.0+cu111   numpy 1.22.4
hydra-core 1.0.7   omegaconf 2.0.6   ncps 1.0.1
```

**Never substitute a newer torch.** [conv2d_gradfix.py:53](src/torch_utils/ops/conv2d_gradfix.py#L53)
gates the custom op on `['1.7.','1.8.','1.9','1.10']`. On torch 1.9 it is *enabled*; on
2.x it silently falls back to `F.conv2d`, changing the kernels used for the R1
double-backward. Results stop being comparable to existing runs and nothing warns you
beyond one `warnings.warn`. `environment-ampere.yaml` and `environment.yaml` are
upstream leftovers and do not reproduce this env.

### Per-machine setup scripts

- **Huapáctic (LNS-BUAP)** — [src/infra/setup_huapactic.sh](src/infra/setup_huapactic.sh),
  then [src/infra/smoke_test_huapactic.sh](src/infra/smoke_test_huapactic.sh) on a `gpu2`
  node. Config: [configs/env/huapactic.yaml](configs/env/huapactic.yaml).
  `gpu2` = `ampere[01-05]`, 4×A100-80GB each, 96 CPU, 1TB RAM, 15-day walltime.
  Launch with `env=huapactic sbatch_args.partition=gpu2` (without the partition it lands
  on `normal*`, which has no GPUs).
  **CUDA and gcc come from `module load CUDA/11.1.1-GCC-10.2.0`** — an exact match for
  `cu111`, EasyBuild-paired with the gcc 10.2.0 that CUDA 11.x requires. Take CUDA from
  the module, never from conda: conda's `cuda-libraries-dev=11.8.0` is an old-style
  metapackage whose `*_linux-64` dependencies track `cuda-version` and silently resolve
  to CUDA 13, producing nvcc 11.8 + cudart 13.3 + libcublas 13.6 in one prefix. conda
  also lays CUDA out as `targets/x86_64-linux/{include,lib}` rather than the
  `include/` + `lib64/` that torch's `cpp_extension` probes, which is what produces
  `No CUDA runtime is found`. Likewise do not install `gcc_linux-64` — its `activate.d`
  scripts export `CC`/`CXX` on every activation and would override the module's gcc.
- **gpuserver02 (INAOE)** — [configs/env/local.yaml](configs/env/local.yaml). Carries a
  dead-GPU guard for a specific UUID; do not copy that to other clusters.

### Verifying an environment: check correctness, not existence

Six setup failures on Huapáctic all came from checks that confirmed a thing was *present*
rather than *right*. When writing or reviewing an env check:

- **Pinned packages** — compare `importlib.metadata.version(...)` against the pin.
  `import ncps` succeeded at 0.0.2 while the pin said 1.0.1, so the fixing install was skipped.
- **`ncps` specifically** — [motion.py:22-30](src/training/motion.py#L22-L30) wraps the import
  in `try/except ImportError`, so a wrong version leaves `_NCPS_AVAILABLE = False` and
  `motion.py` still imports cleanly. `gen_strategy=ncps_ltc|ncps_cfc` then fails at model
  construction, after the job is queued. Assert `_NCPS_AVAILABLE`, not the import.
- **CUDA headers** — do not size the toolkit by grepping the ops' own `#include`s. They pull
  torch headers that reach further: `bias_act.cu` → `c10/util/complex.h` → `<thrust/complex.h>`,
  and the `.cpp` files → `ATen/cuda/CUDAContext.h` → `<cusparse.h>`, `<cublas_v2.h>`. Check for
  those header files under `$CUDA_HOME/include`, not for the presence of `nvcc`.
- **torch↔numpy** — `import torch` only *warns* when the NumPy bridge is dead. Test
  functionally with `torch.from_numpy(...)`. torch 1.9 requires `numpy<2`.
- **CUDA ops** — compiling is not working. The GPU smoke test must also exercise a
  double-backward through `conv2d_gradfix.conv2d`, the path R1/PL regularization uses.

### Known packaging traps (2022 pins vs. a current index)

| Trap | Why |
|---|---|
| `pip<24.1` | omegaconf 2.0.6 declares `PyYAML (>=5.1.*)`; pip ≥24.1 rejects that metadata, making hydra 1.0.7 uninstallable. Upgrading hydra is a config migration — the configs use hydra 1.0 `# @package _group_` semantics. |
| `setuptools<81` | setuptools ≥81 dropped `pkg_resources`, which torch's `cpp_extension` imports at module scope. Breaks *every* custom-op import. |
| `moviepy<2` | moviepy 2.0 removed `moviepy.editor`, imported by `convert_video_to_dataset.py`. |
| `numpy<2` via a **constraints file** | Per-install pins ping-pong: opencv (4.10+ and 5.x) declares `numpy>=2` and will drag it back. Use `PIP_CONSTRAINT` + `env/pip.conf` so every resolution honours it. |
| `psutil`, `pandas` | Imported by `training_loop.py` and `preprocess_ffs.py`; missing from upstream's yaml. |
| drop `av` | Pinned upstream but never imported here (mp4 goes through `cv2.VideoWriter`). Its sdist needs ffmpeg dev headers and root. |
| no `set -u` in setup scripts | conda's compiler `activate.d`/`deactivate.d` scripts read `$ADDR2LINE`, `$CONDA_BACKUP_CXX` by design. The conda shell function re-sources them after *every* `install`/`update`/`remove`, not just `activate`. |

### JIT-compiled CUDA ops

```bash
export TORCH_CUDA_ARCH_LIST="8.0"   # A100 (sm_80); "7.0" for V100
```

A job killed mid-build (SLURM time limit, Ctrl-C) leaves a `FileBaton` lock in
`TORCH_EXTENSIONS_DIR`. [custom_ops.py:105](src/torch_utils/custom_ops.py#L105) and torch's
own `_jit_compile` then `baton.wait()` **forever** on a lock nobody holds — the retry hangs
instead of rebuilding. Delete stale `lock` files before re-running. `custom_ops.verbosity`
defaults to `'brief'`, which passes `verbose=False` to `cpp_extension.load()`, so a slow
build and a deadlock look identical; set it to `'full'` when diagnosing.

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

Six options, all implemented in [src/training/motion.py](src/training/motion.py). The
last four form a controlled ablation whose only intended variable is the recurrent
core, so they share equalized-LR treatment (see *Learning-rate parity* below):

| Strategy | Description |
|---|---|
| `conv` (default) | Two stacked `EqLRConv1d` layers, `padding=0`. Fast and stable but only sees a fixed local window (`kernel_size`). Needs `num_additional_codes = (kernel_size-1)*2` warmup inputs. |
| `autoregressive` | LSTM. Can diverge for large `t`. |
| `exp_cf` | **Exponential closed-form cell** (`ExpClosedFormCell`). Input-dependent time constants with an exponential gate: `x(t+Δt) = x·exp(-Δt/τ) + f·(1-exp(-Δt/τ))`. Inspired by LTC (Hasani 2021) and adopting CfC's closed-form strategy (Hasani 2022), but identical to neither — it is **not** a solver-based conductance model, and `ode_unfolds` here just subdivides Δt into successive closed-form steps. `ltc` remains a deprecated alias. |
| `cfc` | **Closed-form Continuous-time network** (Hasani et al., Nature MI 2022). Same idea as `exp_cf` but the gate is a sigmoid: `x(t+Δt) = σ(-τ·Δt)⊙g + (1-σ(-τ·Δt))⊙h`. A single closed-form update per step, avoiding the inner ODE loop and its accumulated discretisation error — not an exact solution of the liquid ODE. |
| `ncps_ltc` | Reference LTC from the `ncps` library — the genuine solver-based conductance model, semi-implicit Euler. **Must stay fp32**: its update divides by `cm/Δt + gleak + w_act`, which is not bounded away from zero. Memory scales as `B · units² · T · ode_unfolds`; set `ncps_ltc_use_checkpoint=true` to trade compute for it. |
| `ncps_cfc` | Reference CfC from the `ncps` library. Supports `mode` (`default` / `pure` / `no_gate`) and, unlike `ncps_ltc`, a per-sample `timespans` tensor, so non-unit Δt works. |

To train with the exponential closed-form cell:

```bash
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=exp_cf \
  model.generator.motion.ltc_hidden_dim=512 \
  model.generator.motion.ltc_ode_unfolds=6
```

`ltc_ode_unfolds` controls how many closed-form sub-steps are taken per
`motion_z_distance` interval; higher values are smoother but slower. `ltc_hidden_dim`
defaults to `motion.z_dim` if omitted. The config keys keep the historical `ltc_`
prefix (as does the `self.ltc` submodule attribute) so existing runs still resume.

### Learning-rate parity across the ablation

Every `FullyConnectedLayer`/`EqLRConv1d` here uses equalized LR: the raw parameter is
stored `1/lr_multiplier` larger and scaled back by `lr_multiplier/sqrt(fan_in)` in the
forward pass. The `ncps` cores are plain `nn.Parameter`s. Because Adam's step is
invariant to gradient scale, the ncps arms would otherwise train orders of magnitude
faster, and any FVD gap would measure the learning rate rather than the cell.
`construct_optimizer_params` in
[src/training/training_loop.py](src/training/training_loop.py) therefore puts the ncps
cores in their own param group at `lr * lr_multiplier`. This makes effective LRs
comparable, **not** identical — equalized LR also rescales the init, and a
`1/sqrt(fan_in)` residual remains.

The related `log_tau_0` asymmetry inside `ExpClosedFormCell`/`CfCCell` (a raw parameter
at full LR shadowing an equalized bias) is left in place by default but can be ablated
with `ltc_learn_tau_bias=false` / `cfc_learn_tau_bias=false`.

To train with CfC:

```bash
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=cfc \
  model.generator.motion.cfc_hidden_dim=512
```

CfC needs no `ode_unfolds` — each step is a single closed-form update rather than an
inner ODE loop. `cfc_hidden_dim` defaults to `motion.z_dim` if omitted.

To train with the reference `ncps` implementations (requires **`ncps==1.0.1`** — the
version all completed runs used; earlier docs said 0.0.2, which is wrong and has an
incompatible API):

```bash
# ncps CfC — backbone defaults to hidden_dim, not ncps' 128, to keep capacity comparable
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=ncps_cfc \
  model.generator.motion.ncps_cfc_hidden_dim=512

# ncps LTC — fp32 only; checkpointing cuts activation memory
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_exp env=local dataset=ffs \
  dataset.resolution=256 num_gpus=4 \
  model.generator.motion.gen_strategy=ncps_ltc \
  model.generator.motion.ncps_ltc_hidden_dim=128 \
  model.generator.motion.ncps_ltc_use_checkpoint=true \
  model.generator.motion.ncps_ltc_checkpoint_every=4
```

Note that `ncps_*_wiring=ncp` (sparse `AutoNCP`) reduces *parameter* count but not
activation memory — ncps stores the wiring as a dense adjacency mask, so masked
synapses are zeros rather than absent. Use `ncps_ltc_use_checkpoint` for memory.
