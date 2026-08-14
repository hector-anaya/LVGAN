# Experiment Report — Continuous-Time Motion Generators for StyleGAN-V

**Status:** protocol / pre-registration (no results yet)
**Date:** 2026-08-11
**Codebase:** `stylegan-v` @ branch `master`
**Owner:** hector.96.uno@gmail.com

---

## 1. Objective and hypothesis

StyleGAN-V generates the motion latent trajectory with a 2-layer `Conv1d` network
([src/training/motion.py:359-367](src/training/motion.py#L359-L367)). That network has a **fixed
receptive field** (`kernel_size=11`, `padding=0`), so every motion code only sees a local window of
the noise trajectory and the model has no persistent state across the video.

This repo adds four continuous-time alternatives
([src/training/motion.py](src/training/motion.py)):

| `gen_strategy` | Implementation | Where |
|---|---|---|
| `conv` | 2× `EqLRConv1d`, fixed window (baseline) | [motion.py:359](src/training/motion.py#L359) |
| `autoregressive` | LSTM (baseline) | [motion.py:351](src/training/motion.py#L351) |
| `ltc` | Liquid Time-Constant cell, `ode_unfolds` Euler sub-steps | [motion.py:47-118](src/training/motion.py#L47-L118) |
| `cfc` | Closed-form Continuous-time cell, analytic Δt gate | [motion.py:141-204](src/training/motion.py#L141-L204) |
| `ncps_ltc` | `ncps` reference LTC (+ optional NCP wiring) | [motion.py:229](src/training/motion.py#L229) |
| `ncps_cfc` | `ncps` reference CfC | [motion.py:293](src/training/motion.py#L293) |

**H1 (quality).** Continuous-time motion generators (`ltc`, `cfc`) match or beat `conv` on short-horizon
FVD (`fvd2048_16f`) at equal image quality (`fid50k_full`).

**H2 (long horizon).** Because the hidden state persists over the whole trajectory, `ltc`/`cfc` degrade
less at long horizons: the gap in `fvd2048_128f` and `fvd2048_128f_subsample8f` should be larger than
the gap in `fvd2048_16f`. This is the main claim to defend.

**H3 (cost).** `cfc` reaches `ltc` quality at lower cost, since it removes the inner `ode_unfolds`
loop. Measured in s/kimg and peak VRAM.

**H4 (robustness).** The advantage is larger on datasets with slow / long-range motion
(SkyTimelapse, TaiChi) than on datasets whose motion is essentially local (FaceForensics).

---

## 2. What "3 times per dataset" means here

Two different variances exist, and they are not interchangeable:

| Repetition | How | Cost | What it measures |
|---|---|---|---|
| **Training seeds** (`training.seed=0,1,2`) | 3 independent trainings per cell | 3× full training | Run-to-run variance — this is the one reviewers ask for |
| **Evaluation repeats** (`--num_runs 3` in [calc_metrics.py:111](src/scripts/calc_metrics.py#L111)) | 3 metric passes on the same checkpoint | minutes | Sampling noise of FVD/FID only |

**Plan: do both.** 3 training seeds per (dataset × strategy) cell, and `--num_runs 3` on the selected
checkpoint of each run. Report `mean ± std` over the 3 seeds (evaluation repeats are averaged first,
per run). If the compute budget in §7 does not allow 3 seeds everywhere, drop to 1 seed on Tier-B
datasets and say so explicitly in the paper — do **not** silently report `--num_runs 3` as if it were
run-to-run variance.

---

## 3. Prerequisites before launching anything

1. **Datasets** must be in frame-folder-zip format (`dataset/video1/frame1.jpg ...`, packed as
   `.zip`) at `${env.datasets_dir}/<dataset.name>.zip` and/or `data/<dataset.name>.zip`.
   - `ffs_256_unstable.zip` — via `src/scripts/preprocess_ffs.py`
   - `sky_timelapse_256.zip`
   - `ucf101_256.zip` — `preprocess_ucf101.py` (untracked, in repo root)
   - `taichi_128.zip` — **not yet prepared**; TaiChi-HD needs to be downloaded and cropped
2. **`configs/dataset/taichi.yaml`** — did not exist; created in this change set. Verify `fps` and
   `resolution` against the frames you actually extracted.
3. **Commit the code first.** The experiment directory is named
   `${dataset.name}_${model.name}_${sampling.name}_${exp_suffix}-${git_hash}`
   ([configs/infra.yaml:10](configs/infra.yaml#L10)). A dirty tree makes `create_project_dir` complain
   (`ignore_uncommited_changes=true` to bypass), and **`training.resume=latest` only finds a run whose
   git hash and `exp_suffix` match**. Every run in the comparison must come from the same commit.
4. **Set `env=local` paths** — `configs/env/local.yaml` currently points `python_bin` at
   `${env.project_path}/env/bin/python`, i.e. a POSIX conda env inside the repo.
5. The default `gen_strategy` in [configs/model/stylegan-v.yaml:20](configs/model/stylegan-v.yaml#L20)
   is **`ltc`**, not `conv`. The baseline therefore needs an explicit
   `model.generator.motion.gen_strategy=conv` override — otherwise the "baseline" silently trains LTC.

---

## 4. Parameters to modify

### 4.1 Per-dataset parameters

These are the only knobs that should differ **between datasets**. Everything else is held fixed
across the whole study.

| Parameter | FaceForensics | SkyTimelapse | UCF101 | TaiChi |
|---|---|---|---|---|
| `dataset=` | `ffs` | `sky_timelapse` | `ucf101` | `taichi` |
| `dataset.resolution` | `256` | `256` | `256` | `128` |
| `dataset.fps` (config) | 30 | 25 | 25 | 25 |
| `model.generator.time_enc.min_period_len` | `16` | `256` | `32` | `32` |
| `training.mirror` | `true` | `true` | `true` | `true` |
| `training.aug` / `training.target` | `ada` / `0.6` | `ada` / `0.6` | `ada` / `0.6` | `ada` / `0.6` |
| `dataset.c_dim` | 0 | 0 | 0 (keep unconditional) | 0 |
| `training.kimg` | 25000 | 25000 | 25000 | 25000 |

Notes on the non-obvious ones:

- **`min_period_len` is the single most important per-dataset knob.** It is the shortest period of
  the Fourier time embedding *and*, through
  [configs/model/stylegan-v.yaml:17](configs/model/stylegan-v.yaml#L17), it *is*
  `motion.motion_z_distance` (the spacing between motion codes). Bigger value ⇒ slower, smoother
  motion. `16` (FFS) and `256` (Sky) are the values used by the original paper; `32` for UCF101 and
  TaiChi is a starting point and **must be confirmed by the pilot sweep in §6.5** before the main
  runs — those two datasets were not tuned upstream.
- **`min_period_len` also sets the LTC/CfC unroll length.** `max_traj_len ≈ max_num_frames /
  motion_z_distance + 2` ([motion.py:434-437](src/training/motion.py#L434-L437)). With
  `max_num_frames=1024`: `min_period_len=16` ⇒ ~66 sequential cell steps per forward pass (×
  `ode_unfolds=6` ⇒ ~396 sequential FC evaluations for LTC), while `min_period_len=256` ⇒ ~6 steps.
  FaceForensics is therefore the **most expensive** LTC configuration, not the cheapest. Budget
  accordingly and log s/kimg per dataset.
- **UCF101 conditioning** is left off (`c_dim: 101` is commented out in
  [configs/dataset/ucf101.yaml](configs/dataset/ucf101.yaml)) so all four datasets use the same
  unconditional setup. Class-conditional UCF101 would change the motion network input dim
  ([motion.py:375](src/training/motion.py#L375)) and is out of scope.
- **`dataset.max_num_frames: 1024`** stays at its default everywhere; it defines the temporal support
  of the metrics and of the trajectory length.

### 4.2 Parameters that change per experimental condition

| Parameter | Values | Purpose |
|---|---|---|
| `model.generator.motion.gen_strategy` | `conv` / `autoregressive` / `ltc` / `cfc` / `ncps_ltc` / `ncps_cfc` | The condition under test |
| `model.generator.motion.ltc_ode_unfolds` | `6` (main), `{1,3,6,12}` (ablation) | Euler sub-steps per interval |
| `model.generator.motion.ltc_hidden_dim` | `512` (main), `{128,256,512}` (ablation) | LTC capacity |
| `model.generator.motion.cfc_hidden_dim` | `512` (main), `{128,256,512}` (ablation) | CfC capacity |
| `training.seed` | `0`, `1`, `2` | Repetitions |
| `exp_suffix` | e.g. `ltc_s0` | Names the output dir — **must encode strategy + seed** |

### 4.3 Parameters held fixed (do not touch)

`sampling=random`, `sampling.num_frames_per_video=3`, `sampling.max_dist=32`, `training.batch_size=64`,
`num_gpus=4`, `training.cfg=auto`, `training.snap=200`, `model.generator.z_dim=512`,
`model.generator.w_dim=512`, `motion.z_dim=512`, `motion.v_dim=512`, `motion.kernel_size=11`,
`motion.use_fractional_t=true`, `motion.fourier=true`, `time_enc.dim=256`,
`time_enc.max_period_len=1024`, `time_enc.phase_dropout_std=1.0`, `discriminator.concat_res=16`,
`discriminator.num_frames_div_factor=2`, `discriminator.dummy_c=false`, `loss_kwargs.*`.

If `num_gpus` must change (fewer GPUs available), change it for **all** runs and keep
`training.batch_size` divisible by `num_gpus`; a mixed setup invalidates the comparison.

---

## 5. Main experiment

### 5.1 Matrix

4 datasets × 3 strategies (`conv` baseline, `ltc`, `cfc`) × 3 seeds = **36 training runs**.
`autoregressive`, `ncps_ltc`, `ncps_cfc` are covered in the ablation (§6), 1 seed, one dataset.

### 5.2 Launch commands

```bash
# ---- baseline (conv) ----------------------------------------------------
python src/infra/launch.py hydra.run.dir=. env=local num_gpus=4 \
  dataset=ffs dataset.resolution=256 \
  model.generator.time_enc.min_period_len=16 \
  model.generator.motion.gen_strategy=conv \
  training.seed=0 exp_suffix=conv_s0

# ---- LTC ----------------------------------------------------------------
python src/infra/launch.py hydra.run.dir=. env=local num_gpus=4 \
  dataset=ffs dataset.resolution=256 \
  model.generator.time_enc.min_period_len=16 \
  model.generator.motion.gen_strategy=ltc \
  model.generator.motion.ltc_hidden_dim=512 \
  model.generator.motion.ltc_ode_unfolds=6 \
  training.seed=0 exp_suffix=ltc_s0

# ---- CfC ----------------------------------------------------------------
python src/infra/launch.py hydra.run.dir=. env=local num_gpus=4 \
  dataset=ffs dataset.resolution=256 \
  model.generator.time_enc.min_period_len=16 \
  model.generator.motion.gen_strategy=cfc \
  model.generator.motion.cfc_hidden_dim=512 \
  training.seed=0 exp_suffix=cfc_s0
```

Full sweep (add `slurm=true` to submit instead of running locally):

```bash
for ds in "ffs 256 16" "sky_timelapse 256 256" "ucf101 256 32" "taichi 128 32"; do
  set -- $ds; name=$1; res=$2; mpl=$3
  for strat in conv ltc cfc; do
    for seed in 0 1 2; do
      python src/infra/launch.py hydra.run.dir=. env=local num_gpus=4 \
        hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
        dataset=$name dataset.resolution=$res \
        model.generator.time_enc.min_period_len=$mpl \
        model.generator.motion.gen_strategy=$strat \
        training.seed=$seed exp_suffix=${strat}_s${seed}
    done
  done
done
```

### 5.3 Evaluation protocol

1. Training already logs `fvd2048_16f`, `fvd2048_128f`, `fvd2048_128f_subsample8f`, `fid50k_full`
   every `snap=200` kimg ([configs/training/base.yaml:8](configs/training/base.yaml#L8)).
2. **Checkpoint selection:** best `fvd2048_16f`, chosen automatically by passing `--networks_dir`
   ([calc_metrics.py:152-159](src/scripts/calc_metrics.py#L152-L159)). Same rule for every run.
3. **Final numbers**, 3 evaluation repeats, cache disabled:

```bash
python src/scripts/calc_metrics.py \
  --networks_dir experiments/<EXPERIMENT_DIR>/output \
  --metrics fvd2048_16f,fvd2048_128f,fvd2048_128f_subsample8f,fid50k_full \
  --gpus 4 --num_runs 3 --use_cache 0 --verbose 0
```

4. **Qualitative / long-horizon** (same `--seed` for every strategy so content is comparable):

```bash
python src/scripts/generate.py \
  --networks_dir experiments/<EXPERIMENT_DIR>/output \
  --num_videos 25 --as_grids true --save_as_mp4 true \
  --fps 25 --video_len 1024 --batch_size 25 --seed 42 \
  --outdir results/<EXPERIMENT_DIR>_long --truncation_psi 0.9
```

`--video_len 1024` (≫ the 128 frames used by the metrics) is the qualitative counterpart of **H2**:
it is where a fixed conv window is expected to show looping / drift and where the recurrent state
should not.

5. **Content/motion decomposition grid** (`--moco_decomposition 1`) for one checkpoint per dataset,
   to show that the LTC/CfC motion code stays disentangled from content.
6. **Cost** — record from the training log: s/kimg, peak VRAM, total wall clock, and generator
   parameter count per strategy.

### 5.4 Results table (to fill)

Report `mean ± std` over seeds 0/1/2. Lower is better for all four metrics.

**FaceForensics 256² (`min_period_len=16`)**

| Strategy | FVD16 ↓ | FVD128 ↓ | FVD128-sub8 ↓ | FID50k ↓ | s/kimg | Params |
|---|---|---|---|---|---|---|
| `conv` (baseline) | | | | | | |
| `ltc` | | | | | | |
| `cfc` | | | | | | |

**SkyTimelapse 256² (`min_period_len=256`)** — same columns
**UCF101 256² (`min_period_len=32`)** — same columns
**TaiChi 128² (`min_period_len=32`)** — same columns

Add a derived column **ΔFVD128 − ΔFVD16** (improvement over baseline at long vs short horizon); that
number, not raw FVD, is the direct test of H2.

---

## 6. Ablation study

All ablations run on **FaceForensics 256²**, **1 seed (`training.seed=0`)**, and a **shortened budget
of `training.kimg=5000`** (~1/5 of a full run). Every ablation is compared against an
`ltc`/`cfc` reference trained under the identical shortened budget — never against a 25000-kimg number.
Reported metric: `fvd2048_16f` + `fvd2048_128f` + `fid50k_full`, `--num_runs 3`.

Ablation A is the one that must be in the paper; B–F are supporting.

### 6.1 A — Motion generator family (6 runs)

`model.generator.motion.gen_strategy ∈ {conv, autoregressive, ltc, cfc, ncps_ltc, ncps_cfc}`

Isolates "continuous-time recurrence" from "recurrence" (LSTM) and from "local convolution", and
checks our LTC/CfC implementations against the reference `ncps` ones. `ncps_*` needs `pip install ncps`
([motion.py:234](src/training/motion.py#L234)); `ncps_ltc` additionally exposes
`ncps_ltc_wiring ∈ {full, ncp}` ([motion.py:404](src/training/motion.py#L404)).

### 6.2 B — ODE solver granularity (4 runs)

`model.generator.motion.ltc_ode_unfolds ∈ {1, 3, 6, 12}` with `gen_strategy=ltc`.

Quality vs cost of the Euler discretisation. `ode_unfolds=1` degenerates to a plain gated RNN step and
is the natural bridge to `cfc` — if `cfc` ≈ `ltc@6` at the cost of `ltc@1`, H3 holds.

### 6.3 C — Capacity (4 runs)

`ltc_hidden_dim ∈ {128, 256, 512}` and `cfc_hidden_dim = 512` at matched parameter count.

Rules out "LTC is better only because it has more parameters than the 2-layer conv".

### 6.4 D — Time conditioning (4 runs)

| Run | Override |
|---|---|
| no Fourier time embedding | `model.generator.motion.fourier=false` |
| no phase dropout | `model.generator.time_enc.phase_dropout_std=0.0` |
| integer time positions | `model.generator.motion.use_fractional_t=false` |
| no time conditioning in D | `model.discriminator.dummy_c=true` |

Tests whether the continuous-time cell *replaces* part of the hand-designed positional machinery. If
`ltc` survives `fourier=false` much better than `conv` does, that is a strong, cheap result.

### 6.5 E — Temporal scale sweep (4 runs per dataset, also serves as the §4.1 pilot)

`model.generator.time_enc.min_period_len ∈ {16, 32, 64, 256}` (this moves `motion_z_distance` with it).

Run this **first** on UCF101 and TaiChi, at `kimg=5000`, to fix their `min_period_len` before the main
runs. On FFS it doubles as the sensitivity analysis: the claim is that `ltc`/`cfc` are *less* sensitive
to this hyper-parameter than `conv`, because the recurrent state is not tied to a fixed window.

### 6.6 F — Frame sampling (3 runs)

`sampling.num_frames_per_video ∈ {2, 3, 4}` and `sampling.max_dist ∈ {32, 128}`.

Checks whether the LTC advantage depends on how far apart the discriminator's frames are sampled —
i.e. whether it survives when the discriminator can no longer see long-range inconsistency.

### 6.7 Ablation results table (to fill)

| Ablation | Setting | FVD16 ↓ | FVD128 ↓ | FID50k ↓ | s/kimg |
|---|---|---|---|---|---|
| — | `ltc` reference @5000 kimg | | | | |
| A | `conv` | | | | |
| A | `autoregressive` | | | | |
| A | `cfc` | | | | |
| A | `ncps_ltc` | | | | |
| A | `ncps_cfc` | | | | |
| B | `ode_unfolds=1 / 3 / 12` | | | | |
| C | `hidden_dim=128 / 256` | | | | |
| D | `fourier=false` | | | | |
| D | `phase_dropout_std=0` | | | | |
| D | `use_fractional_t=false` | | | | |
| D | `dummy_c=true` | | | | |
| E | `min_period_len=32 / 64 / 256` | | | | |
| F | `num_frames_per_video=2 / 4` | | | | |

---

## 7. Compute budget

Reference point from the README: FaceForensics 256², 4× GPU, 25000 kimg ≈ **2 days**.

| Block | Runs | Days/run (4 GPUs) | GPU-days |
|---|---|---|---|
| Main, 4 datasets × 3 strategies × 3 seeds | 36 | ~2 (TaiChi 128² ≈ 0.6) | ~250 |
| Ablation A–D, F @5000 kimg | ~21 | ~0.4 | ~34 |
| Ablation E pilots (3 datasets × 4) | 12 | ~0.4 | ~19 |
| **Total** | **69** | | **≈ 300 GPU-days** |

That is the honest number for the full plan. Two reductions, in order of preference:

1. **Tier the seeds.** 3 seeds on FaceForensics + SkyTimelapse (the two datasets the paper's claim
   rests on), 1 seed on UCF101 + TaiChi. 36 → 20 runs, ~140 GPU-days. State it in the paper.
2. **Shorten the non-primary datasets** to `training.kimg=10000`, keeping the budget identical across
   strategies within a dataset. Cross-dataset numbers then stop being comparable to published FVDs —
   only within-dataset comparisons remain valid.

LTC on FaceForensics (`min_period_len=16`, 66 unroll steps × 6 unfolds) is the slowest cell in the
matrix; measure one `conv` and one `ltc` run for 200 kimg first and scale the table above from the
measured s/kimg rather than from this estimate.

---

## 8. Bookkeeping

- One commit for the whole study; record the short hash here: `__________`.
- `exp_suffix` convention: `<strategy>_s<seed>` for main runs, `abl<letter>_<setting>` for ablations.
  Full dir name = `<dataset.name>_stylegan-v_random3_max32_<exp_suffix>-<git_hash>`.
- Per run, archive: `experiment_config.yaml`, `training_cmd.sh`, `output/metric-*.jsonl`,
  `output/log.txt`, and the selected `network-snapshot-*.pkl`.
- Aggregate all `metric-*.jsonl` into one CSV (`run_id, dataset, strategy, seed, kimg, metric, value`)
  before making any table or plot.
- Interrupted runs: relaunch the identical command with `training.resume=latest` — it matches on git
  hash + `exp_suffix`, so both must be unchanged.

## 9. Risks / threats to validity

| Risk | Mitigation |
|---|---|
| Default `gen_strategy` is `ltc`, so a forgotten override yields a fake "baseline" | Always pass `gen_strategy` explicitly; verify in the saved `experiment_config.yaml` |
| Parameter count differs between `conv` and `ltc`/`cfc` | Ablation C (matched capacity) |
| `min_period_len` untuned for UCF101 / TaiChi | Ablation E run as a pilot, per strategy, before the main runs |
| FVD checkpoint selection on `fvd2048_16f` may favour short-horizon models | Also report the metric at the final checkpoint, not only the best one |
| Long-horizon claim rests on 128-frame FVD only | Add the 1024-frame qualitative grids and a temporal-consistency plot (per-frame LPIPS to frame 0 vs t) |
| TaiChi at 128² is not comparable to the 256² datasets | Report it in a separate table; never average FVD across resolutions |
| `ncps` unavailable in the env | `ncps_*` conditions are optional; drop them and note it |
