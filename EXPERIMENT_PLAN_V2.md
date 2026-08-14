# Experiment Plan V2 — Datasets, Baselines, Metrics, A100 Budget

**Status:** planning (supersedes §4.1/§5/§7 of [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md); §2, §3, §6, §8 of that
document still hold)
**Date:** 2026-08-13
**Trigger:** access to 80 GB A100 GPUs → scope can grow beyond the original 4-dataset / FVD-only plan.

---

## 0. What changed and why

The original protocol was: 4 datasets (FFS, SkyTimelapse, UCF101, TaiChi), 3 motion generators
(`conv`/`ltc`/`cfc`), FVD + FID. Three things make that insufficient now:

1. **The metric cannot see the claim.** Ge et al. (CVPR 2024) show FVD is biased toward per-frame
   content, and their headline case study is *literally StyleGAN-V with LSTM motion codes*: it produces
   visibly repeated temporal patterns, and I3D-FVD does not penalise it. Recomputing with VideoMAE-v2
   features restores agreement with humans. Our entire contribution is a change to the motion code
   generator — reporting only I3D-FVD hands a reviewer the argument that our metric is blind to our
   own thesis.
2. **The dataset suite does not span the axis we claim to improve.** FFS/UCF101 are local-motion
   datasets. The hypothesis (H2) is about long-range temporal dependency. Only SkyTimelapse tests
   that, and it tests it on nearly-static scenes.
3. **Prior art already owns "continuous-time motion + GAN."** MODE-GAN (arXiv 2112.10960) does
   unconditional video generation with a *neural-ODE* motion generator and appearance/motion
   decomposition. Novelty must therefore rest on the LTC/CfC-specific inductive bias (input-dependent
   τ, solver-free closed form) *measured against a plain Neural-ODE motion generator and against an
   SSM*, not on "continuous time" as such.

This document fixes all three.

---

## 1. Positioning (decide this first — it determines everything below)

**The paper is a controlled architectural study of the motion generator inside a fixed video-GAN
backbone. It is not a bid for video-generation SOTA.**

Current SOTA on this benchmark suite is diffusion (AR-Diffusion, CVPR 2025: FVD₁₆ 40.8 on
SkyTimelapse, 66.3 on TaiChi-HD, 186.6 on UCF-101). No GAN will approach that, and no 2025/2026
GAN successor to StyleGAN-V exists at CVPR/ICCV/NeurIPS. Trying to compete head-on loses. What is
defensible and currently unclaimed:

> Given a fixed generator/discriminator/loss, the *choice of latent motion-trajectory model* is a
> first-class design decision, and continuous-time recurrent cells (LTC/CfC) dominate the fixed-window
> convolution used by StyleGAN-V — increasingly so as the horizon grows — at equal image quality and
> lower cost than a Neural ODE.

Everything is then an apples-to-apples internal comparison, external numbers are context rather than
the claim, and the diffusion gap is a stated scope limit rather than a weakness. Write that sentence
into the intro before running anything; it decides which datasets and metrics are *required* versus
merely nice.

---

## 2. Datasets

### 2.1 Selection principle

Order datasets by **motion horizon** — how many frames of context are needed to predict the next
motion increment — not by size or popularity. That axis *is* H2. A suite that spans it makes the
paper's central plot (advantage vs. horizon) fall out of the dataset choice itself.

| Horizon | Regime | Dataset |
|---|---|---|
| Short (~10s of frames) | local, quasi-Markov facial motion | FaceForensics, CelebV-HQ |
| Medium | articulated body motion, periodic + aperiodic | TaiChi-HD, Jester |
| Long | smooth, monotone, globally-coherent trajectories | RealEstate10K (camera), SkyTimelapse (clouds) |
| Mixed / adversarial | in-the-wild, diverse, hard | UCF101, RainbowJelly |

### 2.2 Verdict on the four datasets you proposed

| Proposed | Verdict | Reasoning |
|---|---|---|
| **RealEstate10K** | **Adopt — Tier A** | The best of the four by a wide margin *for this paper*. ~65 K train / 7 K test YouTube walkthroughs with camera poses. Camera motion is smooth, continuous and coherent over hundreds of frames — exactly the regime where a persistent recurrent state should beat a fixed `kernel_size=11` window, and where `conv` should visibly loop. Already the de-facto long-horizon benchmark in the world-model literature (Geometry Forcing reports 16-frame *and* 256-frame FVD on it). Caveats: it is *camera* motion, not object motion — say so; YouTube link rot means you must record and publish your clip list. |
| **Kinetics-600** | **Drop as a main dataset** | ~500 K clips / 600 classes. It is a *conditional / large-model* benchmark: DVD-GAN needed enormous compute and still produced poor unconditional samples. A StyleGAN-V-scale unconditional GAN will produce mush, and a mush-vs-mush FVD comparison tests nothing about motion. Link rot also breaks reproducibility. Diversity is already covered by UCF101. *If* you want it, use it as a single 128² stress test on a 10–20 class motion-heavy subset, 1 seed, explicitly labelled as a capacity probe. |
| **ActivityNet** | **Drop** | ~20 K *untrimmed* videos, ~2 min average, with **shot cuts inside single videos**. A frame-sampling video discriminator will treat a hard cut as legitimate motion, which actively teaches the generator to produce discontinuities — the opposite of what we are measuring. Salvageable only with shot-boundary detection and re-segmentation, which is a project in itself for no benchmark payoff. |
| **VoxCeleb** | **Replace with CelebV-HQ** | Vox is a legitimate face-video benchmark (it is one of the four in the CelebV-HQ unconditional-generation table), but at 224² crops it is largely redundant with FaceForensics and adds no new horizon regime. **CelebV-HQ** dominates it: 35 666 clips / 15 653 identities / ~65 h at ≥512², richer motion, *and* it ships a published unconditional benchmark over VideoGPT, MoCoGAN-HD, DIGAN and **StyleGAN-V** — i.e. free external baseline numbers on the exact backbone we modify. |

### 2.3 Recommended suite

| # | Dataset | Res | Scale | Tier | Why it is in | External StyleGAN-V-protocol numbers exist? |
|---|---|---|---|---|---|---|
| 1 | **FaceForensics** (`ffs`) | 256 | 704 vids / 364 K frames | **A** | Short-horizon control. Expect *small* gains — that is a feature, it shows we are not just adding capacity. Everyone reports it. | Yes (MoCoGAN-HD, DIGAN, StyleGAN-V, MoStGAN-V, StyleInV, Latte, AR-Diffusion) |
| 2 | **RealEstate10K** | 256 | ~65 K clips | **A** | **The H2 dataset.** Long smooth camera trajectories. New to this benchmark family. | No — internal baselines only |
| 3 | **SkyTimelapse** | 256 | 2 114 vids / 1.17 M frames | **A** | Long-period, slow motion (`min_period_len=256`). Established. | Yes |
| 4 | **TaiChi-HD** | 128 | ~2.7–3.1 K vids | **A** | Articulated human motion; **trackable keypoints, so FVMD is meaningful here** (it is near-degenerate on Sky). Cheap at 128². | Yes |
| 5 | **UCF101** | 256 | 13 320 clips / 101 classes | B | Diversity + comparability. Hard for GANs (FVD ~10³); do not expect a win, report it anyway. | Yes |
| 6 | **RainbowJelly** | 256 | — | B | **Config already exists in this repo** (`configs/dataset/rainbow_jelly.yaml`). It is StyleGAN-V's *own* long-video dataset and MoStGAN-V reports it. Cheapest possible long-horizon evidence. | Yes (StyleGAN-V, MoStGAN-V) |
| 7 | **CelebV-HQ** | 256 | 35 666 clips | B | Replaces VoxCeleb. Published StyleGAN-V unconditional numbers. Larger/harder face data than FFS. | Yes (CelebV-HQ paper) |
| 8 | **Jester** *(or SSv2)* | 128 | 148 K clips / 27 classes | C | **Diagnostic.** Built so that classes differ *only* by motion, not appearance — no appearance shortcut. The cleanest possible test of H4. No generation baselines exist, so frame it as a new motion-centric probe, not a comparison. | No |

Also already configured and free to add if a face-heavy story is wanted: `mead` (`c_dim: 77`).
**DeeperForensics** (732 vids / 194 K frames, stabilised FFHQ crops, used by StyleInV) is the drop-in
alternative to CelebV-HQ if CelebV-HQ download proves painful.

### 2.4 Preprocessing work required (this is the schedule risk, not the training)

| Dataset | State | Work |
|---|---|---|
| `ffs_256_unstable` | script exists (`src/scripts/preprocess_ffs.py`) | run |
| `sky_timelapse_256` | standard | download + `convert_videos_to_frames.py` |
| `ucf101_256` | `preprocess_ucf101.py` in repo root (untracked) | **commit it**, run |
| `taichi_128` | config created, data **not prepared** | download TaiChi-HD, crop per FOMM protocol |
| `realestate10k_256` | **new** | YouTube download + per-clip frame extraction + 256 centre-crop; expect link rot → **freeze and publish the clip list**; drop clips <256 frames so the long-horizon protocol is valid |
| `rainbow_jelly_256` | config exists, data unknown | source from StyleGAN-V release |
| `celebvhq_256` | **new** | official downloader + 256 crop |
| `jester_128` | **new** | direct download (no link rot), resize 128 |

Everything lands in the existing format (`<root>/<video>/<frame>.jpg`, zipped) at
`${env.datasets_dir}/<dataset.name>.zip`. Budget **2–3 weeks of wall-clock** for RealEstate10K +
CelebV-HQ download/extraction and start it *now*, in parallel with Phase 1 below — it is the longest
lead-time item in the whole plan and it is not GPU-bound.

---

## 3. Baselines and related work

### 3.1 Internal baselines — this is where the paper is won

The external table is context. The *argument* is the motion-generator family sweep, at **matched
parameter count**, on a fixed backbone. Expand ablation A from 6 conditions to 9:

| # | `gen_strategy` | Status | Why it must be there |
|---|---|---|---|
| 1 | `conv` | implemented | StyleGAN-V default; the thing we replace |
| 2 | `autoregressive` (LSTM) | implemented | Isolates *recurrence* from *continuous time*. Bonus: this is the exact variant Ge et al. used to expose FVD's blindness — running it lets us **reproduce their finding on our own metric stack**, which is a strong, cheap credibility result |
| 3 | `gru` | **to implement** (~20 lines) | Cheaper recurrent control; guards "the win is just gating" |
| 4 | `ode_rnn` / Neural-ODE | **to implement** | **Mandatory.** MODE-GAN owns ODE-motion-generator + GAN. Without this row, the reviewer's question "is LTC better than a plain neural ODE, or is it just continuous time?" is unanswered and the paper is rejected |
| 5 | `s4` or `mamba` | **to implement** | **Mandatory in 2026.** A selective SSM is a linear-time-varying continuous-time model with a hardware-aware parallel scan — the obvious competitor, and strictly faster than our sequential cells. If LTC/CfC lose here, that is still a publishable, honest finding about *sequential* vs *parallel* continuous-time models |
| 6 | `liquid_s4` | optional | Liquid-S4 (Hasani et al., ICLR 2023) is the literal bridge between rows 4–5 and 7–8; the most direct "same family, parallelisable" comparison |
| 7 | `ltc` | implemented | ours |
| 8 | `cfc` | implemented | ours |
| 9 | `ncps_ltc` / `ncps_cfc` | implemented | reference-implementation check on our hand-rolled cells |

Rows 3–6 are the main *code* work in this plan (~1 week). They are worth more than another dataset.

### 3.2 External comparison table

Report these as published numbers under the **StyleGAN-V FVD protocol** (sample videos first, *then*
extract fixed-length clips — not the older "pre-chunk the dataset" protocol; the two are not
comparable). State 2048 samples explicitly.

*GAN era (direct lineage):* MoCoGAN, MoCoGAN-HD, DIGAN, **StyleGAN-V** (our backbone), **MoStGAN-V**
(CVPR 2023 — the closest competitor: it also improves StyleGAN-V's motion, via temporal motion-style
attention; it reports a *smaller FVD₁₂₈−FVD₁₆ gap* as its long-horizon evidence, i.e. it competes on
exactly our H2), **StyleInV** (ICCV 2023), **LongVideoGAN**, RAVEN, INR-V.

*Autoregressive / diffusion (context only):* VideoGPT, TATS, PVDM, LVDM, Latte, HVDM, FVDM,
**AR-Diffusion** (CVPR 2025, current SOTA).

Reference FVD₁₆ on the standard suite for calibration — FaceForensics / SkyTimelapse / UCF101 /
TaiChi: MoCoGAN 124.7 / 206.6 / 2886.9 / —; MoCoGAN-HD 111.8 / 164.1 / 1729.6 / 128.1;
VideoGPT 185.9 / 222.7 / 2880.6 / —; PVDM 355.9 / 75.5 / 1141.9 / 540.2; Latte 77.7 / 110.5 / 604.6 /
267.1; FVDM 55.0 / 106.1 / 468.2 / 194.6. **Verify each against its source paper before it goes in a
table** — these are secondhand.

### 3.3 Prior art to explicitly position against

| Work | Overlap | Required response in the paper |
|---|---|---|
| **MODE-GAN** (2112.10960) | Neural-ODE motion generator, unconditional, motion/appearance split | Baseline row 4 + a paragraph: input-dependent τ vs. a fixed vector field |
| **Vid-ODE** (AAAI 2021) | continuous-time video, ODE-ConvGRU | Distinguish: conditional interpolation/extrapolation, not unconditional synthesis |
| **MoStGAN-V** (CVPR 2023) | improves StyleGAN-V motion, same backbone | Cite as closest competitor; if compute allows, retrain it under our metric suite |
| **DIGAN** (ICLR 2022) | INR-based continuous video | Cite; standard baseline |
| **CfC invertibility caveat** (Nature MI 2022) | authors note CfCs only approximate ODEs and are not bijections | Non-issue for us (we use it as a trajectory *generator*, not a normalizing flow) — but pre-empt it in one sentence |
| **LiquidTAD / liquid-net sequential bottleneck** | serial cost on long high-dim sequences | This is our §5.3 cost story; report it honestly |

---

## 4. Metrics — is FVD enough?

**No.** Not for a paper whose claim is specifically about motion. FVD is necessary for comparability
and insufficient as evidence.

### 4.1 The three documented failures

1. **Content bias.** I3D features weight per-frame appearance over temporal structure. Demonstrated
   failure case: StyleGAN-V + LSTM motion codes generates repeated temporal patterns and FVD does not
   penalise it. There is an exploitable null space in which scores can be lowered by resampling; with
   VideoMAE-v2 features that null space largely disappears.
2. **Broken statistical assumptions + sample inefficiency.** I3D's feature space is non-Gaussian, so
   the Fréchet form is misspecified; reliable estimation needs far more than the 2048 samples the
   whole field uses. The common "split into overlapping short clips" workaround itself biases the
   estimate.
3. **Length ceiling.** FVD is structurally tied to a 16-frame input window; long-horizon behaviour is
   only reachable through window aggregation, which mixes drift with window artefacts.

### 4.2 Proposed metric suite

| Tier | Metric | Measures | Implementation | Required? |
|---|---|---|---|---|
| **1 — comparability** | `fvd2048_16f`, `fvd2048_128f`, `fvd2048_128f_subsample8f`, `fid50k_full` | field-standard | **already in repo** (`src/metrics/`) | yes |
| **2 — fix FVD** | **cd-FVD** (VideoMAE-v2 features) | temporal coherence without content bias | `pip`, offline on dumped videos | **yes — this is the single most important addition** |
| | **JEDi** (V-JEPA feats + polynomial-kernel MMD, ICLR 2025) | human-aligned; no Gaussian assumption; converges at ~16 % of FVD's sample count (+34 % human agreement) | reference impl from project page — verify package name | yes |
| | **FVMD** (PIPs++ keypoint velocity/acceleration Fréchet distance, ICML-W 2024) | **motion consistency directly** | `pip install fvmd`; CLI `python -m fvmd --log_dir <d> <gen> <gt>`; **accepts folder-of-clips-of-frames = this repo's native format** | yes on TaiChi / faces / Jester / UCF101; report as N/A on Sky (no trackable points) — say why |
| **3 — temporal, disentangled** | **STREAM-T** (ICLR 2024) | temporal naturalness via FFT of frame-wise change, spatial-independent, **no length limit** | `github.com/pro2nit/STREAM` | yes — it is the only distributional metric that survives past 128 frames |
| | **VBench** subject consistency, background consistency, temporal flickering, motion smoothness | per-axis temporal quality | `vbench` custom-video mode | yes |
| | **VBench dynamic degree** (RAFT flow) | **anti-gaming guard — see §4.3** | same | **mandatory, non-negotiable** |
| **4 — long horizon (H2)** | **FVD(t) curve**, TATS protocol | drift | small script over dumped videos | **yes — this is the paper's money plot** |
| | per-frame LPIPS to frame 0; motion-latent FFT / self-similarity matrix | looping & collapse | small script | yes |
| **5 — cost / human** | s/kimg, peak VRAM, params, inference frames/s | H3 | training log | yes |
| | 2AFC human study (~20 raters × 50 long-video pairs) | ground truth | small web form | yes if targeting a top venue |

### 4.3 The anti-gaming guard (the biggest threat to this paper)

Every temporal-consistency metric — subject consistency, background consistency, temporal flickering,
motion smoothness, arguably STREAM-T — **is trivially maximised by a near-static video.** VBench added
dynamic degree precisely for this, and VBench++ documents the empirical trade-off (models strong on
consistency tend to have low dynamic degree).

This is not a hypothetical risk for us. An LTC/CfC cell with a large learned τ is an over-damped
system: its natural failure mode is *producing less motion*. "Smoother" and "deader" look identical to
most of Tier 3. So:

**Rule: no smoothness/consistency number is reported without, on the same row, (a) VBench dynamic
degree and (b) mean RAFT optical-flow magnitude, both compared against the real-data value.** A win is
only a win if flow statistics *match real data* while consistency improves.

Add two controls, both cheap:

- **Static-video floor.** `src/scripts/construct_static_videos_dataset.py` already freezes each video
  to a single repeated frame. Run the *entire* metric suite on it, per dataset, and print it as the
  top row of every table. Any metric on which the frozen dataset beats real data is a metric that
  cannot support a motion claim — and this makes that visible to the reader instead of to Reviewer 2.
- **Real-data ceiling.** Same suite, held-out real split vs. train split, via the existing
  `src/scripts/calc_metrics_for_dataset.py`. Gives every metric a meaningful scale.

### 4.4 Long-horizon protocol (the direct test of H2)

Follow TATS, which is what the long-video literature compares against:

1. Generate **512 videos × 1024 frames** per checkpoint (`src/scripts/generate.py --video_len 1024`,
   fixed `--seed` across strategies so content is matched).
2. Compute FVD on **non-overlapping 16-frame windows**, stride 16, at t = 0, 16, 32, …, 1008.
3. Report **both** absolute `FVD(t)` **and** normalised `FVD(t)/FVD(0)` — the normalisation is what
   separates "bad model" from "drifting model."
4. Second pass at stride 8 to confirm the curve is not a window-boundary aliasing artefact.
5. Per-window subject/background consistency + a sharpness measure, to distinguish the three
   degradation modes (blur, over-exposure, repetition/collapse) that FVD(t) alone conflates.
6. Storage: dump mp4, not PNG — 2048 videos × 160 frames is ~32 GB as PNG vs ~60 MB as mp4.

Expected result if H2 holds: `conv` shows a rising curve with **periodic structure** (its window is
fixed and its positional encoding is the only source of long-range variation), `ltc`/`cfc` show a
flatter curve. The periodicity is the more striking figure — pair the FVD(t) plot with a frame
self-similarity matrix and an FFT of the motion latent `v_t`.

### 4.5 Implementation plan for the metric stack

All Tier 2–4 metrics consume *videos*, not model internals, so **none of them touch the training
loop.** One new script plus one harness:

1. `src/scripts/export_videos.py` — dump N generated videos into the repo's native
   `<root>/<video>/<frame>.jpg` layout (+ mp4). Mostly a thin wrapper over `generate.py`.
2. `src/scripts/eval_suite.py` — take `(gen_dir, real_dir)`, run cd-FVD / JEDi / FVMD / STREAM /
   VBench / FVD(t), emit one tidy CSV row per `(run_id, dataset, strategy, seed, metric, value)`.

Keep the external metric packages in a **separate conda env** — VBench/JEDi/FVMD pull recent
torch/transformers versions that will fight this repo's pinned CUDA-10/11 environment. The interface
between the two envs is a directory of frames, so this costs nothing.

Only Tier 1 stays in-training (`configs/training/base.yaml: metrics`), unchanged, for checkpoint
selection. **Checkpoint selection stays on `fvd2048_16f`** for all strategies — changing the selection
rule to a metric we introduce would be circular. Note the resulting bias (short-horizon selection may
disfavour our own long-horizon claim, i.e. it is conservative) and also report final-checkpoint numbers.

---

## 5. Compute plan on 80 GB A100s

### 5.1 What the A100s change

| Setting | Now | On A100-80GB | Note |
|---|---|---|---|
| `training.allow_tf32` | `false` | **`true`** | Free ~1.3–1.5× on Ampere. **Set it once, for every run in the study** — mixing TF32 and FP32 across conditions invalidates the comparison |
| `training.batch_size` | 64 | **keep 64** | Do *not* raise it. 64 is what the published numbers use; 80 GB buys you fewer gradient-accumulation rounds per step (i.e. speed), not a bigger batch |
| `training.num_workers` | 3 | 8–12 | At 256² with TF32 the loader becomes the bottleneck |
| `training.nobench` | `false` | keep | cuDNN autotune helps |
| `sbatch_args.mem` | 64 GB/GPU | 80–96 GB/GPU | `src.infra.utils.num_gpus_to_mem` takes `mem_per_gpu` |
| resolution headroom | 256² | 512² feasible | **Do not use it.** 512² breaks comparability with every published number and costs 4×. Out of scope |

Net: expect **~1.5–2.5× the reference throughput** (README: FFS 256², 25000 kimg, 4 GPUs ≈ 2 days) →
~1–1.3 days per `conv` run. **Treat that as a hypothesis and measure it in Phase 1.**

### 5.2 The LTC latency bottleneck — the one real engineering risk

80 GB does not help here; LTC is **latency-bound, not memory-bound**.
[motion.py:433-436](src/training/motion.py#L433-L436):

```python
def get_max_traj_len(self, t):
    max_t = max(self.cfg.sampling.max_num_frames - 1, t.max().item())
    return np.ceil(max_t / self.cfg.motion.motion_z_distance).astype(int).item() + 2
```

With `max_num_frames=1024` and `motion_z_distance=min_period_len=16` (FaceForensics), that is
**66 sequential cell steps × 6 `ode_unfolds` ≈ 396 sequential FC evaluations on every forward pass**,
while the discriminator only ever sees 3 frames. FaceForensics is therefore the *most expensive* LTC
configuration in the matrix, not the cheapest.

Two things to try, in Phase 1, before committing 300 GPU-days:

- **Drop the `max_num_frames` floor.** The `max(max_num_frames - 1, t.max())` term forces the
  worst-case unroll on every step even when the sampled `t` are small. For a causal cell (LTC, CfC,
  LSTM, GRU, and `conv` with `padding=0`), motion codes beyond `ceil(t.max()/d)+2` are computed and
  discarded, so unrolling only to `t.max()` is **output-identical**. Savings depend on the sampled `t`
  distribution — `t` are absolute frame indices, so on long videos `t.max()` is often near
  `max_num_frames` and the saving is modest; on short-video datasets it is large. **Measure it, do not
  assume it.** If adopted, adopt it for *all* strategies.
- **`torch.compile` / CUDA-graph the cell.** 396 tiny sequential matmuls is a kernel-launch-bound
  workload; capture is worth more here than any hardware upgrade.

Whatever you decide, apply it identically to every condition and record it — this is exactly the kind
of asymmetry that silently invalidates an architecture comparison.

### 5.3 Budget — 4×A100-80GB for 12 months

**Available: 4 GPUs × 365 days = 1460 GPU-days gross.** Assume 65–75 % effective utilisation after
maintenance, preemption, crashed runs and debugging → **≈ 950–1100 usable GPU-days.**

Per-run cost, 25000 kimg, estimated (measure in Phase 1 and rescale the whole table):

| Dataset | Res | `conv` | `ltc`/`cfc` | GPU-days per run (≈) |
|---|---|---|---|---|
| FFS, Sky, RE10K, UCF101, RainbowJelly, CelebV-HQ | 256 | ~1.2 d | ~1.5–2.0 d | 5–8 |
| TaiChi, Jester | 128 | ~0.4 d | ~0.6 d | 1.6–2.4 |

| Scenario | Main runs | Ablation runs | GPU-days | Fraction of the year |
|---|---|---|---|---|
| **Full** — Tier A×3 seeds, Tier B×3 seeds, Tier C, ablations A–F | 48 | ~38 | **≈ 340** | ~33 % |
| Recommended — 3 seeds on FFS+RE10K only, 1 elsewhere | 36 | ~38 | ≈ 200 | ~20 % |
| Minimum defensible | 24 | 9 | ≈ 110 | ~11 % |

**Take Full.** At 340 GPU-days it consumes about a third of the budget, which leaves the right kind of
slack: roughly 30 % of runs in a study this size get thrown away (bad `min_period_len`, diverged GAN,
a bug found at kimg 8000), and reviewers reliably demand one more dataset or ablation at rebuttal
time. Budgeting 340 planned + ~200 wasted/rework + ~150 rebuttal ≈ 700 GPU-days still fits inside
1100 with margin. Going Recommended here would be leaving capacity unused for no gain.

**Do still run the study risk-ordered** (§5.4): sequence it so that FaceForensics + RealEstate10K ×
3 seeds and ablation A are finished by roughly month 6. If access ends early or something breaks
badly, that subset alone is the Minimum-defensible paper.

Report `mean ± std` over seeds. Never report `--num_runs 3` (evaluation repeats) as though it were
run-to-run variance — see [EXPERIMENT_REPORT.md §2](EXPERIMENT_REPORT.md).

### 5.4 Scheduling — the 4-GPU-specific decision

With 48+ runs and only 4 GPUs, **job shape matters more than per-run speed**:

| Shape | Per-run wall-clock (256², 25000 kimg) | Runs in flight | Aggregate |
|---|---|---|---|
| 1 job × 4 GPUs (`num_gpus=4`) | ~1.2–2 d | 1 | loses 10–20 % to inter-GPU comm |
| 4 jobs × 1 GPU (`num_gpus=1`) | ~4–7 d | 4 | no comm overhead; risks becoming dataloader/disk-bound |

Total `training.batch_size` stays **64** in both shapes (gradient accumulation absorbs the
difference), and StyleGAN2 has no batch-norm, so the two are statistically equivalent *provided the
minibatch-std group size resolved by `cfg=auto` is the same*. Verify that in Phase 1 by diffing the
two saved `experiment_config.yaml` files.

**Measure both in Phase 1, pick one, and freeze it for the entire study.** A mixed setup invalidates
the comparison. Single-GPU jobs are the likely winner for throughput and are far simpler to queue,
but only if the loader keeps up — hence `num_workers` 8–12 and datasets on local NVMe, not network
storage.

### 5.5 Twelve-month calendar

| Months | Phase | GPU-days | Content | Gate |
|---|---|---|---|---|
| **1–2** | 0 — data + code | ~15 | RealEstate10K, CelebV-HQ, Jester, TaiChi preprocessing (off the GPU critical path). Implement §3.1 rows 3–6 (`gru`, `ode_rnn`, `mamba`, optionally `liquid_s4`). Build `export_videos.py` + `eval_suite.py` | zips load; new strategies train for 100 kimg without NaN |
| **2** | 1 — calibration | ~10 | 200-kimg `conv` vs `ltc` on FFS: s/kimg, VRAM, TF32 gain, `get_max_traj_len` change, 1-GPU vs 4-GPU shape. Validate every Tier 2–4 metric against the **static-video floor** and **real-vs-real ceiling** | §5.3 rescaled from measurement; metric suite emits a CSV; **commit freeze** |
| **3** | 2 — pilots | ~19 | Ablation E (`min_period_len ∈ {16,32,64,256}`) @5000 kimg on RE10K, UCF101, TaiChi, Jester | one `min_period_len` fixed per dataset |
| **4–6** | 3a — critical path | ~110 | FFS + RealEstate10K × {conv, ltc, cfc} × 3 seeds (18 runs) + ablation A (9 runs @5000 kimg) | **the minimum paper exists from here on** |
| **6–9** | 3b — remainder | ~140 | Sky, TaiChi, UCF101, RainbowJelly, CelebV-HQ × 3 seeds; Jester × 1 seed | full main table |
| **7–9** | 4 — ablations B–F | ~30 | @5000 kimg on FFS, interleaved with 3b | ablation table filled |
| **9–10** | 5 — long horizon + human | ~15 | 512 × 1024-frame generation, FVD(t) curves, looping/self-similarity analysis, 2AFC study | H2 figure |
| **10–12** | write-up + rebuttal | ~150 reserve | paper, plus whatever reviewers ask for | — |

Phases 2–4 must all run from the **same frozen git hash** — `training.resume=latest` matches on git
hash + `exp_suffix`, so any commit mid-study orphans every in-flight run. That is why the commit
freeze sits at the end of month 2, after the new strategies and the TF32/unroll decisions have landed.

---

## 6. Additions to the risk table

| Risk | Mitigation |
|---|---|
| **LTC wins on smoothness by generating less motion** | §4.3 guard: dynamic degree + RAFT flow magnitude on every row; static-video floor as the top row of every table |
| **I3D-FVD cannot see the effect we claim** | cd-FVD + JEDi + FVMD + STREAM-T (§4.2); reproduce the Ge et al. LSTM finding on our own stack as validation |
| **Reviewer: "this is just MODE-GAN with a different cell"** | Neural-ODE baseline row (§3.1 #4) at matched params |
| **Reviewer: "why not Mamba?"** | SSM baseline row (§3.1 #5); an honest loss here is still a result |
| **RealEstate10K link rot → unreproducible** | freeze + publish clip list and the extracted-zip checksum |
| **RE10K has no published unconditional baselines** | it carries the *internal* comparison only; comparability lives on FFS/Sky/UCF101/TaiChi |
| **External metric packages break this repo's env** | separate conda env; frames-on-disk interface |
| **Diffusion SOTA is far ahead** | §1 positioning, stated as a scope limit in the intro, not buried |
| **TF32 / unroll optimisation applied unevenly** | one frozen commit + assert the setting in every saved `experiment_config.yaml` |
| **Preprocessing slips and starves the GPUs** | Phase 0 starts immediately and is off the critical path of Phase 1 |

---

## 7. Code isolation: branch, not a separate repository

The goal — don't disturb the current working code — is right. But **a branch in this repo beats a
separate repo**, for a reason specific to this codebase: experiment directories are named
`${dataset.name}_${model.name}_${sampling.name}_${exp_suffix}-${git_hash}`
([configs/infra.yaml:10](configs/infra.yaml#L10)), and `training.resume=latest` finds a prior run by
matching **git hash + `exp_suffix`**. Consequences:

- A branch keeps `get_git_hash()` ([src/infra/utils.py:86](src/infra/utils.py#L86)) meaningful and
  keeps every run traceable to a commit that shares history with `master`. A fresh repo starts a
  disjoint history, so you can never `git diff` a result against the StyleGAN-V baseline that produced
  it.
- `create_project_dir` **refuses to run on a dirty tree** unless `ignore_uncommited_changes=true`
  ([src/infra/utils.py:64-68](src/infra/utils.py#L64-L68)). Never pass that flag during the study — it
  is the mechanism that guarantees the hash in the directory name actually describes the code that
  ran.

Recommended setup:

```bash
git checkout -b ltc-motion            # all study code lives here
# ... implement §3.1 rows 3-6, eval_suite.py, TF32 + unroll decisions ...
git commit -m "..."                   # ONE commit; this hash names every run in the study
git rev-parse --short HEAD            # record it in EXPERIMENT_REPORT.md §8
```

**Clean the tree first — it is currently dirty in ways that will bite.** `git status` shows ~20
untracked `desktop.ini` files, a `.vs/` directory, and two OneDrive conflict copies
(`configs/model/stylegan-v (1).yaml`, `src/training/motion (1).py`). Add to `.gitignore`:

```
desktop.ini
.vs/
*[0-9]).py
*[0-9]).yaml
```

The two `(1)` files are almost certainly accidental sync duplicates of files you have already edited —
check them against their originals and delete them rather than ignoring them, so nobody trains from
the wrong `motion.py`.

Use a separate repository only if you intend to publish the code standalone at submission time; in
that case, fork at the end from the frozen study commit, not now.

## 8. Open decisions

1. ~~GPU count~~ — **resolved: 4×A100-80GB for ~12 months → Full scenario (§5.3).**
2. **Target venue and deadline** — the calendar in §5.5 assumes ~10 months to a submission. A tighter
   deadline cuts Tier B first (Sky/TaiChi/UCF101 to 1 seed), then the human study.
3. **Is RealEstate10K acceptable as a Tier-A dataset given it tests camera rather than object motion?**
   It is the strongest H2 evidence available, but a reviewer may call it a special case. The hedge is
   RainbowJelly + SkyTimelapse as object-motion long-horizon companions.

---

## Sources

- [On the Content Bias in Fréchet Video Distance (CVPR 2024)](https://content-debiased-fvd.github.io/)
- [Beyond FVD / JEDi (ICLR 2025), arXiv:2410.05203](https://arxiv.org/abs/2410.05203) · [project page](https://oooolga.github.io/JEDi.github.io/)
- [Fréchet Video Motion Distance (ICML 2024 workshop), arXiv:2407.16124](https://arxiv.org/abs/2407.16124) · [code](https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance)
- [STREAM (ICLR 2024), arXiv:2403.09669](https://arxiv.org/abs/2403.09669) · [code](https://github.com/pro2nit/STREAM)
- [VBench (CVPR 2024), arXiv:2311.17982](https://arxiv.org/pdf/2311.17982) · [VBench++](https://arxiv.org/html/2411.13503v1)
- [TATS — Long Video Generation with Time-Agnostic VQGAN, arXiv:2204.03638](https://arxiv.org/pdf/2204.03638)
- [MoStGAN-V (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_MoStGAN-V_Video_Generation_With_Temporal_Motion_Styles_CVPR_2023_paper.pdf)
- [StyleInV (ICCV 2023), arXiv:2308.16909](https://arxiv.org/pdf/2308.16909)
- [MODE-GAN — Continuous-Time Video Generation via Neural ODE, arXiv:2112.10960](https://arxiv.org/pdf/2112.10960)
- [Vid-ODE (AAAI 2021), arXiv:2010.08188](https://arxiv.org/abs/2010.08188)
- [CelebV-HQ (ECCV 2022)](https://celebv-hq.github.io/)
- [AR-Diffusion (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_AR-Diffusion_Asynchronous_Video_Generation_with_Auto-Regressive_Diffusion_CVPR_2025_paper.pdf)
- [Closed-form Continuous-time Neural Networks (Nature MI 2022)](https://www.nature.com/articles/s42256-022-00556-7)
- [StyleGAN-V (CVPR 2022), arXiv:2112.14683](https://arxiv.org/abs/2112.14683)
