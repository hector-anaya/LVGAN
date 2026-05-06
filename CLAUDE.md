# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for an autoregressive video-generation GAN (LTC-driven, U-Net generator, 3D-Conv WGAN-GP discriminator) trained on the TaiChi dataset (and previously Sky-Timelapse). Not a packaged project — there are no tests, lint, build, or CI. The "interface" is running `train.py` directly.

## Running

```bash
python train.py            # current entry point — TaiChi
python test_sky_1st.py     # legacy single-file version (Sky-Timelapse)
```

Both scripts are hardcoded:
- GPU pinned via `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` at the top of the file.
- Dataset path hardcoded near the top (e.g. `BASE_DIR = '/mnt/d/PHD/Experiments/Dataset/UCFCLASS/TaiChi'`). Different datasets have different on-disk layouts (TaiChi: `BASE_DIR/<video>/*.jpg` flat; Sky-Timelapse: one extra nesting level) — switching datasets means editing both `BASE_DIR` and `build_dataset()` accordingly.
- Output folder hardcoded as `folder = "exp_test_claude_taichi/"` in [train.py](train.py).

There is no `requirements.txt`. Required packages observed: `tensorflow` (2.15-era APIs), `ncps` (`ncps.tf.LTCCell`, `ncps.wirings`), `numpy`, `opencv-python` (`cv2`), `matplotlib`, `scipy`. Keras is imported as both `tf.keras.*` and bare `keras.models.Model` — keep both available.

## Repository layout

The root holds the **current working version** of the model. The `exp_*/` subdirectories are **per-run snapshots**: at the start of every training run, [train.py:277-281](train.py#L277-L281) copies `train.py`, `models.py`, `losses.py`, `utils.py` into the run folder alongside checkpoints, plots, videos, and `log.csv`. Treat anything inside an `exp_*/` directory as a frozen artifact of a past run — edit the root files, not the snapshots.

- [train.py](train.py) — config, dataset loading, train step (`@tf.function` with `tf.while_loop` for 3× disc updates), main loop, checkpointing.
- [models.py](models.py) — `define_generator` (U-Net), `CreateARLTCGAN` (autoregressive wrapper), `build_discriminator_3d`, `TemporalSelfAttention`.
- [losses.py](losses.py) — WGAN + L1 + perceptual (VGG19 `block4_conv2`) + temporal-consistency (VGG19 `block3_conv3`) + 2nd-order gradient-temporal + frame-diff losses, `WarmupCosineDecay` schedule, the two Adam optimizers, and the WGAN-GP `gradient_penalty`.
- [utils.py](utils.py) — `summarize_performance` (saves plots / videos / weights), `log_loss_to_csv`, `save_video`, plus several unused legacy layers (`ReshapeLayer`, `BlurredNoise`, `ResizeLayer`, etc.) kept around from earlier experiments.
- [test_sky_1st.py](test_sky_1st.py) — earlier single-file version of the whole pipeline. Header comments (`# exp 1 ... exp 10`) are the running changelog of design decisions; useful context when reasoning about why the current code looks the way it does.

## Architecture notes that span files

- **Autoregressive generation loop** lives in `CreateARLTCGAN.call()` ([models.py:231-305](models.py#L231-L305)). Frame `i+1` is produced by feeding the previous frame + sampled `z[i]` + LTC hidden state into the U-Net generator. The hidden state is updated by `ltccell_image`, and every `ATTN_EVERY=4` steps it is rewritten in place by `TemporalSelfAttention` over the full prefix — this is the mechanism intended to fight long-horizon content drift. Both this loop and `sample_z_m()` use `tf.while_loop` (not Python loops) specifically to avoid graph unrolling at `SEQ_LENGHT=128`; preserve that when editing.
- **Teacher forcing** is on inside the generator loop ([models.py:279-293](models.py#L279-L293)) and driven by a global `tf_step_counter` and `get_teacher_forcing_rate()` defined in [train.py](train.py). It linearly decays from `TF_RATE_INIT=1.0` to `TF_MIN=0.1` over `TF_DECAY_STEPS=100_000` global steps.
- **Two-level batching.** `minibatch = (8, 16)` means each outer step (`train_step_multi`) runs 8 inner `train_step` calls, each over a 16-frame mini-clip sliced with stride `minibatch[1] - 1` to overlap by one frame. The per-call disc-vs-gen ratio is **3 disc updates → 1 gen update**, implemented via `tf.while_loop` inside `train_step` ([train.py:144-205](train.py#L144-L205)).
- **Loss weights** (`LAMBDA, TEMP, GRAD, FRAME, PERC, PEN`) are tuned per experiment and the comment block at the top of [test_sky_1st.py](test_sky_1st.py) documents prior values per `exp_N`. They appear both in [train.py](train.py) and (referenced) in [losses.py](losses.py); when adjusting, check both files agree.
- **Best-checkpoint criterion** is `frame_loss * FRAME` ([train.py:259-267](train.py#L259-L267)) — i.e. the frame-difference loss, not the total generator loss. Intentional; don't "fix" it to use total loss without asking.
- **Input range** for everything (dataset, generator output `tanh`, discriminator, losses) is `[-1, 1]`. Loss functions that hand frames to VGG do their own `(x + 1) * 127.5` rescale + `preprocess_input`.

## Conventions worth preserving

- Inside `@tf.function` the code is written in graph-style (`tf.while_loop`, `tf.cond`, `tf.TensorArray`) rather than Pythonic loops. This is deliberate (graph-unrolling cost at `SEQ_LENGHT=128`). Don't replace these with Python `for`/`if` even though autograph would accept them.
- The discriminator was deliberately reverted to **float32** (no `LossScaleOptimizer`); see comment in [losses.py:50-53](losses.py#L50-L53) and the `exp 9 reverted disc` note. Don't reintroduce mixed precision without a reason.
- Decoder skip connections at d5/d6/d7 are intentionally `skip=False` ([models.py:181-184](models.py#L181-L184)) — this was the `exp 10` change. Re-enabling them changes the model meaningfully.
