"""
gan_logger.py — TensorFlow-efficient loss logger for progressive temporal GAN
==============================================================================
Uses tf.summary (TensorBoard) as the primary backend:
  - All scalars are written with a single summary writer → zero Python overhead
    inside @tf.function traced code.
  - A lightweight CSV fallback is written every N steps in pure Python
    (outside the graph), so you always have a plain-text record.
  - Phase transitions and α are logged as extra scalars automatically.

Usage
-----
    from gan_logger import GANLogger

    logger = GANLogger(log_dir='exp_progressive/logs', csv_every=50)

    # inside your training loop (outside @tf.function):
    logger.log_step(
        step      = z,
        phase     = phase,
        alpha     = alpha,
        n_frames  = n_frames,
        tf_rate   = tf_rate_val,
        losses    = dict(
            G_total   = a,
            G_gan     = b,
            G_l1      = c,
            D_total   = d,
            GP        = e,
            temporal  = f,
            perceptual= g,
            grad_temp = h,
            frame_diff= i,
        )
    )

    # at the end of each epoch (optional):
    logger.log_epoch(epoch, mean_losses_dict)

Launch TensorBoard with:
    tensorboard --logdir exp_progressive/logs
"""

import csv
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf


class GANLogger:
    """
    Thin wrapper around tf.summary + CSV for GAN training metrics.

    Parameters
    ----------
    log_dir     : directory for TensorBoard event files and the CSV
    csv_every   : write a CSV row every this many steps (default 50)
    flush_every : flush the summary writer every this many steps (default 200)
    """

    LOSS_KEYS = [
        'Generator Total Loss',
        'Generator Loss',
        'L1 Loss',
        'Discriminator Loss',
        'Penalty',
        'Temporal Loss',
        'Perceptual Loss',
        'Grad temp loss',
        'Frame Loss',
    ]

    def __init__(self, log_dir: str = 'logs',
                 csv_every: int = 50,
                 flush_every: int = 200):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir     = log_dir
        self.csv_every   = csv_every
        self.flush_every = flush_every
        self._step       = 0
        self._t0         = time.time()

        # ── TensorBoard writer ────────────────────────────────────────────────
        self._writer = tf.summary.create_file_writer(log_dir)

        # ── CSV ───────────────────────────────────────────────────────────────
        self._csv_path = os.path.join(log_dir, 'losses.csv')
        self._csv_file = open(self._csv_path, 'a', newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)

        # Write header only if file is new / empty
        if os.path.getsize(self._csv_path) == 0:
            self._csv_writer.writerow(
                ['Step', 'Epoch', 'phase', 'alpha', 'n_frames',
                 'tf_rate', 'steps_per_sec']
                + self.LOSS_KEYS
            )

        # ── Running mean accumulators (for epoch-level summaries) ─────────────
        self._accum: dict = defaultdict(list)
        self._last_phase  = -1

    # ── public API ────────────────────────────────────────────────────────────

    def log_step(self,
                 step: int,
                 phase: int,
                 alpha: float,
                 n_frames: int,
                 tf_rate: float,
                 losses: dict,
                 epoch: int = 0):
        """
        Call once per train_step (outside @tf.function).

        `losses` must be a dict with keys matching LOSS_KEYS (float values).
        Unknown keys are silently ignored; missing keys are logged as NaN.
        """
        self._step = step
        sps        = self._steps_per_sec()

        # ── Phase-transition marker ───────────────────────────────────────────
        if phase != self._last_phase:
            with self._writer.as_default(step=step):
                tf.summary.text(
                    'phase_transition',
                    f'Phase {phase} | {n_frames} frames | α={alpha:.3f}',
                    step=step)
            self._last_phase = phase

        # ── TensorBoard scalars (every step — cheap) ──────────────────────────
        with self._writer.as_default(step=step):
            # Training schedule
            tf.summary.scalar('schedule/phase',    phase,    step=step)
            tf.summary.scalar('schedule/alpha',    alpha,    step=step)
            tf.summary.scalar('schedule/n_frames', n_frames, step=step)
            tf.summary.scalar('schedule/tf_rate',  tf_rate,  step=step)
            tf.summary.scalar('perf/steps_per_sec', sps,     step=step)

            # Losses — each in its own group for clean TB layout
            _groups = {
                'Generator Total Loss': 'generator',
                'Generator Loss':       'generator',
                'L1 Loss':              'generator',
                'Discriminator Loss':   'discriminator',
                'Penalty':              'discriminator',
                'Temporal Loss':        'aux_losses',
                'Perceptual Loss':      'aux_losses',
                'Grad temp loss':       'aux_losses',
                'Frame Loss':           'aux_losses',
            }
            for key in self.LOSS_KEYS:
                val = float(losses.get(key, float('nan')))
                tf.summary.scalar(f'{_groups[key]}/{key}', val, step=step)
                self._accum[key].append(val)

        # ── CSV (every csv_every steps) ───────────────────────────────────────
        if step % self.csv_every == 0:
            row = ([step, epoch, phase, round(alpha, 4),
                    n_frames, round(tf_rate, 4), round(sps, 2)]
                   + [round(float(losses.get(k, float('nan'))), 6)
                      for k in self.LOSS_KEYS])
            self._csv_writer.writerow(row)

        # ── Periodic flush ────────────────────────────────────────────────────
        if step % self.flush_every == 0:
            self._writer.flush()

    def log_epoch(self, epoch: int, step: int, phase: int):
        """
        Write epoch-level mean of all accumulated losses to TensorBoard.
        Clears accumulators afterward.
        """
        if not self._accum:
            return
        with self._writer.as_default(step=step):
            for key, vals in self._accum.items():
                tf.summary.scalar(f'epoch/{key}',
                                  float(np.mean(vals)),
                                  step=epoch)
        self._accum.clear()
        self._writer.flush()

    def log_phase_sample(self, step: int, fake_video, real_video,
                         max_videos: int = 2):
        """
        Write a small batch of generated vs real frames as TensorBoard images.
        `fake_video` and `real_video` are tensors of shape (B, T, H, W, C)
        in [-1, 1].  Logs the first frame, middle frame, and last frame
        of each video — cheap but informative.
        """
        def _to_uint8(t):
            return tf.cast(tf.clip_by_value((t + 1.0) * 127.5, 0, 255),
                           tf.uint8)

        B = min(int(tf.shape(fake_video)[0]), max_videos)
        T = int(tf.shape(fake_video)[1])
        picks = [0, T // 2, T - 1]

        with self._writer.as_default(step=step):
            for label, vid in [('generated', fake_video),
                                ('real',      real_video)]:
                for t in picks:
                    img = _to_uint8(vid[:B, t])          # (B, H, W, C)
                    tf.summary.image(f'frames/{label}_t{t}', img,
                                     step=step, max_outputs=B)
        self._writer.flush()

    def close(self):
        self._writer.flush()
        self._writer.close()
        self._csv_file.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _steps_per_sec(self):
        elapsed = time.time() - self._t0
        if elapsed < 1e-6:
            return 0.0
        return self._step / elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Integration patch for train_gan_progressive
# ──────────────────────────────────────────────────────────────────────────────
#
# Replace your train_gan_progressive print block with this:
#
#   logger = GANLogger(log_dir=folder + 'logs', csv_every=50)
#
#   # inside the step loop, after train_step returns:
#   a, b, c, d, e, f, g, h, i = [l.numpy() for l in losses]
#
#   logger.log_step(
#       step     = z,
#       epoch    = epoch,
#       phase    = phase,
#       alpha    = alpha,
#       n_frames = n_frames,
#       tf_rate  = get_teacher_forcing_rate().numpy(),
#       losses   = {
#           'Generator Total Loss' : a,
#           'Generator Loss'       : b,
#           'L1 Loss'              : c,
#           'Discriminator Loss'   : d,
#           'Penalty'              : e,
#           'Temporal Loss'        : f,
#           'Perceptual Loss'      : g,
#           'Grad temp loss'       : h,
#           'Frame Loss'           : i,
#       }
#   )
#
#   # optionally, log a frame sample every 100 steps:
#   if z % 100 == 0:
#       logger.log_phase_sample(z, gen_output, target)
#
#   # at end of epoch:
#   logger.log_epoch(epoch, step=z, phase=phase)
#
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test  (python gan_logger.py)
# ──────────────────────────────────────────────────────────────────────────────

