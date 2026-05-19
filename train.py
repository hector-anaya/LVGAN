#!/usr/bin/env python
# coding: utf-8
"""
train.py
========
Entry point for exp_test_claude_taichi.

  - Dataset loading (TaiChi flat structure: BASE_DIR/video_name/*.jpg)
  - Teacher-forcing schedule
  - train_step (tf.while_loop for disc updates, @tf.function)
  - train_step_multi (@tf.function)
  - train_gan (main loop)
"""

import argparse
import os

parser = argparse.ArgumentParser(description='Train .')
parser.add_argument('--dataset', type=str, required=True, default='taichi', help='Dataset to use (e.g., "taichi", "sky", "ucf101", "ffs")')
parser.add_argument('--folder', type=str, default='exp_new_test', help='Folder to save logs and models')
parser.add_argument('--gpu', type=str, default='0', help='GPU device to use (e.g., "0", "1", "0,1")')
parser.add_argument('--islocal', type=bool, default=0, help='Flag to indicate local execution (for debugging)')
parser.add_argument('--steps', type=int, default=300_000, help='Number of training steps to run (the dataset repeats indefinitely)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob

# -- local modules ----------------------------------------------------------
from models import (
    define_generator,
    CreateARLTCGAN,
    build_discriminator_3d,
)
from losses import *
from utils import summarize_performance, log_loss_to_csv,GANCheckpointManager


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

folder     = args.folder.rstrip('/') + '/'
SEQ_LENGHT = 128
FRAME_SIZE = (256, 256)
minibatch  = (8, 16)
MAX_STRIDE = 1   # stride fixed to 1 for TaiChi

# TaiChi dataset has a flat structure: BASE_DIR/video_name/*.jpg
# BASE_DIR   = '/home/hector.anaya/Datasets/sky_timelapse/sky_train'

if args.islocal:
    BASE_DIR='/mnt/d/PHD/Experiments/Dataset/UCFCLASS/TaiChi'
else:
    if args.dataset == 'taichi':
        BASE_DIR='/usuarios03/hector.anaya1/Datasets/TaiChi'
    elif args.dataset == 'sky':
        BASE_DIR='/usuarios03/hector.anaya1/Datasets/sky_timelapse/sky_train'
    elif args.dataset == 'ucf101':
        BASE_DIR='/usuarios03/hector.anaya1/Datasets/UCF101processed'
    elif args.dataset == 'ffs':
        BASE_DIR='/usuarios03/hector.anaya1/Datasets/ffs_train'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")  
# Teacher-forcing schedule
TF_RATE_INIT   = 1.0
TF_MIN         = 0.1
TF_DECAY_STEPS = 100_000
tf_step_counter = tf.Variable(0, trainable=False, dtype=tf.float32)

# ──────────────────────────────────────────────────────────────────────────────
#  LOSS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA = 1
TEMP   = 0.01
GRAD   = 10
FRAME  = 10000
PERC   = 1000
PEN    = 10


# ──────────────────────────────────────────────────────────────────────────────
#  LR SCHEDULE
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_STEPS  = 300_000
WARMUP_STEPS = 2_000


def get_teacher_forcing_rate():
    """Linear decay from TF_RATE_INIT → TF_MIN over TF_DECAY_STEPS."""
    progress = tf.minimum(tf_step_counter / TF_DECAY_STEPS, 1.0)
    return TF_RATE_INIT - progress * (TF_RATE_INIT - TF_MIN)


# ──────────────────────────────────────────────────────────────────────────────
#  DATASET
#  TaiChi is a flat structure: each sub-folder directly contains *.jpg frames.
#  (Sky timelapse has one extra nesting level — adjust BASE_DIR if switching.)
# ──────────────────────────────────────────────────────────────────────────────


MAX_STRIDE = 2  # adjust to your value

# ── random `seq_length`-frame window sampler ──────────────────────────────
# Wrapped in tf.py_function so .repeat() draws a *fresh* random window per
# video on every pass through the dataset (instead of one fixed window).
def _sample_window_paths(video_dir, seq_length, max_stride):
    video_dir = video_dir.numpy().decode("utf-8")
    frame_paths = sorted(glob(os.path.join(video_dir, "*.jpg")))
    total = len(frame_paths)

    stride = np.random.randint(1, max_stride + 1)
    required_span = 1 + (seq_length - 1) * stride

    if total >= required_span:
        start = np.random.randint(0, total - required_span + 1)
        indices = start + np.arange(seq_length) * stride
    else:
        start = np.random.randint(0, total)
        indices = (start + np.arange(seq_length) * stride) % total

    return np.array([frame_paths[i].encode("utf-8") for i in indices])


# collect video dirs (videos must have at least `seq` frames)
def make_paths(BASE_DIR, seq):
    video_dirs = []
    for name in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, name)) and \
           len(glob(os.path.join(BASE_DIR, name, "*.jpg"))) >= seq:
            video_dirs.append(os.path.join(BASE_DIR, name))
    video_dirs.sort()
    print(f"Found {len(video_dirs)} video directories.")
    return video_dirs

def load_frame(path):
    """Pure TF op — safe for prefetch / parallel map."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, FRAME_SIZE)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img                                      # (H, W, 3)


def make_dataset(video_dirs, seq_length, max_stride):
    """
    Infinite dataset of single `seq_length`-frame clips (one clip per element).

    from_tensor_slices(video_dirs) → shuffle → repeat → a py_function picks a
    fresh random window per video on each pass → decode frames. Because the
    stream is infinite, the training loop is responsible for stopping after a
    fixed number of steps (args.steps).
    """
    path_ds = (tf.data.Dataset
               .from_tensor_slices(video_dirs)
               .shuffle(len(video_dirs), reshuffle_each_iteration=True)
               .repeat())

    def _sample(video_dir):
        paths = tf.py_function(
            func=lambda vd: _sample_window_paths(vd, seq_length, max_stride),
            inp=[video_dir],
            Tout=tf.string,
        )
        paths.set_shape([seq_length])
        return paths

    path_ds = path_ds.map(_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # map load_frame over the seq_length paths → (seq_length, H, W, C)
    video_ds = path_ds.map(
        lambda paths: tf.map_fn(
            load_frame,
            paths,
            fn_output_signature=tf.TensorSpec(shape=[FRAME_SIZE[0], FRAME_SIZE[1], 3],
                                          dtype=tf.float32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return (video_ds
            .batch(1)
            .prefetch(tf.data.AUTOTUNE))

  
#  ──────────────────────────────────────────────────────────────────────────────
#  OPTIMIZERS
#  Note: plain Adam (no LossScaleOptimizer) because discriminator is float32
# ──────────────────────────────────────────────────────────────────────────────

generator_optimizer = tf.keras.optimizers.Adam(
    WarmupCosineDecay(0.0002, WARMUP_STEPS, TOTAL_STEPS),
    beta_1=0.0, beta_2=0.999,
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    WarmupCosineDecay(0.0001, WARMUP_STEPS, TOTAL_STEPS),
    beta_1=0.0, beta_2=0.999,
)


def generator_loss(disc_generated_output, gen_output, target,
                   minibatch_size=16):
    gan_loss   = -tf.reduce_mean(disc_generated_output)       # WGAN
    l1_loss    = tf.reduce_mean(tf.abs(target - gen_output))
    temp       = temporal_consistency_loss(gen_output, target)
    grad_temp  = gradient_temporal_loss(gen_output, target)
    frame_loss = frame_difference_loss(gen_output, target)
    perp       = perceptual_loss(gen_output, target, minibatch_size)

    total = (
        gan_loss
        + LAMBDA * l1_loss
        + PERC   * perp
        + TEMP   * temp
        + GRAD   * grad_temp
        + FRAME  * frame_loss
    )
    return total, gan_loss, l1_loss, temp, perp, grad_temp, frame_loss

# 
# ──────────────────────────────────────────────────────────────────────────────
#  TRAIN STEPS
# ──────────────────────────────────────────────────────────────────────────────

@tf.function
def train_step(target, generator, discriminator):
    tf_rate = get_teacher_forcing_rate()

    # ── 3× discriminator updates via tf.while_loop ───────────────────────────
    def _disc_cond(i, penalty, disc_loss):
        return i < 3

    def _disc_body(i, penalty, disc_loss):
        with tf.GradientTape() as disc_tape:
            gen_output            = generator(
                n=minibatch[1], og_frames=target[0],
                noise=tf.random.uniform((1, 400)),
            )
            disc_real_output      = discriminator(target,     training=True)
            disc_generated_output = discriminator(gen_output, training=True)
            penalty               = gradient_penalty(
                discriminator, target, gen_output
            )
            disc_loss = (discriminator_loss(disc_real_output,
                                            disc_generated_output)
                         + penalty * PEN)

        disc_grads = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(
            zip(disc_grads, discriminator.trainable_variables)
        )
        return i + 1, penalty, disc_loss

    _, penalty, disc_loss = tf.while_loop(
        _disc_cond, _disc_body,
        loop_vars=[tf.constant(0), tf.constant(0.0), tf.constant(0.0)],
        parallel_iterations=1,
    )

    # ── 1× generator update ──────────────────────────────────────────────────
    with tf.GradientTape() as gen_tape:
        gen_output = generator(
            n=minibatch[1], og_frames=target[0],
            teacher_forcing_rate=tf_rate,
            noise=tf.random.uniform((1, 400)),
            training=True,
        )
        disc_generated_output = discriminator(gen_output)
        (gen_total_loss, gen_gan_loss, gen_l1_loss,
         temp, perp, grad, fram) = generator_loss(
            disc_generated_output, gen_output, target,
            minibatch_size=minibatch[1],
        )

    gen_grads = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gen_grads, generator.trainable_variables)
    )

    tf_step_counter.assign_add(1.0)

    return (gen_total_loss, gen_gan_loss, gen_l1_loss,
            disc_loss, penalty, temp, perp, grad, fram)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_gan(generator, discriminator, dataset, steps=None):
    """Step-based loop. `dataset` is the infinite stream from make_dataset()
    yielding one `minibatch[1]`-frame clip per element; we run for `steps`
    global steps (default args.steps) rather than iterating an epoch."""
    if steps is None:
        steps = args.steps
    os.makedirs(folder, exist_ok=True)
    best_loss = np.inf

    for global_step, batch in enumerate(dataset):
        if global_step >= steps:
            break
        X_train = batch[0]                       # (minibatch[1], H, W, 3)

        a, b, c, d, e, f, g, h, i = train_step(
            tf.expand_dims(X_train, axis=0), generator, discriminator
        )

        log_loss_to_csv(
            0, global_step, a, b, c, d, e, f, g, h, i,
            csv_filename=os.path.join(folder, 'log.csv'),
        )

        if global_step % 1000 == 0:
            tf_rate_val = get_teacher_forcing_rate()
            print(
                f"[Step {global_step:06d}/{steps:06d}]"
                f"  gen={float(a):.4f}  gan={float(b):.4f}"
                f"  l1={float(c):.4f}  disc={float(d):.4f}"
                f"  gp={float(e):.4f}  temp={float(f):.4f}"
                f"  perc={float(h):.4f}  frame={float(i):.4f}"
                f"  TF={float(tf_rate_val):.3f}"
            )
            summarize_performance(
                global_step, X_train, generator, discriminator,
                folder, n_samples=minibatch[1] // 2, g_loss=float(a),
            )

        if float(i) * FRAME < best_loss:
            best_loss = float(i) * FRAME
            generator.save_weights(
                os.path.join(folder, 'best_generator.h5')
            )
            print(
                f"  ↳ New best saved"
                f" (frame_loss={float(i):.4f})"
            )


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(folder, exist_ok=True)
    os.system(f'cp train.py {folder}/')
    os.system(f'cp models.py {folder}/')
    os.system(f'cp losses.py {folder}/')
    os.system(f'cp utils.py {folder}/')

    # -- build models --------------------------------------------------------
    generator_base = define_generator(
        z_shape=(400,),
        image_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
    )
    generator_base.summary()

    generator     = CreateARLTCGAN(generator_base,
                                   dim_z_motion=200,
                                   dim_z_content=200)
    discriminator = build_discriminator_3d(
        vid_shape=(minibatch[1], FRAME_SIZE[0], FRAME_SIZE[1], 3)
    )
    discriminator.summary()
    ckpt_manager = GANCheckpointManager(
        folder         = folder,
        generator      = generator,
        discriminator  = discriminator,
        discriminator2 = None,
        gen_opt        = generator_optimizer,
        disc_opt       = discriminator_optimizer,
        disc2_opt      = None,
        n_frames       = minibatch[1],
        # build_disc_fn  = lambda: build_stylegan_v_discriminator(
        #                      frame_size=FRAME_SIZE, k_frames=K_FRAMES),
        build_disc_fn=lambda: build_discriminator_3d(
        vid_shape=(minibatch[1], FRAME_SIZE[0], FRAME_SIZE[1], 3)),
        build_disc2_fn = None,
        save_every     = 5000,
        keep_last      = 3,
    )
    start_z, start_epoch,restored_generator,restored_disc = ckpt_manager.restore_if_available()
    if restored_generator is not None:
        generator=restored_generator
    if restored_disc is not None:
        discriminator=restored_disc
 
    # -- dataset & train -----------------------------------------------------
    video_dirs = make_paths(BASE_DIR, minibatch[1])
    video_dataset    = make_dataset(video_dirs, minibatch[1], 4)
    for video in video_dataset.take(1):
        print(f"[DEBUG] sample video shape={video.shape}", flush=True)
    # Quick shape check
    for v in video_dataset.take(1):
        print("Sample video shape:", v.shape)

    train_gan(generator, discriminator, video_dataset)
