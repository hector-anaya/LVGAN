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
parser.add_argument('--folder', type=str, default='exp_new_test', help='Folder to save logs and models')
parser.add_argument('--gpu', type=str, default='0', help='GPU device to use (e.g., "0", "1", "0,1")')
parser.add_argument('--islocal', type=bool, default=0, help='Flag to indicate local execution (for debugging)')
parser.add_argument('--profile', action='store_true', help='Profile a short window of steps to <folder>/profile/ and exit')
parser.add_argument('--profile_warmup', type=int, default=5, help='Steps to warm up before profiler starts')
parser.add_argument('--profile_steps', type=int, default=10, help='Number of steps to capture in the profile')
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
    BASE_DIR='/usuarios03/hector.anaya1/Datasets/UCF101processed'
# Teacher-forcing schedule
TF_RATE_INIT   = 1.0
TF_MIN         = 0.1
TF_DECAY_STEPS = 100_000
# Step number for profiler step markers — assigned each iteration in train_gan
# so the @tf.function sees a fresh value without retracing.
_profile_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
tf_step_counter = tf.Variable(0, trainable=False, dtype=tf.float32)

# ──────────────────────────────────────────────────────────────────────────────
#  LOSS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA = 1
TEMP   = 0.01
GRAD   = 10
FRAME  = 10000
PERC   = 1000

# ── R1 regularization (replaces WGAN-GP gradient penalty) ────────────────────
# Computed every disc step (lazy R1 via tf.cond was attempted but the nested
# 2nd-order tape returns None in graph mode — see train_step for details).
R1_GAMMA = 1.0


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

# ── pre-compute strided frame paths in pure Python (once) ─────────────────
def sample_frame_paths(video_dir, seq_length, max_stride):
    frame_paths = sorted(glob(os.path.join(video_dir, "*.jpg")))
    total = len(frame_paths)
    if total == 0:
        return None

    stride = np.random.randint(1, max_stride + 1)
    required_span = 1 + (seq_length - 1) * stride

    if total >= required_span:
        start = np.random.randint(0, total - required_span + 1)
        indices = start + np.arange(seq_length) * stride
    else:
        start = np.random.randint(0, total)
        indices = (start + np.arange(seq_length) * stride) % total

    return [frame_paths[i] for i in indices]


# collect video dirs (same logic as before)
def make_paths(BASE_DIR,seq):
    #video_dir=[]
    #for name in os.listdir(BASE_DIR):
     #   if os.path.isdir(os.path.join(BASE_DIR, name)):
      #      # print(name)
       #     for name_in in os.listdir(os.path.join(BASE_DIR, name)):
        #        if os.path.isdir(os.path.join(BASE_DIR, name, name_in)) and len(glob(os.path.join(BASE_DIR, name, name_in, "*.jpg"))) >= seq:
                    # print(name_in)
         #           video_dir.append(os.path.join(BASE_DIR, name, name_in))
#    video_dir.sort()
#    print(f"Found {len(video_dir)} video directories.")
 #   return video_dir
    video_dirs=[]
    for name in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, name)):
            if os.path.isdir(os.path.join(BASE_DIR, name)) and len(glob(os.path.join(BASE_DIR, name, "*.jpg"))) >= SEQ_LENGHT:
                # print(name_in)
                video_dirs.append(os.path.join(BASE_DIR, name))
    video_dirs.sort()
    print(f"Found {len(video_dirs)} video directories.")
   # print(f"Found {len(video_dir)} video directories.")
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
    Re-samples strides every time make_dataset() is called.
    Call once per epoch to get fresh random strides.
    """
    # sample all paths in Python — O(n_videos), negligible cost
    all_selected = []
    for vd in video_dirs:
        paths = sample_frame_paths(vd, seq_length, max_stride)
        if paths is not None:
            all_selected.append(paths)          # list of SEQ_LENGTH strings

    # shape: (n_videos, SEQ_LENGTH) — from_tensor_slices gives one row per video
    path_ds = tf.data.Dataset.from_tensor_slices(all_selected)

    # map load_frame over each row (SEQ_LENGTH paths) → (SEQ_LENGTH, H, W, C)
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
    gan_loss   = generator_gan_loss(disc_generated_output)    # non-saturating logistic
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

    # ── Generate fakes once for the disc update (no gradient into generator) ─
    # Cuts the AR generator from 4 calls/step (3 disc + 1 gen) down to 2.
    fake_for_disc = tf.stop_gradient(generator(
        n=minibatch[1], og_frames=target[0],
        noise=tf.random.uniform((1, 400)),
    ))

    # ── 1× discriminator update with R1 regularization ───────────────────────
    # Uses tf.gradients (graph-mode native) for the input gradient instead of
    # a nested GradientTape — the tape's watch() of a function-arg tensor does
    # not always connect through Keras Model __call__, returning None. This is
    # the same pattern used in Nvidia's reference StyleGAN2 TF implementation.
    # The outer disc_tape still records tf.gradients' output and computes the
    # second-order gradient wrt disc params correctly.
    with tf.profiler.experimental.Trace('disc_update'):
        with tf.GradientTape() as disc_tape:
            d_real      = discriminator(target, training=True)
            real_grads  = tf.gradients(tf.reduce_sum(d_real), target)[0]
            r1 = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3, 4])
            )

            d_fake    = discriminator(fake_for_disc, training=True)
            disc_loss = discriminator_loss(d_real, d_fake)
            d_total   = disc_loss + R1_GAMMA * r1

        d_grads = disc_tape.gradient(d_total, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables)
        )

    # ── 1× generator update ──────────────────────────────────────────────────
    with tf.profiler.experimental.Trace('gen_update'):
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
            disc_loss, r1, temp, perp, grad, fram)


@tf.function
def train_step_multi(target, generator, discriminator):
    """Average losses over minibatch[0] mini-clip steps.
    Outer Trace is INSIDE the @tf.function so the step marker is captured on
    the device timeline (host-side wrapping in eager mode ends before async
    GPU work finishes, which is why TB's Overview page sees "no step marker").
    Inner mini-clip loop is Python-unrolled so each iteration emits its own
    Trace region.
    """
    with tf.profiler.experimental.Trace('train_step', step_num=_profile_step_var):
        aa = tf.zeros(9)
        for j in range(minibatch[0]):
            k      = j * (minibatch[1] - 1)
            X_mini = target[k : minibatch[1] + k]
            with tf.profiler.experimental.Trace('mini_clip', step_num=j):
                aa = aa + train_step(
                    tf.expand_dims(X_mini, axis=0), generator, discriminator
                )
        return aa / tf.cast(minibatch[0], tf.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_gan(generator, discriminator, dataset, epochs=1000):
    os.makedirs(folder, exist_ok=True)
    global_step = 0
    best_loss   = np.inf

    profile_dir   = os.path.join(folder, 'profile')
    profile_start = args.profile_warmup
    profile_stop  = args.profile_warmup + args.profile_steps

    for epoch in range(epochs):
        video_dirs = make_paths(BASE_DIR, SEQ_LENGHT)
        video_dataset    = make_dataset(video_dirs, SEQ_LENGHT, 1)
        for video in video_dataset.take(1):
            print(f"[DEBUG] sample video shape={video.shape}", flush=True)
        for step, batch in enumerate(video_dataset):
            X_train = batch[0]

            if args.profile and global_step == profile_start:
                tf.profiler.experimental.start(profile_dir)
                print(f"[profiler] started — capturing {args.profile_steps} steps to {profile_dir}")
            _profile_step_var.assign(global_step)
            a, b, c, d, e, f, g, h, i = train_step_multi(
                    X_train, generator, discriminator
                )

            if args.profile and global_step == profile_stop - 1:
                tf.profiler.experimental.stop()
                print(f"[profiler] stopped — view with: tensorboard --logdir {profile_dir}")
                return

            log_loss_to_csv(
                epoch, global_step, a, b, c, d, e, f, g, h, i,
                csv_filename=os.path.join(folder, 'log.csv'),
            )

            if global_step % 1000 == 0:
                tf_rate_val = get_teacher_forcing_rate()
                print(
                    f"[Epoch {epoch:04d} | Step {step:05d} | GS {global_step:06d}]"
                    f"  gen={float(a):.4f}  gan={float(b):.4f}"
                    f"  l1={float(c):.4f}  disc={float(d):.4f}"
                    f"  r1={float(e):.4f}  temp={float(f):.4f}"
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

            global_step += 1


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
        n_frames       = SEQ_LENGHT,
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
    video_dirs = make_paths(BASE_DIR, SEQ_LENGHT)
    video_dataset    = make_dataset(video_dirs, SEQ_LENGHT, 1)
    for video in video_dataset.take(1):
        print(f"[DEBUG] sample video shape={video.shape}", flush=True)
    # Quick shape check
    for v in video_dataset.take(1):
        print("Sample video shape:", v.shape)

    train_gan(generator, discriminator, video_dataset, epochs=1000)
