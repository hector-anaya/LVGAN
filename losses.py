#!/usr/bin/env python
# coding: utf-8
"""
losses.py
=========
All loss functions, optimizers, and LR schedules for exp_test_claude_taichi.

  - WarmupCosineDecay (LR schedule)
  - generator_optimizer / discriminator_optimizer  (plain Adam, no LossScale)
  - perceptual_loss    (VGG19 block4_conv2, single random frame)
  - temporal_consistency_loss  (VGG19 block3_conv3, per-frame + temporal-diff)
  - gradient_temporal_loss     (second-order finite differences)
  - frame_difference_loss
  - generator_loss
  - discriminator_loss
  - gradient_penalty           (WGAN-GP, discriminator passed explicitly)
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
# from train import LAMBDA, PERC, TEMP, GRAD, FRAME


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, peak_lr, warmup_steps, total_steps,
                 min_lr_fraction=0.1):
        self.peak   = peak_lr
        self.warmup = warmup_steps
        self.total  = total_steps
        self.min_lr = peak_lr * min_lr_fraction

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.peak * (step / self.warmup)
        progress  = tf.minimum(
            (step - self.warmup) / (self.total - self.warmup), 1.0
        )
        cosine_lr = self.min_lr + 0.5 * (self.peak - self.min_lr) * (
            1.0 + tf.cos(3.14159265 * progress)
        )
        return tf.where(step < self.warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(peak_lr=self.peak, warmup_steps=self.warmup,
                    total_steps=self.total)


#──────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTORS  (built once at import time)
# ──────────────────────────────────────────────────────────────────────────────

def _build_vgg19_perceptual():
    """Full VGG19 for perceptual loss on 256×256 frames."""
    vgg = tf.keras.applications.VGG19(
        weights='imagenet', input_shape=(256, 256, 3), include_top=False
    )
    vgg.trainable = False
    return vgg

def _build_vgg19_temporal():
    """VGG19 up to block3_conv3 for temporal consistency."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return tf.keras.Model(inputs=vgg.input,
                          outputs=vgg.get_layer('block3_conv3').output)

vgg_perceptual    = _build_vgg19_perceptual()
feature_extractor = _build_vgg19_temporal()


# ──────────────────────────────────────────────────────────────────────────────
#  INDIVIDUAL LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def perceptual_loss(fake, real, minibatch_size=16):
    """
    L1 on VGG19 (block4_conv2) features for one randomly chosen frame.
    fake / real : (B, T, H, W, 3)  in [-1, 1]
    """
    idx = tf.random.uniform(
        shape=(), minval=0, maxval=minibatch_size, dtype=tf.int32
    )

    def to_vgg(vid):
        frame = tf.image.resize(vid[:, idx], [256, 256])
        frame = tf.cast((frame + 1.0) * 127.5, tf.uint8)
        return preprocess_input(tf.cast(frame, tf.float32))

    fake_feat = vgg_perceptual(to_vgg(fake), training=False)
    real_feat = vgg_perceptual(to_vgg(real), training=False)
    return tf.reduce_mean(tf.abs(fake_feat - real_feat))


def temporal_consistency_loss(video_real, video_generated):
    """
    VGG19 (block3_conv3) temporal consistency.
    Penalises per-frame reconstruction error + adjacent-frame delta mismatch.
    video_real / video_generated : (B, T, H, W, C)  in [-1, 1]
    """
    B, T, H, W, C = tf.unstack(tf.shape(video_generated))

    def extract(video):
        frames = tf.reshape(video, [B * T, H, W, C])
        frames = tf.keras.applications.vgg19.preprocess_input(
            (frames + 1.0) * 127.5
        )
        feats  = feature_extractor(frames, training=False)
        _, fh, fw, fch = tf.unstack(tf.shape(feats))
        return tf.reshape(feats, [B, T, fh, fw, fch])

    fr = extract(video_real)
    fg = extract(video_generated)

    l_recon    = tf.reduce_mean(tf.square(fg - fr))
    l_temporal = tf.reduce_mean(
        tf.square((fg[:, 1:] - fg[:, :-1]) - (fr[:, 1:] - fr[:, :-1]))
    )
    return l_recon + l_temporal


def gradient_temporal_loss(video, gen):
    """Second-order finite difference (acceleration) consistency."""
    dt   = video[:, 2:] - 2 * video[:, 1:-1] + video[:, :-2]
    dt_g =   gen[:, 2:] - 2 *   gen[:, 1:-1] +   gen[:, :-2]
    return tf.reduce_mean(tf.abs(dt - dt_g))


def frame_difference_loss(gen, gt):
    """First-order temporal difference consistency."""
    return tf.reduce_mean(
        tf.square((gen[:, 1:] - gen[:, :-1]) - (gt[:, 1:] - gt[:, :-1]))
    )


# ──────────────────────────────────────────────────────────────────────────────
#  COMPOSITE LOSSES
# ──────────────────────────────────────────────────────────────────────────────


def discriminator_loss(disc_real_output, disc_generated_output):
    return (-tf.reduce_mean(disc_real_output)
            + tf.reduce_mean(disc_generated_output))


def gradient_penalty(discriminator, x, x_gen):
    """
    WGAN-GP gradient penalty.
    discriminator is passed explicitly so this function has no global deps.
    """
    x     = tf.cast(x,     tf.float32)
    x_gen = tf.cast(x_gen, tf.float32)
    eps   = tf.random.uniform([tf.shape(x)[0], 1, 1, 1, 1], 0.0, 1.0)
    x_hat = x + eps * (x_gen - x)

    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator(x_hat, training=True)

    grads = t.gradient(d_hat, x_hat)
    ddx   = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
    return tf.reduce_mean((ddx - 1.0) ** 2)
