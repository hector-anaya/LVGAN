#!/usr/bin/env python
# coding: utf-8
"""
models.py
=========
All model architecture definitions for exp_test_claude_taichi.

Changes vs exp8:
  - Reverted to standard float32 discriminator (build_discriminator_3d)
  - tf.while_loop in CreateARLTCGAN.call() and sample_z_m() — no graph unrolling
  - Removed blur_frame (commented out in original)
  - Decoder skip connections set to False for d5/d6/d7

  - TemporalSelfAttention
  - myreshape
  - define_encoder_block / decoder_block / bottleneck
  - define_generator
  - CreateARLTCGAN
  - build_discriminator_3d
  - sample_random_stride_subsequence
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Conv3D, Dropout,
    Activation, BatchNormalization, Concatenate,
    UpSampling2D, Add, MultiHeadAttention, LayerNormalization,
    LeakyReLU,
)
from keras.models import Model
from ncps.tf import LTCCell
import ncps


# ──────────────────────────────────────────────────────────────────────────────
#  DATA HELPER
# ──────────────────────────────────────────────────────────────────────────────

def sample_random_stride_subsequence(video, seq_length, max_stride=4):
    """
    Draw `seq_length` frames from video [T, H, W, C] with a random temporal
    stride. Uses circular indexing when the clip is too short.
    """
    total_frames = tf.shape(video)[0]
    stride = tf.random.uniform(
        shape=[], minval=1, maxval=max_stride + 1, dtype=tf.int32
    )
    required = seq_length * stride

    def enough_frames():
        max_start = total_frames - required
        start = tf.random.uniform(
            shape=[], minval=0, maxval=max_start + 1, dtype=tf.int32
        )
        return tf.gather(video, start + tf.range(seq_length) * stride)

    def not_enough_frames():
        start = tf.random.uniform(
            shape=[], minval=0, maxval=total_frames, dtype=tf.int32
        )
        return tf.gather(video,
                         (start + tf.range(seq_length) * stride) % total_frames)

    return tf.cond(total_frames >= required, enough_frames, not_enough_frames)


# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM LAYERS
# ──────────────────────────────────────────────────────────────────────────────

class TemporalSelfAttention(tf.keras.layers.Layer):
    """Causal multi-head self-attention over a sequence of LTC hidden states."""

    def __init__(self, num_heads=4, key_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        embed_dim  = num_heads * key_dim   # 256
        self.attn  = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                        dropout=dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.ff1   = Dense(embed_dim * 2, activation='relu')
        self.ff2   = Dense(512)           # must match hidden-state dim (512)
        self.drop  = tf.keras.layers.Dropout(dropout)

    def call(self, h_sequence, training=False):
        x       = tf.expand_dims(h_sequence, axis=0)   # (1, seq_len, 512)
        seq_len = tf.shape(x)[1]

        # TF-2.15-compatible causal mask (float)
        causal_mask = (1.0 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )) * -1e9

        attn_out = self.attn(x, x, attention_mask=causal_mask, training=training)
        x = self.norm1(x + self.drop(attn_out, training=training))
        x = self.norm2(x + self.drop(self.ff2(self.ff1(x)), training=training))
        return tf.squeeze(x, axis=0)                    # (seq_len, 512)


class myreshape(tf.keras.layers.Layer):
    def call(self, x, shape1=8, shape2=8, shape3=128):
        return tf.reshape(x, [-1, shape1, shape2, shape3])


# ──────────────────────────────────────────────────────────────────────────────
#  GENERATOR BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer=init)(layer_in)
    g = tf.keras.layers.AveragePooling2D(2)(g)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    return LeakyReLU(alpha=0.2)(g)


def decoder_block(layer_in, skip_in, n_filters, dropout=True, skip=False):
    init = tf.random_normal_initializer(0., 0.02)
    g = UpSampling2D()(layer_in)
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    if skip:
        g = Concatenate()([g, skip_in])
    return Activation('relu')(g)


def bottleneck(layer_in, skip_in, n_filters):
    init = tf.random_normal_initializer(0., 0.02)
    g = Conv2D(n_filters, (4, 4), strides=(1, 1), padding='same',
               kernel_initializer=init)(layer_in)
    g = LeakyReLU(alpha=0.2)(g)
    g = Dropout(0.2)(g)
    return Add()([g, skip_in])


# ──────────────────────────────────────────────────────────────────────────────
#  U-NET GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def define_generator(z_shape, image_shape=(256, 256, 3),
                     in_encode_shape=(1, 1, 512)):
    init      = tf.random_normal_initializer(0., 0.02)
    z         = Input(shape=z_shape)
    in_encode = Input(shape=in_encode_shape)
    in_image  = Input(shape=image_shape)

    # Encoder  256→128→64→32→16→8→4→2→1
    e1 = define_encoder_block(in_image,  64, batchnorm=False)
    e2 = define_encoder_block(e1,  128)
    e3 = define_encoder_block(e2,  256)
    e4 = define_encoder_block(e3,  512)
    e5 = define_encoder_block(e4,  512)
    e6 = define_encoder_block(e5,  512)
    e7 = define_encoder_block(e6,  512)
    e8 = define_encoder_block(e7,  512)

    # Latent z → 8×8×128 spatial tensor
    zb = myreshape()(
        Dense(8 * 8 * 128, kernel_initializer=init)(z),
        shape1=8, shape2=8, shape3=128,
    )

    # Decode content code (hidden state) up to 8×8
    d_encode1 = decoder_block(in_encode, e4, 128, skip=False)
    d_encode2 = decoder_block(d_encode1, e4, 256, skip=False)
    d_encode3 = decoder_block(d_encode2, e4, 512, skip=False)

    # Fuse z + e5 + decoded content code at 8×8
    d1 = Concatenate()([zb, e5, d_encode3])
    b1 = bottleneck(d1, d1, 1152)
    b2 = bottleneck(b1, b1, 1152)
    b3 = bottleneck(b2, b2, 1152)

    # NOTE: d5/d6/d7 have skip=False (exp10 change vs exp8)
    d4 = decoder_block(b3, e4, 512,               skip=True)    # 16
    d5 = decoder_block(d4, e3, 256, dropout=False, skip=False)  # 32
    d6 = decoder_block(d5, e2, 128, dropout=False, skip=False)  # 64
    d7 = decoder_block(d6, e1,  64, dropout=False, skip=False)  # 128

    g = UpSampling2D()(d7)                                       # 256
    g = Conv2D(3, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)

    return Model([in_image, z, in_encode], [out_image, e8])


# ──────────────────────────────────────────────────────────────────────────────
#  VAE ENCODER  (first-frame -> z_content during training)
# ──────────────────────────────────────────────────────────────────────────────

def build_vae_encoder(image_shape=(256, 256, 3), z_dim=200):
    """CNN that encodes a single frame to (mu, logvar) of dim z_dim.

    Used by CreateARLTCGAN at TRAINING time only. At inference, z_content is
    sampled directly from N(0, I) — see CreateARLTCGAN.call().
    """
    init = tf.random_normal_initializer(0., 0.02)
    in_image = Input(shape=image_shape)

    x = Conv2D(64,  4, strides=2, padding='same', kernel_initializer=init)(in_image)   # 128
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init)(x)          # 64
    x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init)(x)          # 32
    x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(x)          # 16
    x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(x)          # 8
    x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)                                    # (B, 512)

    mu     = Dense(z_dim, kernel_initializer=init, name='vae_mu')(x)
    logvar = Dense(z_dim, kernel_initializer=init, name='vae_logvar')(x)
    return Model(in_image, [mu, logvar], name='vae_encoder')


# ──────────────────────────────────────────────────────────────────────────────
#  AUTOREGRESSIVE LTC-GAN WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class CreateARLTCGAN(tf.keras.Model):
    """
    Autoregressive video generator driven by LTC cells.

    Key change in exp10: Python for-loops replaced with tf.while_loop in
    both call() and sample_z_m() to prevent graph unrolling at long sequences.
    """

    def __init__(self, generator, dim_z_motion, dim_z_content,
                 image_shape=(256, 256, 3)):
        super().__init__()
        self.gen           = generator
        self.dim_z_motion  = tf.constant(dim_z_motion)
        self.dim_z_content = tf.constant(dim_z_content)
        self._dim_z_content_int = int(dim_z_content)   # Python int for shapes

        # VAE encoder: first_frame -> (mu, logvar) for z_content during training.
        # Replaces the old sample_z_content() random sampling at train time.
        self.vae_encoder = build_vae_encoder(image_shape, dim_z_content)

        # Learnable start frame — replaces og_frames[0] as step-0 input to the
        # U-Net. The model is now unconditional: at inference no real frame is
        # ever fed in.
        self.start_frame = tf.Variable(
            tf.random.normal(image_shape, stddev=0.02),
            trainable=True, name='start_frame'
        )

        wiring = ncps.wirings.Random(
            dim_z_motion, output_dim=dim_z_motion, sparsity_level=0.5
        )
        self.ltccell = LTCCell(wiring)

        wiring2 = ncps.wirings.Random(512, output_dim=512, sparsity_level=0.5)
        self.ltccell_image = LTCCell(wiring2)

        self.temporal_attn = TemporalSelfAttention(
            num_heads=4, key_dim=64, dropout=0.1
        )

    # ── VAE encode + reparameterize ──────────────────────────────────────────
    def encode_first_frame(self, first_frame):
        """Run the VAE on a single (H,W,3) frame; return (z_content, mu, logvar)."""
        x = tf.expand_dims(first_frame, 0)               # (1, H, W, 3)
        mu, logvar = self.vae_encoder(x, training=True)  # each (1, dim_z_content)
        eps = tf.random.normal(tf.shape(mu))
        z_content = mu + tf.exp(0.5 * logvar) * eps
        return z_content, mu, logvar

    # ── attention refinement ─────────────────────────────────────────────────
    def _refine_h_sequence(self, h_array, length, training):
        h_seq = tf.squeeze(h_array.stack(), axis=1)  # (length, 512)
        h_seq = self.temporal_attn(h_seq, training=training)
        return h_seq[-1]                              # (512,)

    # ── forward pass (tf.while_loop — no graph unrolling) ────────────────────
    def call(self, n, og_frames=None, teacher_forcing_rate=0,
             noise=None, training=False):
        # ── z_content from VAE (training, og_frames given) or N(0,I) (inference) ─
        if og_frames is not None:
            z_content_single, mu, logvar = self.encode_first_frame(og_frames[0])
        else:
            z_content_single = tf.random.normal((1, self._dim_z_content_int))
            mu     = tf.zeros((1, self._dim_z_content_int))
            logvar = tf.zeros((1, self._dim_z_content_int))

        z_content = tf.repeat(z_content_single, n, axis=0)   # (n, dim_z_content)
        z_motion  = self.sample_z_m(1, n)                    # (n, dim_z_motion)
        z         = tf.concat([z_content, z_motion], axis=1) # (n, total)

        frames = tf.TensorArray(dtype=tf.float32, size=n)
        h_array = tf.TensorArray(
            dtype=tf.float32, size=n + 1, clear_after_read=False
        )
        h_array     = h_array.write(0, tf.random.normal((1, 512)))
        input_frame = self.start_frame                       # learnable, not og_frames[0]
        ATTN_EVERY  = 4

        def _gen_cond(i, h_array, frames, input_frame):
            return i < n

        def _gen_body(i, h_array, frames, input_frame):
            h_prev = h_array.read(i)

            fake, e_t = self.gen(
                [
                    tf.expand_dims(input_frame, axis=0),
                    tf.expand_dims(z[i],        axis=0),
                    tf.expand_dims(tf.expand_dims(h_prev, axis=0), axis=0),
                ],
                training=training,
            )

            content_code = 0.9 * h_prev + 0.1 * e_t[0][0]
            h, _ = self.ltccell_image(content_code, h_prev)
            h_array = h_array.write(i + 1, h)

            # Periodic attention refinement (inside while_loop via tf.cond)
            do_refine = tf.logical_and(i > 0, tf.equal((i + 1) % ATTN_EVERY, 0))
            h_array = tf.cond(
                do_refine,
                lambda: h_array.write(
                    i + 1,
                    tf.expand_dims(
                        self._refine_h_sequence(h_array, i + 2, training), 0
                    ),
                ),
                lambda: h_array,
            )

            frames = frames.write(i, fake[0])

            # Teacher forcing / autoregressive input selection
            if training:
                def _teacher_or_student():
                    use_real  = tf.less(
                        tf.random.uniform([], 0.0, 1.0), teacher_forcing_rate
                    )
                    safe_idx  = tf.minimum(i + 1, n - 1)
                    real_next = og_frames[safe_idx]
                    return tf.cond(use_real,
                                   lambda: real_next,
                                   lambda: fake[0])
                next_input = tf.cond(
                    i + 1 < n, _teacher_or_student, lambda: fake[0]
                )
            else:
                next_input = fake[0]

            return i + 1, h_array, frames, next_input

        _, h_array, frames, _ = tf.while_loop(
            _gen_cond,
            _gen_body,
            loop_vars=[tf.constant(0), h_array, frames, input_frame],
            parallel_iterations=1,
        )

        _ = h_array.stack()
        return tf.expand_dims(frames.stack(), axis=0), mu, logvar

    # ── latent sampling ───────────────────────────────────────────────────────
    def get_ltc_initial_state(self, num_samples):
        return tf.random.normal((num_samples, self.dim_z_motion))

    def get_iteraion_noise(self, num_samples):
        return tf.random.normal((num_samples, self.dim_z_motion))

    def sample_z_content(self, num_samples, n):
        return tf.repeat(
            tf.random.normal((num_samples, self.dim_z_content)), n, axis=0
        )

    def sample_z_m(self, num_samples, n):
        """Motion latent sequence via LTC cell (tf.while_loop)."""
        h_array = tf.TensorArray(
            dtype=tf.float32, size=n + 1, clear_after_read=False
        )
        h_array = h_array.write(0, self.get_ltc_initial_state(num_samples))

        def _zm_cond(i, h_array):
            return i < n

        def _zm_body(i, h_array):
            h_prev  = h_array.read(i)
            h, _    = self.ltccell(self.get_iteraion_noise(num_samples), h_prev)
            h_array = h_array.write(i + 1, h)
            return i + 1, h_array

        _, h_array = tf.while_loop(
            _zm_cond, _zm_body,
            loop_vars=[tf.constant(0), h_array],
            parallel_iterations=1,
        )
        return h_array.concat()[1:]   # skip initial state


# ──────────────────────────────────────────────────────────────────────────────
#  DISCRIMINATOR  (float32, reverted from fp16 in exp9)
# ──────────────────────────────────────────────────────────────────────────────

def build_discriminator_3d(vid_shape=(16, 256, 256, 3)):
    """
    3-D convolutional video discriminator (float32).
    vid_shape: (frames, height, width, channels)
    """
    init   = tf.random_normal_initializer(0., 0.02)
    in_src = Input(shape=vid_shape)

    x = Conv3D(128,  3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(in_src)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(256,  3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(256,  3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(512,  3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(1024, 3, strides=(1, 2, 2), padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(1, 3, padding='same', kernel_initializer=init)(x)

    return Model(inputs=in_src, outputs=x)
