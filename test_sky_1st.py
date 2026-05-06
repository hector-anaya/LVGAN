#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# from tensorflow.keras.datasets import mnist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import numpy as np
from ncps.tf import LTC,LTCCell
import ncps
import einops
from glob import glob
from keras.models import Model



from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from ncps.wirings import AutoNCP
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer, LeakyReLU,Input,Lambda,UpSampling3D, Layer, Dense, Conv2D, Dropout,Flatten,Reshape,Conv2DTranspose,Activation,BatchNormalization,Conv3D,Concatenate,UpSampling2D,Add,SpectralNormalization,MultiHeadAttention,LayerNormalization

#from tensorflow_addons.layers import SpectralNormalization


# In[2]:


from utils import *
import matplotlib
matplotlib.use("Agg")   # use the headless backend

# In[3]:
#exp_test_claude added Temporal self attention, teacher enforcing, minibatch to 8,16, added prefetch, mod encoder con2d, bigger disc, addedd blur to final image, added mixed_float16
#exp 2 increased tf deacay steps to 10000->50000
#tf.keras.mixed_precision.set_global_policy('mixed_float16')   #claude idea
#exp 3 added newtemporal loss, added paralell dataset, 
#exp 4 increased frame size to 256x256, modified generator, and modified bottleneck of generator
#exp 5 added random stride training and more folders to dataset loader.
#exp 6 addded optmizer with lr schedule, 1 10 10 10 10 10, increment teacher forcing to 100000, added float16 discriminator
#exp 7 modified training loop 1 100 10 100 1000 10
#exp 8 1 0.1 10 100 1000 10
#exp 9 reverted disc 
#exp 10 deleted some skips, modifed so everything inside tf.function is a tf.whileloop

folder="exp_test_claude_taichi/"
loss_object = tf.keras.losses.BinaryCrossentropy()
LAMBDA = 1                                                              
TEMP=0.01
GRAD=10
FRAME=10000
PERC=1000
PEN=10
SEQ_LENGHT=128
FRAME_SIZE=(256,256)
minibatch=(8,16)
MAX_STRIDE = 1


# -- Teacher forcing schedule -------------------------------------------
# Starts at 1.0 (always use real frame as input) and decays toward
# TF_MIN over TF_DECAY_STEPS global steps.
# High TF_RATE early = stable gradients + content anchoring.
# Low TF_RATE later  = model learns to handle its own outputs.
TF_RATE_INIT  = 1.0    # start: always feed real frames
TF_MIN        = 0.1    # floor: always keep some real-frame grounding
TF_DECAY_STEPS = 100000 # steps to decay from init ? min
tf_step_counter = tf.Variable(0, trainable=False, dtype=tf.float32)

def get_teacher_forcing_rate():
    """Linear decay from TF_RATE_INIT ? TF_MIN over TF_DECAY_STEPS."""
    progress = tf.minimum(tf_step_counter / TF_DECAY_STEPS, 1.0)
    return TF_RATE_INIT - progress * (TF_RATE_INIT - TF_MIN)



generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5,beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5,beta_2=0.999)

if not os.path.exists(folder):
	os.mkdir(folder)
os.system(f'cp test_sky_1st.py {folder}/')
# In[4]:
#BASE_DIR='/home/hector.anaya/Datasets/sky_timelapse/sky_train'
BASE_DIR='/home/hector.anaya/Datasets/TaiChi'

def load_video_py(path):
    path = path.numpy().decode("utf-8")
    frame_paths = sorted(glob(os.path.join(path, "*.jpg")))

    total_frames = len(frame_paths)
    if total_frames == 0:
        raise ValueError(f"No frames found in {path}")

    # choose stride
    stride = 1

    # needed temporal span
    required_span = 1 + (SEQ_LENGHT - 1) * stride

    if total_frames >= required_span:
        start = np.random.randint(0, total_frames - required_span + 1)
        indices = start + np.arange(SEQ_LENGHT) * stride
    else:
        # circular indexing if not enough frames
        start = np.random.randint(0, total_frames)
        indices = (start + np.arange(SEQ_LENGHT) * stride) % total_frames

    selected_paths = [frame_paths[i] for i in indices]

    frames = []
    for f in selected_paths:
        img_raw = tf.io.read_file(f)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, FRAME_SIZE)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        frames.append(img)
    return tf.stack(frames)
# Get all video directories

#for name in os.listdir(BASE_DIR):
 #   if os.path.isdir(os.path.join(BASE_DIR, name)):
  #      # print(name)
   #     for name_in in os.listdir(os.path.join(BASE_DIR, name)):
     #       if os.path.isdir(os.path.join(BASE_DIR, name, name_in)) and len(glob(os.path.join(BASE_DIR, name, name_in, "*.jpg"))) >= 128:
    #            # print(name_in)
      #          video_dirs.append(os.path.join(BASE_DIR, name, name_in))
video_dirs=[]
for name in os.listdir(BASE_DIR):
    if os.path.isdir(os.path.join(BASE_DIR, name)):
        if os.path.isdir(os.path.join(BASE_DIR, name)) and len(glob(os.path.join(BASE_DIR, name, "*.jpg"))) >= 128:
                # print(name_in)
            video_dirs.append(os.path.join(BASE_DIR, name))
video_dirs.sort()
print(f"Found {len(video_dirs)} video directories.")


def load_video_tf(path):
    video=tf.py_function(func=load_video_py,inp=[path],Tout=tf.float32)
    # video.set_shape([None, FRAME_SIZE[0], FRAME_SIZE[1], 3])
    return video
# Build dataset
video_dataset = tf.data.Dataset.from_tensor_slices(video_dirs)
video_dataset = video_dataset.map(load_video_tf, num_parallel_calls=tf.data.AUTOTUNE) #new addition, parallel loading
video_dataset = video_dataset.cache()          # cache after first epoch
# Example usage
for video in video_dataset.take(1):
    print(video.shape)  # (num_frames, height, width, channels)


# # In[4]:


video_dataset = video_dataset.batch(1)
video_dataset = video_dataset.prefetch(tf.data.AUTOTUNE)


# # In[5]:


a=video_dataset.take(1)
x=[_ for _ in a]
# #---------------------------------------------------------------------------------------------------


# ── LR schedules ──────────────────────────────────────────────
TOTAL_STEPS   = 300_000   # adjust to your expected training length
WARMUP_STEPS  = 2_000
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr_fraction=0.1):
        self.peak      = peak_lr
        self.warmup    = warmup_steps
        self.total     = total_steps
        self.min_lr    = peak_lr * min_lr_fraction

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        # warmup phase
        warmup_lr = self.peak * (step / self.warmup)
        # cosine phase
        progress  = (step - self.warmup) / (self.total - self.warmup)
        progress  = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak - self.min_lr) * (
                        1.0 + tf.cos(3.14159265 * progress))
        return tf.where(step < self.warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(peak_lr=self.peak, warmup_steps=self.warmup,
                    total_steps=self.total)

generator_optimizer     = tf.keras.optimizers.Adam(
    WarmupCosineDecay(0.0002, WARMUP_STEPS, TOTAL_STEPS),
    beta_1=0.0, beta_2=0.999
)
discriminator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.Adam(
        WarmupCosineDecay(0.0001, WARMUP_STEPS, TOTAL_STEPS),
        beta_1=0.0, beta_2=0.999
    )
)
# In[6]:


#idea with autoregresive

def sample_random_stride_subsequence(video, seq_length, max_stride=4):
    """
    video: Tensor [T, H, W, C]
    seq_length: int
    max_stride: maximum temporal stride

    returns: Tensor [seq_length, H, W, C]
    """
    total_frames = tf.shape(video)[0]

    # choose stride
    stride = tf.random.uniform(
        shape=[],
        minval=1,
        maxval=max_stride + 1,
        dtype=tf.int32
    )

    # effective span needed
    required = seq_length * stride

    def enough_frames():
        max_start = total_frames - required
        start = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=max_start + 1,
            dtype=tf.int32
        )
        indices = start + tf.range(seq_length) * stride
        return tf.gather(video, indices)

    def not_enough_frames():
        # loop (circular) indexing
        start = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=total_frames,
            dtype=tf.int32
        )
        indices = (start + tf.range(seq_length) * stride) % total_frames
        return tf.gather(video, indices)

    return tf.cond(total_frames >= required, enough_frames, not_enough_frames)

# ---------------------------------------------
#  TEMPORAL SELF-ATTENTION
#  Sits on top of the LTC hidden state sequence.
#  Each timestep attends to ALL previous hidden
#  states ? directly fights content drift.
# ---------------------------------------------
class TemporalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim   = key_dim
        embed_dim      = num_heads * key_dim   # = 256

        self.attn  = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                         dropout=dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.ff1   = Dense(embed_dim * 2, activation='relu')  # 512
        self.ff2   = Dense(512)           # ? must match input dim (512), not embed_dim
        self.drop  = tf.keras.layers.Dropout(dropout)

    def call(self, h_sequence, training=False):
        # h_sequence: (seq_len, 512)
        x = tf.expand_dims(h_sequence, axis=0)        # (1, seq_len, 512)
    
        seq_len = tf.shape(x)[1]
    
        # TF 2.15 compatible mask  use float mask instead of bool
        # 0.0 = attend, 1e9 = mask out
        causal_mask = 1.0 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        causal_mask = causal_mask * -1e9               # (seq_len, seq_len)
    
        attn_out = self.attn(x, x, attention_mask=causal_mask, training=training)
        x = self.norm1(x + self.drop(attn_out, training=training))
    
        ff_out = self.ff2(self.ff1(x))
        x = self.norm2(x + self.drop(ff_out, training=training))
    
        return tf.squeeze(x, axis=0)                  # (seq_len, 512)
class myreshape(tf.keras.layers.Layer):
    def call(self,x,shape1=8,shape2=8,shape3=128):
        return tf.reshape(x,[-1,shape1,shape2,shape3])
        



def bottleneck(layer_in,skip_in,n_filters):
    init = tf.random_normal_initializer(0.,0.02)
    g=Conv2D(n_filters,(4,4),strides=(1,1),padding='same',kernel_initializer=init)(layer_in)
    g=LeakyReLU(alpha=0.2)(g)
    g=Dropout(0.2)(g)
    g=Add()([g,skip_in])
    return g


def define_encoder_block(layer_in, n_filters, batchnorm=True):

    init =tf.random_normal_initializer(0., 0.02)
    g = Conv2D(n_filters, (3,3), strides=(1,1),  padding='same',
         kernel_initializer=init)(layer_in)
    g=tf.keras.layers.AveragePooling2D(2)(g)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
      
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True,skip=False):
    # weight initialization
    init =tf.random_normal_initializer(0., 0.02)
# add upsampling layer
    g=UpSampling2D()(layer_in)
    g=Conv2D(n_filters,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(g)
# add batch normalization
    g = BatchNormalization()(g, training=True)
# conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    if skip:
        g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g   

def bottleneck(layer_in,skip_in,n_filters):
    init = tf.random_normal_initializer(0.,0.02)
    g=Conv2D(n_filters,(4,4),strides=(1,1),padding='same',kernel_initializer=init)(layer_in)
    g=LeakyReLU(alpha=0.2)(g)
    g=Dropout(0.2)(g)
    g=Add()([g,skip_in])
    return g


def define_generator(z_shape,image_shape=(256,256,3),in_encode_shape=(1,1,512)):
# weight initialization
    init =tf.random_normal_initializer(0., 0.02)
    # image input
    z=Input(shape=z_shape)
    in_encode=Input(shape=in_encode_shape) 
    in_image = Input(shape=image_shape) #128 256
    # encoder model: C64-C128-C256-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)  #64  128
    print(e1.shape)
    e2 = define_encoder_block(e1, 128) #32 64
    print(e2.shape)
    e3 = define_encoder_block(e2, 256)  #16  32
    print(e3.shape)
    e4 = define_encoder_block(e3, 512)  #8  16
    print(e4.shape)
    e5 = define_encoder_block(e4, 512)  #4  8
    print(e5.shape)
    e6 = define_encoder_block(e5, 512)  #2  4
    print(e6.shape)
    e7 = define_encoder_block(e6, 512)  #1  2
    print(e7.shape)
    e8 = define_encoder_block(e7, 512)  #-  1
    print(e8.shape)
    # bottleneck, no batch norm and relu
   # zdense = Dense(4*4*64, kernel_initializer=init)(z)
   # zl=LeakyReLU(alpha=0.2)(zdense)

    zdense = Dense(8*8*128, kernel_initializer=init)(z)
   # zl=LeakyReLU(alpha=0.2)(zdense)
    #zb=tf.reshape(zdense,[-1,8,8,128])
    zb=myreshape()(zdense,shape1=8,shape2=8,shape3=128) 
    
    # in_encode_dense= Dense(8*8*128, kernel_initializer=init)(in_encode)
    d_encode1 = decoder_block(in_encode, e4, 128,skip=False) #2 
    d_encode2 = decoder_block(d_encode1, e4, 256,skip=False) #4 
    d_encode3 = decoder_block(d_encode2, e4, 512,skip=False) #8 
    # in_encode_dense=myreshape()(in_encode_dense,shape1=8,shape2=8,shape3=128)
    #in_encode_dense=tf.reshape(in_encode_dense,[-1,8,8,128])  
    print(zb.shape)
    # decoder model: CD512-CD512-CD512-C512-C256-128-64
    print(zb.shape)
    print(e6.shape) 
    d1=Concatenate()([zb,e5])
    print(d1.shape)
    d1=Concatenate()([d1,d_encode3])
    print(d1.shape)
    b1=bottleneck(d1,d1,1152)
    print(b1.shape)
    b2=bottleneck(b1,b1,1152)
    print(b2.shape)
    b3=bottleneck(b2,b2,1152)
    print(b3.shape)


 #   d2 = decoder_block(d1, e6, 512) #2
   # print(d2.shape)
  #  d3 = decoder_block(d2, e5, 512,skip=True) #4
    d4 = decoder_block(b3, e4, 512,skip=True) #8  16
    d5 = decoder_block(d4, e3, 256, dropout=False,skip=False) #16 32
    d6 = decoder_block(d5, e2, 128, dropout=False,skip=False)#32  64
    d7 = decoder_block(d6, e1, 64, dropout=False,skip=False)#64  128
    # output  
    g=UpSampling2D()(d7) #128  256
    g=Conv2D(3,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model([in_image,z,in_encode], [out_image,e8])
    return model





   
#experiment 12_5      
class CreateARLTCGAN(tf.keras.Model):
    def __init__(self,generator,dim_z_motion,dim_z_content):
        super().__init__()
        self.gen=generator
        self.dim_z_motion=tf.constant(dim_z_motion)
        self.dim_z_content=tf.constant(dim_z_content)
        # self._blur=blur_frame


        wiring= ncps.wirings.Random(dim_z_motion, output_dim=dim_z_motion, sparsity_level=0.5)
        self.ltccell=LTCCell(wiring)
        wiring2= ncps.wirings.Random(512, output_dim=512, sparsity_level=0.5)
        self.ltccell_image=LTCCell(wiring2)
        self.temporal_attn = TemporalSelfAttention(num_heads=4, key_dim=64,
                                                    dropout=0.1)
    # -- refine the full hidden-state history with attention ----------
    def _refine_h_sequence(self, h_array, length, training):
        h_seq = h_array.stack()                        # (length, 1, 512) extra dim!
        h_seq = tf.squeeze(h_seq, axis=1)              # (length, 512)  fix here
        h_seq = self.temporal_attn(h_seq, training=training)
        return h_seq[-1]                               # (512,)
    
        
    def call(self,n,og_frames,teacher_forcing_rate=0,noise=None,training=False):
        frames = tf.TensorArray(dtype=tf.float32, size=n)  # Initialize TensorArray
        first_frame=og_frames[0]  # 256,256,3
        z_content=self.sample_z_content(1,n)
        z_motion=self.sample_z_m(1,n)
        
        
        z=tf.concat([z_content,z_motion],1)


        h_array = tf.TensorArray(dtype=tf.float32, size=n + 1,clear_after_read=False)  # +1 to store the initial state if needed
        
        h_t =tf.random.normal((1,512))
        h_array = h_array.write(0, h_t)  # write initial state
        
        input_frame=first_frame

        ATTN_EVERY = 4

        def _gen_cond(i, h_array, frames, input_frame):
            return i < n

        def _gen_body(i, h_array, frames, input_frame):
            h_prev = h_array.read(i)

            fake, e_t = self.gen(
                [tf.expand_dims(input_frame, axis=0),
                 tf.expand_dims(z[i], axis=0),
                 tf.expand_dims(tf.expand_dims(h_prev, axis=0), axis=0)],
                training=training,
            )
            alpha = 0.9
            content_code = alpha * h_prev + (1 - alpha) * e_t[0][0]

            h, _ = self.ltccell_image(content_code, h_prev)
            h_array = h_array.write(i + 1, h)

            do_refine = tf.logical_and(i > 0, tf.equal((i + 1) % ATTN_EVERY, 0))
            h_array = tf.cond(
                do_refine,
                lambda: h_array.write(
                    i + 1,
                    tf.expand_dims(self._refine_h_sequence(h_array, i + 2, training), 0),
                ),
                lambda: h_array,
            )

            frames = frames.write(i, fake[0])

            if training:
                def _teacher_or_student():
                    use_real = tf.less(
                        tf.random.uniform([], 0.0, 1.0), teacher_forcing_rate
                    )
                    safe_idx = tf.minimum(i + 1, n - 1)
                    real_next = og_frames[safe_idx]
                    return tf.cond(
                        use_real,
                        lambda: real_next,
                        lambda: fake[0],
                    )
                next_input = tf.cond(i + 1 < n, _teacher_or_student, lambda: fake[0])
            else:
                next_input = fake[0]

           # next_input = self.blur_frame(next_input)
            return i + 1, h_array, frames, next_input

        _, h_array, frames, _ = tf.while_loop(
            _gen_cond,
            _gen_body,
            loop_vars=[tf.constant(0), h_array, frames, input_frame],
            parallel_iterations=1,
        )

        _ = h_array.stack()
        frames = frames.stack()
        return tf.expand_dims(frames, axis=0)
        

        
    def get_ltc_initial_state(self,num_samples):
        return tf.random.normal((num_samples,self.dim_z_motion))
    def get_iteraion_noise(self,num_samples):
        return tf.random.normal((num_samples,self.dim_z_motion))
    def sample_z_content(self,num_samples,n):
        video_len=n
        # content=np.random.normal(0,1,).astype(np.float32)
        content=tf.random.normal((num_samples,self.dim_z_content))
        content=tf.repeat(content,video_len,axis=0)
        # content=np.repeat(content,video_len,axis=0)
        # content=tf.constant(content)
        return content
    def sample_z_m(self, num_samples, n):
        video_len = n
        h_array = tf.TensorArray(dtype=tf.float32, size=video_len + 1,clear_after_read=False)  # +1 to store the initial state if needed
    
        h_t = self.get_ltc_initial_state(num_samples)
        h_array = h_array.write(0, h_t)  # write initial state

        def _zm_cond(i, h_array):
            return i < video_len

        def _zm_body(i, h_array):
            e_t = self.get_iteraion_noise(num_samples)
            h_prev = h_array.read(i)
            h, _ = self.ltccell(e_t, h_prev)
            h_array = h_array.write(i + 1, h)
            return i + 1, h_array

        _, h_array = tf.while_loop(
            _zm_cond,
            _zm_body,
            loop_vars=[tf.constant(0), h_array],
            parallel_iterations=1,
        )
        z_m_t = h_array.concat()[1:]  # skip the initial state
        # z_m_t = tf.reshape(z_m_t, [video_len * num_samples, -1])
        return z_m_t    


    
generator_base = define_generator(z_shape=(400,),image_shape=(FRAME_SIZE[0],FRAME_SIZE[1],3))
generator_base.summary()
        


generator=CreateARLTCGAN(generator_base,200,200)




# ---------------------------------------------------------------------------
# 2.  Drop-in replacement for build_discriminator_3d
# ---------------------------------------------------------------------------


#DISCRIMINATOR FOR VIDEOCONDIDITIONED GENERATOR
def build_discriminator_3d(vid_shape=(SEQ_LENGHT, 64, 64, 1)):
    """
    input_shape: (frames, height, width, channels)
    returns: Keras Model
    """
    init =tf.random_normal_initializer(0., 0.02)
    in_src_image = Input(shape=vid_shape) # 16,256,256

    
    x = Conv3D(128, kernel_size=3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(in_src_image)  #8,128,256
    x = LeakyReLU(0.2)(x)
    
    x = Conv3D(256, kernel_size=3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(x) #4,64,128
    x = LeakyReLU(0.2)(x)
    
    x = Conv3D(256, kernel_size=3, strides=(2, 2, 2), padding='same', kernel_initializer=init)(x) #2,32,64
    x = LeakyReLU(0.2)(x)
    
    x = Conv3D(512, kernel_size=3, strides=(2, 2,2), padding='same', kernel_initializer=init)(x) #1,16 ,32
    x = LeakyReLU(0.2)(x)

    x = Conv3D(1024, kernel_size=3, strides=(1,2,2), padding='same', kernel_initializer=init)(x) #1,16 ,16
    x = LeakyReLU(0.2)(x)
        
    x = Conv3D(1,kernel_size=3, padding='same', kernel_initializer=init)(x)

    
    model = Model(inputs=in_src_image, outputs=x)
    return model



discriminator=build_discriminator_3d((minibatch[1],FRAME_SIZE[0],FRAME_SIZE[1],3))
discriminator.summary()


vgg = tf.keras.applications.VGG19(
        weights='imagenet',
        input_shape=(512, 512, 3),
        include_top=False
    )
vgg.trainable = False
CONTENT_LAYER = ('block4_conv2',)

intermediate_model = tf.keras.models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in CONTENT_LAYER])

def perceptual_loss(fake, real):
    random_int=tf.random.uniform(shape=(), minval=0, maxval=minibatch[1], dtype=tf.int32)
    fake_rgb = (tf.image.resize(fake[:,random_int],[512,512]) + 1.0) * 127.5
    real_rgb = (tf.image.resize(real[:,random_int],[512,512]) + 1.0) * 127.5
    fake_uint8=tf.cast(fake_rgb,tf.uint8)
    real_uint8=tf.cast(real_rgb,tf.uint8)
    fake_features = vgg(preprocess_input(fake_uint8))
    real_features = vgg(preprocess_input(real_uint8))

    return tf.reduce_mean(tf.abs(fake_features - real_features))
'''
def get_content_features(image):
    image = (image[0] + 1) * 127.5
    img_features = {}

    if image.shape[2] != 512:
        image = tf.image.resize(image, [512, 512])


    img_pre = tf.keras.applications.vgg19.preprocess_input(image)

    net = intermediate_model(img_pre)

    for i, layer in enumerate(CONTENT_LAYER):
        img_features[layer] = net[i]

    return img_features


def perceptual_loss(img, syn):

    img_features = get_content_features(img)
    syn_features = get_content_features(syn)

    content_lossE = 0
    for content_layer in CONTENT_LAYER:
        coff = float(1.0 / len(CONTENT_LAYER))
        img_content = img_features[content_layer]
        syn_content = syn_features[content_layer]
        content_lossE += coff * tf.reduce_mean(tf.abs(img_content - syn_content))

    content_loss = tf.reduce_mean(content_lossE)

    return content_loss

'''

def _build_feature_extractor():
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    return tf.keras.Model(inputs=vgg.input,
                          outputs=vgg.get_layer("block3_conv3").output)

feature_extractor = _build_feature_extractor()  # ← runs immediately on import


def temporal_consistency_loss(video_real, video_generated):
    """
    video_real      : (B, T, H, W, C) in [-1, 1] — ground truth
    video_generated : (B, T, H, W, C) in [-1, 1] — model output
    Penalizes perceptual difference between real and generated at every frame.
    """
    B, T, H, W, C = tf.unstack(tf.shape(video_generated))

    def extract_features(video):
        frames = tf.reshape(video, [B * T, H, W, C])
        frames_255 = (frames + 1.0) * 127.5
        frames_pre = tf.keras.applications.vgg19.preprocess_input(frames_255)
        feats = feature_extractor(frames_pre, training=False)  # (B*T, h, w, ch)
        _, fh, fw, fch = tf.unstack(tf.shape(feats))
        return tf.reshape(feats, [B, T, fh, fw, fch])          # (B, T, h, w, ch)

    feats_real = extract_features(video_real)
    feats_gen  = extract_features(video_generated)

    # 1) Reconstruction: generated vs real at every frame
    l_reconstruction = tf.reduce_mean(tf.square(feats_gen - feats_real))

    # 2) Temporal: adjacent frame differences should match between real and generated
    def temporal_diff(feats):
        return feats[:, 1:, ...] - feats[:, :-1, ...]          # (B, T-1, h, w, ch)

    l_temporal = tf.reduce_mean(tf.square(temporal_diff(feats_gen) - temporal_diff(feats_real)))

    return l_reconstruction + l_temporal



    
def gradient_temporal_loss(video,gen):
    # First derivative over time
    dt = video[:,2:] - 2 * video[:,1:-1] + video[:,:-2]
    dt_g= gen[:,2:] - 2 * gen[:,1:-1] + gen[:,:-2]
    print(video[:,2:].shape)
    return tf.reduce_mean(tf.abs(dt-dt_g))

# In[27]:
def frame_difference_loss(gen, gt):
    return tf.reduce_mean(tf.square((gen[:,1:] - gen[:,:-1]) - (gt[:,1:] - gt[:,:-1])))

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss=-tf.reduce_mean(disc_generated_output) #WGAN-gp
    temp=temporal_consistency_loss(gen_output,target)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    grad_temp=gradient_temporal_loss(gen_output,target)
    frame_loss=frame_difference_loss(gen_output, target)
    perp=perceptual_loss(gen_output,target)
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)+perp*PERC+temp*TEMP+grad_temp*GRAD+frame_loss*FRAME
    
    return total_gen_loss, gan_loss, l1_loss,temp,perp,grad_temp,frame_loss




#define the gradient penalty function
def gradient_penalty(x, x_gen):
    x = tf.cast(x, tf.float32)
    x_gen = tf.cast(x_gen, tf.float32)
    epsilon = tf.random.uniform([x.shape[0],1, 1, 1, 1], 0.0, 1.0) #added time dimension, maybe change later
    diff=x_gen-x
    x_hat = x+(epsilon*diff)
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator(x_hat,training=True)
    gradients = t.gradient(d_hat, x_hat)

    ddx = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3,4]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
    return d_regularizer



def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss= -tf.reduce_mean(disc_real_output)
    
    generated_loss=tf.reduce_mean(disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss




@tf.function
def train_step(target):
    tf_rate = get_teacher_forcing_rate()

    def _disc_cond(i, penalty, disc_loss):
        return i < 3

    def _disc_body(i, penalty, disc_loss):
        with tf.GradientTape() as disc_tape:
            gen_output = generator(n=minibatch[1], og_frames=target[0],
                                   noise=tf.random.uniform((1, 400)))
            disc_real_output = discriminator(target, training=True)
            disc_generated_output = discriminator(gen_output, training=True)
            penalty = gradient_penalty(target, gen_output)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output) + penalty * PEN

        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        return i + 1, penalty, disc_loss

    _, penalty, disc_loss = tf.while_loop(
        _disc_cond,
        _disc_body,
        loop_vars=[tf.constant(0), tf.constant(0.0), tf.constant(0.0)],
        parallel_iterations=1,
    )
    with tf.GradientTape() as gen_tape:
    
        gen_output = generator(n=minibatch[1],og_frames=target[0],teacher_forcing_rate=tf_rate,noise=tf.random.uniform((1,400)), training=True)

        disc_generated_output = discriminator(gen_output)

        gen_total_loss, gen_gan_loss, gen_l1_loss,temp,perp,grad,fram= generator_loss(disc_generated_output, gen_output, target)


    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)

    
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    tf_step_counter.assign_add(1.0)
    return(gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss,penalty,temp,perp,grad,fram)



# In[32]:
# Add this AFTER creating the generator object, BEFORE train_step is called
# (right after the line: generator = CreateARLTCGAN(generator_base, 200, 200))

# Force-build all LTC cells in eager mode so their weights/masks
# are created before @tf.function traces the graph

# Training Fnction
epochs=100
batch_size=1
@tf.function
def train_step_multi(target):
    k=tf.constant(0,dtype=tf.int32)
    aa = tf.zeros(9)
    for _ in tf.range(minibatch[0]):   # tf.range, not Python range
        X_train_mini=target[k:minibatch[1]+k]
        k=k+minibatch[1]-1
        aa=aa+train_step(tf.expand_dims(X_train_mini,axis=0))
    return aa / tf.cast(minibatch[0], tf.float32)

def train_gan(generator, discriminator,dataset, epochs=50, batch_size=128):
    z=0
    best_loss=np.inf
    for epoch in range(epochs):
        for step, batch in enumerate(dataset):
            X_train = batch[0]
            a,b,c,d,e,f,g,h,i=train_step_multi(X_train)
            log_loss_to_csv(epoch,z,a,b,c,d,e,f,g,h,i,csv_filename=folder+'log.csv')     
            if z % 1000 ==0:
                tf_rate_val = get_teacher_forcing_rate()
                print(f'gen_total_loss: {a},gen_gan_loss: {b},gen_l1_loss: {c},disc_loss: {d}, penalty: {e}, temporal: {f}, grad: {g}, perc={h}, frame_loss {i}')
                print(f"Epoch {epoch} | Step {step} | TF rate: {tf_rate_val:.3f}")
                summarize_performance(z,X_train, generator,discriminator,folder, n_samples=16//2,g_loss=a)
            z=z+1              
            if i*FRAME < best_loss:
                best_loss = i*FRAME
                generator.save_weights(os.path.join(folder, 'best_generator.h5'))
#                discriminator.save_weights(os.path.join(folder, 'best_discriminator.h5'))
                print(f"Epoch {step}: New best generator saved with loss {i:.4f}")


# In[ ]:


train_gan(generator, discriminator,video_dataset, epochs=1000,batch_size=1)

