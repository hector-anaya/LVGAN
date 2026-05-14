import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt 

import glob as _glob
import csv
import os
def log_loss_to_csv(epoch,step,a,b,c,d,e,f,g,h,i,csv_filename,j=None,k=None):
    # Check if the CSV file already exists; if not, create it with the header
    write_header = not os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header if the file is newly created
        if write_header:
            if j is None and k is None:
                writer.writerow(['Epoch','Step', 'Generator Total Loss', 'Generator Loss', 'L1 Loss', 'Discriminator Loss','Penalty','Temporal Loss','Perceptual Loss','Grad temp loss','Frame Loss'])
            elif k is None :
                writer.writerow(['Epoch','Step', 'Generator Total Loss', 'Generator Loss', 'L1 Loss', 'Discriminator Loss','Penalty','Temporal Loss','Perceptual Loss','Grad temp loss','Frame Loss','Generator image loss'])
            else:
                writer.writerow(['Epoch','Step', 'Generator Total Loss', 'Generator Loss', 'L1 Loss', 'Discriminator Loss','Penalty','Temporal Loss','Perceptual Loss','Grad temp loss','Frame Loss','Generator image loss','Penalty image'])
        if j is None and k is None:
            writer.writerow([epoch,step, a.numpy(), b.numpy(),c.numpy(),d.numpy(),e.numpy(),f.numpy(),g.numpy(),h.numpy(),i.numpy()])
        # Write the loss values for the current epoch
        elif k is None:
            writer.writerow([epoch,step, a.numpy(), b.numpy(),c.numpy(),d.numpy(),e.numpy(),f.numpy(),g.numpy(),h.numpy(),i.numpy(),j.numpy()])
        else:
            writer.writerow([epoch,step, a.numpy(), b.numpy(),c.numpy(),d.numpy(),e.numpy(),f.numpy(),g.numpy(),h.numpy(),i.numpy(),j.numpy(),k.numpy()])


class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self,x,y,filters, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.x = x
        self.y=y
        self.filters=filters

    def call(self, inputs):
        new_size = [-1,self.x,self.y,self.filters]
        return tf.reshape(inputs, new_size)

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({"x": self.x})
        config.update({"y":self.y})
        config.update({"filters":self.filters})
        return config


def generate_fake_samples(g_model, img,act256,act64,z, patch_shape):
# generate fake instance
    X = g_model([img,act256,act64,z],training=True)
    # y = zeros((len(X),patch_shape,patch_shape, 1))
    return X

        
def resizeandrescale(img):
    img=img.astype(np.float32)
    img=img/255
    img= (img - 0.5) * 2.0
    img=img[...,::-1]
    return img
def generate_real_samples(dataset, n_samples, patch_shape):
# unpack dataset
    trainA, trainB,trainC = dataset
    
    # choose random instances
    ix =  np.random.randint(0, len(trainA), n_samples)
    # retrieve selected images
    X1, X2, X3 = trainA[ix[0]], trainB[ix[0]], trainC
    X1=resizeandrescale(cv2.imread(X1))
    X2=resizeandrescale(cv2.imread(X2))
    X3=resizeandrescale(cv2.imread(X3))
    X1=tf.expand_dims(X1,axis=0)
    X2=tf.expand_dims(X2,axis=0)
    # y = ones((n_samples,patch_shape,patch_shape,  1))
    return [X1, X2, X3]

def summarize_performance(step,X_train, g_model,d_model,folder,n_samples=1,g_loss=None):
# select a sample of input images
    # generate a batch of fake samples
        # Normalize shape
    if len(X_train.shape) == 5 and X_train.shape[0]==2:
        video = X_train   # (T, H, W, C)
    else:
        video   = X_train    # already (T, H, W, C)
    X_fakeB, _, _ = g_model(n=32,og_frames=video,noise=tf.random.uniform((1,400)))
   # syn = ((X_fakeB+ 1) / 2.0)*((mask+1)/2)
   # syn=syn.numpy()
  #  cv2.imwrite(folder+'/'+str(step)+'.png',  cv2.cvtColor(syn[0]*255, cv2.COLOR_BGR2RGB))

    # scale all pixels from [-1,1] to [0,1]
    X_train = X_train
    X_fakeB = X_fakeB[0]

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    j=0
    for i, ax in enumerate(axes.flat): 
        if i < n_samples*2:  # Select first 9 images
            if i<n_samples:
                ax.imshow((X_fakeB[j]+1)/2, cmap='gray')
                ax.axis('off')  # Hide axes
            else:
                ax.imshow((X_train[j-n_samples]+1)/2,cmap='gray')
                ax.axis('off')  # Hide axes
            j=j+1
    # Save the grid as an image
        # save plot to file
    filename1 =folder+'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
 #   filename3=folder+'discriminator.h5'
  #  d_model.save(filename3)
    filename2 = folder+f'generator_{step}.h5' 
    if step%100==0:
        save_video(X_fakeB.numpy(),folder+f"vid_{step+1}.mp4")
    if step%5000==0:
        g_model.save_weights(filename2)
# In[52]:



class ReshapeLayer_time(tf.keras.layers.Layer):
    def __init__(self,x,y,filters,t, **kwargs):
        super(ReshapeLayer_time, self).__init__(**kwargs)
        self.t=t
        self.x = x
        self.y=y
        self.filters=filters

    def call(self, inputs):
        new_size = [tf.shape(inputs)[0], self.t, self.x, self.y, self.filters]
        # new_size = [-1,-1,self.x,self.y,self.filters]
        return tf.reshape(inputs, new_size)

    def get_config(self):
        config = super(ReshapeLayer_time, self).get_config()
        config.update({"t": self.t})
        config.update({"x": self.x})
        config.update({"y":self.y})
        config.update({"filters":self.filters})
        return config


# In[53]:


class ResizeLayer_time(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(ResizeLayer_time, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        new_size = [2**(self.n + 1), 2**(self.n + 1)]
        return tf.image.resize(inputs[0], new_size, method='nearest')

    def get_config(self):
        config = super(ResizeLayer_time, self).get_config()
        config.update({"n": self.n})
        return config


# In[54]:


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        new_size = [2**(self.n + 1), 2**(self.n + 1)]
        return tf.image.resize(inputs, new_size, method='nearest')

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"n": self.n})
        return config
import scipy.signal

class BlurredNoise(tf.keras.layers.Layer):
    def __init__(
        self,
        channels=1024,
        min_sampling_rate=250,
        max_sampling_rate=10000,
        blur_widths=19,
        cutoff=2.0,
        width=12.0,
        sampling_rate_base=2.0,
        normalize_per_filter=1.0,
        **kwargs
    ):
        super(BlurredNoise, self).__init__(**kwargs)
        self.channels = channels
        self.min_sampling_rate = min_sampling_rate
        self.max_sampling_rate = max_sampling_rate
        self.blur_widths = blur_widths
        self.cutoff = cutoff
        self.width = width
        self.sampling_rate_base = sampling_rate_base
        self.normalize_per_filter = normalize_per_filter

        assert self.channels % self.blur_widths == 0, "channels must be divisible by blur_widths"
        self.noise_channels = self.channels // self.blur_widths
        self.kernel_size = int(np.ceil(self.max_sampling_rate / 2))
        self.conv1d = tf.keras.layers.Conv1D(self.channels,self.kernel_size,groups=self.blur_widths )

    def build(self, input_shape):
        blur_filters = np.zeros((self.blur_widths, self.kernel_size), dtype=np.float32)

        if self.sampling_rate_base > 1:
            log_min_sampling_rate = np.log(self.min_sampling_rate) / np.log(self.sampling_rate_base)
            log_max_sampling_rate = np.log(self.max_sampling_rate) / np.log(self.sampling_rate_base)
            log_sampling_rates = np.linspace(log_min_sampling_rate, log_max_sampling_rate, self.blur_widths)
            sampling_rates = np.clip(self.sampling_rate_base ** log_sampling_rates,
                                     self.min_sampling_rate, self.max_sampling_rate)
        else:
            sampling_rates = np.linspace(self.min_sampling_rate, self.max_sampling_rate, self.blur_widths)

        for i, sampling_rate in enumerate(sampling_rates):
            num_taps = int(np.ceil(sampling_rate / 2))
            blur_filter = scipy.signal.firwin(numtaps=num_taps,
                                              cutoff=self.cutoff,
                                              width=self.width,
                                              fs=sampling_rate)
            blur_filters[i, -num_taps:] = blur_filter

      #  blur_filters = blur_filters[:, :, np.newaxis]  # (blur_widths, kernel_size, 1)


        if self.normalize_per_filter>0:
            self.output_scale = tf.math.rsqrt(tf.reduce_sum(blur_filters**2, axis=1, keepdims=True))
            self.output_scale = einops.rearrange(self.output_scale, "c 1-> 1 c 1")
            
        #blur_filters = einops.rearrange(blur_filters, "c k -> c 1 k")
       # self.blur_filters = tf.constant(blur_filters, dtype=tf.float32)
        blur_filters = np.expand_dims(blur_filters, axis=2)  # [channels, kernel, 1]
        blur_filters = np.transpose(blur_filters, (1, 2, 0))  # [kernel, 1, channels]
        blur_filters = np.expand_dims(blur_filters, axis=3)   # [kernel, 1, channels]
        self.blur_filters = tf.constant(blur_filters, dtype=tf.float32) 
        super(BlurredNoise, self).build(input_shape)
    def call(self, batch_size, seq_length):
        input_seq_len = seq_length + self.kernel_size - 1
        noise = tf.random.normal([batch_size, input_seq_len, self.channels])  # [B, T, C]

        # Prepare input for depthwise_conv2d: [B, H=T, W=1, C]
        x = tf.expand_dims(noise, axis=2)
        # print(f"x_shape {x.shape}")
        # print(f"blur filters shape {tf.expand_dims(self.blur_filters,axis=0).shape}")

        # Apply grouped conv (per-channel) with depthwise_conv2d
        out = tf.nn.depthwise_conv2d(
            x,
            self.blur_filters,
            strides=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC'
        )  # [B, T_out, 1, C]

        out = tf.squeeze(out, axis=2)  # [B, T_out, C]
        out = tf.transpose(out, [0, 2, 1])  # [B, C, T_out]

        if self.normalize_per_filter:
            out *= (1 + self.normalize_per_filter * (self.output_scale - 1))
        # print(out.shape)
        return out   

from tensorflow.keras.layers import Lambda

def noise_generator_layer(seq_len):
    return Lambda(lambda x: blurred_noise(tf.shape(x)[0], seq_len))        

from tensorflow.keras.layers import Layer, Dense, Reshape, Permute, Softmax

class TemporalSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.depth = embed_dim // num_heads

        self.q_dense = Dense(embed_dim)
        self.k_dense = Dense(embed_dim)
        self.v_dense = Dense(embed_dim)
        self.out_dense = Dense(embed_dim)
        self.softmax = Softmax()

    def split_heads(self, x, batch_size):
        # [B, T, H*W, embed_dim] -> [B, num_heads, T, H*W, depth]
        x = tf.reshape(x, (batch_size, -1, x.shape[2], self.num_heads, self.depth))
        return tf.transpose(x, [0, 3, 1, 2, 4])  # [B, heads, T, HW, depth]

    def call(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        x_reshaped = tf.reshape(x, [B, T, H * W, C])  # [B, T, HW, C]

        q = self.q_dense(x_reshaped)  # [B, T, HW, C]
        k = self.k_dense(x_reshaped)
        v = self.v_dense(x_reshaped)

        # compute attention scores along the time axis
        q = tf.transpose(q, [0, 2, 1, 3])  # [B, HW, T, C]
        k = tf.transpose(k, [0, 2, 3, 1])  # [B, HW, C, T]
        v = tf.transpose(v, [0, 2, 1, 3])  # [B, HW, T, C]

        scores = tf.matmul(q, k) / tf.math.sqrt(tf.cast(C, tf.float32))  # [B, HW, T, T]
        weights = self.softmax(scores)  # [B, HW, T, T]

        attended = tf.matmul(weights, v)  # [B, HW, T, C]
        attended = tf.transpose(attended, [0, 2, 1, 3])  # [B, T, HW, C]
        attended = tf.reshape(attended, [B, T, H, W, C])  # back to original spatial shape

        return self.out_dense(attended)
def save_video(frames, output_path, fps=15):
    """
    Save a stack of frames (NumPy array) as a video.
    
    Parameters:
        frames: NumPy array of shape (num_frames, height, width, 3)
        output_path: where to save the video (e.g., 'output.mp4')
        fps: frames per second (default: 15)
    """
    # Convert from [-1, 1] back to [0, 255]
    frames = ((frames + 1.0) * 127.5).astype(np.uint8)

    height, width = frames[0].shape[:2]
    
    # Define video codec (use 'XVID' or 'mp4v' for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR

    out.release()
    print(f"Saved video to {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _rmtree(path):
    try:
        shutil.rmtree(path)
        print(f'[CKPT] Removed old checkpoint → {path}')
    except Exception as e:
        print(f'[CKPT] Warning: could not remove {path}: {e}')


def _warm_up_generator(gen, n_frames, frame_size=(256, 256)):
    """One dummy forward pass so Keras creates all weight tensors."""
    dummy = tf.zeros((n_frames, frame_size[0], frame_size[1], 3))
    gen(n=n_frames, og_frames=dummy,
        noise=tf.zeros((1, 400)), training=True)


# ──────────────────────────────────────────────────────────────────────────────
#  GAN CHECKPOINT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class GANCheckpointManager:
    """
    Handles periodic saving and full restoration of a fixed-frame GAN
    training run with DUAL discriminators (3D video + 2D frame).

    Parameters
    ----------
    folder          : experiment root (checkpoints written to folder/checkpoints/)
    generator       : LatentARLTCGAN instance
    discriminator   : 3D video discriminator
    discriminator2  : 2D frame discriminator
    gen_opt         : generator Adam optimiser
    disc_opt        : 3D discriminator Adam optimiser
    disc2_opt       : 2D discriminator Adam optimiser
    n_frames        : fixed frame count used for the entire run
    build_disc_fn   : callable() -> fresh 3D disc model
    build_disc2_fn  : callable() -> fresh 2D frame disc model
    save_every      : save a checkpoint every this many global steps
    keep_last       : number of recent checkpoints to keep on disk
    """

    _SIDECAR = 'state.json'

    def __init__(self,
                 folder,
                 generator,
                 discriminator,
                 gen_opt,
                 disc_opt,
                 n_frames,
                 build_disc_fn,
                 discriminator2=None,
                 disc2_opt=None,
                 build_disc2_fn=None,
                 save_every=500,
                 keep_last=3):
        self.folder         = folder
        self.ckpt_dir       = os.path.join(folder, 'checkpoints')
        self.gen            = generator
        self.disc           = discriminator
        self.disc2          = discriminator2
        self.gen_opt        = gen_opt
        self.disc_opt       = disc_opt
        self.disc2_opt      = disc2_opt
        self.n_frames       = n_frames
        self.build_disc_fn  = build_disc_fn
        self.build_disc2_fn = build_disc2_fn
        self.save_every     = save_every
        self.keep_last      = keep_last
        self._saved_steps   = []

        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, z: int, epoch: int):
        """Call once per training step (after losses are computed)."""
        if z > 0 and z % self.save_every == 0:
            self.save(z, epoch)

    def save(self, z: int, epoch: int):
        """Force-save a checkpoint right now."""
        tag      = f'step_{z:07d}'
        step_dir = os.path.join(self.ckpt_dir, tag)
        os.makedirs(step_dir, exist_ok=True)

        # ── 1. model weights ─────────────────────────────────────────────────
        self.gen.save_weights(
            os.path.join(step_dir, 'generator.weights.h5'))
        self.disc.save_weights(
            os.path.join(step_dir, 'discriminator.weights.h5'))
        if self.disc2 is not None:
            self.disc2.save_weights(
                os.path.join(step_dir, 'discriminator2d.weights.h5'))

        # ── 2. tf.train.Checkpoint (optimizer slot variables) ────────────────
        ckpt_kwargs = {'gen_opt': self.gen_opt, 'disc_opt': self.disc_opt}
        if self.disc2_opt is not None:
            ckpt_kwargs['disc2_opt'] = self.disc2_opt
        tf_ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        tf_ckpt.write(os.path.join(step_dir, 'optimizers'))

        # ── 3. training state (JSON sidecar) ─────────────────────────────────
        state = {
            'z'       : z,
            'epoch'   : epoch,
            'n_frames': self.n_frames,
            'gen_lr'  : float(self.gen_opt.learning_rate),
            'disc_lr' : float(self.disc_opt.learning_rate),
        }
        if self.disc2_opt is not None:
            state['disc2_lr'] = float(self.disc2_opt.learning_rate)
        with open(os.path.join(step_dir, self._SIDECAR), 'w') as f:
            json.dump(state, f, indent=2)

        print(f'[CKPT] Saved checkpoint → {step_dir}')

        # ── 4. rolling cleanup ───────────────────────────────────────────────
        self._saved_steps.append(z)
        if len(self._saved_steps) > self.keep_last:
            old_z   = self._saved_steps.pop(0)
            old_dir = os.path.join(self.ckpt_dir, f'step_{old_z:07d}')
            _rmtree(old_dir)

    def restore_if_available(self, frame_size=(256, 256)):
        """
        Scan checkpoint dir for the latest checkpoint and restore everything.

        Returns (start_z, start_epoch) — both 0 if no checkpoint exists.
        """
        latest = self._find_latest_checkpoint()
        if latest is None:
            print('[CKPT] No checkpoint found — starting from scratch.')
            return 0, 0,None,None

        print(f'[CKPT] Restoring from {latest}')

        with open(os.path.join(latest, self._SIDECAR)) as f:
            state = json.load(f)

        # ── learning rates ───────────────────────────────────────────────────
        self.gen_opt.learning_rate.assign(state['gen_lr'])
        self.disc_opt.learning_rate.assign(state['disc_lr'])
        if self.disc2_opt is not None:
            self.disc2_opt.learning_rate.assign(
                state.get('disc2_lr', state['disc_lr']))

        # ── generator weights ────────────────────────────────────────────────
        _warm_up_generator(self.gen, self.n_frames, frame_size=frame_size)
        self.gen.load_weights(os.path.join(latest, 'generator.weights.h5'))

        # ── primary discriminator ────────────────────────────────────────────
        new_disc = self.build_disc_fn()
        new_disc.load_weights(os.path.join(latest, 'discriminator.weights.h5'))
        self.disc = new_disc

        # ── secondary discriminator (optional) ───────────────────────────────
        if self.build_disc2_fn is not None:
            new_disc2 = self.build_disc2_fn()
            disc2_path = os.path.join(latest, 'discriminator2d.weights.h5')
            if os.path.exists(disc2_path):
                new_disc2.load_weights(disc2_path)
            else:
                print('[CKPT] WARNING: no 2D disc weights — starting 2D disc from scratch.')
            self.disc2 = new_disc2

        # ── optimiser slot variables ─────────────────────────────────────────
        ckpt_kwargs = {'gen_opt': self.gen_opt, 'disc_opt': self.disc_opt}
        if self.disc2_opt is not None:
            ckpt_kwargs['disc2_opt'] = self.disc2_opt
        tf_ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        tf_ckpt.read(os.path.join(latest, 'optimizers')).expect_partial()

        z, epoch = state['z'], state['epoch']
        print(f'[CKPT] Restored  z={z}  epoch={epoch}  n_frames={self.n_frames}')
        return z + 1, epoch,self.gen,self.disc  # resume from NEXT step

    # ── internals ─────────────────────────────────────────────────────────────

    def _find_latest_checkpoint(self):
        """Return the path of the checkpoint with the highest step number."""
        dirs = [
            d for d in _glob.glob(os.path.join(self.ckpt_dir, 'step_*'))
            if os.path.isdir(d) and
               os.path.exists(os.path.join(d, self._SIDECAR))
        ]
        if not dirs:
            return None
        dirs.sort(key=lambda p: int(os.path.basename(p).split('_')[1]))
        return dirs[-1]
