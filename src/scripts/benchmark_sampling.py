"""Measures sampling wall-clock and parameter counts for the motion-generator configs.

Produces the numbers for the "Computational Cost" table: motion-network params,
total generator params, discriminator params, seconds per 128-frame clip and peak
GPU memory, for one `model.generator.motion.gen_strategy` at a time.

The generator is rebuilt from a resolved training config (`experiment_config.yaml`
written into every experiment dir by `src/infra/launch.py`) so that the two
configs differ *only* in the motion module. Weights do not affect wall-clock, but
`--network_pkl` loads a trained `G_ema` instead if you want to time the exact
checkpoint.

Example -- both configs, same protocol, one GPU:

    python src/scripts/benchmark_sampling.py \
      --config PATH_TO_EXPERIMENT/experiment_config.yaml \
      --gen_strategy conv --video_len 128 --warmup 5 --iters 20

    python src/scripts/benchmark_sampling.py \
      --config PATH_TO_EXPERIMENT/experiment_config.yaml \
      --gen_strategy ltc --video_len 128 --warmup 5 --iters 20


      python src/scripts/benchmark_sampling.py   --config experiments/sky_timelapse_256_stylegan-v_random3_max32_ltc_experiment_sky_2-3fecd69/experiment_config.yaml --gen_strategy conv   --video_len 128 --warmup 5 --iters 25 --json_out conv.json
 - export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$(date +%s)
 - mkdir -p $TORCH_EXTENSIONS_DIR
"""

import sys; sys.path.extend(['.', 'src'])
import argparse
import copy
import json
import time
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from src import dnnlib
from src.torch_utils.ops import bias_act, upfirdn2d

torch.set_grad_enabled(False)

#----------------------------------------------------------------------------

def resolve_config(cfg):
    """Resolves every interpolation in place.

    `OmegaConf.resolve` only exists in omegaconf >= 2.1; the training environment
    pins omegaconf 2.0 through hydra-core 1.0.7, so fall back to the round-trip.
    """
    if hasattr(OmegaConf, 'resolve'):
        OmegaConf.resolve(cfg)
        return cfg
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def build_generator(cfg, gen_strategy: str, device: torch.device):
    """Rebuilds G exactly as src/train.py does for cfg=auto at this resolution."""
    from src.training.networks import Generator

    # Safe only because `cfg` is already fully resolved: a deepcopy drops the
    # parent link that absolute interpolations such as ${sampling} rely on.
    gen_cfg = copy.deepcopy(cfg.model.generator)
    if gen_strategy is not None:
        gen_cfg.motion.gen_strategy = gen_strategy

    res = cfg.dataset.resolution
    # src/train.py: spec.fmaps = 1 if res >= 512 else 0.5; map=2 for cfg=auto
    fmaps = 1 if res >= 512 else 0.5
    G = Generator(
        c_dim=cfg.dataset.c_dim,
        w_dim=gen_cfg.w_dim,
        img_resolution=res,
        img_channels=3,
        mapping_kwargs=dnnlib.EasyDict(num_layers=2, cfg=gen_cfg),
        synthesis_kwargs=dnnlib.EasyDict(
            channel_base=int(fmaps * 32768), channel_max=512,
            num_fp16_res=4, conv_clamp=256),
        cfg=gen_cfg,
    )
    return G.to(device).eval()


def build_discriminator(cfg, device: torch.device):
    from src.training.networks import Discriminator

    res = cfg.dataset.resolution
    fmaps = 1 if res >= 512 else 0.5
    D = Discriminator(
        c_dim=cfg.dataset.c_dim,
        img_resolution=res,
        img_channels=3,
        channel_base=int(fmaps * 32768), channel_max=512,
        num_fp16_res=4, conv_clamp=256,
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(num_layers=2),
        epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=4),
        cfg=cfg.model.discriminator,
    )
    return D.to(device).eval()

#----------------------------------------------------------------------------

def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def motion_breakdown(G) -> dict:
    """Per-submodule parameter counts of the motion mapping network."""
    me = G.synthesis.motion_encoder
    out = {}
    for name, sub in me.named_children():
        out[name] = count_params(sub)
    out['_motion_encoder_total'] = count_params(me)
    # The strategy-specific sub-network, i.e. everything except the shared
    # Fourier time encoder. This is the number usually quoted as "motion params".
    shared = out.get('time_encoder', 0)
    out['_strategy_subnetwork'] = out['_motion_encoder_total'] - shared
    return out

#----------------------------------------------------------------------------

def generate_one_clip(G, z, c, ts, chunk: int):
    """One clip, optionally split into chunks of `chunk` frames along time.

    When chunked, the motion network is evaluated ONCE for the whole clip and the
    resulting per-frame motion codes are sliced per chunk. Passing `motion_z`
    instead would only share the input noise: `generate_motion_u_codes` would still
    re-run the full trajectory for every chunk, multiplying the motion-network cost
    by the number of chunks and inflating the LTC row specifically -- which is the
    one number this table exists to report.
    """
    if chunk is None or chunk >= ts.shape[1]:
        return G(z=z, c=c, t=ts, noise_mode='const')

    ws = G.mapping(z, c)                                          # [1, num_ws, w_dim]
    motion_v = G.synthesis.motion_encoder(c=c, t=ts)['motion_v']  # [num_frames, v_dim]
    outs, start = [], 0
    for cur_ts in ts.split(chunk, dim=1):
        n = cur_ts.shape[1]
        outs.append(G.synthesis(ws=ws, t=cur_ts, c=c,
                                motion_v=motion_v[start:start + n], noise_mode='const'))
        start += n
    return torch.cat(outs, dim=0)


def count_motion_calls(G, z, c, ts, chunk: int) -> int:
    """How many times the motion network actually runs for one clip (must be 1)."""
    me = G.synthesis.motion_encoder
    orig, n = me.forward, [0]

    def counting(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    me.forward = counting
    try:
        generate_one_clip(G, z, c, ts, chunk)
    finally:
        me.forward = orig
    return n[0]


def time_sampling(G, device, video_len: int, warmup: int, iters: int, chunk: int, seed: int):
    z = torch.randn(1, G.z_dim, device=device)
    c = torch.zeros(1, G.c_dim, device=device)
    ts = torch.arange(video_len, device=device).float().unsqueeze(0)  # [1, video_len]

    for _ in range(warmup):
        generate_one_clip(G, z, c, ts, chunk)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        img = generate_one_clip(G, z, c, ts, chunk)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
        assert img.shape[0] == video_len, f"Expected {video_len} frames, got {img.shape[0]}"
        del img

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return np.array(samples), peak_bytes

#----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Resolved experiment_config.yaml from a training run')
    p.add_argument('--gen_strategy', default=None, help='Override model.generator.motion.gen_strategy')
    p.add_argument('--network_pkl', default=None, help='Time this checkpoint\'s G_ema instead of a freshly built G')
    p.add_argument('--video_len', type=int, default=128, help='Frames per clip')
    p.add_argument('--warmup', type=int, default=5, help='Warmup clips, discarded')
    p.add_argument('--iters', type=int, default=20, help='Timed clips to average over')
    p.add_argument('--chunk', type=int, default=0, help='Frames per synthesis forward; 0 = whole clip at once')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip_discriminator', action='store_true')
    p.add_argument('--json_out', default=None, help='Also write the results as JSON here')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg = resolve_config(OmegaConf.load(args.config))

    # --- environment provenance -------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plugins = {'bias_act': bool(bias_act._init()), 'upfirdn2d': bool(upfirdn2d._init())}

    # Free VRAM before we allocate anything: on a shared node another job may already
    # hold the card, which forces chunking and contaminates the timing.
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        free_GiB = round(free_b / 2**30, 2)
    except AttributeError:  # torch < 1.10 has no mem_get_info
        free_GiB = None

    env = {
        'gpu_name': torch.cuda.get_device_name(device),
        'gpu_capability': '.'.join(map(str, torch.cuda.get_device_capability(device))),
        'gpu_total_mem_GiB': round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2),
        'gpu_free_mem_GiB_at_start': free_GiB,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'fused_cuda_plugins': plugins,
        # StyleGAN2 uses native fp16 in the top `num_fp16_res` blocks (here 4, i.e.
        # resolutions 32..256). There is no torch.cuda.amp autocast anywhere.
        'precision': 'StyleGAN2 native mixed precision (fp16 in top 4 synthesis blocks, fp32 elsewhere); no torch.cuda.amp',
        'torch_amp_autocast': False,
    }

    # --- networks ---------------------------------------------------------------
    if args.network_pkl:
        import legacy
        with dnnlib.util.open_url(args.network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
        source = f'G_ema from {args.network_pkl}'
        strategy = G.cfg.motion.gen_strategy
    else:
        G = build_generator(cfg, args.gen_strategy, device)
        source = f'constructed from {args.config}'
        strategy = G.cfg.motion.gen_strategy

    params = {
        'generator_total': count_params(G),
        'generator_mapping': count_params(G.mapping),
        'motion': motion_breakdown(G),
    }
    params['generator_synthesis_backbone'] = (
        params['generator_total'] - params['generator_mapping'] - params['motion']['_motion_encoder_total'])

    if not args.skip_discriminator:
        D = build_discriminator(cfg, device)
        params['discriminator_total'] = count_params(D)
        del D
        torch.cuda.empty_cache()

    # --- timing -----------------------------------------------------------------
    chunk = args.chunk if args.chunk > 0 else None
    while True:
        try:
            samples, peak_bytes = time_sampling(
                G, device, args.video_len, args.warmup, args.iters, chunk, args.seed)
            break
        except RuntimeError as e:
            # torch.cuda.OutOfMemoryError only exists from torch 1.13; it subclasses
            # RuntimeError, so match on the message to cover the 1.9 training env too.
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            chunk = args.video_len // 2 if chunk is None else chunk // 2
            if chunk < 1:
                raise
            print(f'[OOM] retrying with chunk={chunk} frames per synthesis forward', flush=True)

    # Must be 1: the motion network is a per-clip cost, not a per-chunk one.
    motion_calls = count_motion_calls(
        G, torch.randn(1, G.z_dim, device=device), torch.zeros(1, G.c_dim, device=device),
        torch.arange(args.video_len, device=device).float().unsqueeze(0), chunk)

    result = {
        'gen_strategy': strategy,
        'network_source': source,
        'motion_net_calls_per_clip': motion_calls,
        'video_len': args.video_len,
        'resolution': cfg.dataset.resolution,
        'clips_batch_size': 1,
        'frames_per_forward': chunk if chunk is not None else args.video_len,
        'chunked': chunk is not None,
        'warmup_clips': args.warmup,
        'timed_clips': args.iters,
        'sec_per_clip_mean': float(samples.mean()),
        'sec_per_clip_std': float(samples.std(ddof=1)),
        'sec_per_clip_min': float(samples.min()),
        'sec_per_clip_max': float(samples.max()),
        'peak_gpu_mem_GiB': peak_bytes / 2**30,
        'params': params,
        'env': env,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()

