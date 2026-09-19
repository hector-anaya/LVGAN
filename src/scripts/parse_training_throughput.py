"""Derives steady-state training throughput (kimg/h) from recorded training logs.

Reads the per-tick lines that `src/training/training_loop.py` writes to
`<run>/output/log.txt` (and, when present, the same series in `stats.jsonl`) and
reports the steady-state rate, excluding startup ticks and ticks adjacent to a
metric evaluation.

`sec/kimg` in those lines is `(tick_end - tick_start) / nimg_this_tick * 1e3`
(training_loop.py:424) -- pure training time, with snapshotting and metric
evaluation accounted separately as `maintenance`. So kimg/h = 3600 / sec_per_kimg.

Example:

    python src/scripts/parse_training_throughput.py \
      --run PATH/ffs_..._conv_exp_original-3fecd69 \
      --run PATH/ffs_..._ltc_experiment_ffs_2-3fecd69 \
      --skip_ticks 10
"""

import argparse
import json
import os
import re
import statistics

TICK_RE = re.compile(
    r'^tick\s+(?P<tick>\d+)\s+'
    r'kimg\s+(?P<kimg>[\d.]+)\s+'
    r'time\s+(?P<time>.+?)\s{2,}'
    r'sec/tick\s+(?P<sec_tick>[\d.]+)\s+'
    r'sec/kimg\s+(?P<sec_kimg>[\d.]+)\s+'
    r'maintenance\s+(?P<maint>[\d.]+)\s+'
    r'cpumem\s+(?P<cpumem>[\d.]+)\s+'
    r'gpumem\s+(?P<gpumem>[\d.]+)\s+'
    r'augment\s+(?P<augment>[\d.]+)')


def parse_log(path):
    ticks = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = TICK_RE.match(line.strip())
            if m:
                ticks.append({
                    'tick': int(m['tick']),
                    'kimg': float(m['kimg']),
                    'sec_per_tick': float(m['sec_tick']),
                    'sec_per_kimg': float(m['sec_kimg']),
                    'maintenance': float(m['maint']),
                    'gpumem': float(m['gpumem']),
                })
    return ticks


def run_metadata(run_dir):
    """Pulls the facts that must match for the comparison to be valid."""
    meta = {}
    log = os.path.join(run_dir, 'output', 'log.txt')
    if os.path.isfile(log):
        with open(log, 'r', errors='replace') as f:
            head = [next(f, '') for _ in range(200)]
        for line in head:
            if line.startswith('Num videos:'):
                meta['num_videos'] = int(line.split(':')[1].strip())
            elif line.startswith('Image shape:'):
                meta['image_shape'] = line.split(':', 1)[1].strip()
            elif 'Distributing across' in line:
                meta['num_gpus'] = int(re.search(r'across (\d+) GPU', line).group(1))
            # Which motion sub-network was actually instantiated.
            for key in ('conv', 'ltc', 'cfc', 'rnn', 'ncps_ltc', 'ncps_cfc'):
                if line.startswith(f'synthesis.motion_encoder.{key} '):
                    meta['motion_module'] = key
                    meta['motion_params'] = int(line.split()[1])
        # Totals: first is G, second is D.
        totals = [int(l.split()[1]) for l in open(log, errors='replace')
                  if l.startswith('Total ') and l.split()[1].isdigit()]
        if len(totals) >= 2:
            meta['G_params'], meta['D_params'] = totals[0], totals[1]

    cfg = os.path.join(run_dir, 'experiment_config.yaml')
    if os.path.isfile(cfg):
        text = open(cfg, errors='replace').read()
        for key in ('gen_strategy', 'ltc_hidden_dim', 'ltc_ode_unfolds',
                    'min_period_len', 'batch_size', 'num_gpus', 'resolution', 'kimg'):
            m = re.search(rf'^\s*{key}:\s*(\S+)\s*$', text, re.M)
            if m:
                meta.setdefault(f'cfg_{key}', m.group(1))
    else:
        # Runs launched before experiment_config.yaml was written keep the
        # resolved configs in the copied `configs/` tree.
        for rel, keys in [('configs/training/base.yaml', ('batch_size', 'kimg')),
                          ('configs/infra.yaml', ('num_gpus',)),
                          ('configs/dataset/base.yaml', ('resolution',)),
                          ('configs/model/stylegan-v.yaml', ('min_period_len', 'ltc_hidden_dim', 'ltc_ode_unfolds'))]:
            p = os.path.join(run_dir, rel)
            if os.path.isfile(p):
                text = open(p, errors='replace').read()
                for key in keys:
                    m = re.search(rf'^\s*{key}:\s*(\S+)\s*$', text, re.M)
                    if m:
                        meta.setdefault(f'cfg_{key}', m.group(1))
    return meta


def steady_state(ticks, skip_ticks: int, maint_threshold: float):
    """Excludes startup ticks and ticks adjacent to a snapshot/metric evaluation.

    `maintenance` on tick N covers the work done between tick N-1 and tick N
    (training_loop.py:550), so a large value flags the *pair* (N-1, N).
    """
    metric_ticks = {t['tick'] for t in ticks if t['maintenance'] > maint_threshold}
    excluded = set(metric_ticks) | {t - 1 for t in metric_ticks}
    kept = [t for t in ticks
            if t['tick'] >= skip_ticks and t['tick'] not in excluded]
    return kept, sorted(excluded)


def summarize(run_dir, skip_ticks, maint_threshold):
    log = os.path.join(run_dir, 'output', 'log.txt')
    ticks = parse_log(log)
    kept, excluded = steady_state(ticks, skip_ticks, maint_threshold)
    vals = [t['sec_per_kimg'] for t in kept]

    med = statistics.median(vals)
    mean = statistics.mean(vals)
    return {
        'run_dir': run_dir,
        'log': log,
        'meta': run_metadata(run_dir),
        'total_ticks': len(ticks),
        'last_kimg': ticks[-1]['kimg'] if ticks else None,
        'skipped_startup_ticks': skip_ticks,
        'excluded_metric_ticks': len([t for t in excluded if t >= skip_ticks]),
        'ticks_used': len(kept),
        'sec_per_kimg_mean': mean,
        'sec_per_kimg_median': med,
        'sec_per_kimg_std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
        'kimg_per_h_from_mean': 3600.0 / mean,
        'kimg_per_h_from_median': 3600.0 / med,
        'peak_gpumem_GB': max(t['gpumem'] for t in kept),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, help='Experiment run dir (repeatable)')
    p.add_argument('--skip_ticks', type=int, default=10,
                   help='Drop ticks below this index (startup, ADA warmup, kernel compilation)')
    p.add_argument('--maint_threshold', type=float, default=60.0,
                   help='maintenance seconds above which a tick is treated as adjacent to a metric eval')
    p.add_argument('--json_out', default=None)
    args = p.parse_args()

    results = [summarize(r, args.skip_ticks, args.maint_threshold) for r in args.run]

    for r in results:
        m = r['meta']
        print(f"\n=== {os.path.basename(r['run_dir'])}")
        print(f"  motion module      : {m.get('motion_module')} ({m.get('motion_params')} params)")
        print(f"  GPUs / batch / res : {m.get('num_gpus')} / {m.get('cfg_batch_size')} / {m.get('cfg_resolution')}")
        print(f"  min_period_len     : {m.get('cfg_min_period_len')}   ltc_hidden={m.get('cfg_ltc_hidden_dim')} K={m.get('cfg_ltc_ode_unfolds')}")
        print(f"  dataset videos     : {m.get('num_videos')}   image shape {m.get('image_shape')}")
        print(f"  ticks total/used   : {r['total_ticks']} / {r['ticks_used']}  (skipped first {r['skipped_startup_ticks']}, excluded {r['excluded_metric_ticks']} metric-adjacent)")
        print(f"  last kimg reached  : {r['last_kimg']}")
        print(f"  sec/kimg           : mean {r['sec_per_kimg_mean']:.3f}  median {r['sec_per_kimg_median']:.3f}  sd {r['sec_per_kimg_std']:.3f}")
        print(f"  kimg/h             : {r['kimg_per_h_from_mean']:.1f} (from mean)   {r['kimg_per_h_from_median']:.1f} (from median)")
        print(f"  peak gpumem (GB)   : {r['peak_gpumem_GB']:.2f}")

    if len(results) == 2:
        a, b = results
        for label, key in [('mean', 'kimg_per_h_from_mean'), ('median', 'kimg_per_h_from_median')]:
            slow = (a[key] - b[key]) / a[key] * 100.0
            print(f"\nRelative slowdown of [{os.path.basename(b['run_dir'])}] vs "
                  f"[{os.path.basename(a['run_dir'])}] ({label}): {slow:+.1f}% throughput loss "
                  f"({a[key]:.1f} -> {b[key]:.1f} kimg/h)")

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
