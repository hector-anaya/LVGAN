#!/usr/bin/env python3
"""
preprocess_ucf101.py
====================
Preprocesses UCF-101 into the flat frame-folder layout used by StyleGAN-V,
DIGAN, MoStGAN-V, PVDM, and AR-Diffusion for the canonical 256×256 benchmark.

Protocol (matches all published FVD/IS numbers):
  - Train + Test combined  (no class split, fully unconditional)
  - All 101 classes thrown into one flat pool of video folders
  - Center-crop to square, then resize to 256×256
  - Extract at native FPS (≈25–30) — NOT resampled to a fixed rate
    (StyleGAN-V uses sparse / random-stride sampling at training time)
  - Saved as JPEG quality=95 to match the literature
  - Output layout:
        <output_dir>/
            v_ApplyEyeMakeup_g01_c01/
                frame_000001.jpg
                frame_000002.jpg
                ...
            v_Archery_g01_c01/
                frame_000001.jpg
                ...

Usage:
    # Step 1 — extract the RAR (requires `unrar` or `7z` installed on WSL)
    unrar x UCF101.rar /path/to/ucf101_videos/

    # Step 2 — run this script
    python preprocess_ucf101.py \
        --src  /path/to/ucf101_videos \
        --dst  /path/to/ucf101_256 \
        --size 256 \
        --workers 8 \
        --min_frames 17        # drop clips shorter than this (rare)

Requirements:
    pip install opencv-python-headless tqdm
"""

import argparse
import os
import sys
import glob
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def center_crop_and_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Square-center-crop then resize to (size × size).
    Identical to torchvision.transforms.CenterCrop + Resize used in StyleGAN-V.
    """
    h, w = frame.shape[:2]
    # Crop to square along the longer axis
    short = min(h, w)
    top  = (h - short) // 2
    left = (w - short) // 2
    frame = frame[top:top+short, left:left+short]
    # Resize
    if short != size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return frame


def process_one_video(args_tuple):
    """Worker function — called in a subprocess."""
    avi_path, out_dir, size, min_frames, jpeg_quality = args_tuple

    out_dir = Path(out_dir)
    # Skip if already done (idempotent re-runs)
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= min_frames:
        return out_dir.name, "skipped"

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return Path(avi_path).stem, f"ERROR: could not open {avi_path}"

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV reads BGR → convert to RGB before crop/resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = center_crop_and_resize(frame, size)
        frames.append(frame)
    cap.release()

    if len(frames) < min_frames:
        return Path(avi_path).stem, f"SKIP: only {len(frames)} frames"

    out_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    for idx, f in enumerate(frames):
        # Back to BGR for imwrite
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"frame_{idx+1:06d}.jpg"
        cv2.imwrite(str(fname), bgr, encode_params)

    return out_dir.name, len(frames)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def collect_avi_paths(src: Path):
    """
    UCF-101 after unrar has two common layouts:
      A) flat:  <src>/*.avi                       (unrar e)
      B) nested: <src>/<ClassName>/<clip>.avi      (unrar x)
    We handle both and merge into a flat list.
    """
    avis = list(src.rglob("*.avi"))
    if not avis:
        # Some builds extract as .AVI
        avis = list(src.rglob("*.AVI"))
    return sorted(avis)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UCF-101 to 256×256 frame folders (StyleGAN-V protocol).")
    parser.add_argument("--src",        required=True,
                        help="Root directory containing .avi files (or class sub-dirs)")
    parser.add_argument("--dst",        required=True,
                        help="Output directory for frame folders")
    parser.add_argument("--size",       type=int, default=256,
                        help="Output spatial resolution (default 256)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="Parallel worker processes (default 8)")
    parser.add_argument("--min_frames", type=int, default=17,
                        help="Drop clips with fewer than this many frames (default 17)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality 0-100 (default 95, matches StyleGAN-V)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for .avi files in: {src}")
    avis = collect_avi_paths(src)
    if not avis:
        sys.exit(f"No .avi files found under {src}. "
                 "Did you run 'unrar x UCF101.rar <src>' first?")
    print(f"Found {len(avis):,} clips → extracting to {dst}")

    # Build work list
    jobs = []
    for avi in avis:
        clip_name = avi.stem          # e.g. v_ApplyEyeMakeup_g01_c01
        out_dir   = dst / clip_name
        jobs.append((str(avi), str(out_dir),
                     args.size, args.min_frames, args.jpeg_quality))

    # ── Parallel extraction ───────────────────────────────────────────────────
    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_video, j): j for j in jobs}
        with tqdm(total=len(jobs), unit="clip", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                name, result = fut.result()
                if isinstance(result, int):
                    n_ok += 1
                elif result == "skipped":
                    n_skip += 1
                else:
                    n_err += 1
                    tqdm.write(f"  {name}: {result}")
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

    # ── Summary ───────────────────────────────────────────────────────────────
    video_dirs = [d for d in dst.iterdir() if d.is_dir()]
    total_frames = sum(len(list(d.glob("*.jpg"))) for d in video_dirs)

    print("\n" + "="*60)
    print(f"  Done!")
    print(f"  Video folders written : {len(video_dirs):,}")
    print(f"  Total frames on disk  : {total_frames:,}")
    print(f"  Clips processed       : {n_ok:,}")
    print(f"  Clips skipped (exist) : {n_skip:,}")
    print(f"  Clips with errors     : {n_err:,}")
    print(f"  Output root           : {dst}")
    print("="*60)
    print()
    print("Expected layout:")
    print(f"  {dst}/")
    print(f"  ├── v_ApplyEyeMakeup_g01_c01/")
    print(f"  │   ├── frame_000001.jpg")
    print(f"  │   ├── frame_000002.jpg")
    print(f"  │   └── ...")
    print(f"  ├── v_Archery_g01_c01/")
    print(f"  └── ...")
    print()
    print("This matches the VideoFramesFolderDataset layout expected by:")
    print("  StyleGAN-V · DIGAN · MoStGAN-V · PVDM · AR-Diffusion")
    print()
    print("Next steps:")
    print("  1. Point your make_dataset() / make_paths() to:", dst)
    print("  2. For FVD evaluation, point StyleGAN-V calc_metrics_for_dataset.py to:", dst)
    print("  3. No train/test split needed — use all folders unconditionally.")


if __name__ == "__main__":
    main()
