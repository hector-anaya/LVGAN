#!/bin/bash
# Bootstrap the stylegan-v conda env on LNS-BUAP "Huapáctic".
# Run from the repo root ON THE LOGIN NODE (LNSHEAD). Creates ./env.
#
#   bash src/infra/setup_huapactic.sh           # build, or resume a partial build
#   FORCE=1 bash src/infra/setup_huapactic.sh   # wipe ./env and start clean
#
# Every step checks for its own artifact first, so re-running after an interrupted
# or stalled step picks up where it left off instead of starting over.
#
# --- Why these pins -----------------------------------------------------------
# They mirror the gpuserver02 env that produced the 18 completed runs, verbatim:
#
#     torch==1.9.0+cu111   torchvision==0.10.0+cu111   numpy==1.22.4
#     hydra-core==1.0.7    omegaconf==2.0.6            ncps==1.0.1
#
# This is not nostalgia. conv2d_gradfix.py:53 gates the custom op on
# ['1.7.','1.8.','1.9','1.10']; on torch 1.9 it is ENABLED, on torch 2.x it
# silently falls back to F.conv2d. That changes the kernels used for the R1
# double-backward, so a 2.x env would not be comparable to the existing results.
# Staying on 1.9 also avoids torch-2.x risk in persistence.py and the gradfix
# shims, which this repo was never tested against.
#
# --- Why the toolchain looks like this ----------------------------------------
#  - No conda on PATH; Anaconda3/2022.05 is provided as a module.
#  - torch 1.9's cpp_extension._check_cuda_version REQUIRES the nvcc major version
#    to match torch.version.cuda (11). A minor mismatch is only a warning, so any
#    CUDA 11.x nvcc works; CUDA 12 would hard-fail. The env therefore pulls an
#    11.x nvcc from conda rather than the cluster's default CUDA.
#  - CUDA 11.x rejects host compilers newer than gcc 11, and Huapáctic's default
#    is gcc 12.2.0. The compiler also comes from conda, so this never depends on
#    which GCC modules happen to exist.
#  - Driver is 530.30.02 (CUDA 12.1). It runs cu111 binaries fine: the driver is
#    backward compatible, and cu111 already emits sm_80 for the A100.

# NOTE: deliberately `-eo pipefail` WITHOUT `-u`.
# conda's compiler packages ship etc/conda/{activate,deactivate}.d scripts that
# read $ADDR2LINE, $CONDA_BACKUP_CXX and friends to save/restore prior values.
# Those are unset in a clean shell, so `-u` turns them fatal. It is not enough to
# guard `conda activate`: the conda shell function calls __conda_reactivate()
# after every install/update/remove into the active prefix, which sources those
# same scripts, so `conda install` trips it too. Errors still abort the script
# via -e; only the unset-variable check is off.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="$PROJECT_DIR/env"
PY="$ENV_DIR/bin/python"

# --- Pins, matched to gpuserver02 ---------------------------------------------
PY_VER=3.9
TORCH_VER=1.9.0+cu111
TV_VER=0.10.0+cu111
NUMPY_VER=1.22.4
NCPS_VER=1.0.1
CUDA_VER=11.8      # nvcc only; major must equal torch's 11
GCC_VER=10         # CUDA 11.x supports gcc <= 11
TORCH_FIND_LINKS="https://download.pytorch.org/whl/torch_stable.html"
# ------------------------------------------------------------------------------

step() { echo; echo "==> $*"; }
have() { "$PY" -c "import $1" >/dev/null 2>&1; }

conda_on()  { conda activate "$1"; }
conda_off() { conda deactivate; }

module purge
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "${FORCE:-0}" = "1" ] && [ -d "$ENV_DIR" ]; then
    step "FORCE=1 — removing $ENV_DIR"
    rm -rf "$ENV_DIR"
fi

# --- 1. interpreter -----------------------------------------------------------
# The python version is part of the pin: torch 1.9 has no wheels past cp39, so a
# 3.10 env (as an earlier build of this script produced) cannot be repaired in
# place and must be rebuilt.
CURRENT_PY_VER=""
if [ -x "$PY" ]; then
    CURRENT_PY_VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
fi

if [ "$CURRENT_PY_VER" = "$PY_VER" ]; then
    step "python $CURRENT_PY_VER present — skipping create"
else
    if [ -d "$ENV_DIR" ]; then
        if [ -n "$CURRENT_PY_VER" ]; then
            step "env has python $CURRENT_PY_VER but torch $TORCH_VER needs $PY_VER — rebuilding"
        else
            step "$ENV_DIR exists but has no working python — removing and recreating"
        fi
        rm -rf "$ENV_DIR"
    fi
    step "Creating env at $ENV_DIR (python $PY_VER)"
    conda create -y -p "$ENV_DIR" "python=$PY_VER" pip
fi

conda_on "$ENV_DIR"

# --- 1a. constraints ----------------------------------------------------------
# numpy is pinned exactly to gpuserver02's version. Enforcing that per-install
# does NOT work: pip resolves one command at a time, so a later install is free to
# pull a different numpy in to satisfy its own dependency (this is exactly how
# opencv broke an earlier build of this env). A constraints file applies to every
# resolution, transitive ones included, forcing packages to backtrack instead.
CONSTRAINTS="$ENV_DIR/constraints.txt"
printf 'numpy==%s\n' "$NUMPY_VER" > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

# Persist it. pip reads <sys.prefix>/pip.conf as its site-level config, so this
# survives beyond this script: a later `pip install <anything>` inside the env
# still cannot move numpy off the pin.
cat > "$ENV_DIR/pip.conf" <<EOF
[install]
constraint = $CONSTRAINTS
EOF
step "Constraints active (persisted to env/pip.conf): $(tr '\n' ' ' < "$CONSTRAINTS")"

# --- 1b. pip version ----------------------------------------------------------
# hydra-core==1.0.7 requires omegaconf<2.1, and omegaconf 2.0.6 declares
# `PyYAML (>=5.1.*)` in its metadata. A `.*` suffix is only legal with == or !=,
# so pip >= 24.1 rejects that wheel outright and hydra 1.0.7 becomes
# uninstallable. Upgrading hydra is not an option: the configs use hydra 1.0
# package semantics (`# @package _group_`), dropped in 1.1.
NEEDS_OLD_PIP="$("$PY" - <<'PY'
import re, pip
m = re.match(r"(\d+)\.(\d+)", pip.__version__)
print(0 if m and (int(m.group(1)), int(m.group(2))) < (24, 1) else 1)
PY
)"
if [ "$NEEDS_OLD_PIP" = "1" ]; then
    step "Downgrading pip (<24.1) for hydra-core 1.0.7 / omegaconf 2.0.6 metadata"
    "$PY" -m pip install "pip<24.1"
else
    step "pip $("$PY" -m pip --version | awk '{print $2}') already < 24.1 — skipping"
fi

# --- 1c. setuptools -----------------------------------------------------------
# setuptools >= 81 removed pkg_resources, which torch's cpp_extension imports at
# module scope. Without it every custom-op import dies before it compiles
# anything (bias_act -> custom_ops -> torch.utils.cpp_extension).
if ! "$PY" -c "import pkg_resources" >/dev/null 2>&1; then
    step "Installing setuptools<81 (cpp_extension needs pkg_resources)"
    pip install "setuptools<81"
else
    step "pkg_resources available — skipping setuptools pin"
fi

# --- 2. compiler + nvcc -------------------------------------------------------
# Both come from conda so the build never depends on cluster module names.
if [ -x "$ENV_DIR/bin/nvcc" ] && [ -n "$(ls "$ENV_DIR"/bin/*-g++ 2>/dev/null || true)" ]; then
    step "nvcc + conda compiler present — skipping"
else
    step "Installing gcc/g++ $GCC_VER (CUDA 11.x rejects gcc > 11; system default here is 12.2)"
    conda install -y -c conda-forge \
        "gcc_linux-64=$GCC_VER" "gxx_linux-64=$GCC_VER" "sysroot_linux-64=2.17"

    step "Installing nvcc $CUDA_VER + CUDA runtime headers"
    conda install -y -c nvidia "cuda-nvcc=$CUDA_VER" "cuda-cudart-dev=$CUDA_VER"
fi

# Re-activate so conda's compiler activation scripts export CC/CXX/CFLAGS with the
# correct sysroot. Installing the packages alone does not set them in this shell.
conda_off
conda_on "$ENV_DIR"

# Fallback: don't depend on the activation dance having worked. The compiler
# binaries from gcc_linux-64 / gxx_linux-64 have deterministic names, so CC/CXX
# can be set directly. torch's cpp_extension reads these to pick the host
# compiler for nvcc; if they stay unset it silently uses the system gcc 12.2,
# which CUDA 11.x rejects.
if [ -z "${CXX:-}" ]; then
    CC="$(ls "$ENV_DIR"/bin/*-linux-gnu-gcc 2>/dev/null | head -1 || true)"
    CXX="$(ls "$ENV_DIR"/bin/*-linux-gnu-g++ 2>/dev/null | head -1 || true)"
    export CC CXX
fi

export CUDA_HOME="$ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
step "CC=${CC:-<unset>}"
step "CXX=${CXX:-<unset>}"

if [ -z "${CXX:-}" ] || [ ! -x "${CXX:-/nonexistent}" ]; then
    echo "ERROR: no conda host compiler found in $ENV_DIR/bin." >&2
    echo "       nvcc would fall back to system gcc $(gcc -dumpversion 2>/dev/null), which CUDA $CUDA_VER rejects." >&2
    exit 1
fi

# --- 3. torch -----------------------------------------------------------------
# The +cu111 local-version wheels are only on the legacy torch_stable index, so
# this needs -f rather than --index-url.
if have torch && [ "$("$PY" -c 'import torch; print(torch.__version__)')" = "$TORCH_VER" ]; then
    step "torch $TORCH_VER present — skipping"
else
    step "Installing torch $TORCH_VER / torchvision $TV_VER"
    pip install -f "$TORCH_FIND_LINKS" "torch==$TORCH_VER" "torchvision==$TV_VER"
fi

# --- 4. project deps ----------------------------------------------------------
if have hydra && have ncps && have psutil; then
    step "project deps present — skipping"
else
    step "Installing project dependencies"
    # av==9.2.0 is pinned by upstream's environment.yaml but is never imported
    # anywhere in this codebase (mp4 goes through cv2.VideoWriter, frames through
    # imageio). Dropped: its sdist needs ffmpeg dev headers, which would need root.
    #
    # opencv-python-headless (not -python) is deliberate: no GUI libs on the nodes,
    # and logging.py already keeps cv2.destroyAllWindows() commented out for that
    # reason. Left unpinned so PIP_CONSTRAINT can backtrack it onto numpy 1.22.4.
    #
    # psutil     -> imported by training_loop.py; training dies without it
    # pandas     -> preprocess_ffs.py
    # moviepy<2  -> convert_video_to_dataset.py imports `moviepy.editor`,
    #               a module moviepy 2.0 removed
    pip install --prefer-binary \
        "hydra-core==1.0.7" "click==7.1.2" "ncps==$NCPS_VER" \
        "numpy==$NUMPY_VER" scipy pillow requests imageio imageio-ffmpeg \
        opencv-python-headless \
        ninja tensorboard tqdm gitpython gpustat matplotlib \
        "joblib==1.1.0" "torch-tb-profiler==0.3.1" \
        psutil pandas "moviepy<2"
fi

# pyspng is a fast PNG decode path in dataset.py, guarded by try/except
# ImportError with a PIL fallback. Best-effort: the build continues without it.
if ! have pyspng; then
    step "Trying pyspng (optional fast PNG decode; PIL fallback if unavailable)"
    pip install --only-binary=:all: pyspng || echo "    pyspng unavailable — PIL fallback, this is fine"
fi

if "$PY" -c "import src" >/dev/null 2>&1; then
    step "project installed — skipping"
else
    step "Installing project (editable)"
    pip install -e "$PROJECT_DIR"
fi

# --- 5. verify ----------------------------------------------------------------
# Everything above only checks that pip succeeded. This checks the env actually
# works, on the login node, before anything reaches the queue.
step "Verifying environment"
"$PY" - <<PY
import numpy, torch, sys
print(f"  python      {sys.version.split()[0]}")
print(f"  numpy       {numpy.__version__}")
print(f"  torch       {torch.__version__}  (cuda {torch.version.cuda})")
assert numpy.__version__ == "$NUMPY_VER", f"numpy {numpy.__version__} != $NUMPY_VER"
assert torch.__version__ == "$TORCH_VER", f"torch {torch.__version__} != $TORCH_VER"

torch.from_numpy(numpy.zeros(4, dtype=numpy.float32))
print("  torch<->numpy bridge OK")

import pkg_resources, setuptools
print(f"  setuptools  {setuptools.__version__} (pkg_resources present)")

import hydra, omegaconf, ncps, psutil, cv2, imageio, scipy, PIL
print(f"  hydra       {hydra.__version__}")
print(f"  omegaconf   {omegaconf.__version__}")
print(f"  ncps        {ncps.__version__ if hasattr(ncps,'__version__') else 'installed'}")
print(f"  cv2         {cv2.__version__}")

from moviepy.editor import VideoFileClip
print("  moviepy.editor OK")

# The whole reason for staying on 1.9: this must report True.
from src.torch_utils.ops import conv2d_gradfix
gate = any(torch.__version__.startswith(x) for x in ['1.7.','1.8.','1.9','1.10'])
print(f"  conv2d_gradfix enabled: {gate}")
assert gate, "gradfix would fall back -- results not comparable to gpuserver02"

from src.training.motion import MotionMappingNetwork
print("  src.training.motion OK")
print("\n  CPU-side checks passed.")
PY

step "Env built at $ENV_DIR ($(du -sh "$ENV_DIR" | cut -f1))"
echo
echo "Still unverified: the ninja/nvcc build of bias_act + upfirdn2d, which can"
echo "only happen on a GPU node."
echo
echo "Verify on a GPU node:"
echo "  srun -p gpu2 --gres=gpu:1 --cpus-per-gpu=8 --time=00:20:00 --pty \\"
echo "       bash src/infra/smoke_test_huapactic.sh"
