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
#  - CUDA and the host compiler come from the cluster's CUDA/11.1.1-GCC-10.2.0
#    module, which is an EXACT match for torch 1.9.0+cu111 (11.1, not merely
#    "some 11.x"), and is EasyBuild-paired with gcc 10.2.0 -- the right host
#    compiler, since CUDA 11.x rejects anything newer than gcc 11 and Huapáctic's
#    default is 12.2.0.
#  - Earlier revisions of this script took CUDA from conda instead. Do not go
#    back to that. conda's `cuda-libraries-dev=11.8.0` is an old-style
#    metapackage whose `*_linux-64` dependencies are new-style and track
#    `cuda-version`, so the env silently ended up with nvcc 11.8 alongside
#    cudart 13.3 and libcublas 13.6. It also lays headers out as
#    targets/x86_64-linux/include with libs in lib/ (not lib64/), which is not
#    where torch's cpp_extension looks -- hence "No CUDA runtime is found".
#    A module is one coherent toolkit with the layout torch expects.
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
CUDA_MODULE="CUDA/11.1.1-GCC-10.2.0"   # exact match for cu111; pulls gcc 10.2.0
TORCH_FIND_LINKS="https://download.pytorch.org/whl/torch_stable.html"
# ------------------------------------------------------------------------------

step() { echo; echo "==> $*"; }
have() { "$PY" -c "import $1" >/dev/null 2>&1; }

conda_on()  { conda activate "$1"; }
conda_off() { conda deactivate; }

module purge
module load "$CUDA_MODULE"
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"

# EasyBuild exports EBROOTCUDA for the module's install prefix.
CUDA_HOME="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: $CUDA_MODULE did not provide a usable CUDA_HOME (got '${CUDA_HOME:-unset}')." >&2
    echo "       Available: module avail 2>&1 | tr ' ' '\\n' | grep -i '^CUDA/'" >&2
    exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
step "CUDA from module: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -1))"

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

# --- 2. purge conda's CUDA and compilers --------------------------------------
# CUDA and gcc now come from $CUDA_MODULE. Any conda copies must go, for two
# reasons beyond mere redundancy:
#   1. The conda CUDA set is internally inconsistent (nvcc 11.8 + cudart 13.3 +
#      libcublas 13.6, all held together by a cuda-version 13.3 metapackage).
#   2. gcc_linux-64's activate.d scripts EXPORT CC/CXX on every activation, which
#      would silently override the module's gcc 10.2.0 and point nvcc back at a
#      compiler it was never paired with.
# --force-remove because the broken CUDA set's own dependency graph is what got
# us here; we are deleting it wholesale, not satisfying it.
CONDA_CUDA_PKGS="$(conda list -p "$ENV_DIR" 2>/dev/null \
    | awk '$1 ~ /^(cuda|libcublas|libcufft|libcurand|libcusolver|libcusparse|libnvjitlink|gcc_linux|gxx_linux|binutils_linux|sysroot_linux)/ {print $1}' \
    | tr '\n' ' ')"

if [ -n "$(echo "$CONDA_CUDA_PKGS" | tr -d '[:space:]')" ]; then
    step "Removing conda CUDA/compiler packages (module provides these):"
    echo "    $CONDA_CUDA_PKGS"
    conda remove -y -p "$ENV_DIR" --force-remove $CONDA_CUDA_PKGS || \
        echo "    (some were already absent — continuing)"
else
    step "no conda CUDA/compiler packages to remove — skipping"
fi

# Host compiler comes from the CUDA module's GCC. Clear anything conda may still
# be exporting so cpp_extension does not pick up a stale conda toolchain.
unset CC CXX
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
step "CC=$CC ($("$CC" -dumpversion))"
step "CXX=$CXX"

case "$("$CC" -dumpversion)" in
    1[2-9]*|[2-9][0-9]*)
        echo "ERROR: gcc $("$CC" -dumpversion) is too new for CUDA 11.x (needs <= 11)." >&2
        echo "       $CUDA_MODULE should have supplied gcc 10.2.0 — check 'module list'." >&2
        exit 1 ;;
esac

# The headers the ops actually need, verified where nvcc will look for them.
for h in include/cuda_runtime.h include/thrust/complex.h include/cublas_v2.h; do
    if [ ! -f "$CUDA_HOME/$h" ]; then
        echo "ERROR: $CUDA_MODULE is missing $CUDA_HOME/$h" >&2
        exit 1
    fi
done
step "CUDA headers verified under $CUDA_HOME/include"

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
# Presence is NOT enough. An earlier partial build left ncps at 0.0.2 while the pin
# says 1.0.1, and a bare `have ncps` check happily skipped the install that would
# have corrected it -- so the pin silently never applied. ncps 0.0.2 and 1.0.1 do
# not share an API, and motion.py imports LTC / CfC / AutoNCP from it. Check the
# pinned versions themselves, not just importability.
deps_ok() {
    "$PY" - <<PY >/dev/null 2>&1
import importlib.metadata as md
assert md.version("ncps") == "$NCPS_VER"
assert md.version("numpy") == "$NUMPY_VER"
assert md.version("hydra-core") == "1.0.7"
import psutil, pandas, cv2, imageio, scipy
from moviepy.editor import VideoFileClip
PY
}

if deps_ok; then
    step "project deps present at pinned versions — skipping"
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

import importlib.metadata as md
import hydra, omegaconf, ncps, psutil, cv2, imageio, scipy, PIL
print(f"  hydra       {hydra.__version__}")
print(f"  omegaconf   {omegaconf.__version__}")
# ncps.__version__ is unreliable -- report what pip actually installed, and hold
# it to the pin. 0.0.2 and 1.0.1 differ in API, and motion.py imports LTC/CfC/AutoNCP.
print(f"  ncps        {md.version('ncps')}")
assert md.version("ncps") == "$NCPS_VER", f"ncps {md.version('ncps')} != $NCPS_VER"
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

# motion.py:22-30 wraps the ncps import in try/except ImportError, so a broken or
# wrong-version ncps leaves _NCPS_AVAILABLE False and motion.py still imports
# cleanly. The ncps_ltc / ncps_cfc strategies would then fail at model
# construction, i.e. after a job is already queued. Assert it here instead.
from src.training import motion as _m
print(f"  ncps strategies available: {_m._NCPS_AVAILABLE}")
assert _m._NCPS_AVAILABLE, (
    "ncps.torch / ncps.wirings did not import -- gen_strategy=ncps_ltc|ncps_cfc would fail")
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
