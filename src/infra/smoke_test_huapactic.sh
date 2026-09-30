#!/bin/bash
# Prove the env works on a gpu2 node BEFORE queuing a real run.
# The thing that actually breaks on a new cluster is the ninja/nvcc JIT build of
# bias_act and upfirdn2d, and that can only happen on a GPU node.
#
#   srun -p gpu2 --gres=gpu:1 --cpus-per-gpu=8 --time=00:20:00 --pty \
#        bash src/infra/smoke_test_huapactic.sh

# `-eo pipefail` without `-u`: conda's compiler activate.d/deactivate.d scripts
# read unset variables ($ADDR2LINE, $CONDA_BACKUP_CXX) by design, and the conda
# shell function sources them on activate, deactivate AND after any install.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

module purge
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env"

# A100 = sm_80. Pin it so the extension build does not probe the device.
export TORCH_CUDA_ARCH_LIST="8.0"
# nvcc and the host compiler both live in the env (see setup_huapactic.sh).
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
# Override to a node-local path to rule out NFS, e.g.
#   TORCH_EXTENSIONS_DIR=/tmp/te_$USER srun ... bash src/infra/smoke_test_huapactic.sh
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_DIR/.torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# A job killed mid-build (SLURM time limit, Ctrl-C) leaves the FileBaton lock
# behind. custom_ops.py:105 and torch's own _jit_compile then call baton.wait(),
# which loops forever on a lock whose owner is long gone -- the retry hangs
# instead of rebuilding. Nothing else holds these locks between jobs, so any that
# exist now are stale by definition.
STALE=$(find "$TORCH_EXTENSIONS_DIR" -name lock 2>/dev/null | wc -l)
if [ "$STALE" -gt 0 ]; then
    echo "== clearing $STALE stale build lock(s) from a killed run =="
    find "$TORCH_EXTENSIONS_DIR" -name lock -print -delete
fi

echo "== node =="; hostname; nvidia-smi -L
echo "== nvcc =="; nvcc --version | tail -2
echo "== host compiler =="; "${CXX:-g++}" --version | head -1; echo "CXX=${CXX:-<unset, using system g++>}"

# -u: unbuffered, so progress appears live instead of arriving all at once at the
# end (or not at all, if the job is killed first).
# Fail fast and legibly if the CUDA dev headers are incomplete, rather than
# spending minutes in nvcc to reach the same conclusion.
for h in include/thrust/complex.h include/cublas_v2.h include/cuda_runtime.h; do
    if [ ! -f "$CONDA_PREFIX/$h" ]; then
        echo "ERROR: missing $CONDA_PREFIX/$h" >&2
        echo "       CUDA dev headers incomplete -- re-run src/infra/setup_huapactic.sh" >&2
        exit 1
    fi
done
echo "== CUDA headers present =="

python -u - <<'PY'
import time
_t = time.time()
def lap(what):
    global _t
    print(f"    [{time.time() - _t:6.1f}s] {what}", flush=True)
    _t = time.time()

import torch
lap("import torch")
print("torch             :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
print("device            :", torch.cuda.get_device_name(0))
print("capability        :", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available(), "no CUDA device visible inside the allocation"

# sm_80 must be in the wheel's compiled arch list, or every kernel silently JITs
# through PTX (or fails). cu111 does include it; this asserts rather than assumes.
print("arch list         :", torch.cuda.get_arch_list())
assert "sm_80" in torch.cuda.get_arch_list(), "wheel has no sm_80 -> wrong torch build for A100"

# custom_ops defaults to verbosity='brief', which passes verbose=False to
# cpp_extension.load() -- so a slow nvcc build and a deadlock look identical from
# outside. 'full' streams the ninja command lines.
from src.torch_utils import custom_ops
custom_ops.verbosity = 'full'
print(f"\nTORCH_EXTENSIONS_DIR: {__import__('os').environ.get('TORCH_EXTENSIONS_DIR')}")
print("building custom CUDA ops (first run on a new cluster takes a few minutes)...\n", flush=True)

# The real test: compiles the two custom CUDA extensions with ninja + nvcc.
from src.torch_utils.ops import bias_act, upfirdn2d
x = torch.randn(4, 8, 16, 16, device="cuda")
b = torch.randn(8, device="cuda")
y = bias_act.bias_act(x, b, act="lrelu")
print("bias_act          :", tuple(y.shape), "OK")
lap("build+run bias_act")

f = upfirdn2d.setup_filter([1, 3, 3, 1], device="cuda")
z = upfirdn2d.upsample2d(x, f)
print("upfirdn2d         :", tuple(z.shape), "OK")
lap("build+run upfirdn2d")

# conv2d_gradfix must take the custom path, not the F.conv2d fallback -- that is
# the whole reason this env is pinned to torch 1.9 (see conv2d_gradfix.py:53).
from src.torch_utils.ops import conv2d_gradfix
assert conv2d_gradfix._should_use_custom_op(x), \
    "conv2d_gradfix fell back to F.conv2d -- not comparable to the gpuserver02 runs"
print("conv2d_gradfix    : custom op active")

# Double-backward is what gradfix exists for (R1 / path-length regularization).
w = torch.randn(8, 8, 3, 3, device="cuda", requires_grad=True)
xr = x.detach().requires_grad_(True)
out = conv2d_gradfix.conv2d(xr, w, padding=1)
g = torch.autograd.grad(out.sum(), xr, create_graph=True)[0]
torch.autograd.grad(g.sum(), w)
print("double-backward   : OK")

import ncps
print("ncps              :", getattr(ncps, "__version__", "installed"))
PY

echo
echo "== smoke test passed =="
