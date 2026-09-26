#!/bin/bash
# Prove the env works on a gpu2 node BEFORE queuing a real run.
# The thing that actually breaks on a new cluster is the ninja/nvcc JIT build of
# bias_act and upfirdn2d, and that can only happen on a GPU node.
#
#   srun -p gpu2 --gres=gpu:1 --cpus-per-gpu=8 --time=00:20:00 --pty \
#        bash src/infra/smoke_test_huapactic.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

module purge
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"

# `set -u` must be off across conda activation. The compiler packages install
# etc/conda/activate.d/activate-binutils_linux-64.sh, which reads $ADDR2LINE (and
# friends) to stash their prior values before overwriting them. Those are unset in
# a clean shell, so -u makes activation fatal. Same applies to module commands.
set +u
conda activate "$PROJECT_DIR/env"
set -u

# A100 = sm_80. Pin it so the extension build does not probe the device.
export TORCH_CUDA_ARCH_LIST="8.0"
# nvcc and the host compiler both live in the env (see setup_huapactic.sh).
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/.torch_extensions"

echo "== node =="; hostname; nvidia-smi -L
echo "== nvcc =="; nvcc --version | tail -2
echo "== host compiler =="; "${CXX:-g++}" --version | head -1; echo "CXX=${CXX:-<unset, using system g++>}"

python - <<'PY'
import torch
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

# The real test: compiles the two custom CUDA extensions with ninja + nvcc.
from src.torch_utils.ops import bias_act, upfirdn2d
x = torch.randn(4, 8, 16, 16, device="cuda")
b = torch.randn(8, device="cuda")
y = bias_act.bias_act(x, b, act="lrelu")
print("bias_act          :", tuple(y.shape), "OK")

f = upfirdn2d.setup_filter([1, 3, 3, 1], device="cuda")
z = upfirdn2d.upsample2d(x, f)
print("upfirdn2d         :", tuple(z.shape), "OK")

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
