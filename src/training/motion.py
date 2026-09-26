import inspect
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import DictConfig

from src.torch_utils import misc
from src.torch_utils import persistence
from src.training.layers import (
    MappingNetwork,
    EqLRConv1d,
    FullyConnectedLayer,
)

# Optional: the reference Neural Circuit Policies library (Hasani & Lechner et al.).
# Only required for the `ncps_ltc` / `ncps_cfc` generation strategies, so we keep the
# import soft to avoid breaking training with the other strategies when it is absent.
try:
    from ncps.torch import LTC as NCPSLTCModule, CfC as NCPSCfCModule
    from ncps.wirings import AutoNCP as NCPSAutoNCP
    _NCPS_AVAILABLE = True
except ImportError:
    NCPSLTCModule = None
    NCPSCfCModule = None
    NCPSAutoNCP = None
    _NCPS_AVAILABLE = False

# `use_reentrant` was added to torch.utils.checkpoint.checkpoint in PyTorch 1.11, but
# this repo's environments pin 1.7.1 / 1.9 (environment.yaml, environment-ampere.yaml),
# where the same call raises "Unexpected keyword arguments: use_reentrant". The legacy
# reentrant path also needs a gradient anchor -- see NCPSLiquidMotionNetwork.forward.
_CKPT_SUPPORTS_USE_REENTRANT = 'use_reentrant' in inspect.signature(
    torch.utils.checkpoint.checkpoint).parameters

#----------------------------------------------------------------------------

@persistence.persistent_class
class ExpClosedFormCell(nn.Module):
    """
    Closed-form recurrent cell with input-dependent time constants.

    Each step advances the state by Δt with an exponential gate:
        x(t+Δt) = x(t) * exp(-Δt/τ(x,u)) + f(x,u) * (1 - exp(-Δt/τ(x,u)))

    τ adapts to the input, letting the network slow down or speed up its
    dynamics to match the content — which is what produces temporally
    coherent motion trajectories.

    This cell is *inspired by* LTC (Hasani et al., "Liquid Time-constant
    Networks", AAAI 2021) in its input-dependent time constants, and adopts the
    closed-form update strategy of CfC (Hasani et al., "Closed-form
    Continuous-time Neural Networks", Nature Machine Intelligence 2022) — but it
    is identical to neither. In particular it is *not* the solver-based
    conductance model of the LTC paper: there is no synaptic conductance term and
    no numerical ODE solver. `ode_unfolds` here simply subdivides Δt into that
    many successive closed-form steps; it is not an ODE integration scheme.

    Relative to CfCCell below, the difference under ablation is the gate: an
    exponential decay `exp(-Δt/τ)` here vs. a sigmoid `σ(-τ·Δt)` there.
    """
    def __init__(self, input_dim: int, hidden_dim: int, ode_unfolds: int = 6, lr_multiplier: float = 0.01,
                 *, learn_tau_bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_unfolds = ode_unfolds
        self.learn_tau_bias = learn_tau_bias

        # Both layers use equalized learning rates to match the rest of the codebase
        self.W_tau = FullyConnectedLayer(input_dim + hidden_dim, hidden_dim, activation='linear', lr_multiplier=lr_multiplier)
        self.W_f   = FullyConnectedLayer(input_dim + hidden_dim, hidden_dim, activation='linear', lr_multiplier=lr_multiplier)

        # NOTE: `log_tau_0` is mathematically redundant -- it is a per-channel additive
        # constant on the output of `W_tau`, which already carries its own bias. It is not
        # redundant in *training dynamics*, however: `W_tau`'s bias is an equalized-LR
        # parameter scaled by `lr_multiplier` in the forward pass, while `log_tau_0` is a
        # raw nn.Parameter at the full optimizer LR. So it acts as one fast-learning
        # scalar per channel inside an otherwise ~100x slower cell. `learn_tau_bias=False`
        # removes it, which is the controlled ablation for that asymmetry. (`log_A` below
        # is a genuine amplitude, not redundant, but carries the same LR asymmetry.)
        self.log_tau_0 = nn.Parameter(torch.zeros(hidden_dim)) if learn_tau_bias else None
        self.log_A     = nn.Parameter(torch.zeros(hidden_dim))

    def _ode_step(self, x: torch.Tensor, u: torch.Tensor, delta_t: float) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        # Adaptive time constant τ > 0, baseline is exp(log_tau_0) ≈ 1 frame
        tau_logits = self.W_tau(xu) if self.log_tau_0 is None else self.W_tau(xu) + self.log_tau_0
        tau = F.softplus(tau_logits) + 1e-3                       # [B, H]
        # Steady-state target f, bounded by learnable amplitude A
        f   = torch.tanh(self.W_f(xu)) * F.softplus(self.log_A)   # [B, H]
        decay = torch.exp(-delta_t / tau)
        return x * decay + f * (1.0 - decay)

    def forward(self, u: torch.Tensor, h: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            u: input at current step  [B, input_dim]
            h: previous hidden state  [B, hidden_dim]
            delta_t: elapsed time (normalised — 1.0 == one motion_z_distance unit)
        Returns:
            h_new: [B, hidden_dim]
        """
        step = delta_t / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            h = self._ode_step(h, u, step)
        return h

#----------------------------------------------------------------------------

@persistence.persistent_class
class ExpClosedFormMotionNetwork(nn.Module):
    """
    Processes a noise trajectory through an ExpClosedFormCell recurrent network to
    produce smooth, temporally coherent motion codes.

    Compared to the Conv1d motion network (which sees only a fixed local
    window), the hidden state carries information across the entire sequence and
    adapts its time constant to the content, avoiding the discontinuities that
    can appear at Conv1d window boundaries.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, ode_unfolds: int = 6, lr_multiplier: float = 0.01,
                 *, learn_tau_bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ExpClosedFormCell(input_dim, hidden_dim, ode_unfolds, lr_multiplier, learn_tau_bias=learn_tau_bias)
        self.proj = FullyConnectedLayer(hidden_dim, output_dim, activation='lrelu', lr_multiplier=lr_multiplier)

    def forward(self, trajectory: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            trajectory: [B, T, input_dim] — noise trajectory
            delta_t:    time between consecutive codes (1.0 = one motion_z_distance unit)
        Returns:
            [B, T, output_dim] — motion codes
        """
        B, T, _ = trajectory.shape
        h = trajectory.new_zeros(B, self.hidden_dim)
        outs: List[torch.Tensor] = []
        for t in range(T):
            h = self.cell(trajectory[:, t], h, delta_t)
            outs.append(h)
        out = torch.stack(outs, dim=1)                    # [B, T, hidden_dim]
        out = self.proj(out.reshape(B * T, -1))           # [B*T, output_dim]
        return out.reshape(B, T, -1)                      # [B, T, output_dim]

#----------------------------------------------------------------------------

# Backwards-compatible aliases under the old, inaccurate names. These exist so that
# external scripts which import `LTCCell` / `LiquidMotionNetwork` keep working; they
# are not needed for unpickling old checkpoints, which embed their own copy of this
# module's source and resolve the class name inside it (see torch_utils/persistence.py).
LTCCell = ExpClosedFormCell
LiquidMotionNetwork = ExpClosedFormMotionNetwork

#----------------------------------------------------------------------------

@persistence.persistent_class
class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell.

    A single time-gated evaluation advances the hidden state by Δt,
    interpolating between an early-time response `g` (Δt → 0) and a
    steady-state response `h` (Δt → ∞) through an input-dependent time gate:

        x(t+Δt) = σ(-(τ(x,u)) · Δt) ⊙ g(x,u) + [1 - σ(-(τ(x,u)) · Δt)] ⊙ h(x,u)

    This is a single closed-form update per step, avoiding the inner ODE loop and
    its accumulated discretisation error. Note that it is *not* an exact solution
    of the liquid ODE: the sigmoid gate replaces the exponential from the analytic
    solution for numerical stability, and the derivation makes further simplifying
    assumptions. Relative to ExpClosedFormCell above, this is the sigmoid-gated
    variant of the same closed-form idea.

    Reference: Hasani et al., "Closed-form Continuous-time Neural Networks",
    Nature Machine Intelligence 2022.
    """
    def __init__(self, input_dim: int, hidden_dim: int, lr_multiplier: float = 0.01,
                 *, learn_tau_bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learn_tau_bias = learn_tau_bias

        # Shared backbone, then three heads (matching the official CfC design)
        self.backbone = FullyConnectedLayer(input_dim + hidden_dim, hidden_dim, activation='lrelu', lr_multiplier=lr_multiplier)
        self.g   = FullyConnectedLayer(hidden_dim, hidden_dim, activation='linear', lr_multiplier=lr_multiplier)  # early-time limit
        self.h   = FullyConnectedLayer(hidden_dim, hidden_dim, activation='linear', lr_multiplier=lr_multiplier)  # steady-state limit
        self.tau = FullyConnectedLayer(hidden_dim, hidden_dim, activation='linear', lr_multiplier=lr_multiplier)  # input-dependent rate

        # NOTE: `log_tau_0` is mathematically redundant -- it is a per-channel additive
        # constant on the output of `self.tau`, which already carries its own bias. It is
        # not redundant in *training dynamics*, however: that bias is an equalized-LR
        # parameter scaled by `lr_multiplier` in the forward pass, while `log_tau_0` is a
        # raw nn.Parameter at the full optimizer LR. So it acts as one fast-learning
        # scalar per channel inside an otherwise ~100x slower cell. `learn_tau_bias=False`
        # removes it, which is the controlled ablation for that asymmetry.
        self.log_tau_0 = nn.Parameter(torch.zeros(hidden_dim)) if learn_tau_bias else None

    def forward(self, u: torch.Tensor, h: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            u: input at current step  [B, input_dim]
            h: previous hidden state  [B, hidden_dim]
            delta_t: elapsed time (normalised — 1.0 == one motion_z_distance unit)
        Returns:
            h_new: [B, hidden_dim]
        """
        feat = self.backbone(torch.cat([h, u], dim=-1))   # [B, H]
        g_lim = torch.tanh(self.g(feat))                  # [B, H] response as Δt → 0
        h_lim = torch.tanh(self.h(feat))                  # [B, H] response as Δt → ∞
        tau_logits = self.tau(feat) if self.log_tau_0 is None else self.tau(feat) + self.log_tau_0
        gate = torch.sigmoid(-tau_logits * delta_t)       # [B, H]
        return g_lim * gate + h_lim * (1.0 - gate)

#----------------------------------------------------------------------------

@persistence.persistent_class
class ClosedFormMotionNetwork(nn.Module):
    """
    Processes a noise trajectory through a CfC recurrent network to produce
    smooth, temporally coherent motion codes.

    Like ExpClosedFormMotionNetwork, the hidden state carries context across the
    full sequence, but each step is a single closed-form update (no ode_unfolds
    loop), so it is cheaper while keeping the continuous-time inductive bias.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr_multiplier: float = 0.01,
                 *, learn_tau_bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = CfCCell(input_dim, hidden_dim, lr_multiplier, learn_tau_bias=learn_tau_bias)
        self.proj = FullyConnectedLayer(hidden_dim, output_dim, activation='lrelu', lr_multiplier=lr_multiplier)

    def forward(self, trajectory: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            trajectory: [B, T, input_dim] — noise trajectory
            delta_t:    time between consecutive codes (1.0 = one motion_z_distance unit)
        Returns:
            [B, T, output_dim] — motion codes
        """
        B, T, _ = trajectory.shape
        h = trajectory.new_zeros(B, self.hidden_dim)
        outs: List[torch.Tensor] = []
        for t in range(T):
            h = self.cell(trajectory[:, t], h, delta_t)
            outs.append(h)
        out = torch.stack(outs, dim=1)                    # [B, T, hidden_dim]
        out = self.proj(out.reshape(B * T, -1))           # [B*T, output_dim]
        return out.reshape(B, T, -1)                      # [B, T, output_dim]

#----------------------------------------------------------------------------

@persistence.persistent_class
class NCPSLiquidMotionNetwork(nn.Module):
    """
    Same role as ExpClosedFormMotionNetwork, but the recurrent core is the
    reference LTC implementation from the `ncps` library (Neural Circuit
    Policies, Hasani & Lechner et al.). Unlike the three closed-form arms, this
    one really is the solver-based conductance model of the LTC paper.

    The ncps cell is wrapped so the public interface matches the other motion
    networks: it consumes a [B, T, input_dim] noise trajectory and returns a
    [B, T, output_dim] sequence of motion codes. We add a final equalized-LR
    projection to map the cell's output width to motion.v_dim, keeping the
    rest of the codebase unchanged.

    PRECISION: this arm must stay in fp32. Its semi-implicit update divides by
    `cm / Δt + gleak + w_act`, which is not bounded away from zero, so reduced
    precision can put the denominator at or through zero — the same instability
    already observed in the TensorFlow track. The three closed-form arms have
    bounded gates (exp / sigmoid) and are the safe candidates for fp16/bf16.

    MEMORY: ncps retains a [B, units, units] synapse-activation tensor per ODE
    sub-step for backward, so cost scales with B * units^2 * T * ode_unfolds. At
    B=32, units=512, T=16, ode_unfolds=6 that is several GB for the motion
    network alone. `use_checkpoint` trades compute for memory (see below).
    AutoNCP does *not* help here: ncps stores the wiring as a dense adjacency
    mask, so masked-out synapses are zeros rather than absent and the
    intermediates are exactly the same size.

    `wiring` selects the synaptic connectivity:
      - 'full' (default): every neuron connects to every other neuron.
      - 'ncp': a sparse AutoNCP wiring (sensory → inter → command → motor). The
        recurrent output is the `motor_neurons` motor cells (defaults to
        hidden_dim // 4), which the final projection maps to output_dim. Note
        this reduces parameter count, not activation memory (see above).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, ode_unfolds: int = 6,
                 wiring: str = 'full', motor_neurons: int = None, seed: int = 22222, lr_multiplier: float = 0.01,
                 *, use_checkpoint: bool = False, checkpoint_every: int = 1):
        super().__init__()
        if not _NCPS_AVAILABLE:
            raise ImportError(
                "gen_strategy='ncps_ltc' requires the `ncps` package. Install it with `pip install ncps`."
            )
        assert wiring in ('full', 'ncp'), f"Unknown ncps_ltc wiring: {wiring}"
        assert checkpoint_every >= 1, f"checkpoint_every must be >= 1, got {checkpoint_every}"
        self.hidden_dim = hidden_dim
        self.wiring = wiring
        self.use_checkpoint = use_checkpoint
        self.checkpoint_every = checkpoint_every
        # Recorded so the optimizer can put the (non-equalized) ncps core into its own
        # param group at `lr * lr_multiplier` -- see `recurrent_parameters` below.
        self.lr_multiplier = lr_multiplier

        # `implicit_param_constraints` is passed explicitly rather than relying on the
        # library default. ncps' LTC *module* already defaults it to True (its LTCCell
        # defaults to False), and when it is False the cell requires an
        # `apply_weight_constraints()` call after every optimizer step to keep `cm`,
        # `gleak` and `w` positive -- which nothing in this codebase does. Pinning it
        # here means a change to that library default cannot silently break the arm.
        if wiring == 'ncp':
            # Sparse Neural Circuit Policy wiring. A fixed seed keeps the connectivity
            # reproducible across persistence reconstruction (init args are replayed).
            self.motor_neurons = motor_neurons if motor_neurons is not None else max(1, hidden_dim // 4)
            ncp_wiring = NCPSAutoNCP(hidden_dim, self.motor_neurons, seed=seed)
            self.rnn = NCPSLTCModule(
                input_dim,
                ncp_wiring,
                return_sequences=True,
                batch_first=True,
                ode_unfolds=ode_unfolds,
                implicit_param_constraints=True,
            )
            rnn_out_dim = self.motor_neurons
        else:
            self.motor_neurons = hidden_dim
            self.rnn = NCPSLTCModule(
                input_dim,
                hidden_dim,
                return_sequences=True,
                batch_first=True,
                ode_unfolds=ode_unfolds,
                implicit_param_constraints=True,
            )
            rnn_out_dim = hidden_dim
        self.proj = FullyConnectedLayer(rnn_out_dim, output_dim, activation='lrelu', lr_multiplier=lr_multiplier)

    def recurrent_parameters(self):
        """Parameters of the ncps core, for a separate optimizer param group."""
        return self.rnn.parameters()

    def _run_segment(self, chunk: torch.Tensor, h: torch.Tensor, delta_t: float):
        """
        Drives ncps' LTCCell manually over `chunk`'s time axis.

        ncps' cell signature is `forward(inputs, states, elapsed_time=1.0)` returning
        `(outputs, next_state)`. `outputs` is the wiring-mapped output (motor neurons
        under an AutoNCP), while `next_state` is the full hidden state, so the two
        differ in width in 'ncp' mode and both must be carried.
        """
        outs: List[torch.Tensor] = []
        for t in range(chunk.shape[1]):
            out_t, h = self.rnn.rnn_cell(chunk[:, t], h, delta_t)
            outs.append(out_t)
        return torch.stack(outs, dim=1), h

    def forward(self, trajectory: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            trajectory: [B, T, input_dim] — noise trajectory
            delta_t:    time between consecutive codes (1.0 = one motion_z_distance unit)
        Returns:
            [B, T, output_dim] — motion codes
        """
        B, T, _ = trajectory.shape
        # NOTE: ncps' LTC only supports a *scalar* elapsed time per step — passing a
        # per-sample `timespans` tensor breaks its internal `cm / elapsed_time`
        # broadcast. We always advance by one motion_z_distance unit (delta_t=1.0),
        # which is exactly ncps' default unit step, so we let it use that default.
        # (Unlike ncps' CfC, which does accept a `timespans` tensor — see
        # NCPSClosedFormMotionNetwork.forward. This assert is real, not conservatism.)
        assert float(delta_t) == 1.0, "NCPSLiquidMotionNetwork only supports delta_t=1.0 (ncps unit step)."

        # Checkpointing only pays for itself when a backward pass will actually follow.
        # Skipping it under inference matters for correctness too, not just speed: the
        # legacy gradient anchor below forces requires_grad onto the output, which would
        # otherwise propagate to frozen copies such as G_ema and break the `.numpy()`
        # calls in the image-export and metrics paths.
        will_backward = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        if not (self.use_checkpoint and will_backward):
            out, _ = self.rnn(trajectory)                   # [B, T, rnn_out_dim]
        else:
            # Gradient checkpointing: drive the cell ourselves so each segment of
            # `checkpoint_every` steps is recomputed in backward instead of retaining
            # every [B, units, units] ODE intermediate. Larger `checkpoint_every`
            # recovers more of the forward compute for proportionally less saving.
            # The LTC state is `hidden_dim` wide in both wirings: under an AutoNCP the
            # wiring's unit count is hidden_dim and only the *output* narrows to
            # motor_neurons, so this is correct for 'full' and 'ncp' alike.
            h = trajectory.new_zeros(B, self.hidden_dim)
            segments: List[torch.Tensor] = []

            # `delta_t` is closed over rather than passed through `checkpoint`: the
            # legacy reentrant CheckpointFunction forwards every argument to
            # `save_for_backward`, which accepts tensors only and would reject a float.
            def run_segment(chunk, h_in, _anchor=None):
                return self._run_segment(chunk, h_in, delta_t)

            # On the legacy path, reentrant checkpointing silently yields *no* gradient
            # for a segment unless at least one of its tensor inputs requires grad --
            # and neither does here: `trajectory` is sampled noise (generate_motion_u_codes
            # builds it with torch.randn) and `h` starts as zeros. Since each segment's
            # output feeds the next one's state, that would cascade and leave the whole
            # ncps core untrained behind a single torch UserWarning. The anchor is an
            # unused grad-requiring input that keeps the recomputed graph connected.
            anchor = None if _CKPT_SUPPORTS_USE_REENTRANT else torch.zeros(
                1, dtype=trajectory.dtype, device=trajectory.device, requires_grad=True)

            for start in range(0, T, self.checkpoint_every):
                chunk = trajectory[:, start:start + self.checkpoint_every]
                if _CKPT_SUPPORTS_USE_REENTRANT:
                    seg, h = torch.utils.checkpoint.checkpoint(
                        run_segment, chunk, h, use_reentrant=False)
                else:
                    seg, h = torch.utils.checkpoint.checkpoint(run_segment, chunk, h, anchor)
                segments.append(seg)
            out = torch.cat(segments, dim=1)                # [B, T, rnn_out_dim]

        out = self.proj(out.reshape(B * T, -1))             # [B*T, output_dim]
        return out.reshape(B, T, -1)                        # [B, T, output_dim]

#----------------------------------------------------------------------------

@persistence.persistent_class
class NCPSClosedFormMotionNetwork(nn.Module):
    """
    Same role as ClosedFormMotionNetwork, but the recurrent core is the
    reference CfC implementation from the `ncps` library rather than our
    hand-rolled CfCCell. Interface and final projection match the other
    motion networks (see NCPSLiquidMotionNetwork).

    The final projection is deliberately our own equalized-LR FullyConnectedLayer
    rather than ncps' `proj_size` option: the latter appends a plain nn.Linear,
    which would sit outside the equalized-LR scheme and reintroduce the learning
    rate asymmetry that the optimizer param groups exist to remove.

    `mode` selects the closed-form update (ncps' terminology):
      - 'default': the gated closed-form update, i.e. the CfC of the paper.
      - 'pure':    the direct solution approximation, without the gating.
      - 'no_gate': the update without the second gate.
    The latter two are ablations of the cell itself, not intended defaults.

    `wiring` selects the synaptic connectivity, mirroring NCPSLiquidMotionNetwork:
      - 'full' (default): a dense CfCCell over `hidden_dim` units, fed by a backbone
        of `backbone_layers` x `backbone_units` driving the g/h/time heads.
      - 'ncp': a sparse AutoNCP wiring (sensory -> inter -> command -> motor). The
        recurrent output is the `motor_neurons` motor cells (defaults to
        hidden_dim // 4), which the final projection maps to output_dim.
        NOTE: ncps raises ValueError if any backbone_* argument is supplied
        alongside a Wiring, so the backbone settings do not apply in this mode --
        'ncp' is therefore a capacity change as well as a connectivity change.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr_multiplier: float = 0.01, *,
                 mode: str = 'default', backbone_units: int = None, backbone_layers: int = 1,
                 backbone_dropout: float = 0.0, wiring: str = 'full', motor_neurons: int = None,
                 seed: int = 22222):
        super().__init__()
        if not _NCPS_AVAILABLE:
            raise ImportError(
                "gen_strategy='ncps_cfc' requires the `ncps` package. Install it with `pip install ncps`."
            )
        assert mode in ('default', 'pure', 'no_gate'), f"Unknown ncps_cfc mode: {mode}"
        assert wiring in ('full', 'ncp'), f"Unknown ncps_cfc wiring: {wiring}"
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.wiring = wiring
        # Recorded so the optimizer can put the (non-equalized) ncps core into its own
        # param group at `lr * lr_multiplier` -- see `recurrent_parameters` below.
        self.lr_multiplier = lr_multiplier

        if wiring == 'ncp':
            # Sparse Neural Circuit Policy wiring. A fixed seed keeps the connectivity
            # reproducible across persistence reconstruction (init args are replayed).
            self.motor_neurons = motor_neurons if motor_neurons is not None else max(1, hidden_dim // 4)
            ncp_wiring = NCPSAutoNCP(hidden_dim, self.motor_neurons, seed=seed)
            # No backbone_* here: ncps rejects them outright in wired mode.
            self.rnn = NCPSCfCModule(
                input_dim,
                ncp_wiring,
                return_sequences=True,
                batch_first=True,
                mode=mode,
            )
            rnn_out_dim = self.motor_neurons
        else:
            self.motor_neurons = hidden_dim
            # ncps silently substitutes backbone_units=128 when it is left as None, which
            # would bottleneck the g/h/time heads relative to our hand-rolled
            # ClosedFormMotionNetwork (whose backbone is hidden_dim wide). That is a
            # capacity difference rather than an implementation difference, and it would
            # confound the comparison -- so we default to hidden_dim instead.
            self.rnn = NCPSCfCModule(
                input_dim,
                hidden_dim,
                return_sequences=True,
                batch_first=True,
                mode=mode,
                backbone_units=hidden_dim if backbone_units is None else backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
            )
            rnn_out_dim = hidden_dim
        self.proj = FullyConnectedLayer(rnn_out_dim, output_dim, activation='lrelu', lr_multiplier=lr_multiplier)

    def recurrent_parameters(self):
        """Parameters of the ncps core, for a separate optimizer param group."""
        return self.rnn.parameters()

    def forward(self, trajectory: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """
        Args:
            trajectory: [B, T, input_dim] — noise trajectory
            delta_t:    time between consecutive codes (1.0 = one motion_z_distance unit)
        Returns:
            [B, T, output_dim] — motion codes
        """
        B, T, _ = trajectory.shape
        # Unlike ncps' LTC (which folds the step into a `cm / elapsed_time` broadcast and
        # so only accepts a scalar), ncps' CfC takes a per-sample `timespans` tensor, so a
        # non-unit delta_t is supported here. delta_t=1.0 stays on a fast path: passing no
        # timespans lets ncps use its own unit step instead of us allocating a constant.
        if float(delta_t) == 1.0:
            out, _ = self.rnn(trajectory)                                   # [B, T, rnn_out_dim]
        else:
            timespans = trajectory.new_full((B, T, 1), float(delta_t))      # [B, T, 1]
            out, _ = self.rnn(trajectory, timespans=timespans)              # [B, T, rnn_out_dim]
        out = self.proj(out.reshape(B * T, -1))             # [B*T, output_dim]
        return out.reshape(B, T, -1)                        # [B, T, output_dim]

#----------------------------------------------------------------------------

@persistence.persistent_class
class MotionMappingNetwork(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        assert self.cfg.motion.gen_strategy in ["autoregressive", "conv", "exp_cf", "ltc", "cfc", "ncps_ltc", "ncps_cfc"], f"Unknown generation strategy: {self.cfg.motion.gen_strategy}"

        # 'ltc' is the historical name for the hand-rolled exponential closed-form cell.
        # It is kept as an alias so existing configs and experiment dirs keep working;
        # 'exp_cf' is the canonical name, matching ExpClosedFormMotionNetwork. Everything
        # below branches on `self.gen_strategy`, never on the raw config value.
        self.gen_strategy = 'exp_cf' if self.cfg.motion.gen_strategy == 'ltc' else self.cfg.motion.gen_strategy

        if self.cfg.motion.fourier:
            self.time_encoder = AlignedTimeEncoder(
                cfg=self.cfg,
                latent_dim=self.cfg.motion.v_dim
            )
        else:
            self.mapping = MappingNetwork(
                z_dim=self.cfg.motion.z_dim,
                c_dim=self.cfg.c_dim,
                w_dim=self.cfg.motion.v_dim,
                num_ws=None,
                num_layers=2,
                activation='lrelu',
                w_avg_beta=None,
                cfg=self.cfg,
            )

        if self.gen_strategy == 'autoregressive':
            self.rnn = nn.LSTM(
                input_size=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_size=self.cfg.motion.z_dim,
                bidirectional=False,
                batch_first=True)
            self._parameters_flattened = False
            self.num_additional_codes = 0
        elif self.gen_strategy == 'conv':
            # Using Conv1d without paddings instead of LSTM makes the generations good for any time in t \in (0, +\infty),
            # while LSTM would diverge for large `t`
            # Also, this allows us to use equalized learning rates
            self.conv = nn.Sequential(
                EqLRConv1d(self.cfg.motion.z_dim + self.cfg.c_dim, self.cfg.motion.z_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
                EqLRConv1d(self.cfg.motion.z_dim, self.cfg.motion.v_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
            )
            self.num_additional_codes = (self.cfg.motion.kernel_size - 1) * 2
        elif self.gen_strategy == 'exp_cf':
            # Exponential closed-form recurrent network: the hidden state carries
            # temporal context across the full trajectory and its adaptive time
            # constants enforce smooth, coherent motion without the window-boundary
            # artefacts of Conv1d.
            # The `ltc_*` config keys keep their historical names (see the 'ltc' alias
            # above), as does the `self.ltc` attribute -- renaming the attribute would
            # change the state_dict keys and break `training.resume` for existing runs.
            ltc_hidden_dim = self.cfg.motion.get('ltc_hidden_dim', self.cfg.motion.z_dim)
            ltc_ode_unfolds = self.cfg.motion.get('ltc_ode_unfolds', 6)
            self.ltc = ExpClosedFormMotionNetwork(
                input_dim=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_dim=ltc_hidden_dim,
                output_dim=self.cfg.motion.v_dim,
                ode_unfolds=ltc_ode_unfolds,
                lr_multiplier=0.01,
                learn_tau_bias=self.cfg.motion.get('ltc_learn_tau_bias', True),
            )
            self.num_additional_codes = 0
        elif self.gen_strategy == 'cfc':
            # Closed-form Continuous-time network: same continuous-time inductive
            # bias as the exponential cell, but the gate is a sigmoid rather than an
            # exponential decay, and each step is a single update (no ode_unfolds).
            cfc_hidden_dim = self.cfg.motion.get('cfc_hidden_dim', self.cfg.motion.z_dim)
            self.cfc = ClosedFormMotionNetwork(
                input_dim=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_dim=cfc_hidden_dim,
                output_dim=self.cfg.motion.v_dim,
                lr_multiplier=0.01,
                learn_tau_bias=self.cfg.motion.get('cfc_learn_tau_bias', True),
            )
            self.num_additional_codes = 0
        elif self.gen_strategy == 'ncps_ltc':
            # Same continuous-time LTC inductive bias as 'ltc', but using the
            # reference implementation from the `ncps` library instead of our
            # hand-rolled cell.
            # NOTE: ncps' LTC builds [B, units, units] synapse-activation tensors per ODE
            # sub-step and retains them all for backward, so memory grows ~quadratically
            # with the hidden width — hence the small default hidden dim (128). The sparse
            # 'ncp' wiring does NOT reduce this (ncps stores the wiring as a dense mask);
            # for memory, set ncps_ltc_use_checkpoint=true, which recomputes each segment
            # in backward, and raise ncps_ltc_checkpoint_every to trade the saving back
            # for compute. `ode_unfolds` multiplies both cost and memory linearly.
            ncps_ltc_wiring = self.cfg.motion.get('ncps_ltc_wiring', 'full')
            ncps_ltc_default_hidden = self.cfg.motion.z_dim if ncps_ltc_wiring == 'ncp' else min(128, self.cfg.motion.z_dim)
            ncps_ltc_hidden_dim = self.cfg.motion.get('ncps_ltc_hidden_dim', ncps_ltc_default_hidden)
            ncps_ltc_ode_unfolds = self.cfg.motion.get('ncps_ltc_ode_unfolds', 6)
            ncps_ltc_motor_neurons = self.cfg.motion.get('ncps_ltc_motor_neurons', None)
            self.ncps_ltc = NCPSLiquidMotionNetwork(
                input_dim=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_dim=ncps_ltc_hidden_dim,
                output_dim=self.cfg.motion.v_dim,
                ode_unfolds=ncps_ltc_ode_unfolds,
                wiring=ncps_ltc_wiring,
                motor_neurons=ncps_ltc_motor_neurons,
                lr_multiplier=0.01,
                use_checkpoint=self.cfg.motion.get('ncps_ltc_use_checkpoint', False),
                checkpoint_every=self.cfg.motion.get('ncps_ltc_checkpoint_every', 1),
            )
            self.num_additional_codes = 0
        elif self.gen_strategy == 'ncps_cfc':
            # Same continuous-time CfC inductive bias as 'cfc', but using the
            # reference implementation from the `ncps` library instead of our
            # hand-rolled cell.
            # NOTE: backbone_units defaults to None here, which the network resolves to
            # hidden_dim (not ncps' 128) -- see NCPSClosedFormMotionNetwork. The backbone
            # settings are ignored when ncps_cfc_wiring='ncp', which ncps forbids.
            ncps_cfc_hidden_dim = self.cfg.motion.get('ncps_cfc_hidden_dim', self.cfg.motion.z_dim)
            self.ncps_cfc = NCPSClosedFormMotionNetwork(
                input_dim=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_dim=ncps_cfc_hidden_dim,
                output_dim=self.cfg.motion.v_dim,
                lr_multiplier=0.01,
                mode=self.cfg.motion.get('ncps_cfc_mode', 'default'),
                backbone_units=self.cfg.motion.get('ncps_cfc_backbone_units', None),
                backbone_layers=self.cfg.motion.get('ncps_cfc_backbone_layers', 1),
                backbone_dropout=self.cfg.motion.get('ncps_cfc_backbone_dropout', 0.0),
                wiring=self.cfg.motion.get('ncps_cfc_wiring', 'full'),
                motor_neurons=self.cfg.motion.get('ncps_cfc_motor_neurons', None),
            )
            self.num_additional_codes = 0
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.cfg.motion.gen_strategy}')

    def get_max_traj_len(self, t: torch.Tensor) -> int:
        max_t = max(self.cfg.sampling.max_num_frames - 1, t.max().item()) # [1]
        max_traj_len = np.ceil(max_t / self.cfg.motion.motion_z_distance).astype(int).item() + 2 # [1]
        return max_traj_len

    def generate_motion_u_codes(self, c: torch.Tensor, t: torch.Tensor, motion_z: torch.Tensor=None) -> Dict:
        """
        Arguments:
            - c of shape [batch_size, c_dim]
            - t of shape [batch_size, num_frames]
            - w of shape [batch_size, w_dim]
            - motion_z of shape [batch_size, max_traj_len, motion_z_dim] --- in case we want to reuse some existing motion noise
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        max_traj_len = self.get_max_traj_len(t) + self.num_additional_codes # [1]

        if motion_z is None:
            motion_z = torch.randn(batch_size, max_traj_len, self.cfg.motion.z_dim, device=c.device) # [batch_size, max_traj_len, motion.z_dim]

        # Input motion trajectories are just random noise
        input_trajs = motion_z[:batch_size, :max_traj_len, :self.cfg.motion.z_dim].to(c.device) # [batch_size, max_traj_len, motion.z_dim]

        if self.cfg.c_dim > 0:
            # Different classes might have different motions, so it should be useful to condition on c
            misc.assert_shape(c, [batch_size, None])
            input_trajs = torch.cat([input_trajs, c.unsqueeze(1).repeat(1, max_traj_len, 1)], dim=2) # [batch_size, max_traj_len, motion.z_dim + cond_dim]

        if self.gen_strategy == 'autoregressive':
            # Somehow, RNN parameters do not get flattened on their own and we get a lot of warnings...
            if not self._parameters_flattened:
                self.rnn.flatten_parameters()
                self._parameters_flattened = True
            trajs, _ = self.rnn(input_trajs) # [batch_size, max_traj_len, motion.z_dim]
        elif self.gen_strategy == 'conv':
            trajs = self.conv(input_trajs.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, max_traj_len, motion.v_dim]
        elif self.gen_strategy == 'exp_cf':
            # delta_t=1.0: each step is one motion_z_distance unit (normalised)
            trajs = self.ltc(input_trajs, delta_t=1.0) # [batch_size, max_traj_len, motion.v_dim]
        elif self.gen_strategy == 'cfc':
            # delta_t=1.0: each step is one motion_z_distance unit (normalised)
            trajs = self.cfc(input_trajs, delta_t=1.0) # [batch_size, max_traj_len, motion.v_dim]
        elif self.gen_strategy == 'ncps_ltc':
            # delta_t=1.0: each step is one motion_z_distance unit (normalised)
            trajs = self.ncps_ltc(input_trajs, delta_t=1.0) # [batch_size, max_traj_len, motion.v_dim]
        elif self.gen_strategy == 'ncps_cfc':
            # delta_t=1.0: each step is one motion_z_distance unit (normalised)
            trajs = self.ncps_cfc(input_trajs, delta_t=1.0) # [batch_size, max_traj_len, motion.v_dim]
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.cfg.motion.gen_strategy}')

        # Now, we should select neighbouring codes for each frame
        left_idx = (t / self.cfg.motion.motion_z_distance).floor().long() # [batch_size, num_frames]
        batch_idx = torch.arange(batch_size, device=c.device).unsqueeze(1).repeat(1, num_frames) # [batch_size, num_frames]
        motion_u_left = trajs[batch_idx, left_idx] # [batch_size, num_frames, motion.z_dim]
        motion_u_right = trajs[batch_idx, left_idx + 1] # [batch_size, num_frames, motion.z_dim]

        # Compute `u` codes as the interpolation between `u_left` and `u_right`
        t_left = t - t % self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        t_right = t_left + self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        # Compute interpolation weights `alpha` (we'll use them later)
        interp_weights = ((t % self.cfg.motion.motion_z_distance) / self.cfg.motion.motion_z_distance).unsqueeze(2).to(torch.float32) # [batch_size, num_frames, 1]
        motion_u = motion_u_left * (1 - interp_weights) + motion_u_right * interp_weights # [batch_size, num_frames, motion.z_dim]
        motion_u = motion_u.view(batch_size * num_frames, motion_u.shape[2]).to(torch.float32) # [batch_size * num_frames, motion.z_dim]

        # Save the results into the output dict
        out['motion_u_left'] = motion_u_left # [batch_size, num_frames, motion.z_dim]
        out['motion_u_right'] = motion_u_right # [batch_size, num_frames, motion.z_dim]
        out['t_left'] = t_left # [batch_size, num_frames]
        out['t_right'] = t_right # [batch_size, num_frames]
        out['interp_weights'] = interp_weights # [batch_size, num_frames, 1]
        out['motion_u'] = motion_u # [batch_size * num_frames, motion.z_dim]
        out['motion_z'] = motion_z # [batch_size+, max_traj_len+, motion.z_dim+]

        return out

    def get_dim(self) -> int:
        return self.cfg.motion.v_dim if self.time_encoder is None else self.time_encoder.get_dim()

    def forward(self, c: torch.Tensor, t: torch.Tensor, motion_z: Dict=None) -> Dict:
        assert len(c) == len(t), f"Wrong shape: {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        out = {} # We'll be aggregating the result here
        motion_u_info: Dict = self.generate_motion_u_codes(c, t, motion_z=motion_z) # Dict of tensors
        motion_u = motion_u_info['motion_u'].view(t.shape[0] * t.shape[1], -1) # [batch_size * num_frames, motion.z_dim]

        # Compute the `v` motion codes
        if self.cfg.motion.fourier:
            motion_v = self.time_encoder(
                t=t,
                motion_u_left=motion_u_info['motion_u_left'],
                motion_u_right=motion_u_info['motion_u_right'],
                t_left=motion_u_info['t_left'],
                t_right=motion_u_info['t_right'],
                interp_weights=motion_u_info['interp_weights'],
            ) # [batch_size * num_frames, motion_v_dim]
        else:
            motion_v = self.mapping(z=motion_u, c=c.repeat_interleave(t.shape[1], dim=0)) # [batch_size * num_frames, motion.v_dim]

        out['motion_v'] = motion_v # [batch_size * num_frames, motion.v_dim]
        out['motion_z'] = motion_u_info['motion_z'] # (Any shape)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class AlignedTimeEncoder(nn.Module):
    def __init__(self,
        latent_dim: int=512,
        cfg: DictConfig = {},
    ):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        freqs = construct_linspaced_frequencies(self.cfg.time_enc.dim, self.cfg.time_enc.min_period_len, self.cfg.time_enc.max_period_len)
        self.register_buffer('freqs', freqs) # [1, num_fourier_feats]

        # Creating the affine without bias to prevent motion mode collapse
        self.periods_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        self.phase_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        period_lens = 2 * np.pi / self.freqs # [1, num_fourier_feats]
        phase_scales = self.cfg.time_enc.max_period_len / period_lens # [1, num_fourier_feats]
        self.register_buffer('phase_scales', phase_scales)

        self.aligners_predictor = FullyConnectedLayer(latent_dim, self.freqs.shape[1] * 2, activation='linear', bias=False)

    def get_dim(self) -> int:
        return self.freqs.shape[1] * 2

    def forward(self, t: torch.Tensor, motion_u_left: torch.Tensor, motion_u_right: torch.Tensor, interp_weights: torch.Tensor, t_left: torch.Tensor, t_right: torch.Tensor):
        batch_size, num_frames, motion_u_dim = motion_u_left.shape # [1], [1], [1]

        misc.assert_shape(t, [batch_size, num_frames])
        misc.assert_shape(motion_u_left, [batch_size, num_frames, None])
        misc.assert_shape(motion_u_right, [batch_size, num_frames, None])
        misc.assert_shape(interp_weights, [batch_size, num_frames, 1])
        assert t.shape == t_left.shape == t_right.shape, f"Wrong shape: {t.shape} vs {t_left.shape} vs {t_right.shape}"

        motion_u_left = motion_u_left.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        motion_u_right = motion_u_right.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        periods = self.periods_predictor(motion_u_left).tanh() + 1 # [batch_size * num_frames, feat_dim]
        phases = self.phase_predictor(motion_u_left) # [batch_size * num_frames, feat_dim]
        aligners_left = self.aligners_predictor(motion_u_left) # [batch_size * num_frames, out_dim]
        aligners_right = self.aligners_predictor(motion_u_right) # [batch_size * num_frames, out_dim]

        raw_pos_embs = self.freqs * periods * t.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
        raw_pos_embs_left = self.freqs * periods * t_left.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
        raw_pos_embs_right = self.freqs * periods * t_right.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]

        pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1) # [bf, out_dim]
        pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1) # [bf, out_dim]
        pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1) # [bf, out_dim]

        interp_weights = interp_weights.view(-1, 1) # [bf, 1]
        aligners_remove = pos_embs_left * (1 - interp_weights) + pos_embs_right * interp_weights # [bf, out_dim]
        aligners_add = aligners_left * (1 - interp_weights) + aligners_right * interp_weights # [bf, out_dim]
        time_embs = pos_embs - aligners_remove + aligners_add # [bf, out_dim]

        return time_embs

#----------------------------------------------------------------------------

def construct_linspaced_frequencies(num_freqs: int, min_period_len: int, max_period_len: int) -> torch.Tensor:
    freqs = 2 * np.pi / (2 ** np.linspace(np.log2(min_period_len), np.log2(max_period_len), num_freqs)) # [num_freqs]
    freqs = torch.from_numpy(freqs[::-1].copy().astype(np.float32)).unsqueeze(0) # [1, num_freqs]

    return freqs

#----------------------------------------------------------------------------
