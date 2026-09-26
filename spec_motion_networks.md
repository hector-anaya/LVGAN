# Spec: motion-network ablation cleanup

## Context

The StyleGAN-V-based generator has four interchangeable motion networks selected by
`gen_strategy`. They are intended to form a controlled ablation over the temporal
backbone:

| arm | class | core |
|---|---|---|
| `ltc` | `LiquidMotionNetwork` | hand-rolled, exponential closed-form gate |
| `ncps_ltc` | `NCPSLiquidMotionNetwork` | `ncps` LTC — solver-based, semi-implicit Euler |
| `cfc` | `ClosedFormMotionNetwork` | hand-rolled, sigmoid closed-form gate |
| `ncps_cfc` | `NCPSClosedFormMotionNetwork` | `ncps` CfC — reference closed-form |

Right now the arms differ in learning rate, backbone width, and available wiring
options as well as in the cell math, so the comparison measures several things at
once. This spec closes those gaps so the only variable is the recurrent core.

**Before implementing anything, verify the `ncps` API against the installed
version.** Signatures and defaults have moved across releases and the details
below are from reading, not from the installed package:

```bash
python -c "import ncps, inspect; print(ncps.__version__ if hasattr(ncps,'__version__') else 'unknown')"
python -c "import ncps.torch as t, inspect; print(inspect.signature(t.CfC.__init__))"
python -c "import ncps.torch as t, inspect; print(inspect.signature(t.LTC.__init__))"
python -c "import ncps.torch.ltc_cell as m; print(m.__file__)"
```

If any keyword named below does not exist, stop and report rather than guessing a
substitute.

---

## Task 1 — Citation and naming corrections

1. `CfCCell` docstring cites the CfC paper. Confirm it reads **Nature Machine
   Intelligence 2022**, not AAAI 2022. Fix anywhere else in the repo that cites it
   as AAAI (grep for `AAAI 2022` and `Closed-form`).

2. `CfCCell` docstring claims the update is "exact in Δt rather than a first-order
   approximation." This overstates the paper: the sigmoid gate *replaces* the
   exponential from the analytic solution for numerical stability, and the
   derivation makes simplifying assumptions. Reword to something defensible, e.g.
   "a single closed-form update per step, avoiding the inner ODE loop and its
   accumulated discretisation error."

3. `LTCCell` (the hand-rolled one, with `W_tau` / `W_f` / `log_A`) cites Hasani
   2021 alone, but its update is `x·exp(-Δt/τ) + f·(1-exp(-Δt/τ))` — closed-form,
   not the solver-based conductance model of that paper. Rename the class to
   `ExpClosedFormCell` and `LiquidMotionNetwork` to `ExpClosedFormMotionNetwork`,
   and rewrite the docstring to state it is a closed-form cell with input-dependent
   time constants, inspired by LTC (Hasani 2021) and adopting the closed-form
   strategy of CfC (Hasani 2022), but identical to neither.

   Keep `gen_strategy='ltc'` working as an alias so existing configs and
   checkpoints do not break; add the new name as the canonical one.

---

## Task 2 — `NCPSClosedFormMotionNetwork`

Bring it to the paper's intended configuration and to parity with the LTC wrapper.

- Add `mode: str = 'default'` and pass it through. Assert it is one of
  `('default', 'pure', 'no_gate')`. Document that `'default'` is the gated
  closed-form update and the other two are ablations.
- Add `backbone_units: int = None`, `backbone_layers: int = 1`,
  `backbone_dropout: float = 0.0`. When `backbone_units` is `None`, pass
  `hidden_dim` — **not** ncps' default of 128. The library default bottlenecks the
  g/h/time heads and is a capacity difference, not an implementation difference,
  which would confound the comparison against `ClosedFormMotionNetwork`.
- Add a `wiring: str = 'full' | 'ncp'` branch mirroring
  `NCPSLiquidMotionNetwork`, with `motor_neurons` defaulting to
  `max(1, hidden_dim // 4)` and a fixed `seed` for reproducible connectivity under
  `persistence` replay. In `'ncp'` mode `rnn_out_dim = motor_neurons`.
  **Check first** whether the installed ncps forbids `backbone_*` arguments
  alongside a `Wiring` — if it does, do not pass them in that branch.
- Relax the `delta_t == 1.0` assert. Unlike LTC, ncps' CfC accepts a per-sample
  `timespans` tensor. Keep the fast path (call with no `timespans`) when
  `delta_t == 1.0` to avoid allocating a constant tensor; otherwise build
  `trajectory.new_full((B, T, 1), float(delta_t))` and pass it.
  Leave the assert in `NCPSLiquidMotionNetwork` — there it is real, because
  `timespans` breaks the `cm / elapsed_time` broadcast.

---

## Task 3 — Learning-rate parity (highest priority)

Every `FullyConnectedLayer` in this codebase applies equalized-LR scaling of
`lr_multiplier=0.01`. The `ncps` cores use plain `nn.Parameter`s at the full
optimizer LR. So two arms train ~100× faster than the other two, and any FVD
difference between them is currently uninterpretable.

1. Add to both `NCPSLiquidMotionNetwork` and `NCPSClosedFormMotionNetwork`:

   ```python
   def recurrent_parameters(self):
       """Parameters of the ncps core, for a separate optimizer param group."""
       return self.rnn.parameters()
   ```

2. In whichever module builds the generator optimizer, split into two param
   groups so the ncps core gets `base_lr * lr_multiplier`:

   ```python
   ncps_params, ncps_ids = [], set()
   for m in G.modules():
       if isinstance(m, (NCPSClosedFormMotionNetwork, NCPSLiquidMotionNetwork)):
           for p in m.recurrent_parameters():
               ncps_ids.add(id(p)); ncps_params.append(p)
   other_params = [p for p in G.parameters() if id(p) not in ncps_ids]
   ```

   Preserve the existing betas/eps and any `reg_interval` LR rescaling that
   StyleGAN-V's lazy regularisation applies — apply it to both groups.

3. Note in a comment that this makes the *effective* LRs comparable but not
   identical: equalized LR also rescales the weight init, which the ncps
   parameters do not get.

---

## Task 4 — `log_tau_0` redundancy

Both `CfCCell` and the hand-rolled LTC cell add a per-channel `nn.Parameter`
(`log_tau_0`) to the output of a `FullyConnectedLayer` that already has a bias.
Mathematically redundant — but because it is a raw parameter, it trains at 100×
the LR of that bias. Effectively one fast-learning scalar per channel inside an
otherwise slow-learning cell.

Do **not** silently delete it. Instead:
- Add a comment stating the redundancy and the LR asymmetry explicitly.
- Add an `__init__` flag `learn_tau_bias: bool = True` so it can be switched off
  for an ablation.

---

## Task 5 — `NCPSLiquidMotionNetwork` memory

The `ncps` LTC stores `[B, units, units]` synapse activations per ODE sub-step,
all retained for backward. At `B=32, units=512, T=16, ode_unfolds=6` this is on
the order of several GB for the motion network alone. Note that `AutoNCP` does
**not** help: ncps stores the wiring as a dense adjacency mask, so masked entries
are zeros rather than absent, and the intermediates are the same size.

1. Add `use_checkpoint: bool = False`. When set, drive the cell manually over the
   time axis with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
   instead of calling `self.rnn(trajectory)`. Verify the cell's return signature
   (`rnn_cell.forward` returns `(outputs, next_state)` in current ncps, but this
   has changed) and the correct attribute path to the cell before writing this.
   Add a `checkpoint_every: int = 1` so chunks of N steps can share one segment —
   `checkpoint_every=4` recovers most of the compute for ~4× savings.
2. Expose `ode_unfolds` in the config (it is already an `__init__` arg — make sure
   it is plumbed through from the training config, not hardcoded). Cost is linear
   in it; 6 is the library default, not a tuned value.
3. Check `implicit_param_constraints` on the installed version. If it defaults to
   `False`, the cell expects `apply_weight_constraints()` after every optimizer
   step to keep `cm`, `gleak`, `w` positive. Nothing currently does that. Either
   pass `True` explicitly or add the post-step call.
4. Add a comment that this arm must stay fp32 — the semi-implicit update divides
   by `cm/Δt + gleak + w_act`, which is unbounded below, and this is the same
   instability already observed in the TensorFlow track. The three closed-form
   arms have bounded gates and are safe candidates for reduced precision.

---

## Task 6 — Verification

Add `tests/test_motion_networks.py` (pytest, skip if `ncps` is unavailable):

- Each of the four networks: `[B, T, input_dim]` in → `[B, T, output_dim]` out,
  for `B=2, T=8`. No NaN/Inf, forward and backward.
- `NCPSClosedFormMotionNetwork` with `wiring='ncp'` produces the right output
  width and its parameter count is below the `'full'` variant.
- `recurrent_parameters()` returns a non-empty set that is disjoint from
  `self.proj.parameters()`.
- Optimizer construction: the two param groups partition `G.parameters()` with no
  overlap and no omissions (assert the id-set sizes sum correctly).
- `use_checkpoint=True` and `use_checkpoint=False` give identical output and
  identical gradients on the same seed (`torch.allclose`, atol 1e-5).
- Determinism of `AutoNCP` under a fixed seed: build twice, compare adjacency.

Run a smoke train of ~50 steps per arm at reduced `hidden_dim` and confirm the
loss is finite and the arms are within an order of magnitude of each other.

---

## Constraints

- **Do not change** the generator, discriminator, dataset pipeline, or loss
  functions. Scope is the motion networks and the optimizer param-group split.
- `@persistence.persistent_class` replays `__init__` args on reconstruction. New
  keyword arguments are fine for new pickles, but **checkpoints saved under the
  old signatures will fail to rebuild**. Note this in the PR description and do
  not attempt to migrate old checkpoints.
- Keep every new argument keyword-only with a default that reproduces current
  behaviour, except where this spec says otherwise (backbone width, LR groups).
- Report anything where the installed `ncps` disagrees with this spec instead of
  working around it.
