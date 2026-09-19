import sys; sys.path.extend(['.', 'src'])

import pytest
import torch
import torch.nn as nn

from src.training.motion import (
    ExpClosedFormCell,
    ExpClosedFormMotionNetwork,
    CfCCell,
    ClosedFormMotionNetwork,
    NCPSLiquidMotionNetwork,
    NCPSClosedFormMotionNetwork,
    LTCCell,
    LiquidMotionNetwork,
    _NCPS_AVAILABLE,
)
from src.training.layers import FullyConnectedLayer
from src.training.training_loop import construct_optimizer_params

requires_ncps = pytest.mark.skipif(not _NCPS_AVAILABLE, reason="the `ncps` package is not installed")

B, T, IN, HID, OUT = 2, 8, 16, 32, 24


def _hand_rolled():
    return {
        'exp_cf': ExpClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT),
        'cfc': ClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT),
    }


def _ncps():
    return {
        'ncps_ltc': NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT),
        'ncps_cfc': NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT),
    }


def _assert_forward_backward(net):
    """[B, T, input_dim] -> [B, T, output_dim], finite forward and backward."""
    x = torch.randn(B, T, IN, requires_grad=True)
    y = net(x)
    assert y.shape == (B, T, OUT)
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(x.grad).all()


#----------------------------------------------------------------------------
# Shapes and numerics

@pytest.mark.parametrize('name', ['exp_cf', 'cfc'])
def test_hand_rolled_shapes(name):
    _assert_forward_backward(_hand_rolled()[name])


@requires_ncps
@pytest.mark.parametrize('name', ['ncps_ltc', 'ncps_cfc'])
def test_ncps_shapes(name):
    _assert_forward_backward(_ncps()[name])


#----------------------------------------------------------------------------
# Task 1 -- naming and citations

def test_old_names_remain_importable_as_aliases():
    assert LTCCell is ExpClosedFormCell
    assert LiquidMotionNetwork is ExpClosedFormMotionNetwork


def test_docstrings_do_not_overstate():
    # @persistence.persistent_class subclasses the original, so __doc__ lands one
    # step up the MRO.
    def doc(cls):
        return cls.__doc__ or cls.__mro__[1].__doc__

    cfc, exp = doc(CfCCell), doc(ExpClosedFormCell)
    assert 'Nature Machine Intelligence 2022' in cfc
    assert 'AAAI 2022' not in cfc and 'AAAI 2022' not in exp
    # The CfC update is a closed-form step, not an exact solution of the liquid ODE.
    assert 'exact in' not in cfc
    # The hand-rolled cell must not claim to be the solver-based LTC model.
    assert 'identical to neither' in exp


#----------------------------------------------------------------------------
# Task 2 -- ncps CfC configuration

@requires_ncps
def test_ncps_cfc_ncp_wiring_narrows_output_and_shrinks_params():
    full = NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, wiring='full')
    ncp = NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, wiring='ncp')

    assert ncp.motor_neurons == max(1, HID // 4)
    assert ncp.proj.weight.shape[1] == ncp.motor_neurons   # rnn_out_dim == motor_neurons
    assert ncp(torch.randn(B, T, IN)).shape == (B, T, OUT)
    assert sum(p.numel() for p in ncp.parameters()) < sum(p.numel() for p in full.parameters())


@requires_ncps
@pytest.mark.parametrize('mode', ['default', 'pure', 'no_gate'])
def test_ncps_cfc_modes(mode):
    net = NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, mode=mode)
    assert net(torch.randn(B, T, IN)).shape == (B, T, OUT)


@requires_ncps
@pytest.mark.parametrize('kwargs', [{'mode': 'bogus'}, {'wiring': 'bogus'}])
def test_ncps_cfc_rejects_unknown_options(kwargs):
    with pytest.raises(AssertionError):
        NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, **kwargs)


@requires_ncps
def test_ncps_cfc_accepts_non_unit_delta_t():
    """ncps' CfC takes a `timespans` tensor, so the delta_t == 1.0 assert is gone."""
    net = NCPSClosedFormMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT)
    assert net(torch.randn(B, T, IN), delta_t=0.5).shape == (B, T, OUT)


@requires_ncps
def test_ncps_ltc_still_requires_unit_delta_t():
    """Unlike CfC, `timespans` breaks ncps' LTC `cm / elapsed_time` broadcast."""
    net = NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT)
    with pytest.raises(AssertionError):
        net(torch.randn(B, T, IN), delta_t=0.5)


#----------------------------------------------------------------------------
# Task 3 -- learning-rate parity

@requires_ncps
@pytest.mark.parametrize('name', ['ncps_ltc', 'ncps_cfc'])
def test_recurrent_parameters_disjoint_from_projection(name):
    net = _ncps()[name]
    recurrent = {id(p) for p in net.recurrent_parameters()}
    assert len(recurrent) > 0
    assert recurrent.isdisjoint({id(p) for p in net.proj.parameters()})


@requires_ncps
@pytest.mark.parametrize('name', ['ncps_ltc', 'ncps_cfc'])
def test_optimizer_param_groups_partition_the_module(name):
    lr, lr_multiplier = 0.002, 0.01

    class FakeG(nn.Module):
        def __init__(self, motion):
            super().__init__()
            self.mapping = FullyConnectedLayer(HID, HID, lr_multiplier=lr_multiplier)
            self.motion = motion

    G = FakeG(_ncps()[name])
    groups = construct_optimizer_params(G, lr)
    assert len(groups) == 2

    ids = [id(p) for g in groups for p in g['params']]
    all_ids = [id(p) for p in G.parameters()]
    assert len(ids) == len(set(ids)), "param groups overlap"
    assert set(ids) == set(all_ids) and len(ids) == len(all_ids), "param groups do not partition G"

    assert 'lr' not in groups[0]                       # inherits the optimizer default
    assert groups[1]['lr'] == pytest.approx(lr * lr_multiplier)

    opt = torch.optim.Adam(groups, lr=lr, betas=(0.0, 0.99), eps=1e-8)
    assert opt.param_groups[0]['lr'] == pytest.approx(lr)
    assert opt.param_groups[1]['lr'] == pytest.approx(lr * lr_multiplier)
    assert all(g['betas'] == (0.0, 0.99) and g['eps'] == 1e-8 for g in opt.param_groups)


def test_optimizer_params_unchanged_without_ncps_cores():
    """A generator with no ncps arm must keep the original behaviour."""
    class PlainG(nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = FullyConnectedLayer(HID, HID, lr_multiplier=0.01)

    assert not isinstance(construct_optimizer_params(PlainG(), 0.002), list)


def test_optimizer_params_preserves_params_with_lr_hook():
    """The MoCoGAN discriminator's existing hook must keep priority."""
    class HookedD(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = FullyConnectedLayer(HID, HID)
            self.b = FullyConnectedLayer(HID, HID)

        def params_with_lr(self, lr):
            return [{'params': self.a.parameters()},
                    {'params': self.b.parameters(), 'lr': 0.5 * lr}]

    groups = construct_optimizer_params(HookedD(), 0.002)
    assert len(groups) == 2 and groups[1]['lr'] == pytest.approx(0.001)


#----------------------------------------------------------------------------
# Task 4 -- log_tau_0 redundancy

@pytest.mark.parametrize('factory', [ExpClosedFormMotionNetwork, ClosedFormMotionNetwork])
def test_learn_tau_bias_toggles_the_redundant_parameter(factory):
    on = factory(input_dim=IN, hidden_dim=HID, output_dim=OUT)
    off = factory(input_dim=IN, hidden_dim=HID, output_dim=OUT, learn_tau_bias=False)

    assert on.cell.log_tau_0 is not None, "default must reproduce the previous behaviour"
    assert off.cell.log_tau_0 is None
    # log_tau_0 is one scalar per channel and nothing else changes.
    assert sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters()) == HID

    _assert_forward_backward(off)


#----------------------------------------------------------------------------
# Task 5 -- ncps LTC memory

@requires_ncps
@pytest.mark.parametrize('wiring', ['full', 'ncp'])
@pytest.mark.parametrize('checkpoint_every', [1, 3, T])
def test_checkpointing_matches_the_unchecked_path(wiring, checkpoint_every):
    """use_checkpoint must change memory, never results."""
    def run(use_checkpoint):
        torch.manual_seed(0)
        net = NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, wiring=wiring,
                                      use_checkpoint=use_checkpoint, checkpoint_every=checkpoint_every)
        torch.manual_seed(1)
        x = torch.randn(B, T, IN, requires_grad=True)
        y = net(x)
        y.square().mean().backward()
        # Some ncps parameters can be genuinely unused for a given wiring/version, so
        # grads are keyed by name rather than zipped positionally: *which* parameters
        # receive a gradient is itself part of what must not change.
        return y, {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}, x.grad.clone()

    y_ref, g_ref, xg_ref = run(False)
    y_ckpt, g_ckpt, xg_ckpt = run(True)

    assert torch.allclose(y_ref, y_ckpt, atol=1e-5)
    assert g_ref.keys() == g_ckpt.keys(), (
        f"checkpointing changed which parameters get gradients: "
        f"only-plain={sorted(g_ref.keys() - g_ckpt.keys())}, only-ckpt={sorted(g_ckpt.keys() - g_ref.keys())}")
    assert all(torch.allclose(g_ref[n], g_ckpt[n], atol=1e-5) for n in g_ref)
    assert torch.allclose(xg_ref, xg_ckpt, atol=1e-5)


@requires_ncps
@pytest.mark.parametrize('checkpoint_every', [1, 3, T])
def test_checkpointing_trains_on_a_non_grad_input(checkpoint_every):
    """The real caller feeds sampled noise, which does NOT require grad.

    `generate_motion_u_codes` builds the trajectory with torch.randn and the initial
    state is zeros, so neither checkpointed input requires grad. Under PyTorch's
    legacy reentrant checkpointing that leaves every segment -- and hence the whole
    ncps core -- with no gradient at all, behind a mere UserWarning. The earlier
    equivalence test passes `requires_grad=True` and cannot catch this.
    """
    def trained_core_params(use_checkpoint):
        torch.manual_seed(0)
        net = NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT,
                                      use_checkpoint=use_checkpoint, checkpoint_every=checkpoint_every)
        torch.manual_seed(1)
        x = torch.randn(B, T, IN)                  # no requires_grad, as in training
        assert not x.requires_grad
        net(x).square().mean().backward()
        core = {id(p) for p in net.recurrent_parameters()}
        return {n for n, p in net.named_parameters()
                if id(p) in core and p.grad is not None and p.grad.abs().sum() > 0}

    plain = trained_core_params(False)
    assert plain, "expected the plain path to train some ncps core parameters"
    assert trained_core_params(True) == plain, "checkpointed segments produced no gradient"


@requires_ncps
@pytest.mark.parametrize('freeze', [True, False])
def test_checkpointing_leaves_inference_output_detached(freeze):
    """A frozen copy (G_ema) must not come back requiring grad.

    training_loop exports sample images with `G_ema(...).cpu().numpy()` outside any
    no_grad block, relying on the frozen parameters to keep the output detached. The
    legacy checkpointing path adds a grad-requiring anchor input, so it has to stay
    switched off whenever no backward will follow.
    """
    net = NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT,
                                  use_checkpoint=True, checkpoint_every=1)
    if freeze:
        net.requires_grad_(False)                  # as training_loop does for G_ema
        assert not net(torch.randn(B, T, IN)).requires_grad
    else:
        with torch.no_grad():
            assert not net(torch.randn(B, T, IN)).requires_grad


@requires_ncps
def test_checkpoint_every_must_be_positive():
    with pytest.raises(AssertionError):
        NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, checkpoint_every=0)


@requires_ncps
def test_ode_unfolds_is_configurable():
    """Cost is linear in ode_unfolds, so it must not be hardcoded."""
    net = NCPSLiquidMotionNetwork(input_dim=IN, hidden_dim=HID, output_dim=OUT, ode_unfolds=3)
    # `init_kwargs` comes from @persistence.persistent_class, which records the
    # constructor arguments it will replay on reconstruction.
    assert net.init_kwargs['ode_unfolds'] == 3
    assert net(torch.randn(B, T, IN)).shape == (B, T, OUT)


@requires_ncps
def test_autoncp_wiring_is_deterministic_under_a_fixed_seed():
    """persistence replays __init__ args, so connectivity must rebuild identically."""
    from ncps.wirings import AutoNCP

    a, b = AutoNCP(HID, HID // 4, seed=22222), AutoNCP(HID, HID // 4, seed=22222)
    for w in (a, b):
        w.build(IN)
    assert torch.equal(torch.as_tensor(a.adjacency_matrix), torch.as_tensor(b.adjacency_matrix))
    assert torch.equal(torch.as_tensor(a.sensory_adjacency_matrix),
                       torch.as_tensor(b.sensory_adjacency_matrix))

