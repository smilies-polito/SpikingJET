"""The spikingjet engine: bit operations, fault lists, injection, faults during training, config."""
import numpy as np
import pytest
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjet.bitops import apply_faults, float32_bit_flip, float32_stuck_at, resolve_bits
from spikingjet.config import load_config
from spikingjet.faults import ParamBER, read_fault_list, single_fault_list
from spikingjet.injector import Fault, Injector
from spikingjet.sites import discover
from spikingjet.training import TrainingFaults


class Mixed(nn.Module):
    """Tiny mixed network: conv + ReLU (non-spiking) -> LIF (snntorch, no hook code) -> linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.act = nn.ReLU()
        self.fc = nn.Linear(4 * 6 * 6, 16)
        self.lif = snn.Leaky(beta=0.8)
        self.out = nn.Linear(16, 3)

    def forward(self, x):                       # [B, T, 1, 6, 6]
        mem = self.lif.init_leaky()
        acc = 0
        for t in range(x.shape[1]):
            h = self.act(self.conv(x[:, t])).flatten(1)
            spk, mem = self.lif(self.fc(h), mem)
            acc = acc + self.out(spk + mem)
        return acc / x.shape[1]


def _data(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, 1, 6, 6, generator=g), torch.randint(0, 3, (n,), generator=g)


# ---------------------------------------------------------------- bit operations
def test_bitops_match_scalar_reference():
    rng = np.random.default_rng(0)
    vals = torch.tensor(rng.normal(size=200).astype(np.float32))
    idx = np.arange(200)
    bits = rng.integers(0, 32, 200)
    for mode in (-1, 0, 1):
        t = vals.clone()
        apply_faults(t, idx, bits, np.full(200, mode))
        ref = [float32_bit_flip(float(v), int(b)) if mode < 0 else float32_stuck_at(float(v), int(b), mode)
               for v, b in zip(vals, bits)]
        assert torch.equal(t.view(torch.int32), torch.tensor(ref, dtype=torch.float32).view(torch.int32))


def test_resolve_bits():
    assert resolve_bits("exponent") == tuple(range(23, 31))
    assert resolve_bits([0, 3]) == (0, 1, 2, 3)
    assert resolve_bits([31, 1, 5]) == (1, 5, 31)
    with pytest.raises(ValueError):
        resolve_bits("mantisa")


# ---------------------------------------------------------------- fault lists
def test_single_fault_list_is_unchanged():
    """Same seed -> the same list as the original SpikingJET generator (tests/data, recorded from it)."""
    from networks import load_network, synthetic_batch
    net = load_network("NMNIST", torch.device("cpu"))
    s = discover(net, synthetic_batch("NMNIST", 2)[0], with_activations=False)
    new = single_fault_list(s.params + s.neurons, 51196)
    ref = read_fault_list("tests/data/NMNIST_51196_fault_list.csv")
    assert [(f.site, f.index, f.bit, f.value) for f in new] == [(f.site, f.index, f.bit, f.value) for f in ref]


def test_param_ber_is_paired():
    a, b = Mixed(), Mixed()
    sa, sb = discover(a, _data()[0]), discover(b, _data()[0])
    ma = ParamBER(sa.params).sample(1e-3, "all", ParamBER.rng_for(0, "all", 1e-3, 2, False))
    mb = ParamBER(sb.params).sample(1e-3, "all", ParamBER.rng_for(0, "all", 1e-3, 2, False))
    assert [(s.name, i.tolist(), bt.tolist()) for s, i, bt, _ in ma] == [(s.name, i.tolist(), bt.tolist()) for s, i, bt, _ in mb]


# ---------------------------------------------------------------- injection
def test_sites_of_mixed_network():
    s = discover(Mixed(), _data()[0])
    assert [x.name for x in s.neurons] == ["lif.potential", "lif.spike"] and not s.explicit_hook
    assert {"conv.output", "act.output", "fc.output", "out.output"} <= {x.name for x in s.activations}


def test_param_and_state_faults_restore_exactly():
    torch.manual_seed(0)
    net, (x, _) = Mixed().eval(), _data()
    s = discover(net, x)
    inj = Injector(net, s)
    with torch.no_grad():
        clean = net(x)
        before = {n: p.clone() for n, p in net.named_parameters()}
        inj.set_param_faults([Fault("fc.weight", (3, 7), 30, -1)])
        assert not torch.equal(net(x), clean)
        inj.restore_params()
        assert all(torch.equal(p, before[n]) for n, p in net.named_parameters())
        for site in ("lif.potential", "act.output"):
            inj.set_state_faults([Fault(site, tuple(0 for _ in s.get(site).shape), 30, -1)])
            with inj.enabled():
                assert not torch.equal(net(x), clean), site
            inj.clear_states()
        assert torch.equal(net(x), clean)
    inj.detach()
    assert torch.equal(net(x), clean)


def test_transient_neuron_fault_hits_one_step():
    net, (x, _) = Mixed().eval(), _data()
    s = discover(net, x)
    inj = Injector(net, s)
    with torch.no_grad():
        clean = net(x)
        outs = []
        for step in (0, 3):
            inj.set_state_faults([Fault("lif.potential", (2,), 30, -1, step)])
            with inj.enabled():
                outs.append(net(x))
        assert not torch.equal(outs[0], clean) and not torch.equal(outs[0], outs[1])
    inj.detach()


# ---------------------------------------------------------------- faults during training
def _train(groups, steps=3, opt_cls=torch.optim.Adam):
    torch.manual_seed(0)
    net = Mixed()
    opt = opt_cls(net.parameters(), lr=1e-2)
    x, y = _data()
    tf = TrainingFaults(groups, net, opt, x, steps_per_epoch=steps, seed=1)
    snaps = []
    for _ in range(steps):
        tf.before_forward(0)
        w_fwd = net.fc.weight.detach().clone()
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        tf.after_backward()
        w_bwd = net.fc.weight.detach().clone()
        g = net.fc.weight.grad.clone()
        opt.step()
        tf.after_step()
        snaps.append((w_fwd, w_bwd, g, net.fc.weight.detach().clone()))
    tf.close()
    return net, opt, tf, snaps


def test_training_transient_params_are_removed_before_the_update():
    net, _, tf, snaps = _train([{"target": "params", "select": ["fc.weight"], "ber": 0.05, "bits": "mantissa"}])
    assert tf.summary()["group0_params_transient"] > 0
    for w_fwd, w_bwd, _, _ in snaps:
        assert not torch.equal(w_fwd, w_bwd)              # faulty during forward, clean for the update


def test_training_permanent_stuck_bits_hold_after_every_update():
    net, _, tf, snaps = _train([{"target": "params", "select": ["fc.weight"], "ber": 0.01,
                                 "fault_model": "stuck-at", "persistence": "permanent"}])
    G = tf.groups[0]
    site, idx, bits, vals = G.fixed[0]
    ints = net.fc.weight.detach().view(-1).view(torch.int32)[torch.as_tensor(idx)]
    held = ((ints >> torch.as_tensor(bits, dtype=torch.int32)) & 1).numpy()
    assert (held == vals).all()


def test_training_upset_grads_optimizer_neurons_activations():
    groups = [{"target": "params", "ber": 1e-3, "persistence": "upset", "bits": "mantissa"},
              {"target": "grads", "ber": 1e-2, "bits": "mantissa"},
              {"target": "optimizer", "ber": 1e-2, "persistence": "upset", "bits": "mantissa"},
              {"target": "neurons", "ber": 1e-2, "bits": "mantissa"},
              {"target": "activations", "select": ["act.output"], "ber": 1e-2, "bits": "mantissa"}]
    net, opt, tf, _ = _train(groups)
    counts = tf.summary()
    assert all(v > 0 for v in counts.values()), counts
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_training_without_faults_is_unchanged():
    a, _, _, _ = _train([])
    torch.manual_seed(0)
    b = Mixed()
    opt = torch.optim.Adam(b.parameters(), lr=1e-2)
    x, y = _data()
    for _ in range(3):
        loss = F.cross_entropy(b(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


# ---------------------------------------------------------------- configuration
def test_config_overrides_and_validation(tmp_path):
    f = tmp_path / "e.toml"
    f.write_text('network = "SHD_DELAY"\n[[train.faults]]\ntarget = "grads"\nber = 1e-6\n')
    cfg = load_config(str(f), ["train.faults[0].ber=1e-4", "campaign.bits=[0, 22]", "batch_size=8"])
    assert cfg["train"]["faults"][0]["ber"] == 1e-4 and cfg["campaign"]["bits"] == [0, 22] and cfg["batch_size"] == 8
    assert cfg["train"]["faults"][0]["persistence"] == "transient"           # group defaults filled in
    with pytest.raises(KeyError, match="did you mean 'max_faults'"):
        load_config(None, ["campaign.max_fault=3"])
    f.write_text('[[train.faults]]\ntarget = "weights"\n')
    with pytest.raises(ValueError, match="target"):
        load_config(str(f))
