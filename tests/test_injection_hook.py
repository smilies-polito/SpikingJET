"""Every registered network whose checkpoint is present must (1) run on a random batch shaped by
its `example` (networks.py) and return [batch, classes] scores, and (2) if it is an SNN that calls
FaultInjector.injection, pass (spike, potential, layer, t) in this order: a spike fault must hit
the spike tensor, a 'potential' fault the membrane (README, 'Neuron faults').
A network added to networks.py is checked automatically. No dataset is needed."""
import os

import pytest
import torch

from networks import NETWORKS, checkpoint_path, load_network, synthetic_batch

AVAILABLE = [n for n, s in NETWORKS.items() if s.example is not None and os.path.exists(checkpoint_path(n))]


class Recorder:
    def __init__(self):
        self.calls = []

    def injection(self, spike, pot, layer, t):
        self.calls.append((layer, spike.detach().clone(), pot.detach().clone()))


@pytest.mark.parametrize("name", AVAILABLE)
def test_network_runs_and_hook_order(name):
    net = load_network(name, torch.device("cpu"))
    rec = Recorder()
    net.FaultInjector = rec
    x, _ = synthetic_batch(name, 2)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, NETWORKS[name].example.classes), f"{name}: output {tuple(out.shape)}"
    if not rec.calls:                      # CNN or SNN without neuron faults: weight faults only
        return
    pot_values = set()
    for layer, spike, pot in rec.calls:
        assert set(spike.unique().tolist()) <= {0.0, 1.0}, \
            f"{name}.{layer}: first hook argument is not a spike tensor (arguments swapped?)"
        pot_values |= set(pot.unique().tolist())
    assert not pot_values <= {0.0, 1.0}, f"{name}: second hook argument looks like spikes, not potentials"


def test_every_network_has_an_example():
    missing = [n for n, s in NETWORKS.items() if s.example is None]
    assert not missing, f"add Example(shape, classes) to these entries of networks.py: {missing}"
