#!/usr/bin/env python3
"""
Check a registered network end to end through SpikingJET's own code paths, before a campaign.

    python3 verify_snn.py --network SHD_DELAY          (any name of networks.py)

1. loads the network and its test loader through networks.py (the paths main.py uses)
2. measures clean accuracy in exact FP32 (like main.py / ber_eval.py) and compares it with the
   test accuracy stored in the checkpoint, when there is one
3. finds the fault sites (spikingjet.sites) and injects one fault into a parameter, the
   membrane potential and the spikes of the first spiking layer and the first activation, and
   checks that each changes the output and that the clean output comes back
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def stored_accuracy(path):
    """test_acc saved by train_snn.py / train_shd_delays.py / train_cnn.py, or None."""
    if not path or not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("test_acc") if isinstance(ckpt, dict) else None


def main():
    from networks import NETWORKS, checkpoint_path, get_loader, load_network
    from numerics import set_tf32

    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True, choices=list(NETWORKS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--forbid-cuda", action="store_true")
    a = p.parse_args()

    from spikingjet.injector import Fault, Injector
    from spikingjet.sites import discover

    set_tf32(False, deterministic=False)                   # exact FP32, as in the campaigns
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.forbid_cuda else "cpu")
    net = load_network(a.network, dev)
    loader = get_loader(a.network, "test", a.batch_size)
    stored = stored_accuracy(checkpoint_path(a.network))

    correct = total = 0
    first = None
    with torch.no_grad():
        for x, y in loader:
            if first is None:
                first = x
            out = net(x.to(dev))
            correct += (out.argmax(1).cpu() == y).sum().item()
            total += y.numel()
    acc = 100 * correct / total
    if stored is None:
        verdict = "(no accuracy stored in the checkpoint)"
    else:
        verdict = f"(checkpoint stored {stored:.2f}%)  " + ("OK" if abs(acc - stored) < 0.2 else
                                                           "MISMATCH - check preprocessing / checkpoint")
    print(f"clean accuracy {acc:.2f}% on {total} test samples  {verdict}")

    x = first.to(dev)
    sites = discover(net, x)
    n_param = sum(s.numel for s in sites.params)
    n_neuron = sum(s.numel for s in sites.neurons)
    n_act = sum(s.numel for s in sites.activations)
    tot = 32 * (n_param + n_neuron)
    print(f"fault sites: {32 * n_param:,} parameter bits, {32 * n_neuron:,} neuron-state bits per sample"
          + (f" ({100 * 32 * n_neuron / tot:.2f}% of a params+neurons sample)" if n_neuron else "")
          + f", {32 * n_act:,} activation bits per sample  (python spikingjet.py sites -n {a.network})")
    checks = []
    if sites.neurons:
        how = "explicit FaultInjector hook" if sites.explicit_hook else "automatic hooks on snntorch neurons"
        print(f"neuron layers ({how}):", {s.name: s.shape for s in sites.neurons})
        layer = sites.neurons[0].layer
        checks = [s for s in sites.neurons if s.layer == layer]
    else:
        print("no spiking neurons found: weight and activation faults only")
    if sites.activations:
        checks.append(sites.activations[0])

    inj = Injector(net, sites)
    ok = True
    try:
        with torch.no_grad():
            clean = net(x)
            for s in checks:
                idx = tuple(0 for _ in s.shape)
                inj.set_state_faults([Fault(s.name, idx, 30, -1)])
                with inj.enabled():
                    diff = (net(x) - clean).abs().max().item()
                inj.clear_states()
                ok &= diff > 0
                print(f"injected {s.name}{list(idx)} bit 30 -> max output change {diff:.3g}")
            p = sites.params[0]
            inj.set_param_faults([Fault(p.name, tuple(0 for _ in p.shape), 30, -1)])
            diff = (net(x) - clean).abs().max().item()
            inj.restore_params()
            print(f"injected {p.name}[0] bit 30 -> max output change {diff:.3g}")
            ok &= diff > 0
            restored = torch.equal(net(x), clean)
            print("golden output restored:", restored)
    finally:
        inj.detach()
    if not (ok and restored):
        sys.exit("FAILED: an injection had no effect or the golden output was not restored")


if __name__ == "__main__":
    main()
