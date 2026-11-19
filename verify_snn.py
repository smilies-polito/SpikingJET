#!/usr/bin/env python3
"""
Check a trained SNN baseline end to end through SpikingJET's own code paths, before a campaign.

    python3 verify_snn.py --network SHD_RLIF

1. loads the network through networks.load_network (the path main.py uses)
2. measures clean accuracy on the test loader main.py will use, and compares it with the
   test accuracy stored by train_snn.py
3. records the neuron-state shapes exactly like the clean run does and prints how the
   DATE'09 sample will split between parameters and neuron states
4. injects one membrane-potential and one spike fault and checks the output changes
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    from utils import load_network
    from models.snn_sota import get_loader
    from FaultInjectionManager import FaultInjectionManager
    from FaultGenerators.WeightFault import WeightFault

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_network(a.network, dev)
    loader = get_loader(a.network, "test", a.batch_size)
    stored = torch.load(f"models/pretrained_models/{a.network}.th", map_location="cpu").get("test_acc")

    fim = FaultInjectionManager(network=net, network_name=a.network + "_verify", device=dev, loader=loader)
    net.FaultInjector = fim
    fim.inject = False
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = net(x.to(dev))
            correct += (out.argmax(1).cpu() == y).sum().item(); total += y.numel()
    acc = 100 * correct / total
    print(f"clean accuracy {acc:.2f}%  (train_snn.py stored {stored})  "
          f"{'OK' if stored is None or abs(acc - stored) < 0.05 else 'MISMATCH - check preprocessing'}")

    n_param = sum(p.numel() for p in net.parameters())
    n_neuron = sum(torch.Size(s[1:]).numel() for s in fim.size.values())
    tot = 32 * (n_param + 2 * n_neuron)
    print(f"fault population: {32*n_param:,} parameter bits + {64*n_neuron:,} neuron-state bits "
          f"(potential + spike) -> {100*64*n_neuron/tot:.2f}% of a uniform sample hits neuron states")
    print("recorded neuron layers:", {k: tuple(v[1:]) for k, v in fim.size.items()})

    x, _ = next(iter(loader)); x = x.to(dev)
    with torch.no_grad():
        clean = net(x)
        layer = next(iter(fim.size)); shape = fim.size[layer][1:]
        idx = tuple(0 for _ in shape)
        fim.inject, fim.fault_mode = True, "bit-flip"
        for kind in ("potential", "spike"):
            fim.current = [layer, kind, idx, 30, 0, 1]
            diff = (net(x) - clean).abs().max().item()
            print(f"injected {layer}.{kind}{list(idx)} bit 30 -> max output change {diff:.3g}")
        fim.current, fim.inject = None, False
        print("golden output restored:", torch.allclose(net(x), clean))


if __name__ == "__main__":
    main()
