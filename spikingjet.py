#!/usr/bin/env python3
"""
SpikingJET: fault injection for spiking, conventional and mixed neural networks.

    python spikingjet.py sites    -n NETWORK                      list every fault site of a network
    python spikingjet.py campaign --config EXP.toml               single-fault campaign (DATE'09 sample)
    python spikingjet.py ber      --config EXP.toml               bit-error-rate sweep
    python spikingjet.py train    --config EXP.toml               training, with faults during training
    python spikingjet.py config   [--config EXP.toml]             print the full configuration

Every command reads an experiment file (TOML, examples in experiments/) on top of the defaults;
-n sets the network and --set changes any key:

    python spikingjet.py campaign -n SHD_DELAY --set campaign.fault_model=bit-flip --set campaign.max_faults=50
    python spikingjet.py ber --config experiments/ber_neurons.toml --set "ber.bers=[1e-5, 1e-4]"
    python spikingjet.py train --config experiments/fat_weights.toml --set train.faults[0].ber=1e-4
"""
import argparse
import sys

from spikingjet.config import dump, load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sites", "campaign", "ber", "train", "config"])
    ap.add_argument("--config", "-c", help="experiment file (TOML)")
    ap.add_argument("--network", "-n", help="network of networks.py (overrides the config)")
    ap.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key, e.g. campaign.max_faults=50 (repeatable)")
    ap.add_argument("--synthetic", action="store_true", help="train: random data instead of the dataset (smoke test)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set, network=a.network)

    if a.command == "config":
        print(dump(cfg))
    elif a.command == "sites":
        import torch
        from spikingjet.runtime import device_of, require_network, sites_of
        from networks import load_network, synthetic_batch
        name = require_network(cfg)
        dev = device_of(cfg)
        net = load_network(name, dev)
        print(sites_of(net, cfg, synthetic_batch(name, 2)[0].to(dev)).describe())
    elif a.command == "campaign":
        from spikingjet import campaign
        campaign.run(cfg)
    elif a.command == "ber":
        from spikingjet import ber
        ber.run(cfg)
    elif a.command == "train":
        from spikingjet import trainer
        trainer.run(cfg, synthetic=a.synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
