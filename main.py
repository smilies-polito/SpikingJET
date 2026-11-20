#!/usr/bin/env python3
"""
Single-fault campaign (command-line front end of `python spikingjet.py campaign`).

    python main.py -n SHD_DELAY --use-cuda --fault-mode bit-flip
    python main.py -n NMNIST_HYBRID --use-cuda --targets activations neurons --max-faults 500
    python main.py -n SHD_DELAY --config experiments/campaign_snn.toml      # every option of [campaign]

Results: output/fault_list/<network>/<seed>_fault_list.csv (faults),
<seed>_fault_list_result_batch_size_<b>.csv (per-site summary), output/<network>/pt/ (raw scores).
"""
import sys

from spikingjet import campaign
from spikingjet.config import load_config
from utils import parse_args


def main(args):
    if args.forbid_cuda:
        device = "cpu"
    elif args.use_cuda:
        device = "cuda"
    else:
        device = None                                # config value (default: auto)
    cfg = load_config(args.config, args.set,
                      network=args.network_name,
                      batch_size=args.batch_size,
                      device=device,
                      tf32=True if args.tf32 else None,
                      campaign__fault_model=args.fault_mode,
                      campaign__seed=args.seed,
                      campaign__max_faults=args.max_faults,
                      campaign__targets=args.targets,
                      campaign__neuron_persistence=args.neuron_persistence)
    campaign.run(cfg)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
