#!/usr/bin/env python3
"""
Hardware-fault-model campaigns on a deployment-form SHD_DELAY checkpoint (models/deployed_shd.py).

    python3 hw_campaign.py map     --checkpoint deployed.th [--hw-target neuromorphic] [--persistence mixed]
    python3 hw_campaign.py faults  --checkpoint deployed.th --out faults.csv.gz --fraction 0.01   fault list
    python3 hw_campaign.py catalog --checkpoint deployed.th --out DIR ...   single-fault damage catalog
    python3 hw_campaign.py ber     --checkpoint deployed.th --out DIR ...   BER multi-fault sweep

This CLI is new and independent of main.py: the original FP32 campaigns are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def exact_numerics(tf32: bool = False):
    """Exact FP32 convolutions (no TF32 unless --tf32) and deterministic cuDNN (numerics.py)."""
    from numerics import set_tf32
    return set_tf32(tf32)


def _common(p):
    p.add_argument("--checkpoint", required=True, help="DeployedSHD checkpoint (.th)")
    p.add_argument("--hw-target", default="neuromorphic", choices=["neuromorphic", "gpu"])
    p.add_argument("--persistence", default="mixed",
                   help="preset (permanent | transient | mixed) or JSON {\"M1\": \"permanent\", ...}")
    from numerics import add_argument as add_tf32_argument
    add_tf32_argument(p)


def _mode(text):
    return json.loads(text) if text.strip().startswith("{") else text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("map", help="print the memory map")
    _common(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("faults", help="stratified single-fault list for 'catalog'")
    _common(p)
    p.add_argument("--out", required=True, help="gzip CSV")
    p.add_argument("--fraction", type=float, default=0.01, help="fraction of each stratum to sample")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("catalog", help="single-fault damage catalog (fault-parallel)")
    _common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--faults", required=True, help="gzip CSV fault list (hwfault.draw)")
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--split", default="test")
    p = sub.add_parser("ber", help="BER multi-fault sweep")
    _common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True, help="JSON protocol configuration")
    a = ap.parse_args(argv)

    from hwfault.memory_map import HardwareTargetNotImplemented, check_hw_target
    try:
        check_hw_target(a.hw_target)              # before loading anything or writing results
    except HardwareTargetNotImplemented as e:
        print(f"hw_campaign: {e}", file=sys.stderr)
        return 2
    exact_numerics(a.tf32)
    from models.deployed_shd import DeployedSHD
    from hwfault.memory_map import MemoryMap
    net = DeployedSHD.load(a.checkpoint)
    if a.cmd == "map":
        mm = MemoryMap(net, a.hw_target, _mode(a.persistence))
        print(json.dumps(mm.to_dict(), indent=1) if a.json else mm.table())
        return 0
    if a.cmd == "faults":
        from hwfault.draw import derive_seed, stratified_single_faults, write_faults
        mm = MemoryMap(net, a.hw_target, _mode(a.persistence))
        faults, _ = stratified_single_faults(mm, a.fraction, lambda key: derive_seed("catalog", a.seed, *key))
        write_faults(a.out, faults)
        print(f"{len(faults):,} faults -> {a.out}")
        return 0
    if a.cmd == "catalog":
        from hwfault.catalog_campaign import main_catalog
        return main_catalog(a, net)
    from hwfault.ber_campaign import main_ber
    return main_ber(a, net)


if __name__ == "__main__":
    sys.exit(main())
