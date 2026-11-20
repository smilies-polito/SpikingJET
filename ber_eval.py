#!/usr/bin/env python3
"""
Multi-fault (bit error rate) evaluation for SpikingJET networks.

SpikingJET's DATE'09 campaigns inject ONE fault at a time. The FAT literature
(Stutz et al. RandBET, EDEN, Vinck et al., FitAct, EnforceSNN) instead flips every
stored bit independently with probability p (the bit error rate, BER) and
reports accuracy vs BER and the tolerable BER at a given accuracy loss. This
script adds that view without touching the single-fault pipeline.

For each (BER, realization):
  1. draw the number of flips  n ~ Binomial(total_bits, BER)
  2. draw n positions uniformly over (parameter element, bit) - optionally
     restricted to a bit range (e.g. mantissa only, exponent only)
  3. apply them all at once (XOR on the int32 view; stuck-at sets/clears bits)
  4. run the WHOLE test set with these faults present, then restore exactly
The fault map of a realization depends only on (seed, BER, bit range) and on the
parameter shapes, so two models with the same parameters see the same faults
(paired), exactly like the DATE'09 campaigns.

Fault population (--scope):
  spikingjet : the same parameters SpikingJET's FaultListGenerator injects
               (top-level modules that are or contain Conv2d / Linear /
               snn.Leaky / snn.Synaptic). Default; keeps the two protocols aligned.
  all        : every parameter of the network.

Usage:
    python3 ber_eval.py --network SHD_DELAY NMNIST_CSNN --bits all mantissa \
        --realizations 10 --out ber_results/snn.csv
    python3 ber_report.py ber_results/snn.csv               # table + figure

Any network registered in networks.py works.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")

BIT_RANGES = {
    "all": (0, 31),
    "mantissa": (0, 22),
    "exponent": (23, 30),
    "sign": (31, 31),
    "low-mantissa": (0, 15),
}
DEFAULT_BERS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]


# ----------------------------------------------------------------------------
# Fault population (the injection itself is spikingjet's engine)
# ----------------------------------------------------------------------------
def fault_population(model: nn.Module, scope: str):
    """Ordered list of parameter sites that can receive faults (spikingjet.sites.param_sites)."""
    from spikingjet.sites import param_sites
    return param_sites(model, scope)


def split_tmr(population):
    """Split off the 2nd and 3rd copies of TMR layers ('<layer>.copies.1/2.*').

    The rest (with copy 0 in the position of the original layer) has exactly the shapes and order
    of the plain network's population, so it receives the SAME fault map as the plain network
    (paired comparison). The two extra copies get their own, independent faults at the same BER."""
    extra = [s for s in population if ".copies.1." in s.name or ".copies.2." in s.name]
    base = [s for s in population if not (".copies.1." in s.name or ".copies.2." in s.name)]
    return base, extra


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    """(accuracy %, % of non-finite outputs); a non-finite output is never correct."""
    from spikingjet.ber import evaluate as _evaluate
    return _evaluate(model, loader, device)


def get_loader(network, batch_size, synthetic=False):
    if synthetic:                                  # random batch shaped like the network's input
        from networks import NETWORKS, synthetic_batch
        x, y = synthetic_batch(network, 64)
        if NETWORKS[network].example.time_first:   # [T, B, ...]: one batch, already in model order
            return [(x, y)]
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size)
    from networks import get_loader as registry_loader
    return registry_loader(network, "test", batch_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", nargs="+", required=True)
    ap.add_argument("--bers", type=float, nargs="+", default=DEFAULT_BERS)
    ap.add_argument("--bits", nargs="+", default=["all", "mantissa"], choices=list(BIT_RANGES))
    ap.add_argument("--mode", choices=["bit-flip", "stuck-at"], default="bit-flip")
    ap.add_argument("--scope", choices=["spikingjet", "all"], default="spikingjet")
    ap.add_argument("--realizations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="random inputs instead of the test set (smoke test of the pipeline)")
    from numerics import add_argument as add_tf32_argument, set_tf32
    add_tf32_argument(ap)
    a = ap.parse_args()
    set_tf32(a.tf32, deterministic=False)          # exact FP32 unless --tf32 (numerics.py)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["network", "mode", "scope", "bits", "ber", "realization", "n_faults",
                    "n_params", "clean_acc", "faulty_acc", "nonfinite_pct", "seconds"])

    for name in a.network:
        from networks import load_network
        model = load_network(name, device)
        model.eval()
        loader = get_loader(name, a.batch_size, a.synthetic)
        from spikingjet.faults import ParamBER
        from spikingjet.injector import Injector
        from spikingjet.sites import SiteTable
        pop, extra_pop = split_tmr(fault_population(model, a.scope))
        stuck = a.mode == "stuck-at"
        inj = Injector(model, SiteTable(pop + extra_pop, [], []), hook_activations=False)
        base, extra = ParamBER(pop), (ParamBER(extra_pop) if extra_pop else None)
        clean, _ = evaluate(model, loader, device)
        print(f"{name}: {base.n_elems:,} parameters in the fault population"
              + (f" + {extra.n_elems:,} in the extra TMR copies (faulted independently)" if extra else "")
              + f", clean {clean:.2f}%")

        for bits in a.bits:
            for ber in a.bers:
                for r in range(a.realizations):
                    # same (seed, bits, ber, r) -> same fault map for every model with the same parameters
                    t0 = time.time()
                    arrays = base.sample(ber, bits, ParamBER.rng_for(a.seed, bits, ber, r, stuck), stuck)
                    if extra:
                        arrays += extra.sample(ber, bits, ParamBER.rng_for(a.seed, bits, ber, r, stuck, (7,)), stuck)
                    n_f = sum(len(x[1]) for x in arrays)
                    inj.apply_param_arrays(arrays)
                    try:
                        acc, nf = evaluate(model, loader, device)
                    finally:
                        inj.restore_params()
                    # n_params = the paired (base) population, so ber_report pairs TMR rows with the plain network
                    w.writerow([name, a.mode, a.scope, bits, ber, r, n_f, base.n_elems,
                                f"{clean:.4f}", f"{acc:.4f}", f"{nf:.4f}", f"{time.time() - t0:.1f}"])
                    f.flush()
                print(f"  {bits:9s} BER {ber:.0e}: last realization {acc:6.2f}% "
                      f"({n_f} faults, {nf:.1f}% non-finite outputs)")
        inj.detach()
        # sanity: the restore must be exact
        after, _ = evaluate(model, loader, device)
        if abs(after - clean) > 1e-9:
            print(f"ERROR: clean accuracy changed after the sweep ({clean} -> {after}); restore failed")
            sys.exit(1)
    f.close()


if __name__ == "__main__":
    main()
