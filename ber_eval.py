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
import math
import os
import sys
import time

import numpy as np
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
# Fault population
# ----------------------------------------------------------------------------
def fault_population(model: nn.Module, scope: str):
    """Ordered list of (name, parameter) that can receive faults."""
    params = list(model.named_parameters())
    if scope == "all":
        return params
    classes = [nn.Conv2d, nn.Linear]
    try:
        import snntorch as snn
        classes += [snn.Leaky, snn.Synaptic]
    except ImportError:
        pass
    top = {name.split(".")[0] for name, m in model.named_modules()
           if name and isinstance(m, tuple(classes))}
    return [(n, p) for n, p in params if n.split(".")[0] in top]


def _to_int32(x: torch.Tensor) -> torch.Tensor:
    """uint32 bit pattern held in int64 -> same bits as int32 (bit 31 becomes the sign)."""
    return torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32)


def split_tmr(population):
    """Split off the 2nd and 3rd copies of TMR layers ('<layer>.copies.1/2.*').

    The rest (with copy 0 in the position of the original layer) has exactly the shapes and order
    of the plain network's population, so it receives the SAME fault map as the plain network
    (paired comparison). The two extra copies get their own, independent faults at the same BER."""
    extra = [(n, p) for n, p in population if ".copies.1." in n or ".copies.2." in n]
    base = [(n, p) for n, p in population if not (".copies.1." in n or ".copies.2." in n)]
    return base, extra


class BitErrorInjector:
    """Applies one fault map to a list of float32 parameters and restores it exactly."""

    def __init__(self, population, mode: str = "bit-flip"):
        self.population = population
        self.sizes = np.array([p.numel() for _, p in population], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n_elems = int(self.offsets[-1])
        self.mode = mode
        self._saved = []
        for n, p in population:
            if p.dtype != torch.float32:
                raise TypeError(f"{n} is {p.dtype}; only float32 parameters are supported")

    def sample(self, ber: float, bit_lo: int, bit_hi: int, rng: np.random.Generator):
        """Faults on the contiguous bit range bit_lo..bit_hi."""
        return self.sample_bits(ber, np.arange(bit_lo, bit_hi + 1), rng)

    def sample_bits(self, ber: float, bits, rng: np.random.Generator):
        """Every (element, bit) position with bit in `bits` fails independently with
        probability `ber`. Returns (element index, bit, stuck value) arrays."""
        bits = np.asarray(bits, dtype=np.int64)
        nb = bits.size
        total_bits = self.n_elems * nb
        n = int(rng.binomial(total_bits, ber)) if ber > 0 else 0
        if n == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
        flat = rng.integers(0, total_bits, size=n, dtype=np.int64)
        flat = np.unique(flat)                     # collisions are negligible; avoids double flips
        elem = flat // nb
        bit = bits[flat % nb]
        values = rng.integers(0, 2, size=elem.size, dtype=np.int64)   # used by stuck-at only
        return elem, bit, values

    @torch.no_grad()
    def apply(self, elem, bit, values):
        self._saved = []
        if elem.size == 0:
            return
        which = np.searchsorted(self.offsets, elem, side="right") - 1
        for k in np.unique(which):
            sel = which == k
            _, p = self.population[k]
            local = torch.as_tensor(elem[sel] - self.offsets[k], device=p.device)
            ints = p.data.view(-1).view(torch.int32)
            # (element, bit) pairs are unique, so several faults on one element touch different
            # bits and their masks can be combined by summation (sum == OR == XOR here).
            m = torch.as_tensor(np.left_shift(np.int64(1), bit[sel]), device=p.device)
            uniq, inv = torch.unique(local, return_inverse=True)
            self._saved.append((ints, uniq, ints[uniq].clone()))
            if self.mode == "bit-flip":
                flip = torch.zeros(uniq.numel(), dtype=torch.int64, device=p.device).index_add_(0, inv, m)
                ints[uniq] = ints[uniq] ^ _to_int32(flip)
            else:                                       # stuck-at-1 sets the bit, stuck-at-0 clears it
                v = torch.as_tensor(values[sel], device=p.device)
                zeros = torch.zeros(uniq.numel(), dtype=torch.int64, device=p.device)
                set_m = zeros.clone().index_add_(0, inv, m * v)
                clr_m = zeros.clone().index_add_(0, inv, m * (1 - v))
                ints[uniq] = (ints[uniq] | _to_int32(set_m)) & ~_to_int32(clr_m)

    @torch.no_grad()
    def restore(self):
        for ints, idx, orig in self._saved:
            ints[idx] = orig
        self._saved = []


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    correct = total = nonfinite = 0
    for x, y in loader:
        out = model(x.to(device))
        bad = ~torch.isfinite(out).all(dim=1)
        pred = out.argmax(1).cpu()
        pred[bad.cpu()] = -1                       # a non-finite output is never correct
        correct += (pred == y).sum().item()
        nonfinite += bad.sum().item()
        total += y.numel()
    return 100.0 * correct / total, 100.0 * nonfinite / total


def get_loader(network, batch_size, synthetic=False):
    if synthetic:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (64,), generator=g)
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
        pop, extra_pop = split_tmr(fault_population(model, a.scope))
        inj = BitErrorInjector(pop, a.mode)
        inj_x = BitErrorInjector(extra_pop, a.mode) if extra_pop else None
        clean, _ = evaluate(model, loader, device)
        print(f"{name}: {inj.n_elems:,} parameters in the fault population"
              + (f" + {inj_x.n_elems:,} in the extra TMR copies (faulted independently)" if inj_x else "")
              + f", clean {clean:.2f}%")

        for bits in a.bits:
            lo, hi = BIT_RANGES[bits]
            for ber in a.bers:
                for r in range(a.realizations):
                    # same (seed, bits, ber, r) -> same fault map for every model with the same parameters
                    key = (a.seed, lo, hi, int(round(-math.log10(ber) * 1000)), r, a.mode == "stuck-at")
                    rng = np.random.default_rng(np.random.SeedSequence([int(v) for v in key]))
                    elem, bit, vals = inj.sample(ber, lo, hi, rng)
                    n_f = elem.size
                    t0 = time.time()
                    inj.apply(elem, bit, vals)
                    if inj_x:
                        rng_x = np.random.default_rng(np.random.SeedSequence([int(v) for v in key] + [7]))
                        ex, bx, vx = inj_x.sample(ber, lo, hi, rng_x)
                        inj_x.apply(ex, bx, vx)
                        n_f += ex.size
                    try:
                        acc, nf = evaluate(model, loader, device)
                    finally:
                        inj.restore()
                        if inj_x:
                            inj_x.restore()
                    # n_params = the paired (base) population, so ber_report pairs TMR rows with the plain network
                    w.writerow([name, a.mode, a.scope, bits, ber, r, n_f, inj.n_elems,
                                f"{clean:.4f}", f"{acc:.4f}", f"{nf:.4f}", f"{time.time() - t0:.1f}"])
                    f.flush()
                print(f"  {bits:9s} BER {ber:.0e}: last realization {acc:6.2f}% "
                      f"({n_f} faults, {nf:.1f}% non-finite outputs)")
        # sanity: the restore must be exact
        after, _ = evaluate(model, loader, device)
        if abs(after - clean) > 1e-9:
            print(f"ERROR: clean accuracy changed after the sweep ({clean} -> {after}); restore failed")
            sys.exit(1)
    f.close()


if __name__ == "__main__":
    main()
