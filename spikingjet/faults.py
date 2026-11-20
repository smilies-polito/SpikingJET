"""Fault lists and fault maps.

single_fault_list   statistical sample of single faults (DATE'09 formula), over any sites
read / write_fault_list   the CSV format of output/fault_list/<network>/<seed>_fault_list.csv
ParamBER            a random fault map at a bit error rate over a population of parameters

The single-fault sample is drawn exactly like SpikingJET always has (numpy default_rng(seed),
choice without replacement over the (site, element, bit) population in order, then a stuck-at
value per fault), but without materialising the population, so the same seed gives the same list
and large networks need no memory.
"""
from __future__ import annotations

import csv
import math
import os
from ast import literal_eval

import numpy as np

from .bitops import resolve_bits, sample_ber
from .injector import Fault


def date09_n(N: int, e: float = 0.03, t: float = 2.58, p: float = 0.5) -> float:
    """Number of faults for error margin e at confidence t (Leveugle et al., DATE 2009)."""
    return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))


def single_fault_list(sites, seed: int, bits="all", e=0.03, t=2.58, p=0.5, max_faults=None,
                      n_steps=None, transient=False):
    """DATE'09 sample over every (site element, bit). Returns a list of Fault (value = the stuck-at
    value, used only by stuck-at campaigns; bit-flip campaigns set it to -1 when injecting).
    transient: neuron / activation faults get a random timestep in [0, n_steps)."""
    bits = np.asarray(resolve_bits(bits), dtype=np.int64)
    nb = bits.size
    sizes = np.array([s.numel * nb for s in sites], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    N = int(offsets[-1])
    rng = np.random.default_rng(seed=seed)
    n = date09_n(N, e, t, p)
    if max_faults is not None and max_faults < n:
        print(f"WARNING: max_faults {max_faults}: using {max_faults} of the {int(n)} faults of the DATE'09 sample")
        n = max_faults
    flat = rng.choice(N, int(n), replace=False)
    stuck = rng.integers(low=0, high=2, size=len(flat))
    which = np.searchsorted(offsets, flat, side="right") - 1
    faults = []
    step_rng = np.random.default_rng(seed + 1)
    for g, k, v in zip(flat, which, stuck):
        site = sites[k]
        local = int(g - offsets[k])
        elem, bit = divmod(local, nb)
        index = tuple(int(i) for i in np.unravel_index(elem, site.shape)) if site.shape else ()
        step = None
        if transient and site.kind != "param":
            step = int(step_rng.integers(0, n_steps or 1))
        faults.append(Fault(site.name, index, int(bits[bit]), int(v), step))
    return faults


def write_fault_list(path, faults):
    """CSV with the historical columns Injection, Layer, TensorIndex, Bit, Value (+ Step)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Injection", "Layer", "TensorIndex", "Bit", "Value", "Step"])
        for i, ft in enumerate(faults):
            w.writerow([i, ft.site, ft.index, ft.bit, ft.value, "" if ft.step is None else ft.step])


def read_fault_list(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, rows = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(head)}
    out = []
    for r in rows:
        value = int(r[col["Value"]]) if "Value" in col and r[col["Value"]] != "" else -1
        step = int(r[col["Step"]]) if "Step" in col and r[col["Step"]] != "" else None
        out.append(Fault(r[col["Layer"]], tuple(literal_eval(r[col["TensorIndex"]])), int(r[col["Bit"]]), value, step))
    return out


class ParamBER:
    """Fault maps at a bit error rate over a list of parameter sites, treated as one memory
    (every stored bit of the population fails independently). The draws are those of
    ber_eval.py, so (seed, BER, bits, realization) gives the same map for any network with the
    same parameter shapes (paired comparison)."""

    def __init__(self, sites):
        self.sites = list(sites)
        self.sizes = np.array([s.numel for s in self.sites], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n_elems = int(self.offsets[-1])

    @staticmethod
    def rng_for(seed, bits, ber, realization, stuck, extra=()):
        b = resolve_bits(bits)
        key = (seed, b[0], b[-1], int(round(-math.log10(ber) * 1000)) if ber > 0 else 0, realization, stuck) + tuple(extra)
        return np.random.default_rng(np.random.SeedSequence([int(v) for v in key]))

    def sample(self, ber, bits, rng, stuck=False):
        """[(site, local flat idx, bits, values)] for Injector.apply_param_arrays."""
        elem, bit, vals = sample_ber(self.n_elems, resolve_bits(bits), ber, rng, stuck)
        return self.split(elem, bit, vals)

    def split(self, elem, bit, vals):
        out = []
        if elem.size == 0:
            return out
        which = np.searchsorted(self.offsets, elem, side="right") - 1
        for k in np.unique(which):
            sel = which == k
            out.append((self.sites[k], elem[sel] - self.offsets[k], bit[sel], vals[sel]))
        return out
