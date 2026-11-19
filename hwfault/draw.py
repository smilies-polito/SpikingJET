"""
Counter-based fault drawing (design D4) and stratified single-fault lists.

Every potential fault position has a stable key  (site, coordinates..., bit)  where site is
(class, layer, field) and the coordinates are the cell index plus, for transient faults, the
global sample index and the timestep. A fault is present iff  U(seed, key) < rate,  with U a
counter-based uniform (splitmix64 chain on int64 tensors, vectorized on the GPU). Consequences:
  * the same seed gives the same faults (reproducible, no generator state);
  * a baseline and a grown network share every fault on the cells they have in common
    (pairing across widths), and the extra cells get independent draws at the same rate;
  * permanent faults are drawn once per realization (no sample / timestep in the key), transient
    faults per sample (M1/M2, with a hashed onset) or per sample and timestep (M3/M4).

Per cell the draw is done in two exact stages: (1) the cell holds at least one faulty bit with
probability 1 - (1 - r)**nbits; (2) for those cells, the first faulty bit follows the truncated
geometric law and every later bit is Bernoulli(r). The joint law of the bits is exactly that of
independent Bernoulli(r) bits, at ~1/nbits of the hashing cost.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import math
import os
from dataclasses import dataclass

import torch

from .injector import Fault, unravel
from .memory_map import CLASSES, LIF

_M64 = 1 << 64


def _s(x):        # unsigned 64-bit constant -> signed int64
    return x - _M64 if x >= (1 << 63) else x


_GOLD, _C1, _C2 = _s(0x9E3779B97F4A7C15), _s(0xBF58476D1CE4E5B9), _s(0x94D049BB133111EB)


def _srl(x, k):
    return (x >> k) & ((1 << (64 - k)) - 1)


def mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer on int64 tensors (wrap-around arithmetic)."""
    x = x + _GOLD
    x = (x ^ _srl(x, 30)) * _C1
    x = (x ^ _srl(x, 27)) * _C2
    return x ^ _srl(x, 31)


def salt(*parts) -> int:
    """Stable 63-bit integer for a tuple of names / numbers (independent of PYTHONHASHSEED)."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") >> 1


def derive_seed(namespace: str, base_seed: int, *key) -> int:
    """Seed of a namespace / base seed / key tuple."""
    import json
    msg = json.dumps([namespace, int(base_seed), *key], separators=(",", ":"), default=str).encode()
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "little") >> 1


FIELD_BITS = 16                 # every key coordinate must be < 2**16 (samples, steps, neurons)


def pack(coords) -> torch.Tensor:
    """Coordinates -> one int64 key with fixed 16-bit fields (the last coordinate in the lowest
    field). The packing does not depend on tensor sizes, so the key of a cell is the same in a
    baseline and in a grown network."""
    key = None
    n = len(coords)
    for k, c in enumerate(coords):
        c = c.to(torch.int64)
        term = c << (FIELD_BITS * (n - 1 - k))
        key = term if key is None else key | term
    return key


def hash_keys(seed: int, site_salt: int, coords) -> torch.Tensor:
    """coords: list of int64 tensors (broadcastable). Returns int64 hashes of the same shape."""
    dev = coords[0].device if coords else "cpu"
    h0 = mix(torch.as_tensor(_s((seed ^ site_salt) % _M64), dtype=torch.int64, device=dev))
    return mix(h0 ^ pack(coords))


def uniform(h: torch.Tensor) -> torch.Tensor:
    """int64 hash -> float64 uniform in [0, 1) (53 bits)."""
    return _srl(h, 11).to(torch.float64) * (2.0 ** -53)


def _sub(h: torch.Tensor, tag: int) -> torch.Tensor:
    """Derived independent hash of the same key (stage-2 draws, types, onsets)."""
    return mix(h ^ _s(salt("sub", tag) % _M64))


def draw_bits(seed, site_salt, coords, nbits: int, rate: float):
    """Bernoulli(rate) faults on every (cell, bit) of the cells given by coords (broadcast to the
    cell grid). Returns (flat cell indices [K], bits [K]) of the faulty positions."""
    cells, bits, _ = draw_bits_h(seed, site_salt, coords, nbits, rate)
    return cells, bits


def draw_bits_h(seed, site_salt, coords, nbits: int, rate: float):
    """draw_bits that also returns the stage-1 hash of every faulty position's cell (used to
    derive fault types and onsets per (cell, bit) without re-hashing the grid)."""
    dev = coords[0].device
    empty = torch.empty(0, dtype=torch.int64, device=dev)
    if rate <= 0:
        return empty, empty, empty
    rate = min(float(rate), 1.0)
    p_any = 1.0 - (1.0 - rate) ** nbits if rate < 1 else 1.0
    if rate < 1e-12:
        p_any = nbits * rate
    h = hash_keys(seed, site_salt, coords).reshape(-1)
    cells = torch.nonzero(uniform(h) < p_any).squeeze(1)
    if cells.numel() == 0:
        return empty, empty, empty
    hc = h[cells]
    if rate >= 1:
        first = torch.zeros_like(cells)
    else:
        u1 = uniform(_sub(hc, 0))
        # CDF(j) = (1 - q^(j+1)) / p_any; smallest j with CDF(j) > u: j = floor(log(1 - u p_any) / log q)
        first = torch.floor(torch.log1p(-u1 * p_any) / math.log(1.0 - rate)).clamp(0, nbits - 1).to(torch.int64)
    out_c, out_b, out_h = [cells], [first], [hc]
    for b in range(1, nbits):               # later bits: independent Bernoulli(rate)
        later = first < b
        if not bool(later.any()):
            break
        hit = uniform(_sub(hc[later], b)) < rate
        out_c.append(cells[later][hit])
        out_b.append(torch.full_like(cells[later][hit], b))
        out_h.append(hc[later][hit])
    return torch.cat(out_c), torch.cat(out_b), torch.cat(out_h)


def choice_from(h: torch.Tensor, bits: torch.Tensor, tag: int, n: int) -> torch.Tensor:
    """Deterministic integer in [0, n) per (cell hash, bit) (fault type, onset)."""
    return torch.floor(uniform(_sub(h ^ (bits << 40), 100 + tag)) * n).to(torch.int64)


def hashed_choice(seed, site_salt, coords, n: int) -> torch.Tensor:
    """Deterministic integer in [0, n) per key."""
    return torch.floor(uniform(_sub(hash_keys(seed, site_salt, coords), 99)) * n).to(torch.int64)


# ----------------------------------------------------------------------------------------------
# BER drawing
# ----------------------------------------------------------------------------------------------
DEFAULT_TYPES = {      # fault type drawn at BER, per (class, persistence)
    ("M1", "permanent"): "stuck-at", ("M2", "permanent"): "stuck-at", ("M3", "permanent"): "stuck-at",
    ("M1", "transient"): "flip", ("M2", "transient"): "flip", ("M3", "transient"): "flip",
    ("M4", "permanent"): "dead-saturated", ("M4", "transient"): "flip",
}


@dataclass
class DrawSpec:
    seed: int                       # realization seed
    rates: dict                     # class -> BER (bits for M1-M3, events for M4)
    types: dict | None = None       # (class, persistence) -> "flip" | "stuck-at" | ...


class FaultDrawer:
    """Draws the faults of one realization on a memory map (see module docstring)."""

    def __init__(self, mm, spec: DrawSpec):
        self.mm = mm
        self.spec = spec
        self.types = dict(DEFAULT_TYPES, **(spec.types or {}))
        for c in spec.rates:
            if c not in CLASSES:
                raise ValueError(f"unknown class {c}")

    def rate(self, cls):
        return float(self.spec.rates.get(cls, 0.0))

    def site_salt(self, site):
        return salt(site.cls, site.layer, site.field, site.persistence)

    @staticmethod
    def _grid(shape, device):
        """Index tensors of a cell grid, broadcastable (one per dimension)."""
        out = []
        for d, n in enumerate(shape):
            sh = [1] * len(shape)
            sh[d] = n
            out.append(torch.arange(n, device=device).reshape(sh))
        return out

    def _type_ids(self, site, h, bits):
        """0 flip, 1 sa0, 2 sa1 (bits); M4 permanent: 1 dead, 2 saturated."""
        kind = self.types[(site.cls, site.persistence)]
        if kind == "flip":
            return torch.zeros_like(bits)
        if kind in ("stuck-at", "dead-saturated"):
            return 1 + choice_from(h, bits, 1, 2)
        raise ValueError(kind)

    def permanent_faults(self, device="cpu") -> list:
        """All permanent faults of the realization (any class whose persistence is permanent)."""
        faults = []
        for site in self.mm.sites:
            r = self.rate(site.cls)
            if r <= 0 or site.persistence != "permanent":
                continue
            nb = 1 if site.cls == "M4" else site.bits
            cells, bits, h = draw_bits_h(self.spec.seed, self.site_salt(site), self._grid(site.shape, device), nb, r)
            if cells.numel() == 0:
                continue
            idx = unravel(cells, site.shape).tolist()
            ty = self._type_ids(site, h, bits).tolist()
            names = ("dead", "dead", "saturated") if site.cls == "M4" else ("flip", "sa0", "sa1")
            for k, b in enumerate(bits.tolist()):
                faults.append(Fault(site.cls, site.layer, site.field, tuple(idx[k]),
                                    0 if site.cls == "M4" else b, names[ty[k]], "permanent"))
        return faults

    # transient M1/M2: per sample, onset hashed uniformly over the layer's output steps
    def transient_static(self, layer: str, samples: torch.Tensor, steps: int):
        """Vectorized transient M1/M2 faults of `layer` for the given global sample ids.
        Returns a list of (site, rows [K], cell index [K, d], bits [K], t0 [K], types [K] (0 flip,
        1 sa0, 2 sa1))."""
        out = []
        dev = samples.device
        for site in self.mm.sites:
            if site.layer != layer or site.cls not in ("M1", "M2") or site.persistence != "transient":
                continue
            r = self.rate(site.cls)
            if r <= 0:
                continue
            coords = [samples.reshape([-1] + [1] * len(site.shape))] + \
                     [g.unsqueeze(0) for g in self._grid(site.shape, dev)]
            cells, bits, h = draw_bits_h(self.spec.seed, self.site_salt(site), coords, site.bits, r)
            if cells.numel() == 0:
                continue
            idx = unravel(cells, (samples.numel(),) + tuple(site.shape))
            t0 = choice_from(h, bits, 2, steps)
            out.append((site, idx[:, 0], idx[:, 1:], bits, t0, self._type_ids(site, h, bits)))
        return out

    # transient M3/M4: per sample and timestep, drawn for a whole layer at once
    def transient_state_layer(self, lif: str, samples: torch.Tensor, steps: int, n: int, t_chunk: int = 48,
                              t_start: int = 0):
        """Transient M3/M4 faults of layer `lif` for the given global sample ids and every step.
        Returns {"membrane": (t, row, col, xor, set, clear) or None, "spike": (t, row, col) or None},
        each sorted by timestep."""
        layer = {v: k for k, v in LIF.items()}[lif]
        dev = samples.device
        res = {"membrane": None, "spike": None}
        for site in self.mm.sites:
            if site.layer != layer or site.persistence != "transient" or site.cls not in ("M3", "M4"):
                continue
            r = self.rate(site.cls)
            if r <= 0:
                continue
            nb = 1 if site.cls == "M4" else site.bits
            parts = []
            for t_lo in range(t_start, steps, t_chunk):
                tt = torch.arange(t_lo, min(steps, t_lo + t_chunk), device=dev)
                coords = [samples.reshape(-1, 1, 1), tt.reshape(1, -1, 1), torch.arange(n, device=dev).reshape(1, 1, -1)]
                cells, bits, h = draw_bits_h(self.spec.seed, self.site_salt(site), coords, nb, r)
                if cells.numel():
                    ix = unravel(cells, (samples.numel(), tt.numel(), n))
                    parts.append((ix[:, 1] + t_lo, ix[:, 0], ix[:, 2], bits, h))
            if not parts:
                continue
            t, row, col, bits, h = (torch.cat([p[k] for p in parts]) for k in range(5))
            order = torch.argsort(t, stable=True)
            t, row, col, bits, h = t[order], row[order], col[order], bits[order], h[order]
            if site.cls == "M4":
                res["spike"] = (t, row, col)
                continue
            m = torch.ones_like(bits) << bits
            ty = self._type_ids(site, h, bits)
            zero = torch.zeros_like(m)
            res["membrane"] = (t, row, col, torch.where(ty == 0, m, zero), torch.where(ty == 2, m, zero),
                               torch.where(ty == 1, m, zero))
        return res

    def transient_state(self, lif: str, t: int, samples: torch.Tensor, n: int):
        """Dense masks of one timestep (inspection / tests): ((xor, set, clear) [B, n] or None,
        spike flip mask [B, n] or None)."""
        d = self.transient_state_layer(lif, samples, t + 1, n)
        B = samples.numel()
        mem, spk = None, None
        if d["membrane"] is not None:
            tt, r, c, xor, setm, clr = d["membrane"]
            sel = tt == t
            out = []
            for q in (xor, setm, clr):
                m = torch.zeros(B, n, dtype=torch.int64, device=samples.device)
                m.index_put_((r[sel], c[sel]), q[sel], accumulate=True)
                out.append(m)
            mem = tuple(out)
        if d["spike"] is not None:
            tt, r, c = d["spike"]
            sel = tt == t
            spk = torch.zeros(B, n, dtype=torch.bool, device=samples.device)
            spk[r[sel], c[sel]] = True
        return mem, spk


# ----------------------------------------------------------------------------------------------
# Fault-list files (gzip CSV)
# ----------------------------------------------------------------------------------------------
FIELDS = ("cls", "layer", "field", "index", "bit", "ftype", "persistence", "t", "sample")


def write_faults(path: str, faults, extra: dict | None = None):
    """gzip CSV; `extra` columns (same for every row, e.g. ber, realization) are appended."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = extra or {}
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS) + list(extra))
        w.writeheader()
        for ft in faults:
            w.writerow({**ft.to_row(), **extra})


def read_faults(path: str) -> list:
    with gzip.open(path, "rt", newline="") as f:
        return [Fault.from_row(r) for r in csv.DictReader(f)]


# ----------------------------------------------------------------------------------------------
# Stratified single-fault lists (catalog, spec fault-sensitivity-catalog)
# ----------------------------------------------------------------------------------------------
def stratified_single_faults(mm, fraction: float, seed_of, exhaustive_classes=("M2",),
                             dead_saturated: bool = True):
    """Single faults of a memory map, sampled per stratum.

    Stratum = (class, layer, field, bit or event type, persistence). Units of a stratum: its
    cells for permanent faults; (cell, timestep) for transient M3/M4; cells with a seeded onset
    for transient M1/M2. round(fraction * size) units are drawn without replacement with a numpy
    generator seeded by seed_of(stratum_key) (catalog namespace); classes in exhaustive_classes
    are taken whole. Bit faults are flips (stuck-at outcomes are derived from them); M4 transient
    strata are 'drop' and 'spurious'. With dead_saturated, the dead and saturated (permanent)
    fault of every hidden neuron is added, whatever the persistence of M4.
    Returns (faults, coverage rows)."""
    import numpy as np
    faults, coverage = [], []
    for site in mm.sites:
        steps = site.steps
        if site.cls == "M4":
            events = site.fault_types                      # drop/spurious or dead/saturated
            if site.persistence == "permanent":
                continue                                   # covered by dead_saturated below
            bit_list = [(0, e) for e in events]
        else:
            bit_list = [(b, "flip") for b in range(site.bits)]
        cells = site.cells
        per_time = site.cls in ("M3", "M4") and site.persistence == "transient"
        for bit, ftype in bit_list:
            key = (site.cls, site.layer, site.field, bit, ftype, site.persistence)
            size = cells * (steps if per_time else 1)
            exhaustive = site.cls in exhaustive_classes
            k = size if exhaustive else int(round(fraction * size))
            rng = np.random.default_rng(seed_of(key))
            units = np.arange(size) if k == size else np.sort(rng.choice(size, k, replace=False))
            onset_rng = np.random.default_rng(seed_of(key + ("onset",)))
            for u in units.tolist():
                cell, t = (u // steps, u % steps) if per_time else (u, None)
                if site.persistence == "transient" and not per_time:
                    t = int(onset_rng.integers(0, mm.T + 12 * (1 + ("block1", "block2", "block_out").index(site.layer))))
                idx = tuple(int(v) for v in np.unravel_index(cell, site.shape))
                faults.append(Fault(site.cls, site.layer, site.field, idx, bit, ftype, site.persistence, t))
            coverage.append({"cls": site.cls, "layer": site.layer, "field": site.field, "bit": bit,
                             "ftype": ftype, "persistence": site.persistence, "size": size,
                             "measured": int(len(units)), "exhaustive": exhaustive})
    if dead_saturated:
        for site in mm.sites:
            if site.cls != "M4":
                continue
            for ftype in ("dead", "saturated"):
                for j in range(site.cells):
                    faults.append(Fault("M4", site.layer, "spike", (j,), 0, ftype, "permanent"))
                coverage.append({"cls": "M4", "layer": site.layer, "field": "spike", "bit": 0, "ftype": ftype,
                                 "persistence": "permanent", "size": site.cells, "measured": site.cells,
                                 "exhaustive": True})
    return faults, coverage
