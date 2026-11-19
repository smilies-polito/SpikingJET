"""
BER multi-fault campaign on the deployment form.

Each realization is one chip instance: its permanent faults are drawn once (hwfault.draw) and
shared by every sample; its transient faults are drawn per sample (M1/M2, with an onset) and per
sample and timestep (M3/M4), following the persistence of each class. A sweep runs every BER of
the grid with R realizations; the joint sweep puts the chip BER on every class, the isolated
sweeps on one class only. The seed of realization r at BER b is the same for every sweep, so an
isolated sweep sees exactly the faults of its class in the joint sweep.

Records (one per sweep, BER, realization) hold the seed, the number of faults per class
(permanent + transient), and the metrics of hwfault.metrics.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time

import torch

from .draw import DrawSpec, FaultDrawer, write_faults
from .injector import DrawnInjector
from .memory_map import CLASSES, MemoryMap
from .metrics import Accumulator

RECORD_FIELDS = ("sweep", "ber", "realization", "seed", "n_M1", "n_M2", "n_M3", "n_M4", "n_faults",
                 "p_M1", "p_M2", "p_M3", "p_M4",
                 "n", "clean_acc", "acc", "clean_conf", "conf", "dacc", "dconf", "D", "flips", "kl",
                 "nonfinite", "seconds")


def ber_grid(lo=1e-6, hi=1e-1, per_decade=2):
    n = int(round(math.log10(hi / lo) * per_decade))
    return [float(f"{lo * 10 ** (k / per_decade):.3g}") for k in range(n + 1)]


class EvalData:
    """Evaluation split on the device with the clean outputs of the model."""

    def __init__(self, net, loader, device):
        self.x, self.y, self.out = [], [], []
        saved, net.FaultInjector = net.FaultInjector, None
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device).float()
                self.x.append(x)
                self.y.append(y.to(device))
                self.out.append(net(x))
        net.FaultInjector = saved
        self.n = sum(int(y.numel()) for y in self.y)
        self.clean_acc = sum(int((o.argmax(1) == y).sum()) for o, y in zip(self.out, self.y)) / max(self.n, 1)


def sweep_rates(sweep: str, ber: float, rates: dict | None = None) -> dict:
    """Per-class rates of a sweep: 'joint' -> chip BER (or the per-class override) on every class;
    'M1'..'M4' -> only that class."""
    base = {c: (rates or {}).get(c, ber) for c in CLASSES}
    if sweep == "joint":
        return base
    return {sweep: base[sweep]}


@torch.no_grad()
def run_realization(net, data: EvalData, mm: MemoryMap, seed: int, rates: dict, types=None,
                    lam: float = 1.0, fault_log: str | None = None, max_logged: int = 200_000,
                    select=None) -> dict:
    """One chip instance over the whole split. `select(permanent_faults) -> faults` optionally
    sub-samples the permanent faults (equal-fault-count view)."""
    drawer = FaultDrawer(mm, DrawSpec(seed, rates, types))
    inj = DrawnInjector(net, drawer)
    if select is not None:
        inj.permanent = select(inj.permanent)
        inj.faults = list(inj.permanent)
        inj._index()
    acc = Accumulator(1, device=data.y[0].device)
    t0 = time.time()
    start = 0
    with inj:
        for x, y, out in zip(data.x, data.y, data.out):
            inj.set_batch(start, x.shape[0])
            acc.add(out, net(x), y)
            start += x.shape[0]
    r = acc.result(lam)[0]
    counts = {c: 0 for c in CLASSES}
    for f in inj.permanent:
        counts[f.cls] += 1
    r.update({f"p_{c}": counts[c] for c in CLASSES})            # permanent faults per class
    for c, k in inj.transient_counts.items():
        counts[c] += k
    if fault_log is not None and len(inj.permanent) <= max_logged:
        write_faults(fault_log, inj.permanent, {"seed": seed})
    r.update({f"n_{c}": counts[c] for c in CLASSES})
    r["n_faults"] = sum(counts.values())
    r["seconds"] = round(time.time() - t0, 3)
    r["seed"] = seed
    r["permanent_logged"] = fault_log is not None and len(inj.permanent) <= max_logged
    return r


def done_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    return {(r["sweep"], float(r["ber"]), int(r["realization"])) for r in csv.DictReader(lines)}


def run_sweeps(net, data: EvalData, mm: MemoryMap, sweeps, bers, realizations: int, seed_of,
               out_csv: str, fault_dir: str | None = None, rates=None, types=None, lam=1.0,
               log=print, select_for=None, max_logged=200_000) -> int:
    """All (sweep, BER, realization) records not yet in out_csv; seed_of(ber, r) gives the seed.
    Resumable: existing records are skipped (a partial last line is dropped)."""
    from .catalog_campaign import _repair_tail
    _repair_tail(out_csv)
    done = done_keys(out_csv)
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    n = 0
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for sweep in sweeps:
            for ber in bers:
                t_b = time.time()
                accs = []
                for r in range(realizations):
                    if (sweep, float(ber), r) in done:
                        continue
                    seed = seed_of(ber, r)
                    flog = None
                    if fault_dir and sweep == "joint":
                        flog = os.path.join(fault_dir, f"perm_ber{ber:g}_r{r}.csv.gz")
                    sel = select_for(sweep, ber, r) if select_for else None
                    rec = run_realization(net, data, mm, seed, sweep_rates(sweep, ber, rates), types, lam,
                                          flog, max_logged, sel)
                    rec.update({"sweep": sweep, "ber": ber, "realization": r})
                    w.writerow(rec)
                    fh.flush()
                    accs.append(rec["acc"])
                    n += 1
                if accs:
                    log(f"{sweep:5s} BER {ber:.2e}: {len(accs)} realizations, mean acc {100 * sum(accs) / len(accs):6.2f}% "
                        f"({time.time() - t_b:.0f} s)")
    return n


def main_ber(args, net) -> int:
    """hw_campaign.py ber: --config JSON with bers, realizations, sweeps, base_seed, persistence."""
    from .draw import derive_seed
    from models.snn_sota import get_loader
    cfg = json.load(open(args.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev).eval()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    data = EvalData(net, get_loader("SHD_DELAY", cfg.get("split", "test"), cfg.get("batch_size", 256),
                                    data_root=data_root), dev)
    mm = MemoryMap(net, args.hw_target, cfg.get("persistence", args.persistence))

    def seed_of(ber, r):
        return derive_seed("eval", cfg["base_seed"], f"{ber:.6g}", r)
    run_sweeps(net, data, mm, cfg.get("sweeps", ["joint"]), cfg.get("bers", ber_grid()),
               cfg.get("realizations", 100), seed_of, os.path.join(args.out, "realizations.csv"),
               os.path.join(args.out, "faults"), cfg.get("rates"), None, cfg.get("lam", 1.0))
    return 0
