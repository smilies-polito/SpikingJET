"""Bit-error-rate sweep: many random faults at once, accuracy vs BER, for any fault groups.

Each [[ber.faults]] group picks a target (params / neurons / activations), sites, bits and fault
model; the swept BER applies to every group. For each (BER, realization) the whole test set runs
with all groups active:
    params        persistence "permanent": one fault map per realization (the stored bits are wrong)
    neurons /     "transient": fresh faults at every timestep / call;
    activations   "permanent": one map per realization, the same positions in every sample and step
Parameter maps are drawn like ber_eval.py, so the same seed gives the same map for any network
with the same parameter shapes (paired comparison). Output: one CSV row per (BER, realization).
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from .bitops import resolve_bits
from .faults import ParamBER
from .injector import Injector
from .runtime import device_of, example_batch, numerics, require_network, sites_of
from .sites import select


@torch.no_grad()
def evaluate(model, loader, device, inj=None):
    """(accuracy %, % of samples with a non-finite output); a non-finite output is never correct."""
    model.eval()
    correct = nonfinite = total = 0
    ctx = inj.enabled() if inj is not None else torch.no_grad()
    with ctx:
        for x, y in loader:
            out = model(x.to(device))
            bad = ~torch.isfinite(out).all(dim=1)
            pred = out.argmax(1).cpu()
            pred[bad.cpu()] = -1
            correct += int((pred == y).sum())
            nonfinite += int(bad.sum())
            total += y.numel()
    return 100.0 * correct / total, 100.0 * nonfinite / total


def _groups(cfg, sites):
    groups = cfg["ber"]["faults"] or [dict(target="params", select=["*"], exclude=[], fault_model="bit-flip",
                                           bits="all", persistence="permanent")]
    out = []
    for i, g in enumerate(groups):
        if g["target"] not in ("params", "neurons", "activations"):
            raise SystemExit(f"ber.faults[{i}].target {g['target']!r}: grads / optimizer exist only in training")
        chosen = select(sites.by_target(g["target"]), g["select"], g["exclude"])
        if not chosen:
            raise SystemExit(f"ber.faults[{i}]: no {g['target']} sites match {g['select']}")
        pers = g["persistence"]
        if g["target"] == "params":
            pers = "permanent"                       # a parameter fault stays for the whole test set
        elif pers == "upset":
            raise SystemExit(f"ber.faults[{i}]: persistence 'upset' exists only in training")
        out.append(dict(g, sites=chosen, persistence=pers, bits=resolve_bits(g["bits"])))
    return out


def run(cfg):
    from networks import get_loader, load_network
    numerics(cfg)
    name = require_network(cfg)
    b = cfg["ber"]
    device = device_of(cfg)
    net = load_network(name, device, cfg["checkpoint"] or None)
    loader = get_loader(name, "test", cfg["batch_size"])
    sites = sites_of(net, cfg, example_batch(name, loader, device))
    groups = _groups(cfg, sites)
    inj = Injector(net, sites, hook_activations=any(g["target"] == "activations" for g in groups))
    out = cfg["out"] or f"ber_results/{name}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    desc = " + ".join(f"{g['target']}:{g['fault_model']}:{'/'.join(g['select'])}:{g['persistence']}" for g in groups)
    clean, _ = evaluate(net, loader, device)
    print(f"{name}: clean {clean:.2f}% | groups: {desc}")
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["network", "groups", "ber", "realization", "n_param_faults", "n_state_faults",
                        "clean_acc", "faulty_acc", "nonfinite_pct", "seconds"])
        try:
            for ber in b["bers"]:
                for r in range(b["realizations"]):
                    t0 = time.time()
                    n_param = 0
                    inj.clear_states()
                    states = []
                    for gi, g in enumerate(groups):
                        stuck = g["fault_model"] == "stuck-at"
                        extra = () if gi == 0 else (gi,)
                        rng = ParamBER.rng_for(cfg["seed"], g["bits"], ber, r, stuck, extra)
                        if g["target"] == "params":
                            arrays = ParamBER(g["sites"]).sample(ber, g["bits"], rng, stuck)
                            inj.apply_param_arrays(arrays)
                            n_param += sum(len(a[1]) for a in arrays)
                        else:
                            states.append(inj.add_state_ber(g["sites"], g["bits"], ber, stuck, rng, g["persistence"]))
                    try:
                        acc, nf = evaluate(net, loader, device, inj)
                    finally:
                        inj.restore_params()
                    n_state = sum(s.count for s in states)
                    w.writerow([name, desc, ber, r, n_param, n_state, f"{clean:.4f}", f"{acc:.4f}", f"{nf:.4f}",
                                f"{time.time() - t0:.1f}"])
                    f.flush()
                print(f"  BER {ber:.0e}: last realization {acc:6.2f}% ({n_param} parameter faults, "
                      f"{n_state} state faults, {nf:.1f}% non-finite)")
        finally:
            inj.detach()
    after, _ = evaluate(net, loader, device)
    if abs(after - clean) > 1e-9:
        raise SystemExit(f"ERROR: clean accuracy changed after the sweep ({clean} -> {after}): restore failed")
    print(f"results -> {out}")
