"""Single-fault campaign: one fault at a time, every fault on every test batch.

Outputs (same layout as SpikingJET always had, so analyse.py / pt_to_csv.py keep working):
    output/fault_list/<net>/<seed>_fault_list.csv                       the faults
    output/fault_list/<net>/<seed>_fault_list_result_batch_size_<b>.csv per-site summary (analyse.py table)
    output/<net>/pt/{label,clean,faulty}/batch_size_<b>/*.pt            raw scores (campaign.save_outputs)
The summary is computed during the run; analyse.py recomputes it from the .pt files.
"""
from __future__ import annotations

import os
import time

import torch
from tqdm import tqdm

from .faults import read_fault_list, single_fault_list, write_fault_list
from .injector import Fault, Injector
from .metrics import LayerSummary
from .runtime import device_of, example_batch, numerics, require_network, sites_of
from .sites import select


def run(cfg):
    from networks import get_loader, load_network
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    print(f"numerics: {numerics(cfg)}")
    name = require_network(cfg)
    c = cfg["campaign"]
    device = device_of(cfg)
    bs = cfg["batch_size"]
    net = load_network(name, device, cfg["checkpoint"] or None)
    loader = get_loader(name, "test", bs)
    with_acts = "activations" in c["targets"]
    sites = sites_of(net, cfg, example_batch(name, loader, device), with_activations=with_acts)
    population = select([s for t in c["targets"] for s in sites.by_target(t)], c["select"], c["exclude"])
    if not population:
        raise SystemExit(f"no fault sites for targets {c['targets']} / select {c['select']}")
    print(f"{name}: {len(population)} fault sites, device {device}")

    out_root = cfg["out"] or "output"
    seed = c["seed"]
    list_dir = f"{out_root}/fault_list/{name}"
    if c["fault_list"]:
        faults = read_fault_list(c["fault_list"])
        print(f"fault list loaded: {c['fault_list']} ({len(faults)} faults)")
    else:
        n_steps = getattr(net, "num_steps", None) or 25
        faults = single_fault_list(population, seed, c["bits"], c["error_margin"], c["confidence"], c["p"],
                                   c["max_faults"] or None, n_steps, c["neuron_persistence"] == "transient")
        write_fault_list(f"{list_dir}/{seed}_fault_list.csv", faults)
        print(f"fault list: {len(faults)} faults -> {list_dir}/{seed}_fault_list.csv")

    stuck = c["fault_model"] == "stuck-at"
    dirs = {k: f"{out_root}/{name}/pt/{k}/batch_size_{bs}" for k in ("label", "clean", "faulty")}
    if c["save_outputs"]:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

    inj = Injector(net, sites, hook_activations=with_acts)
    summary = LayerSummary()
    kinds = {s.name: s.kind for s in sites.all()}
    t0 = time.time()
    changed = total = 0
    try:
        with torch.no_grad():
            pbar = tqdm(total=len(loader) * len(faults), desc="Fault injection campaign", colour="green")
            for b, (x, y) in enumerate(loader):
                x = x.to(device)
                with inj.enabled(False):
                    clean = net(x).cpu()
                if c["save_outputs"]:
                    torch.save(clean, f"{dirs['clean']}/batch_{b}.pt")
                    torch.save(y, f"{dirs['label']}/batch_{b}.pt")
                for i, f in enumerate(faults):
                    f = Fault(f.site, f.index, f.bit, f.value if stuck else -1, f.step)
                    if kinds.get(f.site) == "param":
                        inj.set_param_faults([f])
                    else:
                        inj.set_state_faults([f])
                    try:
                        with inj.enabled():
                            faulty = net(x).cpu()
                    finally:
                        inj.restore_params()
                        inj.clear_states()
                    if c["save_outputs"]:
                        torch.save(faulty, f"{dirs['faulty']}/fault_{i}_batch_{b}.pt")
                    summary.add(f.site, i, faulty, clean, y)
                    changed += int((faulty.argmax(1) != clean.argmax(1)).sum())
                    total += y.numel()
                    pbar.set_postfix({"changed": f"{100 * changed / max(total, 1):.4f}%"})
                    pbar.update(1)
            pbar.close()
    finally:
        inj.detach()
    table = summary.table()
    path = f"{list_dir}/{seed}_fault_list_result_batch_size_{bs}.csv"
    os.makedirs(list_dir, exist_ok=True)
    table.to_csv(path, index=True)
    print(table)
    print(f"per-site results -> {path}  ({time.time() - t0:.0f} s)")
    return table
