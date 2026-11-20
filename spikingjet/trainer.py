"""Generic training of any registered network (SNN, CNN or mixed), with faults during training.

    python spikingjet.py train --config experiments/train_fat.toml

Cross-entropy on the network scores, the optimizer / scheduler of [train], the fault groups of
[[train.faults]] (spikingjet.training). The best test accuracy is kept and saved with the config
and the number of injected faults; [train] eval_bers then measures the accuracy of the trained
network under weight bit-flips (robustness of fault-aware training).
Specialised recipes: train_snn.py, train_shd_delays.py (also accept --faults).
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .ber import evaluate
from .faults import ParamBER
from .runtime import device_of, numerics, require_network
from .sites import param_sites
from .training import TrainingFaults


def make_optimizer(params, t):
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    raise SystemExit(f"train.optimizer {t['optimizer']!r}: use sgd | adam | adamw")


def make_scheduler(opt, t, steps_per_epoch):
    if t["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] * steps_per_epoch)
    if t["scheduler"] == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], epochs=t["epochs"],
                                                   steps_per_epoch=steps_per_epoch, pct_start=0.1)
    if t["scheduler"] == "none":
        return None
    raise SystemExit(f"train.scheduler {t['scheduler']!r}: use cosine | onecycle | none")


def weight_ber_robustness(net, loader, device, bers, realizations, seed=0):
    """Mean accuracy under random weight bit-flips at each BER (same maps as ber_eval.py)."""
    from .injector import Injector
    from .sites import SiteTable
    sites = SiteTable(param_sites(net), [], [])
    inj = Injector(net, sites, hook_activations=False)
    pop = ParamBER(sites.params)
    res = {}
    try:
        for ber in bers:
            accs = []
            for r in range(realizations):
                inj.apply_param_arrays(pop.sample(ber, "all", ParamBER.rng_for(seed, "all", ber, r, False)))
                try:
                    accs.append(evaluate(net, loader, device)[0])
                finally:
                    inj.restore_params()
            res[ber] = float(np.mean(accs))
    finally:
        inj.detach()
    return res


def run(cfg, synthetic=False):
    from networks import NETWORKS, checkpoint_path, get_loader, load_network, synthetic_batch
    numerics(cfg)
    name = require_network(cfg)
    t = cfg["train"]
    torch.manual_seed(cfg["seed"])
    device = device_of(cfg)
    spec = NETWORKS[name]
    out = cfg["out"] or checkpoint_path(name)
    if t["init"] == "checkpoint":
        net = load_network(name, device, cfg["checkpoint"] or None)
    elif t["init"] == "scratch":
        if getattr(spec.build, "loads_weights", False):
            raise SystemExit(f"{name} is built from its checkpoint: use train.init = 'checkpoint' to fine-tune "
                             f"it, or its own recipe (train_snn.py / train_shd_delays.py) to train from scratch")
        net = spec.build(device).to(device)
    else:
        raise SystemExit(f"train.init {t['init']!r}: use scratch | checkpoint")
    if out == checkpoint_path(name) and t["init"] == "checkpoint" and not cfg["out"]:
        out = os.path.splitext(out)[0] + "_finetuned.th"         # never overwrite the starting weights silently
    if synthetic:
        x, y = synthetic_batch(name, 4 * cfg["batch_size"])
        ds = torch.utils.data.TensorDataset(x.transpose(0, 1) if spec.example.time_first else x, y)
        tr = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
        te = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"])
        if spec.example.time_first:                        # back to [T, B, ...] per batch
            tr = [(a.transpose(0, 1), b) for a, b in tr]
            te = [(a.transpose(0, 1), b) for a, b in te]
    else:
        tr = get_loader(name, "train", cfg["batch_size"])
        te = get_loader(name, "test", t["eval_batch_size"])
    opt = make_optimizer([p for p in net.parameters() if p.requires_grad], t)
    sched = make_scheduler(opt, t, len(tr))
    x0 = next(iter(tr))[0].to(device)
    tf = TrainingFaults(cfg, net, opt, x0, steps_per_epoch=len(tr))
    print(f"{name}: {sum(p.numel() for p in net.parameters()):,} parameters | {t['optimizer']} lr {t['lr']} "
          f"| {t['epochs']} epochs | device {device}\nfaults during training:\n{tf.describe()}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log_path = os.path.splitext(out)[0] + "_train_log.csv"
    best = -1.0
    try:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_acc", "seconds"]
                       + [f"faults_group{G.i}" for G in tf.groups])
            for ep in range(t["epochs"]):
                net.train()
                t0, loss_sum, correct, total = time.time(), 0.0, 0, 0
                for x, y in tr:
                    x, y = x.to(device), y.to(device)
                    tf.before_forward(ep)
                    out_ = net(x)
                    loss = F.cross_entropy(out_, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    tf.after_backward()
                    if t["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), t["grad_clip"])
                    opt.step()
                    tf.after_step()
                    if sched is not None:
                        sched.step()
                    n = y.numel()
                    loss_sum += float(loss.detach()) * n if torch.isfinite(loss) else 0.0
                    correct += int((out_.argmax(1) == y).sum())
                    total += n
                acc, _ = evaluate(net, te, device)
                counts = tf.epoch_counts(ep)
                w.writerow([ep, loss_sum / total, 100 * correct / total, acc, round(time.time() - t0, 1)]
                           + [counts.get(G.i, 0) for G in tf.groups])
                f.flush()
                mark = ""
                if acc > best:
                    best = acc
                    torch.save({"state_dict": net.state_dict(), "epoch": ep, "test_acc": acc, "config": cfg,
                                "faults_injected": tf.summary()}, out)
                    mark = " [saved]"
                print(f"[{ep + 1:3d}/{t['epochs']}] loss {loss_sum / total:.4f} train {100 * correct / total:6.2f}% "
                      f"test {acc:6.2f}%{mark}  faults {counts or 0}  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        tf.close()
    print(f"saved {out} (best test accuracy {best:.2f}%), log {log_path}")
    print(f"faults injected during training: {tf.summary()}")
    if t["eval_bers"]:
        ckpt = torch.load(out, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        rob = weight_ber_robustness(net, te, device, t["eval_bers"], t["eval_realizations"], cfg["seed"])
        print("accuracy under weight bit-flips: " + ", ".join(f"BER {b:.0e} {a:.2f}%" for b, a in rob.items()))
        ckpt["robustness"] = rob
        torch.save(ckpt, out)
    return best
