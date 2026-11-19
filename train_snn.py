#!/usr/bin/env python3
"""
Train the SpikingJET-compatible SNN baselines (models/snn_sota.py).

    python3 train_snn.py --network NMNIST_CSNN            # ~40 epochs
    python3 train_snn.py --network SHD_RLIF               # ~150 epochs

The best epoch on a held-out 10 % of the TRAINING set is saved to
models/pretrained_models/<network>.th  ({'state_dict': ...}, the format load_from_dict expects).
The test set is touched only once, at the end, so the reported test accuracy is not tuned on it.
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from models.snn_sota import SNN_SOTA, get_loader

DEFAULTS = {
    "NMNIST_CSNN": dict(epochs=40, lr=2e-3, batch_size=64, weight_decay=1e-4),
    "SHD_RLIF": dict(epochs=150, lr=2e-3, batch_size=128, weight_decay=1e-5),
}


def evaluate(net, loader, device, rates=None, neuron_rates=None):
    """Accuracy (%). If `rates` is a dict and the model records spike rates (net.rates),
    the mean rate per layer over the loader is written into it; `neuron_rates` receives the
    per-neuron mean rates (net.layer_rates)."""
    net.eval()
    correct = total = 0
    acc_rates, acc_neuron, n = {}, {}, 0
    with torch.no_grad():
        for x, y in loader:
            out = net(x.to(device))
            correct += (out.argmax(1).cpu() == y).sum().item()
            total += y.numel()
            for k, v in getattr(net, "rates", {}).items():
                acc_rates[k] = acc_rates.get(k, 0.0) + float(v)
            for k, v in getattr(net, "layer_rates", {}).items():
                acc_neuron[k] = acc_neuron.get(k, 0.0) + v.detach().float().cpu()
            n += 1
    if rates is not None:
        rates.update({k: v / max(n, 1) for k, v in acc_rates.items()})
    if neuron_rates is not None:
        neuron_rates.update({k: v / max(n, 1) for k, v in acc_neuron.items()})
    return 100.0 * correct / max(total, 1)


def diagnostics(net, neuron_rates, grad_norms):
    """Per-epoch state of the things that can make an SNN collapse."""
    d = {}
    for name, m in net.named_modules():
        if hasattr(m, "beta") and isinstance(getattr(m, "beta"), torch.Tensor) and m.beta.dim() > 0:
            b, t = m.beta.detach().float(), m.threshold.detach().float()
            d[name] = {"beta": [round(float(b.min()), 3), round(float(b.mean()), 3), round(float(b.max()), 3)],
                       "thr": [round(float(t.min()), 3), round(float(t.mean()), 3), round(float(t.max()), 3)]}
    for name in ("rec1", "rec2"):
        if hasattr(net, name):
            W = getattr(net, name).weight.detach().float()
            d[name] = {"spectral_norm": round(float(torch.linalg.matrix_norm(W, ord=2)), 3),
                       "diag_mean": round(float(W.diagonal().mean()), 4)}
    for name in ("fc1", "fc2"):
        mod = getattr(net, name, None)
        if isinstance(mod, torch.nn.Sequential) and isinstance(mod[-1], torch.nn.BatchNorm1d):
            bn = mod[-1]
            d[name + ".bn"] = {"gamma_mean": round(float(bn.weight.detach().mean()), 3),
                               "beta_mean": round(float(bn.bias.detach().mean()), 3)}
    for k, r in neuron_rates.items():
        d.setdefault(k, {})["frac_saturated(>0.9)"] = round(float((r > 0.9).float().mean()), 3)
        d[k]["frac_silent(<0.001)"] = round(float((r < 0.001).float().mean()), 3)
        d[k]["rate_quartiles"] = [round(float(q), 3) for q in torch.quantile(r, torch.tensor([0.25, 0.5, 0.75]))]
    if grad_norms:
        g = torch.tensor(grad_norms)
        d["grad_norm"] = {"median": round(float(g.median()), 3), "max": round(float(g.max()), 3),
                          "frac_clipped": round(float((g > 1.0).float().mean()), 3)}
    return d


def clamp_neurons(net):
    """Keep learned neuron parameters in range: beta in the model's beta_range if it defines one
    (SHD_RLIF: time constants 5-25 steps), else [0, 0.999]; threshold >= 0.05."""
    lo, hi = getattr(net, "beta_range", (0.0, 0.999))
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith(".beta"):
                p.clamp_(lo, hi)
            elif name.endswith(".threshold"):
                p.clamp_(min=0.05)


def synthetic_loader(network, n, batch_size):
    shape = (10, 2, 34, 34) if network.startswith("NMNIST") else (100, 1, 700)
    classes = 10 if network.startswith("NMNIST") else 20
    x = (torch.rand((n,) + shape) < 0.05).float()
    y = torch.randint(0, classes, (n,))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True, choices=list(SNN_SOTA))
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--readout", choices=["mean_mem", "softmax"], default=None,
                   help="SHD_RLIF only: readout (default mean_mem). Saved in the checkpoint.")
    p.add_argument("--neuron", choices=["snntorch", "sparch"], default=None,
                   help="SHD_RLIF only: neuron update (default snntorch = round-1 model). Saved in the checkpoint.")
    p.add_argument("--bn-momentum", type=float, default=None, help="SHD_RLIF only (default 0.1)")
    p.add_argument("--rec-gain", type=float, default=None,
                   help="SHD_RLIF only: init scale of the recurrent weights (default 1.0)")
    p.add_argument("--freeze-beta", action="store_true",
                   help="do not train beta (it stays an nn.Parameter, so SpikingJET still injects into it)")
    p.add_argument("--freeze-threshold", action="store_true",
                   help="do not train the threshold (stays an nn.Parameter, still injectable)")
    p.add_argument("--rate-reg", type=float, default=None,
                   help="activity regulariser weight (per-neuron rate outside [--rate-min, --rate-max] "
                        "is penalised, summed over neurons). Default 0 (off; sparch's default is off too)")
    p.add_argument("--rate-min", type=float, default=0.01)
    p.add_argument("--rate-max", type=float, default=0.5)
    p.add_argument("--width", type=int, default=None,
                   help="NMNIST_CSNN only: hidden-width multiplier (default 1). Saved in the checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=None, help="default models/pretrained_models/<network>.th")
    p.add_argument("--synthetic", action="store_true", help="smoke test on random data, no dataset needed")
    a = p.parse_args()
    cfg = {**DEFAULTS[a.network], **{k: v for k, v in vars(a).items()
                                     if k in ("epochs", "lr", "batch_size", "weight_decay") and v is not None}}
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = a.out or f"models/pretrained_models/{a.network}.th"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if a.synthetic:
        tr = synthetic_loader(a.network, 64, cfg["batch_size"])
        va = te = synthetic_loader(a.network, 32, cfg["batch_size"])
    else:
        tr = get_loader(a.network, "train", cfg["batch_size"], seed=a.seed, data_root=a.data_root,
                        augment=not a.no_augment)
        va = get_loader(a.network, "val", cfg["batch_size"], seed=a.seed, data_root=a.data_root)
        te = get_loader(a.network, "test", cfg["batch_size"], data_root=a.data_root)

    model_config = {}
    if a.network == "SHD_RLIF":
        for k, v in (("readout", a.readout), ("neuron", a.neuron), ("bn_momentum", a.bn_momentum),
                     ("rec_gain", a.rec_gain)):
            if v is not None:
                model_config[k] = v
    if a.network == "NMNIST_CSNN" and a.width not in (None, 1):
        model_config["width"] = a.width
    net = SNN_SOTA[a.network](**model_config).to(device)
    rate_reg = a.rate_reg or 0.0
    frozen = []
    for n_, p_ in net.named_parameters():
        if (a.freeze_beta and n_.endswith(".beta")) or (a.freeze_threshold and n_.endswith(".threshold")):
            p_.requires_grad_(False); frozen.append(n_)
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=cfg["epochs"],
                                                steps_per_epoch=len(tr), pct_start=0.1)
    print(f"{a.network}: {sum(p.numel() for p in net.parameters()):,} parameters | {cfg} | "
          f"model {model_config or 'default'} | rate reg {rate_reg} in [{a.rate_min}, {a.rate_max}] | "
          f"frozen {len(frozen)} neuron parameter tensors | device {device}")

    log_path = out_path.replace(".th", "_train_log.csv")
    diag_path = out_path.replace(".th", "_diag.jsonl")
    diag_f = open(diag_path, "w")
    best, best_epoch = -1.0, 0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "seconds", "val_spike_rates"])
        for ep in range(1, cfg["epochs"] + 1):
            net.train(); t0 = time.time()
            loss_sum = correct = total = 0
            grad_norms = []
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                out = net(x)
                loss = F.cross_entropy(out, y)
                if rate_reg > 0 and getattr(net, "layer_rates", None):
                    # sparch activity regulariser: push silent neurons up, bursting neurons down
                    reg = sum(F.relu(a.rate_min - r).sum() + F.relu(r - a.rate_max).sum()
                              for r in net.layer_rates.values())
                    loss = loss + rate_reg * reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norms.append(float(torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in net.parameters() if p_.requires_grad], 1.0)))
                opt.step(); sched.step(); clamp_neurons(net)
                loss_sum += F.cross_entropy(out.detach(), y).item() * y.numel(); total += y.numel()
                correct += (out.argmax(1) == y).sum().item()
            rates, neuron_rates = {}, {}
            val = evaluate(net, va, device, rates, neuron_rates)
            diag_f.write(json.dumps({"epoch": ep, "train_acc": round(100 * correct / total, 2),
                                     "val_acc": round(val, 2), **diagnostics(net, neuron_rates, grad_norms)}) + "\n")
            diag_f.flush()
            rate_txt = " ".join(f"{k}={v:.3f}" for k, v in rates.items())
            marker = ""
            if val > best:
                best, best_epoch = val, ep
                torch.save({"state_dict": net.state_dict(), "epoch": ep, "val_acc": val, "config": cfg,
                            "model_config": model_config}, out_path)
                marker = " [saved]"
            w.writerow([ep, loss_sum / total, 100 * correct / total, val, sched.get_last_lr()[0],
                        time.time() - t0, rate_txt])
            f.flush()
            print(f"[{ep:3d}/{cfg['epochs']}] loss {loss_sum/total:.4f} train {100*correct/total:6.2f}% "
                  f"val {val:6.2f}%{marker}  rates {rate_txt}  ({time.time()-t0:.0f}s)")

    diag_f.close()
    ckpt = torch.load(out_path, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    test = evaluate(net, te, device)
    ckpt["test_acc"] = test
    torch.save(ckpt, out_path)
    print(json.dumps({"network": a.network, "best_epoch": best_epoch, "val_acc": round(best, 2),
                      "test_acc": round(test, 2), "weights": out_path}, indent=2))


if __name__ == "__main__":
    main()
