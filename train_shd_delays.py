#!/usr/bin/env python3
"""
Train SNN-Delays (Hammouamri, Khalfaoui-Hassani & Masquelier, ICLR 2024) on SHD and export the
SpikingJET inference model SHD_DELAY (models/snn_sota.py).

    pip install --no-deps dcls==0.1.1        # DCLS (learnable-spacing convolutions), training only;
                                             # already in the container built from faulty_snn.def
    python3 train_shd_delays.py --synthetic --epochs 2 --out /tmp/shd_delay_smoke.th   # smoke test
    python3 train_shd_delays.py                             # 150 epochs

Hyper-parameters are those of best_config_SHD.py in github.com/Thvnvtos/SNN-delays:
  2 hidden layers x 256 LIF, 140 inputs (700 channels in groups of 5), 10 ms steps, delays up to
  25 steps (250 ms), dropout 0.4, BN, batch 256, 150 epochs; Adam weights lr 1e-3 with one-cycle
  to 5e-3 (stepped per epoch), weight decay 1e-5 (0 on BN); Adam positions lr 0.1 with cosine
  annealing; Gaussian DCLS kernels whose sigma decays from 12 to 0.23 over the first 37 epochs;
  loss = cross-entropy on sum_t softmax(v_t) of the readout membrane.

Differences from the reference code, on purpose:
  * model selection on a 10 % validation split of the TRAINING set (seeded), not on the test set
    (the reference SHD loader passes the test set as validation, so its 95.07 % is the best test
    epoch); the test set is evaluated once, at the end. Expect a slightly lower number.
  * no spikingjelly: the LIF, multi-step BN and shared-mask dropout are re-implemented in
    snn_sota.py with the same equations (checked against the reference definitions).
Every epoch the model is converted to its inference form (positions rounded, one weight and one
integer delay per synapse, exactly what the reference evaluates with DCLS 'max' and sigma 0) and
that converted model is what gets validated, selected and saved.
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn_sota import SHD_DELAY, SHD_DELAY_KERNEL, SHD_DELAY_T, get_loader

CFG = dict(epochs=150, batch_size=256, lr_w=1e-3, max_lr_w=5e-3, lr_pos=0.1, weight_decay=1e-5,
           hidden=256, dropout=0.4, sig_init=SHD_DELAY_KERNEL // 2, sig_final=0.23)


class DclsSynapses(nn.Module):
    """Training form of DelayedSynapses: a Dcls1d layer (learnable weight + learnable continuous
    position per synapse), same padding and [B, T, C] interface."""

    def __init__(self, n_in, n_out, kernel=SHD_DELAY_KERNEL):
        super().__init__()
        try:
            from DCLS.construct.modules import Dcls1d
        except ImportError as e:
            raise SystemExit("train_shd_delays.py needs DCLS: use the container built from faulty_snn.def "
                             "(README, 'Setup') or run  pip install --no-deps dcls==0.1.1") from e
        self.kernel = kernel
        self.dcls = Dcls1d(n_in, n_out, kernel_count=1, groups=1, dilated_kernel_size=kernel,
                           bias=False, version="gauss")

    def forward(self, x):                                    # [B, T, n_in]
        xp = F.pad(x.transpose(1, 2), (self.kernel - 1, (self.kernel - 1) // 2))
        return self.dcls(xp).transpose(1, 2)

    @torch.no_grad()
    def rounded_kernel(self):
        """Dense kernel exactly as the reference evaluates: positions rounded, sigma 0, 'max'."""
        d = self.dcls
        P = d.P.detach().round()
        lim = self.kernel // 2
        P = P.clamp(-lim, lim)
        old = d.DCK.version
        d.DCK.version = "max"
        K = d.DCK(d.weight.detach(), P, torch.zeros_like(d.SIG))
        d.DCK.version = old
        return K                                             # [n_out, n_in, kernel]


def build_train_model(cfg):
    syn = (DclsSynapses(700 // 5, cfg["hidden"]), DclsSynapses(cfg["hidden"], cfg["hidden"]),
           DclsSynapses(cfg["hidden"], 20))
    m = SHD_DELAY(hidden=cfg["hidden"], dropout=cfg["dropout"], synapses=syn)
    lim = SHD_DELAY_KERNEL // 2
    for s in syn:
        nn.init.kaiming_uniform_(s.dcls.weight, nonlinearity="relu")
        nn.init.uniform_(s.dcls.P, a=-SHD_DELAY_KERNEL // 2, b=lim)
        s.dcls.clamp_parameters()
        nn.init.constant_(s.dcls.SIG, cfg["sig_init"])
        s.dcls.SIG.requires_grad = False
    return m, syn


@torch.no_grad()
def export_inference(train_model, syn, cfg, device):
    """Train model (DCLS) -> SHD_DELAY with one weight and one integer delay per synapse."""
    inf = SHD_DELAY(hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    for blk_t, blk_i, s in zip((train_model.block1, train_model.block2, train_model.block_out),
                               (inf.block1, inf.block2, inf.block_out), syn):
        K = s.rounded_kernel()
        k = K.abs().argmax(-1)
        blk_i.delay.syn.weight.copy_(K.gather(-1, k.unsqueeze(-1)).squeeze(-1))
        blk_i.delay.k_idx.copy_(k)
        if blk_t.bn is not None:
            blk_i.bn.load_state_dict(blk_t.bn.state_dict())
        blk_i.lif.load_state_dict(blk_t.lif.state_dict())
    return inf.eval()


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    c = n = 0
    for x, y in loader:
        out = net(x.to(device))
        c += (out.argmax(1).cpu() == y).sum().item()
        n += y.numel()
    return 100.0 * c / max(n, 1)


def synthetic_loader(n, bs, shuffle):
    g = torch.Generator().manual_seed(n)
    x = (torch.rand(n, SHD_DELAY_T, 140, generator=g) < 0.05).float()
    y = torch.randint(0, 20, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=bs, shuffle=shuffle)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="models/pretrained_models/SHD_DELAY.th")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--faults", default=None, help="experiment file with [[train.faults]] groups: faults during training (README)")
    p.add_argument("--val-speakers", type=int, nargs="*", default=None,
                   help="speaker-disjoint validation: these training speakers form the validation set "
                        "and are removed from training (default: random 10%% split)")
    p.add_argument("--select-split", choices=["val", "dev"], default="val",
                   help="split whose accuracy picks the saved epoch: val (default) or dev = official test "
                        "set without --val-speakers, so the held-out speakers stay unseen until the end; "
                        "with dev the held-out accuracy of the saved model is also reported")
    a = p.parse_args()
    if a.select_split == "dev" and not a.val_speakers:
        p.error("--select-split dev needs --val-speakers")
    cfg = {**CFG, "epochs": a.epochs, "batch_size": a.batch_size}
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.synthetic:
        tr, va, te = synthetic_loader(128, 32, True), synthetic_loader(64, 32, False), synthetic_loader(64, 32, False)
    else:
        tr = get_loader("SHD_DELAY", "train", cfg["batch_size"], seed=a.seed, data_root=a.data_root,
                        val_speakers=a.val_speakers)
        va = get_loader("SHD_DELAY", a.select_split, cfg["batch_size"], seed=a.seed, data_root=a.data_root,
                        val_speakers=a.val_speakers)
        te = get_loader("SHD_DELAY", "test", cfg["batch_size"], data_root=a.data_root)

    net, syn = build_train_model(cfg)
    net.to(dev)
    weights = [s.dcls.weight for s in syn]
    positions = [s.dcls.P for s in syn]
    bn = [p_ for blk in (net.block1, net.block2) for p_ in blk.bn.parameters()]
    opt_w = torch.optim.Adam([{"params": weights, "lr": cfg["lr_w"], "weight_decay": cfg["weight_decay"]},
                              {"params": bn, "lr": cfg["lr_w"], "weight_decay": 0.0}])
    opt_p = torch.optim.Adam(positions, lr=cfg["lr_pos"], weight_decay=0.0)
    sch_w = torch.optim.lr_scheduler.OneCycleLR(opt_w, max_lr=cfg["max_lr_w"], total_steps=cfg["epochs"])
    sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=cfg["epochs"])
    final_epoch = cfg["epochs"] // 4
    sig_alpha = (cfg["sig_final"] / cfg["sig_init"]) ** (1 / max(final_epoch, 1))
    n_w = sum(w.numel() for w in weights)
    print(f"SNN-Delays on SHD | {n_w:,} synaptic weights + {n_w:,} delay positions | {cfg} | device {dev}")

    from spikingjet.training import from_file, maybe
    tf = maybe(from_file(a.faults, net, [opt_w, opt_p], next(iter(tr))[0].to(dev), len(tr), a.seed))
    log_path = a.out.replace(".th", "_train_log.csv")
    best, best_ep = -1.0, 0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr_w", "lr_pos", "sigma", "seconds"])
        for ep in range(cfg["epochs"]):
            net.train()
            t0 = time.time()
            loss_sum = correct = total = 0
            for x, y in tr:
                x, y = x.to(dev), y.to(dev)
                tf.before_forward(ep)
                out = net(x)                                  # sum_t softmax(v_t)
                loss = F.cross_entropy(out, y)
                opt_w.zero_grad(set_to_none=True)
                opt_p.zero_grad(set_to_none=True)
                loss.backward()
                tf.after_backward()
                opt_w.step()
                opt_p.step()
                for s in syn:
                    s.dcls.clamp_parameters()
                tf.after_step()
                loss_sum += loss.item() * y.numel()
                total += y.numel()
                correct += (out.argmax(1) == y).sum().item()
            sch_w.step()
            sch_p.step()
            sig = syn[-1].dcls.SIG.detach().flatten()[0].item()
            if ep < final_epoch and sig > cfg["sig_final"]:
                with torch.no_grad():
                    for s in syn:
                        s.dcls.SIG *= sig_alpha

            inf = export_inference(net, syn, cfg, dev)       # the model that will be faulted
            val = evaluate(inf, va, dev)
            mark = ""
            if val > best:
                best, best_ep = val, ep + 1
                torch.save({"state_dict": inf.state_dict(), "epoch": ep + 1, "val_acc": val,
                            "config": cfg, "model_config": {}, "seed": a.seed,
                            "val_split": ({"val_speakers": list(a.val_speakers)} if a.val_speakers
                                          else {"random_fraction": 0.1, "seed": a.seed}),
                            "select_split": a.select_split}, a.out)
                mark = " [saved]"
            w.writerow([ep + 1, loss_sum / total, 100 * correct / total, val, sch_w.get_last_lr()[0],
                        sch_p.get_last_lr()[0], sig, time.time() - t0])
            f.flush()
            print(f"[{ep + 1:3d}/{cfg['epochs']}] loss {loss_sum / total:.4f} train {100 * correct / total:6.2f}% "
                  f"val {val:6.2f}%{mark}  sigma {sig:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    tf.close()
    ck = torch.load(a.out, map_location=dev)
    ck["faults_injected"] = tf.summary()
    inf = SHD_DELAY(hidden=cfg["hidden"], dropout=cfg["dropout"]).to(dev)
    inf.load_state_dict(ck["state_dict"], strict=True)
    ck["test_acc"] = evaluate(inf, te, dev)
    res = {"network": "SHD_DELAY", "best_epoch": best_ep, "val_acc": round(best, 2),
           "test_acc": round(ck["test_acc"], 2), "weights": a.out}
    if a.select_split == "dev" and not a.synthetic:
        ck["heldout_acc"] = evaluate(inf, get_loader("SHD_DELAY", "heldout", cfg["batch_size"],
                                                     data_root=a.data_root, val_speakers=a.val_speakers), dev)
        res["heldout_acc"] = round(ck["heldout_acc"], 2)
    torch.save(ck, a.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
