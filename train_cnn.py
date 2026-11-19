#!/usr/bin/env python3
"""
Train any (non-spiking) network registered in networks.py on its dataset and save the checkpoint at
the path registered for it, so that main.py / ber_eval.py can load it.

    python3 train_cnn.py -n ResNet20 --epochs 200
    python3 train_cnn.py -n VGG9 --epochs 100 --lr 0.05

SGD with momentum, cosine learning rate, cross-entropy; the best test accuracy is kept.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from networks import NETWORKS, get_loader


@torch.no_grad()
def accuracy(net, loader, device):
    net.eval()
    correct = total = 0
    for x, y in loader:
        correct += int((net(x.to(device)).argmax(1).cpu() == y).sum())
        total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network-name", "-n", required=True, choices=list(NETWORKS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the weights path in networks.py")
    a = ap.parse_args()

    spec = NETWORKS[a.network_name]
    out = a.out or spec.weights
    if out is None:
        ap.error(f"{a.network_name} has no weights path in networks.py; pass --out")
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = spec.build(device).to(device)
    train = get_loader(a.network_name, "train", a.batch_size)
    test = get_loader(a.network_name, "test", 256)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = -1.0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for epoch in range(a.epochs):
        net.train()
        for x, y in train:
            opt.zero_grad()
            F.cross_entropy(net(x.to(device)), y.to(device)).backward()
            opt.step()
        sched.step()
        acc = accuracy(net, test, device)
        if acc > best:
            best = acc
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "test_acc": acc}, out)
        print(f"epoch {epoch + 1}/{a.epochs}: test {acc:.2f}% (best {best:.2f}%)", flush=True)
    print(f"saved {out} (test accuracy {best:.2f}%)")


if __name__ == "__main__":
    main()
