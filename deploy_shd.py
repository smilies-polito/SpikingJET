#!/usr/bin/env python3
"""
Convert a trained SHD_DELAY (FP32) into its deployment form (models/deployed_shd.py), the input of
hw_campaign.py: BatchNorm folded into the weights, membrane format calibrated on training data,
every stored quantity encoded in the formats of a precision profile (hwfault/profiles/*.toml).

    python3 deploy_shd.py --profile default --out models/pretrained_models/SHD_DELAY_deployed.th
"""
from __future__ import annotations

import argparse
import math

import torch

from hwfault import codec as C
from models.deployed_shd import LAYERS, DeployedSHD, delayed_current, dense_kernel, lif_scan
from models.snn_sota import SHD_DELAY, SHD_DELAY_KERNEL, get_loader


def load_shd_delay(path, device="cpu") -> SHD_DELAY:
    state = torch.load(path, map_location="cpu")
    net = SHD_DELAY(**dict(state.get("model_config", {})))
    net.load_state_dict({k.replace("module.", ""): v for k, v in state["state_dict"].items()}, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def fold(net: SHD_DELAY) -> list:
    """BatchNorm (eval statistics) folded into the delayed-synapse weights and a per-neuron bias."""
    layers = []
    for name in LAYERS:
        blk = getattr(net, name)
        w = blk.delay.syn.weight.detach().float().cpu().clone()
        d = (SHD_DELAY_KERNEL - 1 - blk.delay.k_idx.detach().cpu()).long()
        L = {"w": w, "delay": d, "beta": blk.lif.beta.detach().float().cpu().clone()}
        if blk.bn is not None:
            bn = blk.bn
            g = bn.weight.detach().float().cpu() / torch.sqrt(bn.running_var.float().cpu() + bn.eps)
            L["w"] = w * g[:, None]
            L["b"] = bn.bias.detach().float().cpu() - g * bn.running_mean.float().cpu()
            L["th"] = blk.lif.threshold.detach().float().cpu().clone()
        layers.append(L)
    return layers


@torch.no_grad()
def membrane_samples(layers, loader, device, n_batches):
    """|h| of every layer on a few batches, computed on the folded FP32 parameters."""
    lif = {"block1": "lif1", "block2": "lif2", "block_out": "lif_out"}
    dev_layers = [{k: v.to(device) for k, v in L.items()} for L in layers]
    samples = {n: [] for n in LAYERS}
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device).float()
        for name, L in zip(LAYERS, dev_layers):
            cur = delayed_current(x, dense_kernel(L["w"], L["delay"]))
            readout = "b" not in L
            if not readout:
                cur = cur + L["b"]
            rec = []
            s, _ = lif_scan(cur, L["beta"], L.get("th"), lambda h: h, lambda *a: None, lif[name],
                            readout=readout, record=rec)
            h = torch.stack(rec).flatten()
            samples[name].append(h[::max(1, h.numel() // 200_000)].cpu())
            x = s
    return samples


def calibrate_mem_frac(layers, loader, profile, device, n_batches=4):
    """Fraction bits per layer: the calibration percentile of |h| (and every bias / threshold) fits
    in the signed membrane word, frac = floor(log2(qmax / p))."""
    mf = profile["membrane"]
    if mf.kind != "fixed":
        return [None] * 3
    samples = membrane_samples(layers, loader, device, n_batches)
    fracs = []
    for n, L in zip(LAYERS, layers):
        h = torch.cat(samples[n]).float()
        p = float(torch.quantile(h[torch.randperm(h.numel())[:1_000_000]], mf.calibration_percentile / 100))
        for key in ("b", "th"):
            if key in L:
                p = max(p, float(L[key].abs().max()))
        fracs.append(int(math.floor(math.log2(mf.qmax / max(p, 1e-12)))))
    return fracs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/pretrained_models/SHD_DELAY.th", help="trained SHD_DELAY")
    ap.add_argument("--profile", default="default", help="precision profile in hwfault/profiles")
    ap.add_argument("--out", default="models/pretrained_models/SHD_DELAY_deployed.th")
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = C.load_profile(a.profile)
    layers = fold(load_shd_delay(a.checkpoint))
    fracs = calibrate_mem_frac(layers, get_loader("SHD_DELAY", "train", a.batch_size), profile, device,
                               a.calib_batches)
    net = DeployedSHD.from_float(layers, profile, fracs).to(device).eval()

    correct = total = 0
    with torch.no_grad():
        for x, y in get_loader("SHD_DELAY", "test", a.batch_size):
            correct += int((net(x.to(device).float()).argmax(1).cpu() == y).sum())
            total += y.numel()
    net.save(a.out, source=a.checkpoint, profile_name=a.profile)
    print(f"deployed ({a.profile}): test accuracy {100 * correct / total:.2f}% -> {a.out}")


if __name__ == "__main__":
    main()
