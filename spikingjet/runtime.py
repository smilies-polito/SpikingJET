"""Shared setup of every command: device, numerics, network, data, fault sites."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def device_of(cfg) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device = 'cuda' but CUDA is not available")
    if d == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return torch.device(d)


def numerics(cfg, deterministic=False):
    from numerics import set_tf32
    return set_tf32(bool(cfg.get("tf32", False)), deterministic=deterministic)


def require_network(cfg):
    from networks import NETWORKS
    name = cfg.get("network")
    if not name:
        raise SystemExit("no network: set `network = \"...\"` in the config or pass -n NAME")
    if name not in NETWORKS:
        raise SystemExit(f"unknown network {name!r}; registered: {', '.join(NETWORKS)}")
    return name


def example_batch(name, loader, device, synthetic=False):
    """A batch to probe the network's sites with: the first of the loader, or a random one."""
    from networks import synthetic_batch
    if synthetic or loader is None:
        x, _ = synthetic_batch(name, 2)
    else:
        x, _ = next(iter(loader))
    return x.to(device)


def sites_of(model, cfg, batch, with_activations=True):
    from .sites import discover
    s = cfg["sites"]
    return discover(model, batch, param_scope=s["param_scope"], activation_types=s["activation_types"],
                    with_activations=with_activations)
