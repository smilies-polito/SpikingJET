"""
Network registry: the ONLY file to edit to run SpikingJET on a new network.

Each entry maps a name (the value of --network-name) to
    build    function(device) -> torch.nn.Module       (the architecture, untrained)
    weights  path of the checkpoint to load, or None (use the weights set by `build`)
    dataset  key of DATASETS (the data the campaign runs on)
    example  optional Example(shape, classes, time_first): the shape of ONE input sample and the
             number of classes. Used for runs on random data (ber_eval.py --synthetic, the tests),
             so a new network can be checked before its dataset is downloaded.

Each DATASETS entry is a function(split, batch_size) -> DataLoader, split in {"train", "test"}.

Adding a network
    1. put the model class in models/<your_model>.py
    2. add an entry below, e.g.
           "MyCNN": Net(lambda d: MyCNN(), "models/pretrained_models/MyCNN.th", "cifar10",
                        Example((3, 32, 32), 10, spikes=False)),
    3. run  python verify_snn.py --network MyCNN, then python main.py -n MyCNN --use-cuda

A model works with every tool as long as forward(x) returns one score vector per sample,
shape [batch, classes]. For neuron-state faults (SNNs only) see "Neuron faults" in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


# ----------------------------------------------------------------------------------------------
# Datasets: name -> function(split, batch_size) -> DataLoader
# ----------------------------------------------------------------------------------------------
def _cifar10(split, batch_size):
    from models.utils import load_CIFAR10_datasets
    train, _, test = load_CIFAR10_datasets(train_batch_size=batch_size, test_batch_size=batch_size)
    return train if split == "train" else test


def _dvsgesture(split, batch_size):
    from models.utils import load_DVSGesture_test_dataset
    return load_DVSGesture_test_dataset(batch_size)


def _nmnist_legacy(split, batch_size):
    from models.utils import load_NMNIST_test_dataset
    return load_NMNIST_test_dataset(batch_size)


def _snn_sota(network):
    def load(split, batch_size):
        from models.snn_sota import get_loader
        return get_loader(network, split, batch_size)
    return load


DATASETS: dict[str, Callable] = {
    "cifar10": _cifar10,                     # downloaded automatically into ./data
    "dvsgesture": _dvsgesture,               # tonic, ./data
    "nmnist": _nmnist_legacy,                # tonic, ./data   (legacy NMNIST model)
    "nmnist_csnn": _snn_sota("NMNIST_CSNN"),
    "shd_rlif": _snn_sota("SHD_RLIF"),
    "shd_delay": _snn_sota("SHD_DELAY"),
}


# ----------------------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------------------
@dataclass
class Example:
    shape: tuple              # one sample, without the batch dimension
    classes: int
    time_first: bool = False  # True: the model takes [T, B, ...] (time dimension first)
    spikes: bool = True       # True: random binary events; False: Gaussian values (images)


@dataclass
class Net:
    build: Callable[[torch.device], torch.nn.Module]
    weights: Optional[str]
    dataset: str
    example: Optional[Example] = None


def _resnet(depth):
    def build(device):
        from models import resnet
        return getattr(resnet, f"resnet{depth}")()
    return build


def _sota(name):
    """SNNs trained with train_snn.py / train_shd_delays.py: the checkpoint also stores the
    constructor arguments, so build() loads it directly (weights=None below)."""
    def build(device, path=None):
        import os
        from models.snn_sota import build_snn_sota
        path = path or f"{PRE}/{name}.th"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: train it first (README.md, 'Train a network')")
        return build_snn_sota(name, device, path)
    build.loads_weights = True
    return build


def _cls(module, name, *args):
    def build(device):
        import importlib
        return getattr(importlib.import_module(f"models.{module}"), name)(*args)
    return build


PRE = "models/pretrained_models"
CIFAR = Example((3, 32, 32), 10, spikes=False)
NETWORKS: dict[str, Net] = {
    # --- SNNs (snntorch) ---------------------------------------------------------------------
    "CSNN":        Net(_cls("CSNN", "CSNN"), f"{PRE}/CSNN.th", "dvsgesture",
                       Example((8, 2, 128, 128), 11, time_first=True)),
    "NMNIST":      Net(_cls("NMNIST", "NMNIST"), f"{PRE}/NMNIST.th", "nmnist",
                       Example((8, 2, 34, 34), 10, time_first=True)),
    "NMNIST_CSNN": Net(_sota("NMNIST_CSNN"), None, "nmnist_csnn", Example((10, 2, 34, 34), 10)),
    "SHD_RLIF":    Net(_sota("SHD_RLIF"), None, "shd_rlif", Example((100, 1, 700), 20)),
    "SHD_DELAY":   Net(_sota("SHD_DELAY"), None, "shd_delay", Example((150, 140), 20)),
    # --- mixed: conventional conv front end + spiking back end ---------------------------------
    "NMNIST_HYBRID": Net(_cls("hybrid", "HybridNMNIST"), f"{PRE}/NMNIST_HYBRID.th", "nmnist_csnn",
                         Example((10, 2, 34, 34), 10)),
    # --- CNNs on CIFAR-10 (train first: python train_cnn.py -n ResNet20) ----------------------
    "ResNet20":    Net(_resnet(20), f"{PRE}/ResNet20.th", "cifar10", CIFAR),
    "ResNet32":    Net(_resnet(32), f"{PRE}/ResNet32.th", "cifar10", CIFAR),
    "ResNet44":    Net(_resnet(44), f"{PRE}/ResNet44.th", "cifar10", CIFAR),
    "ResNet56":    Net(_resnet(56), f"{PRE}/ResNet56.th", "cifar10", CIFAR),
    "ResNet18":    Net(_cls("resnet18", "ResNet18"), f"{PRE}/ResNet18.th", "cifar10", CIFAR),
    "VGG9":        Net(_cls("vgg9", "VGG9"), f"{PRE}/VGG9.th", "cifar10", CIFAR),
}


def load_network(name: str, device: torch.device, weights: Optional[str] = None) -> torch.nn.Module:
    """Build the network, load its weights (the registered checkpoint, or `weights`), move it to
    `device`, eval mode."""
    if name not in NETWORKS:
        raise KeyError(f"unknown network {name!r}; registered: {', '.join(NETWORKS)}")
    spec = NETWORKS[name]
    if getattr(spec.build, "loads_weights", False):         # the builder reads the checkpoint itself
        return spec.build(device, weights).to(device).eval()
    net = spec.build(device)
    path = weights or spec.weights
    if path is not None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: train the network first "
                                    f"(README.md, 'Train a network') or fix its path in networks.py")
        from models.utils import load_from_dict
        load_from_dict(network=net, device=device, path=path)
    return net.to(device).eval()


def get_loader(name: str, split: str, batch_size: int):
    """DataLoader of the dataset registered for network `name`."""
    return DATASETS[NETWORKS[name].dataset](split, batch_size)


def checkpoint_path(name: str) -> str:
    """Checkpoint file of network `name` (the weights path, or the one the builder reads)."""
    return NETWORKS[name].weights or f"{PRE}/{name}.th"


def synthetic_batch(name: str, batch_size: int, seed: int = 0):
    """(x, y): a random batch in the input format of network `name` (its `example`)."""
    ex = NETWORKS[name].example
    if ex is None:
        raise ValueError(f"{name} has no `example` in networks.py: add Example(shape, classes) to its entry")
    g = torch.Generator().manual_seed(seed)
    shape = (batch_size,) + tuple(ex.shape)
    x = (torch.rand(shape, generator=g) < 0.05).float() if ex.spikes else torch.randn(shape, generator=g)
    if ex.time_first:
        x = x.transpose(0, 1).contiguous()
    return x, torch.randint(0, ex.classes, (batch_size,), generator=g)
