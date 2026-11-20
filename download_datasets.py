#!/usr/bin/env python3
"""
Download the datasets of the registered networks (networks.py) into ./data.

    python3 download_datasets.py                    # test sets of every dataset (enough for campaigns)
    python3 download_datasets.py --train            # also the training sets (needed to train)
    python3 download_datasets.py shd dvsgesture     # only some datasets
    python3 download_datasets.py --list             # what is available and what is already on disk

Datasets (sizes of the downloaded archives, test / train):
    shd         Spiking Heidelberg Digits        38 MB / 131 MB   SHD_DELAY, SHD_RLIF
                (the training set is also needed by deploy_shd.py for calibration, so it is always fetched)
    nmnist      Neuromorphic MNIST               170 MB / 1 GB    NMNIST, NMNIST_CSNN
    dvsgesture  IBM DVS128 Gesture               0.6 GB / 2.3 GB  CSNN
    cifar10     CIFAR-10 (train and test)        170 MB           ResNet*, VGG9

Every file comes from the official source used by tonic / torchvision and is checked against its
MD5. DVSGesture is fetched through the figshare public API (api.figshare.com): figshare's
direct links used by tonic answer automated clients with an empty page, which tonic reports as
"File not found or corrupted". The data loaders call ensure() themselves, so running this script
first is optional: it only lets you download everything up front (e.g. before going offline).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

DATA_ROOT = "./data"
DATASETS = ("shd", "nmnist", "dvsgesture", "cifar10")


# ----------------------------------------------------------------------------------------------
# DVSGesture through the figshare API
# ----------------------------------------------------------------------------------------------
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()  # nosec B324 - integrity check against the published checksum, not security
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url, dest, md5):
    """Download url to dest (resuming a partial file), then check the MD5."""
    if os.path.exists(dest) and _md5(dest) == md5:
        print(f"  {dest}: already present, checksum OK")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310 - fixed https URL
        if done and r.status != 206:                    # server ignored the range: start again
            done = 0
        total = done + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f:
            got, last = done, -1
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
                pct = int(100 * got / total) if total else 0
                if pct != last:
                    print(f"\r  {os.path.basename(dest)}: {got / 1e6:,.0f} / {total / 1e6:,.0f} MB ({pct}%)",
                          end="", flush=True)
                    last = pct
    print()
    if _md5(part) != md5:
        os.remove(part)
        raise RuntimeError(f"{dest}: checksum mismatch after download (file removed); run again")
    os.replace(part, dest)


def _figshare_api(url):
    """https://figshare.com/ndownloader/files/<id>  ->  https://api.figshare.com/v2/file/download/<id>"""
    return "https://api.figshare.com/v2/file/download/" + url.rstrip("/").rsplit("/", 1)[-1]


def ensure_dvsgesture(root=DATA_ROOT, train=False):
    import tonic
    D = tonic.datasets.DVSGesture
    folder = os.path.join(root, "DVSGesture")
    for split in (["test", "train"] if train else ["test"]):
        filename, url, md5 = getattr(D, f"{split}_filename"), getattr(D, f"{split}_url"), getattr(D, f"{split}_md5")
        extracted = os.path.join(folder, filename.replace(".tar.gz", ""))
        if os.path.isdir(extracted) and os.listdir(extracted):
            continue
        print(f"DVSGesture {split}:")
        _fetch(_figshare_api(url), os.path.join(folder, filename), md5)
        D(save_to=root, train=(split == "train"))       # tonic extracts the verified archive


# ----------------------------------------------------------------------------------------------
# The others: their libraries download them
# ----------------------------------------------------------------------------------------------
def ensure_shd(root=DATA_ROOT, train=True):
    import tonic
    for t in (False, True):                              # train is small and deploy_shd.py needs it
        tonic.datasets.hsd.SHD(save_to=root, train=t)


def ensure_nmnist(root=DATA_ROOT, train=False):
    import tonic
    for t in ([False, True] if train else [False]):
        tonic.datasets.NMNIST(save_to=root, train=t)


def ensure_cifar10(root=DATA_ROOT, train=True):
    from torchvision.datasets import CIFAR10
    CIFAR10(root, train=False, download=True)            # one archive holds both splits


ENSURE = {"shd": ensure_shd, "nmnist": ensure_nmnist, "dvsgesture": ensure_dvsgesture,
          "cifar10": ensure_cifar10}

ON_DISK = {"shd": ["SHD/shd_test.h5", "SHD/shd_train.h5"],
           "nmnist": ["NMNIST/Test", "NMNIST/Train"],
           "dvsgesture": ["DVSGesture/ibmGestureTest", "DVSGesture/ibmGestureTrain"],
           "cifar10": ["cifar-10-batches-py"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {', '.join(DATASETS)}, all (default: all)")
    ap.add_argument("--train", action="store_true", help="also download the training sets")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--list", action="store_true", help="show what is already downloaded and exit")
    a = ap.parse_args(argv)
    unknown = set(a.datasets) - set(DATASETS) - {"all"}
    if unknown:
        ap.error(f"unknown dataset(s) {', '.join(sorted(unknown))}; choose from {', '.join(DATASETS)}, all")
    names = list(DATASETS) if not a.datasets or "all" in a.datasets else a.datasets
    if a.list:
        for n in DATASETS:
            parts = ", ".join(f"{p} {'yes' if os.path.exists(os.path.join(a.root, p)) else 'no'}"
                              for p in ON_DISK[n])
            print(f"{n:<11} {parts}")
        return 0
    for n in names:
        print(f"== {n}")
        ENSURE[n](a.root, train=a.train)
    print(f"done: {', '.join(names)} in {a.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
