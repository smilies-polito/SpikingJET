"""
State-of-the-art-style SNN baselines for SpikingJET: N-MNIST and SHD.

    NMNIST_CSNN : convolutional LIF network, T = 10 frames of 34x34x2.
                  Literature for comparable conv SNNs: 99.2-99.6 % (PLIF, ICCV 2021: 99.61 %).
    SHD_RLIF    : two recurrent LIF layers (256) + leaky-integrator readout, 100 frames x 700 channels.
                  Literature for recurrent LIF without adaptation: ~89.8-92.5 % (Bittar & Garner 2022).

SpikingJET compatibility rules this file follows
------------------------------------------------
1. Every injectable block is a TOP-LEVEL attribute (fc1, lif1, ...). FaultListGenerator keeps a
   parameter only if the first component of its name belongs to a top-level module that is, or
   contains, a Conv2d / Linear / snn.Leaky / snn.Synaptic. BatchNorm is therefore grouped with its
   Conv/Linear inside one nn.Sequential (conv1 = [Conv2d, BatchNorm2d]) so its parameters are
   faulted too, exactly as in the CNNs.
2. beta and threshold are nn.Parameters (learn_beta / learn_threshold = True). Buffers are not in
   named_parameters() and would never receive faults.
3. Every spiking layer calls  self._inj(spk, mem, '<layer name>', t)  at every time step, with the
   arguments in the order FaultInjector.injection(spike, pot, layer, step) expects (spikingjet.injector).
   The layer name is the attribute name (no dots), so '<name>.potential' / '<name>.spike' parse.
4. FaultInjector may be None (training): the hook is skipped.
5. forward() returns [batch, classes] scores; argmax is the prediction.
6. Train and test use the SAME preprocessing (get_loader below), so the clean accuracy measured
   by SpikingJET equals the one measured after training.
"""

from __future__ import annotations

import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

NMNIST_T = 10      # frames per N-MNIST sample (PLIF used T = 10)
SHD_T = 100        # frames per SHD sample (10 ms bins over ~1 s)
SHD_CHANNELS = 700


class _SNNBase(nn.Module):
    num_steps = None

    def __init__(self):
        super().__init__()
        self.FaultInjector = None

    def _inj(self, spk, mem, name, t):
        if self.FaultInjector is not None:
            self.FaultInjector.injection(spk, mem, name, t)


def _leaky(shape, beta0, *, output=False, reset="subtract", grad=None):
    """LIF with per-neuron (or per-channel) learnable beta and threshold."""
    return snn.Leaky(beta=torch.full(shape, beta0), threshold=torch.ones(shape),
                     learn_beta=True, learn_threshold=True,
                     spike_grad=grad or surrogate.atan(), reset_mechanism=reset, output=output)


# ============================================================================
# Layer-wise helpers
# ============================================================================
# SHD_RLIF: BatchNorm statistics are computed over (batch x time) jointly: every feed-forward
# block (Conv/Linear + BN) runs ONCE on the whole sequence, flattened to [B*T, ...],
# and only the LIF recurrence is unrolled in time, layer by layer.
#
# Why: the first version applied BN inside the time loop, one call per step. In
# training each step was normalised with its own batch statistics, and the running
# statistics (momentum 0.1, 100 updates per batch on SHD) ended up describing the LAST
# time steps only, which on SHD are near-silent. In eval mode the active middle of an
# utterance was then divided by a ~16x too small variance: the network saturated and
# validation stayed at chance while training accuracy rose (SHD_RLIF log, Oct 2026).
# Joint (B*T) statistics are the standard choice for BN in SNNs/RNNs on SHD
# (e.g. Bittar & Garner 2022, Front. Neurosci.).
# NMNIST_CSNN keeps the per-step BN it was trained with: N-MNIST activity is spread evenly over
# its 10 frames, so per-step and joint statistics agree, and its eval accuracy matched training.
#
# Fault injection is unchanged: every LIF layer still calls self._inj(spk, mem, name, t)
# at every step, in place, BEFORE its spikes are used by the next step or the next layer.


def _ff(block, x):
    """Apply a feed-forward block to every time step at once. x: [B, T, ...]."""
    B, T = x.shape[:2]
    y = block(x.reshape(B * T, *x.shape[2:]))
    return y.reshape(B, T, *y.shape[1:])


# ============================================================================
# N-MNIST
# ============================================================================
class NMNIST_CSNN(_SNNBase):
    """[Conv3x3(32)-BN-LIF-MaxPool2] -> [Conv3x3(64)-BN-LIF-MaxPool2] -> FC256-LIF -> FC10-LI readout.
    Input: [B, T, 2, 34, 34] event frames."""
    num_steps = NMNIST_T

    def __init__(self, beta0: float = 0.7, dropout: float = 0.2, width: int = 1):
        # width: multiplier of every hidden width (1 = 32/64/256, the published model; 2, 4 = wider baselines)
        super().__init__()
        c1, c2, h = 32 * width, 64 * width, 256 * width
        self.conv1 = nn.Sequential(nn.Conv2d(2, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1))
        self.lif1 = _leaky((c1, 1, 1), beta0)
        self.conv2 = nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2))
        self.lif2 = _leaky((c2, 1, 1), beta0)
        self.fc1 = nn.Linear(c2 * 8 * 8, h)
        self.lif3 = _leaky((h,), beta0)
        self.fc2 = nn.Linear(h, 10)
        self.lif_out = _leaky((10,), beta0, reset="none")   # leaky integrator: membrane = class score
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float()                                 # [B, T, 2, 34, 34]
        m1 = m2 = m3 = mo = None
        out = 0.0
        for t in range(x.shape[1]):
            c = self.conv1(x[:, t])
            s1, m1 = self.lif1(c, torch.zeros_like(c) if m1 is None else m1)
            self._inj(s1, m1, "lif1", t)
            c = self.conv2(F.max_pool2d(s1, 2))
            s2, m2 = self.lif2(c, torch.zeros_like(c) if m2 is None else m2)
            self._inj(s2, m2, "lif2", t)
            c = self.fc1(self.drop(F.max_pool2d(s2, 2).flatten(1)))
            s3, m3 = self.lif3(c, torch.zeros_like(c) if m3 is None else m3)
            self._inj(s3, m3, "lif3", t)
            c = self.fc2(self.drop(s3))
            so, mo = self.lif_out(c, torch.zeros_like(c) if mo is None else mo)
            self._inj(so, mo, "lif_out", t)
            out = out + mo
        return out / x.shape[1]                       # logits; argmax = prediction


# ============================================================================
# SHD
# ============================================================================
SHD_TAU_RANGE = (5.0, 25.0)          # sparch time-constant range, in steps (Bittar & Garner 2022)
SHD_BETA_RANGE = (math.exp(-1 / SHD_TAU_RANGE[0]), math.exp(-1 / SHD_TAU_RANGE[1]))   # ~(0.82, 0.96)


class SHD_RLIF(_SNNBase):
    """[FC700-BN + recurrent FC] -> LIF(256) -> [FC-BN + recurrent FC] -> LIF(256) -> FC20 -> LI readout.
    Input: [B, T, 1, 700] (tonic ToFrame) or [B, T, 700].

    neuron:
      'snntorch' : u_t = beta*u_{t-1} + I_t - s*thr, beta = 0.905 for all neurons, kept in
                   [0, 0.999]. This is the configuration of the 68%-val run (round 1).
      'sparch'   : u_t = beta*u_{t-1} + (1-beta)*I_t - s*thr  (input AND recurrent current scaled
                   by 1-beta, as in the sparch code of Bittar & Garner 2022), beta drawn per neuron
                   in SHD_BETA_RANGE and kept there. The 1-beta factor keeps the membrane on the
                   scale of the input whatever tau is; without it the drive grows ~1/(1-beta)
                   (x10 at tau=10, x25 at tau=25).
    readout:
      'mean_mem' : time-averaged readout membrane, used as logits.
      'softmax'  : log of the time-averaged per-step softmax (sparch). Returns log-probabilities;
                   argmax is the prediction and cross_entropy on them is the NLL.
    rec_gain:
      scale of the orthogonal initialisation of rec1/rec2 (default 1.0 = rounds 1-3). With 1.0
      the gradient explodes through the recurrence over the 100 steps: on SHD-like input the
      gradient norm of fc1 was 8 / 36 / 335 / ~2e5 at T = 10 / 25 / 50 / 100, against 33 at
      T = 100 without recurrence; gain 0.5 gives ~130, 0.25 ~40. Exploding BPTT is the most
      likely cause of the collapses seen in every SHD run so far.
    All options are stored in the checkpoint ('model_config') and rebuilt by build_snn_sota.
    Parameter names/shapes are identical for every option.
    """
    num_steps = SHD_T

    def __init__(self, hidden: int = 256, dropout: float = 0.1, readout: str = "mean_mem",
                 neuron: str = "snntorch", bn_momentum: float = 0.1, rec_gain: float = 1.0):
        super().__init__()
        if readout not in ("mean_mem", "softmax") or neuron not in ("snntorch", "sparch"):
            raise ValueError((readout, neuron))
        self.readout, self.neuron = readout, neuron
        self.beta_range = SHD_BETA_RANGE if neuron == "sparch" else (0.0, 0.999)

        def lif(n, reset="subtract"):
            if neuron == "sparch":
                lo, hi = SHD_BETA_RANGE
                beta = lo + (hi - lo) * torch.rand(n)
            else:
                beta = torch.full((n,), math.exp(-1 / 10))
            return snn.Leaky(beta=beta, threshold=torch.ones(n), learn_beta=True, learn_threshold=True,
                             spike_grad=surrogate.atan(), reset_mechanism=reset)

        self.fc1 = nn.Sequential(nn.Linear(SHD_CHANNELS, hidden), nn.BatchNorm1d(hidden, momentum=bn_momentum))
        self.rec1 = nn.Linear(hidden, hidden, bias=False)
        self.lif1 = lif(hidden)
        self.fc2 = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden, momentum=bn_momentum))
        self.rec2 = nn.Linear(hidden, hidden, bias=False)
        self.lif2 = lif(hidden)
        self.fc_out = nn.Linear(hidden, 20)
        self.lif_out = lif(20, reset="none")
        self.drop = nn.Dropout(dropout)
        nn.init.orthogonal_(self.rec1.weight, gain=rec_gain)
        nn.init.orthogonal_(self.rec2.weight, gain=rec_gain)
        self.rates, self.layer_rates = {}, {}

    def _gain(self, lif):
        return (1.0 - lif.beta.clamp(0.0, 1.0)) if self.neuron == "sparch" else 1.0

    def _recurrent(self, lif, rec, cur, name):
        """Recurrent LIF layer: input current cur [B, T, H] plus rec(previous spikes)."""
        spk = cur.new_zeros(cur.shape[0], rec.in_features)
        mem = torch.zeros_like(cur[:, 0])
        g = self._gain(lif)
        out = []
        for t in range(cur.shape[1]):
            spk, mem = lif(g * (cur[:, t] + rec(spk)), mem)
            self._inj(spk, mem, name, t)
            out.append(spk)
        out = torch.stack(out, 1)
        self.layer_rates[name] = out.mean((0, 1))          # per neuron, with grad (optional regulariser)
        self.rates[name] = self.layer_rates[name].detach().mean()
        return out

    def forward(self, x):
        x = x.float()
        if x.dim() == 4:
            x = x[:, :, 0, :]                                       # [B, T, 700]
        s1 = self._recurrent(self.lif1, self.rec1, _ff(self.fc1, x), "lif1")
        s2 = self._recurrent(self.lif2, self.rec2, _ff(self.fc2, self.drop(s1)), "lif2")
        cur = _ff(self.fc_out, self.drop(s2))
        g = self._gain(self.lif_out)
        mem = torch.zeros_like(cur[:, 0])
        acc = 0.0
        for t in range(cur.shape[1]):
            spk, mem = self.lif_out(g * cur[:, t], mem)
            self._inj(spk, mem, "lif_out", t)
            acc = acc + (F.softmax(mem, dim=1) if self.readout == "softmax" else mem)
        acc = acc / cur.shape[1]
        return torch.log(acc + 1e-12) if self.readout == "softmax" else acc


# ============================================================================
# SHD, delay model (SNN-Delays, Hammouamri, Khalfaoui-Hassani & Masquelier, ICLR 2024)
# ============================================================================
# Reported 95.07 % on SHD (best test accuracy over epochs; the reference code uses the test set as
# its validation set). The 96.04 % SHD result (Zhang et al., "Temporal Information Reconstruction
# and Non-Aligned Residual in SNNs for Speech Classification") is this model plus two add-on
# modules. Architecture (best_config_SHD.py of github.com/Thvnvtos/SNN-delays):
#   input 700 channels summed in groups of 5 -> 140, 10 ms frames, zero-padded to 150 frames
#   (the reference pads to the longest sample of each batch; see _fixed_length_collate)
#   2 x [delayed synapses 140/256 -> 256 (one weight + one delay per synapse, delays 0..24 steps)
#        -> BatchNorm over batch x time -> LIF -> dropout 0.4 (same mask for all time steps)]
#   -> delayed synapses 256 -> 20 -> non-spiking LIF readout, output = sum_t softmax(v_t)
# LIF = spikingjelly LIFNode(tau = 1.005 steps, decay_input=False, v_reset = 0 (hard reset),
#   detach_reset, ATan(alpha = 5)):  H_t = (1 - 1/tau) V_{t-1} + X_t ;  S_t = [H_t >= 1] ;
#   V_t = H_t (1 - S_t). With tau = 1.005 the membrane keeps 0.5 % of its value: the memory of the
#   network is in the synaptic delays, not in the neurons.
#
# This class is the INFERENCE form used by SpikingJET. Training uses DCLS (learnable continuous
# delay positions, train_shd_delays.py); after training the positions are rounded and every
# synapse becomes (weight, integer delay), exactly as the reference code evaluates. Weights are an
# nn.Linear (faulted by SpikingJET like any other layer); delays are an integer buffer, which the
# standard campaigns do not fault.

SHD_DELAY_BINS = 5            # 700 -> 140 input channels
SHD_DELAY_DT_US = 10_000      # 10 ms frames
SHD_DELAY_KERNEL = 25         # max_delay = 250 ms / 10 ms -> 25 (odd)
SHD_DELAY_TAU = 1.005         # (10.05 ms + 1e-9) / 10 ms, spikingjelly convention
SHD_DELAY_T = 150             # every sample zero-padded to 1.5 s (see _fixed_length_collate)


class DelayedSynapses(nn.Module):
    """y[b, t, o] = sum_i W[o, i] * xpad[b, t + k[o, i], i]: one weight and one integer delay
    (d = KERNEL - 1 - k) per synapse. Same padding and output length (T + (KERNEL-1)//2) as the
    reference Dcls1d layers, so the converted network reproduces the trained one."""

    def __init__(self, n_in: int, n_out: int, kernel: int = SHD_DELAY_KERNEL):
        super().__init__()
        self.kernel = kernel
        self.syn = nn.Linear(n_in, n_out, bias=False)
        self.register_buffer("k_idx", torch.zeros(n_out, n_in, dtype=torch.long))

    def dense_kernel(self):
        W = self.syn.weight
        return W.new_zeros(W.shape[0], W.shape[1], self.kernel).scatter(
            2, self.k_idx.unsqueeze(-1), W.unsqueeze(-1))

    def forward(self, x):                                   # x: [B, T, n_in]
        xp = F.pad(x.transpose(1, 2), (self.kernel - 1, (self.kernel - 1) // 2))
        return F.conv1d(xp, self.dense_kernel()).transpose(1, 2)   # [B, T + 12, n_out]


class MultiStepLIF(nn.Module):
    """spikingjelly-equivalent LIF (see above). beta and threshold are nn.Parameters that are not
    trained (requires_grad=False) but are faulted by SpikingJET, as in the other SNNs here."""

    def __init__(self, n: int, tau: float = SHD_DELAY_TAU, v_th: float = 1.0, alpha: float = 5.0,
                 readout: bool = False):
        super().__init__()
        self.beta = nn.Parameter(torch.full((n,), 1.0 - 1.0 / tau), requires_grad=False)
        self.threshold = nn.Parameter(torch.full((n,), v_th), requires_grad=False)
        self.readout = readout
        self.spike_fn = surrogate.atan(alpha=alpha)

    def forward(self, cur, inj, name):                      # cur: [B, T, n]
        v = torch.zeros_like(cur[:, 0])
        spikes, mems = [], []
        for t in range(cur.shape[1]):
            h = self.beta * v + cur[:, t]
            if self.readout:
                s = torch.zeros_like(h)
                v = h
            else:
                s = self.spike_fn(h - self.threshold)
                v = h * (1.0 - s.detach())
            inj(s, v, name, t)
            spikes.append(s)
            mems.append(v)
        return torch.stack(spikes, 1), torch.stack(mems, 1)


class SharedDropout(nn.Module):
    """Dropout with one mask per (sample, neuron) shared by all time steps (spikingjelly
    multi-step dropout)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):                                   # [B, T, n]
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full_like(x[:, :1], 1 - self.p)) / (1 - self.p)
        return x * mask


class DelayBlock(nn.Module):
    """delayed synapses -> BatchNorm (batch x time) -> LIF -> dropout. One top-level module per
    layer, so SpikingJET's fault population includes its weights, BN and neuron parameters."""

    def __init__(self, n_in, n_out, dropout, readout=False, synapses=None):
        super().__init__()
        self.delay = synapses if synapses is not None else DelayedSynapses(n_in, n_out)
        self.bn = None if readout else nn.BatchNorm1d(n_out)
        self.lif = MultiStepLIF(n_out, v_th=1e9 if readout else 1.0, readout=readout)
        self.drop = None if readout else SharedDropout(dropout)

    def forward(self, x, inj, name):
        c = self.delay(x)                                   # [B, T', n_out]
        if self.bn is not None:
            c = _ff(self.bn, c)                             # statistics over batch x time
        s, v = self.lif(c, inj, name)
        return (s if self.drop is None else self.drop(s)), v


class SHD_DELAY(_SNNBase):
    """Inference form of SNN-Delays for SHD (block comment above). forward() returns
    sum_t softmax(v_t) of the readout membrane, [B, 20]; argmax is the prediction (the reference
    training loss is cross-entropy applied to this sum)."""
    num_steps = SHD_DELAY_T + (SHD_DELAY_KERNEL - 1) // 2   # steps of lif1 (lif2 +12, lif_out +24);
                                                            # only used by transient campaigns

    def __init__(self, hidden: int = 256, dropout: float = 0.4, n_in: int = 700 // SHD_DELAY_BINS,
                 synapses=(None, None, None)):
        super().__init__()
        self.block1 = DelayBlock(n_in, hidden, dropout, synapses=synapses[0])
        self.block2 = DelayBlock(hidden, hidden, dropout, synapses=synapses[1])
        self.block_out = DelayBlock(hidden, 20, 0.0, readout=True, synapses=synapses[2])

    def forward(self, x):
        x = x.float()
        if x.dim() == 4:
            x = x[:, :, 0, :]
        s, _ = self.block1(x, self._inj, "lif1")
        s, _ = self.block2(s, self._inj, "lif2")
        _, v = self.block_out(s, self._inj, "lif_out")
        return F.softmax(v, dim=2).sum(1)


# ============================================================================
# Data (shared by training and by SpikingJET)
# ============================================================================
def get_loader(network: str, split: str, batch_size: int, val_fraction: float = 0.1,
               seed: int = 0, data_root: str = "./data", augment: bool = False, val_speakers=None):
    """split in {'train', 'val', 'test'}. 'train'/'val' split the official training set
    (SHD and N-MNIST have no official validation set). drop_last only for training.
    val_speakers (SHD_DELAY only): validation = every training sample of these speakers, train =
    the other speakers (speaker-disjoint, like the SHD test set); default None = random split.
    SHD_DELAY with val_speakers also offers 'dev' (official test set without these speakers) and
    'heldout' (every sample of these speakers, training and test files)."""
    if network == "SHD_DELAY":
        return _shd_delay_loader(split, batch_size, val_fraction, seed, data_root, val_speakers)
    if val_speakers:
        raise ValueError(f"val_speakers is only supported for SHD_DELAY, not {network}")
    import tonic
    import tonic.transforms as tt

    if network.startswith("NMNIST"):
        sensor = tonic.datasets.NMNIST.sensor_size
        tf = tt.Compose([tt.Denoise(filter_time=10000), tt.ToFrame(sensor_size=sensor, n_time_bins=NMNIST_T)])
        ds = tonic.datasets.NMNIST(save_to=data_root, train=(split != "test"), transform=tf)
        cache = f"{data_root}/cache/nmnist_T{NMNIST_T}/{'test' if split == 'test' else 'train'}"
    else:
        sensor = tonic.datasets.hsd.SHD.sensor_size
        tf = tt.Compose([tt.ToFrame(sensor_size=sensor, n_time_bins=SHD_T)])
        ds = tonic.datasets.hsd.SHD(save_to=data_root, train=(split != "test"), transform=tf)
        cache = f"{data_root}/cache/shd_T{SHD_T}/{'test' if split == 'test' else 'train'}"
    ds = tonic.DiskCachedDataset(ds, cache_path=cache)

    if split in ("train", "val"):
        g = torch.Generator().manual_seed(seed)
        n_val = int(len(ds) * val_fraction)
        tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
        ds = tr if split == "train" else va

    collate = _augment_collate(network) if (augment and split == "train") else None
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                       drop_last=(split == "train"), num_workers=4,
                                       collate_fn=collate)


def _bin_shd_frames(frames):
    """(T, 1, 700) 10 ms frames -> (T, 140): sum of groups of 5 neighbouring channels."""
    f = np.asarray(frames, dtype=np.float32)
    f = f.reshape(f.shape[0], -1)                            # (T, 700)
    return f.reshape(f.shape[0], -1, SHD_DELAY_BINS).sum(-1)


_SHD_TRUNCATED = [0]


def _fixed_length_collate(batch):
    """Zero-pad (at the end) every sample to SHD_DELAY_T frames -> [B, SHD_DELAY_T, 140].
    The reference pads to the longest sample of each batch instead; with a fixed length a sample's
    prediction does not depend on which samples share its batch, so clean and faulty runs, and runs
    with different batch sizes, are comparable sample by sample. SHD utterances are shorter than
    1.4 s, so nothing should be cut; a warning is printed if something is."""
    x = torch.zeros(len(batch), SHD_DELAY_T, 700 // SHD_DELAY_BINS)
    for i, b in enumerate(batch):
        f = torch.as_tensor(np.asarray(b[0]), dtype=torch.float32)
        if f.shape[0] > SHD_DELAY_T:
            _SHD_TRUNCATED[0] += 1
            print(f"WARNING SHD_DELAY: sample with {f.shape[0]} frames cut to {SHD_DELAY_T} "
                  f"({_SHD_TRUNCATED[0]} so far)")
        f = f[:SHD_DELAY_T]
        x[i, :f.shape[0]] = f
    return x, torch.as_tensor([int(b[1]) for b in batch])


def shd_speakers(data_root, train=True):
    """Speaker id of every sample of the SHD training (or test) file, in dataset order."""
    import h5py
    f = os.path.join(data_root, "SHD", "shd_train.h5" if train else "shd_test.h5")
    with h5py.File(f, "r") as h:
        return np.asarray(h["extra"]["speaker"])


POOLED = [-1]                  # val_speakers = [-1]: pooled split, every speaker in every split
POOLED_FRACTIONS = {"val": 0.08, "dev": 0.10, "heldout": 0.10}   # train = the rest (72 %)
POOLED_SEED = 0               # fixed: the same split for every training seed


def shd_labels(data_root, train=True):
    """Class label of every sample of the SHD training (or test) file, in dataset order."""
    import h5py
    f = os.path.join(data_root, "SHD", "shd_train.h5" if train else "shd_test.h5")
    with h5py.File(f, "r") as h:
        return np.asarray(h["labels"])


def pooled_indices(data_root):
    """{split: indices into ConcatDataset([training file, test file])}: both SHD files merged and split per
    (speaker, class) group with POOLED_SEED: heldout 10 %, dev 10 %, val 8 %, train the rest (rounded per group)."""
    spk = np.concatenate([shd_speakers(data_root, True), shd_speakers(data_root, False)])
    lab = np.concatenate([shd_labels(data_root, True), shd_labels(data_root, False)])
    rng = np.random.RandomState(POOLED_SEED)
    out = {k: [] for k in ("train", "val", "dev", "heldout")}
    for key in sorted(set(zip(spk.tolist(), lab.tolist()))):
        idx = np.flatnonzero((spk == key[0]) & (lab == key[1]))
        idx = idx[rng.permutation(len(idx))]
        a = 0
        for k in ("heldout", "dev", "val"):
            n = int(round(POOLED_FRACTIONS[k] * len(idx)))
            out[k] += idx[a:a + n].tolist()
            a += n
        out["train"] += idx[a:].tolist()
    return {k: sorted(v) for k, v in out.items()}


def _shd_delay_loader(split, batch_size, val_fraction, seed, data_root, val_speakers=None):
    """SNN-Delays preprocessing with tonic: 10 ms frames on a grid starting at the first event,
    last partial frame kept (include_incomplete=True; same frames as spikingjelly's SHD
    'duration' integration used by the reference), 700 -> 140 channels, zero-padded to
    SHD_DELAY_T frames. val_speakers = POOLED ([-1]): train / val / dev / heldout of pooled_indices
    ('test' stays the official test file)."""
    if val_speakers is not None and [int(v) for v in val_speakers] == POOLED and split != "test":
        full = torch.utils.data.ConcatDataset([_shd_delay_loader("train", 1, 0.0, 0, data_root).dataset,
                                               _shd_delay_loader("test", 1, 0.0, 0, data_root).dataset])
        ds = torch.utils.data.Subset(full, pooled_indices(data_root)[split])
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                           drop_last=(split == "train"), num_workers=4,
                                           collate_fn=_fixed_length_collate)
    import tonic
    import tonic.transforms as tt
    sensor = tonic.datasets.hsd.SHD.sensor_size
    tf = tt.Compose([tt.ToFrame(sensor_size=sensor, time_window=SHD_DELAY_DT_US,
                                 include_incomplete=True), _bin_shd_frames])
    test_file = split in ("test", "dev")             # dev: test file without val_speakers
    ds = tonic.datasets.hsd.SHD(save_to=data_root, train=not test_file, transform=tf)
    ds = tonic.DiskCachedDataset(ds, cache_path=f"{data_root}/cache/shd_dt10ms_bin5_v2/"
                                                 f"{'test' if test_file else 'train'}")
    T0 = np.asarray(ds[0][0]).shape[0]
    if not 20 <= T0 <= 400:      # SHD utterances last ~0.5-1.4 s -> ~50-140 frames of 10 ms
        raise RuntimeError(f"SHD_DELAY loader: first sample has {T0} frames of 10 ms; check the "
                           f"time unit of tonic's SHD events (expected microseconds)")
    if split in ("dev", "heldout"):
        if not val_speakers:
            raise ValueError(f"split {split!r} needs val_speakers")
        held = [int(v) for v in val_speakers]
        te = ds if split == "dev" else _shd_delay_loader("test", 1, val_fraction, seed, data_root).dataset
        te_held = np.isin(shd_speakers(data_root, train=False), held)
        if split == "dev":
            ds = torch.utils.data.Subset(te, np.flatnonzero(~te_held).tolist())
        else:
            tr_held = np.isin(shd_speakers(data_root), held)
            ds = torch.utils.data.ConcatDataset([torch.utils.data.Subset(ds, np.flatnonzero(tr_held).tolist()),
                                                 torch.utils.data.Subset(te, np.flatnonzero(te_held).tolist())])
    elif split in ("train", "val") and val_speakers:
        spk = shd_speakers(data_root)
        if len(spk) != len(ds):
            raise RuntimeError(f"SHD speaker list ({len(spk)}) does not match the dataset ({len(ds)})")
        held = np.isin(spk, [int(v) for v in val_speakers])
        if not held.any():
            raise ValueError(f"no SHD training sample from speakers {list(val_speakers)}")
        ds = torch.utils.data.Subset(ds, np.flatnonzero(held if split == "val" else ~held).tolist())
    elif split in ("train", "val"):
        g = torch.Generator().manual_seed(seed)
        n_val = int(len(ds) * val_fraction)
        tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
        ds = tr if split == "train" else va
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                       drop_last=(split == "train"), num_workers=4,
                                       collate_fn=_fixed_length_collate)


def _augment_collate(network):
    """Light augmentation on frames: random spatial shift (N-MNIST) / channel shift (SHD)."""
    def collate(batch):
        x = torch.stack([torch.as_tensor(b[0]) for b in batch]).float()
        y = torch.as_tensor([b[1] for b in batch])
        if network.startswith("NMNIST"):
            dx, dy = torch.randint(-2, 3, (2,)).tolist()
            x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        else:
            k = int(torch.randint(-20, 21, (1,)))           # shift frequency channels, no wrap-around
            x = torch.roll(x, shifts=k, dims=-1)
            if k > 0:
                x[..., :k] = 0
            elif k < 0:
                x[..., k:] = 0
        return x, y
    return collate


# ============================================================================
# Registry used by utils.load_network and main.py
# ============================================================================
SNN_SOTA = {
    "NMNIST_CSNN": NMNIST_CSNN,
    "SHD_RLIF": SHD_RLIF,
    "SHD_DELAY": SHD_DELAY,
}


def build_snn_sota(network_name: str, device, path=None):
    path = path or f"models/pretrained_models/{network_name}.th"
    state = torch.load(path, map_location=device)
    net = SNN_SOTA[network_name](**state.get("model_config", {}))   # e.g. {'readout': 'softmax'}
    sd = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    net.load_state_dict(sd, strict=True)          # strict: a wrong class fails loudly
    return net.to(device).eval()
