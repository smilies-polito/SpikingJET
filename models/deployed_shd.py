"""
Deployment form of SHD_DELAY.

The network holds what a memory-based neuromorphic accelerator would store, as integer CODES in
the formats of a precision profile (hwfault/profiles):

    layer      quantity    shape (default)        memory class
    blockX     w_code      [out, in]              M1 synapse memory (weights)
               d_code      [out, in]              M1 synapse memory (delays, d = 0..24 steps)
               b_code      [out]  (hidden only)   M2 neuron configuration (folded BatchNorm)
               th_code     [1] shared / [out]     M2 (hidden only; the readout never spikes)
               leak_code   [1] shared / [out]     M2
    lifX       membrane    [B, out] per step      M3 neuron state (fixed point, saturating)
    lifX       spike       [B, out] per step      M4 spike routing

Forward pass (per layer): build the dense delay kernel from the weight codes and
clamp(delay, 0, 24), conv1d (same padding and output length as SHD_DELAY) in the code domain,
multiply by the weight scale (exact: integer inputs times small integer codes), add the bias,
then the LIF recurrence in the membrane format:
    h_t = leak * v_{t-1} + I_t          (quantized to the membrane format, saturating)
    s_t = [h_t > threshold]             (hidden layers; the readout is a non-spiking accumulator)
    v_t = h_t * (1 - s_t)               (hard reset, as SHD_DELAY)
The injection hook  FaultInjector.injection(s_t, v_t, '<lif name>', t)  is called after every
update, exactly as in SHD_DELAY, so SpikingJET-style injectors work unchanged. The output is
sum_t softmax(v_t) of the readout membrane.

The width of the hidden layers is a constructor argument (grown networks use the same class).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwfault import codec as C
from models.snn_sota import SHD_DELAY_KERNEL

LAYERS = ("block1", "block2", "block_out")
LIF_NAMES = {"block1": "lif1", "block2": "lif2", "block_out": "lif_out"}
DEFAULT_WIDTHS = (140, 256, 256, 20)


def delay_used(d: torch.Tensor) -> torch.Tensor:
    """Stored delay value -> delay applied: rounded and saturated to 0..24 (NaN -> 24)."""
    d = torch.nan_to_num(d.float(), nan=float(C.MAX_DELAY), posinf=float(C.MAX_DELAY), neginf=0.0)
    return torch.round(d).clamp(0, C.MAX_DELAY).long()


def dense_kernel(w: torch.Tensor, d: torch.Tensor, kernel: int = SHD_DELAY_KERNEL) -> torch.Tensor:
    """[out, in] weights + integer delays -> [out, in, kernel]; tap k = kernel-1-d (SHD_DELAY k_idx)."""
    k = (kernel - 1 - d).clamp(0, kernel - 1)
    return w.new_zeros(*w.shape, kernel).scatter(2, k.unsqueeze(-1), w.unsqueeze(-1))


def delayed_current(x: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
    """x [B, T, in] -> [B, T + (K-1)//2, out], identical padding to DelayedSynapses."""
    K = kern.shape[-1]
    xp = F.pad(x.transpose(1, 2), (K - 1, (K - 1) // 2))
    return F.conv1d(xp, kern).transpose(1, 2)


def lif_scan(cur, leak, th, qmem, inj, name, readout=False, spike_fn=None, record=None):
    """LIF recurrence shared by the deployment form, the FAT training form and calibration.
    cur [B, T, n]; leak, th broadcastable to [B, n], or callables t -> tensor; qmem(h) quantizes to
    the membrane format
    (identity for float); spike_fn(h - th) returns spikes (default: Heaviside, strict >).
    record: optional list receiving |h| per step (calibration)."""
    v = torch.zeros_like(cur[:, 0])
    spikes, mems = [], []
    leak_t = leak if callable(leak) else (lambda t: leak)       # callables: time-varying values
    th_t = th if callable(th) else (lambda t: th)              # (transient M2 faults)
    for t in range(cur.shape[1]):
        h = qmem(leak_t(t) * v + cur[:, t])
        if record is not None:
            record.append(h.detach().abs())
        if readout:
            s = torch.zeros_like(h)
            v = h
        else:
            thr = th_t(t)
            s = (h - thr > 0).to(h.dtype) if spike_fn is None else spike_fn(h - thr)
            v = h * (1.0 - s.detach())
        inj(s, v, name, t)
        spikes.append(s)
        mems.append(v)
    return torch.stack(spikes, 1), torch.stack(mems, 1)


class DeployedSHD(nn.Module):
    """SHD_DELAY in deployment form; see the module docstring."""

    def __init__(self, profile: C.Profile | str = "default", widths=DEFAULT_WIDTHS, mem_frac=None,
                 kernel: int = SHD_DELAY_KERNEL):
        super().__init__()
        self.profile = C.load_profile(profile) if isinstance(profile, str) else profile.validate()
        self.widths = tuple(int(w) for w in widths)
        self.kernel = kernel
        self.FaultInjector = None
        mf = self.profile["membrane"]
        if mf.kind == "fixed":
            if mem_frac is None:
                raise ValueError("fixed-point membrane needs mem_frac (one per layer, from calibration)")
            self.mem_frac = [int(f) for f in mem_frac]
        else:
            self.mem_frac = [None] * 3
        self.fmt = {}
        for i, name in enumerate(LAYERS):
            n_in, n_out = self.widths[i], self.widths[i + 1]
            hidden = name != "block_out"
            self.fmt[name] = self._layer_formats(i, hidden)
            self.register_buffer(f"{name}_w_code", torch.zeros(n_out, n_in, dtype=torch.int64))
            ws = torch.ones(n_out, 1) if self.profile["weight"].granularity == "channel" else torch.tensor(1.0)
            self.register_buffer(f"{name}_w_scale", ws)
            self.register_buffer(f"{name}_d_code", torch.zeros(n_out, n_in, dtype=torch.int64))
            if hidden:
                self.register_buffer(f"{name}_b_code", torch.zeros(n_out, dtype=torch.int64))
                nth = 1 if self.profile["threshold"].storage == "shared" else n_out
                self.register_buffer(f"{name}_th_code", torch.zeros(nth, dtype=torch.int64))
            nlk = 1 if self.profile["leak"].storage == "shared" else n_out
            self.register_buffer(f"{name}_leak_code", torch.zeros(nlk, dtype=torch.int64))

    # ------------------------------------------------------------------------------------------
    def _layer_formats(self, i, hidden):
        fm = {}
        for q in C.QUANTITIES:
            f = self.profile[q]
            if f.kind == "fixed" and (f.scale == "membrane" or q == "membrane"):
                f = C.with_frac(f, self.mem_frac[i])
            fm[q] = f
        if not hidden:
            fm.pop("bias")
            fm.pop("threshold")
        return fm

    @property
    def hidden(self):
        return self.widths[1]

    def code(self, layer: str, q: str) -> torch.Tensor:
        """Stored code tensor of quantity q ('weight', 'delay', 'bias', 'threshold', 'leak')."""
        return getattr(self, f"{layer}_{ {'weight': 'w', 'delay': 'd', 'bias': 'b', 'threshold': 'th', 'leak': 'leak'}[q]}_code")

    def scale(self, layer: str, q: str):
        return getattr(self, f"{layer}_w_scale") if q == "weight" and self.fmt[layer]["weight"].kind in ("int", "uint") else None

    def value(self, layer: str, q: str, code: torch.Tensor | None = None) -> torch.Tensor:
        """Decoded value of quantity q (optionally of a replacement code tensor)."""
        c = self.code(layer, q) if code is None else code
        return C.decode(c, self.fmt[layer][q], self.scale(layer, q))

    def stored_quantities(self):
        """[(layer, quantity, shape, format)] of every stored (memory) quantity."""
        out = []
        for name in LAYERS:
            for q in ("weight", "delay", "bias", "threshold", "leak"):
                if q in self.fmt[name]:
                    out.append((name, q, tuple(self.code(name, q).shape), self.fmt[name][q]))
            out.append((LIF_NAMES[name], "membrane", (self.widths[LAYERS.index(name) + 1],), self.fmt[name]["membrane"]))
            if name != "block_out":
                out.append((LIF_NAMES[name], "spike", (self.widths[LAYERS.index(name) + 1],), None))
        return out

    def qmem_fn(self, layer: str):
        f = self.fmt[layer]["membrane"]
        if f.kind == "float":
            return lambda h: h
        s = 2.0 ** -f.frac
        lo, hi = f.qmin * s, f.qmax * s
        if f.saturate:
            return lambda h: torch.round(h / s).clamp(f.qmin, f.qmax) * s
        span = 2.0 ** f.bits

        def wrap(h):
            q = torch.round(h / s)
            return (torch.remainder(q - f.qmin, span) + f.qmin) * s
        return wrap

    # ------------------------------------------------------------------------------------------
    def _inj(self, spk, mem, name, t):
        if self.FaultInjector is not None:
            self.FaultInjector.injection(spk, mem, name, t)

    def weight_operand(self, name: str):
        """(kernel operand [out, in], output scale): integer codes and their scale for int/uint
        weights, so that the delayed-synapse accumulation is exact in FP32 (integer inputs times
        small integer codes); decoded values and scale 1 for float weights."""
        f = self.fmt[name]["weight"]
        if f.kind in ("int", "uint"):
            sc = self.scale(name, "weight")
            return self.code(name, "weight").float(), (sc.reshape(-1) if sc.dim() > 0 else sc)
        return self.value(name, "weight"), None

    def layer_counts(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Delayed-synapse accumulation in the weight-operand domain, [B, T + 12, out]."""
        w, _ = self.weight_operand(name)
        return delayed_current(x, dense_kernel(w, delay_used(self.value(name, "delay")), self.kernel))

    @staticmethod
    def assemble_current(counts, scale, bias):
        """Synaptic current = counts * weight scale (+ bias, [n] or per-row/time [B, T, n])."""
        cur = counts if scale is None else counts * scale
        return cur if bias is None else cur + bias

    def layer_current(self, name: str, x: torch.Tensor) -> torch.Tensor:
        _, sc = self.weight_operand(name)
        bias = self.value(name, "bias") if name != "block_out" else None
        return self.assemble_current(self.layer_counts(name, x), sc, bias)

    def layer_forward(self, name: str, x: torch.Tensor, inj=None, record=None):
        readout = name == "block_out"
        counts = self.layer_counts(name, x)
        _, sc = self.weight_operand(name)
        bias = None if readout else self.value(name, "bias")
        th = None if readout else self.value(name, "threshold")
        leak = self.value(name, "leak")
        fi = self.FaultInjector
        if fi is not None and hasattr(fi, "layer_hook"):        # transient M1/M2 upsets (hwfault)
            counts, bias, leak, th = fi.layer_hook(self, name, x, counts, bias, leak, th)
        cur = self.assemble_current(counts, sc, bias)
        return lif_scan(cur, leak, th, self.qmem_fn(name), inj or self._inj,
                        LIF_NAMES[name], readout=readout, record=record)

    def forward(self, x):
        x = x.float()
        if x.dim() == 4:
            x = x[:, :, 0, :]
        s, _ = self.layer_forward("block1", x)
        s, _ = self.layer_forward("block2", s)
        _, v = self.layer_forward("block_out", s)
        return F.softmax(v, dim=2).sum(1)

    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_float(cls, layers, profile="default", mem_frac=None, kernel=SHD_DELAY_KERNEL):
        """layers: 3 dicts with float tensors 'w' [out,in], 'delay' (int) [out,in], 'beta' [out],
        and for hidden layers 'b' [out], 'th' [out]. Encodes them in the profile's formats."""
        widths = (layers[0]["w"].shape[1],) + tuple(l["w"].shape[0] for l in layers)
        net = cls(profile, widths, mem_frac, kernel)
        with torch.no_grad():
            for name, L in zip(LAYERS, layers):
                fm = net.fmt[name]
                wf = fm["weight"]
                if wf.kind in ("int", "uint"):
                    sc = C.compute_scale(L["w"], wf)
                    getattr(net, f"{name}_w_scale").copy_(sc.reshape(getattr(net, f"{name}_w_scale").shape))
                net.code(name, "weight").copy_(C.encode(L["w"], wf, net.scale(name, "weight")))
                net.code(name, "delay").copy_(C.encode(L["delay"].float(), fm["delay"]))
                for q, key in (("bias", "b"), ("threshold", "th"), ("leak", "beta")):
                    if q not in fm:
                        continue
                    v = L[key].float()
                    if fm[q].storage == "shared" and q in ("threshold", "leak"):
                        if not torch.all(v == v[0]):
                            raise ValueError(f"{name}.{q} differs across neurons; use per-neuron storage")
                        v = v[:1]
                    net.code(name, q).copy_(C.encode(v, fm[q]))
        return net

    def to_float_layers(self):
        """Inverse of from_float on the decoded (quantized) values."""
        out = []
        for name in LAYERS:
            n = self.widths[LAYERS.index(name) + 1]
            L = {"w": self.value(name, "weight").clone(), "delay": delay_used(self.value(name, "delay")),
                 "beta": self.value(name, "leak").expand(n).clone()}
            if name != "block_out":
                L["b"] = self.value(name, "bias").clone()
                L["th"] = self.value(name, "threshold").expand(n).clone()
            out.append(L)
        return out

    def checkpoint(self) -> dict:
        return {"format": "DeployedSHD-v1", "profile": self.profile.to_dict(),
                "profile_name": self.profile.name, "widths": list(self.widths), "mem_frac": self.mem_frac,
                "kernel": self.kernel, "state_dict": copy.deepcopy(self.state_dict())}

    def save(self, path: str, **extra):
        torch.save({**self.checkpoint(), **extra}, path)

    @classmethod
    def from_checkpoint(cls, ck: dict, device="cpu"):
        prof = C.profile_from_dict(ck["profile"], name=ck.get("profile_name"))
        net = cls(prof, ck["widths"], ck["mem_frac"], ck.get("kernel", SHD_DELAY_KERNEL))
        net.load_state_dict(ck["state_dict"])
        return net.to(device).eval()

    @classmethod
    def load(cls, path: str, device="cpu"):
        return cls.from_checkpoint(torch.load(path, map_location="cpu"), device)
