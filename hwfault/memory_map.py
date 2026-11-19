"""
Memory map of the deployment form on the selected hardware target.

neuromorphic (memory-based accelerator):
    M1 synapse memory        weight, delay                       written once
    M2 neuron configuration  bias, threshold, leak               written once
    M3 neuron state          membrane (every layer, readout too) written every timestep
    M4 spike routing         spike events of hidden layers       written every timestep
gpu: reserved for a later change; selecting it raises HardwareTargetNotImplemented.

Persistence (per class, `permanent` | `transient`), presets:
    permanent  all classes permanent
    transient  all classes transient
    mixed      M1/M2 permanent, M3/M4 transient (default)
"""
from __future__ import annotations

from dataclasses import dataclass

from . import codec as C

HW_TARGETS = ("neuromorphic", "gpu")
CLASSES = ("M1", "M2", "M3", "M4")
CLASS_OF = {"weight": "M1", "delay": "M1", "bias": "M2", "threshold": "M2", "leak": "M2",
            "membrane": "M3", "spike": "M4"}
WRITTEN = {"M1": "once", "M2": "once", "M3": "every timestep", "M4": "every timestep"}
PRESETS = {
    "permanent": {"M1": "permanent", "M2": "permanent", "M3": "permanent", "M4": "permanent"},
    "transient": {"M1": "transient", "M2": "transient", "M3": "transient", "M4": "transient"},
    "mixed": {"M1": "permanent", "M2": "permanent", "M3": "transient", "M4": "transient"},
}
FAULT_TYPES = {      # (class, persistence) -> supported fault types
    ("M1", "permanent"): ("flip", "sa0", "sa1"), ("M1", "transient"): ("flip", "sa0", "sa1"),
    ("M2", "permanent"): ("flip", "sa0", "sa1"), ("M2", "transient"): ("flip", "sa0", "sa1"),
    ("M3", "permanent"): ("flip", "sa0", "sa1"), ("M3", "transient"): ("flip", "sa0", "sa1"),
    ("M4", "permanent"): ("dead", "saturated"), ("M4", "transient"): ("drop", "spurious"),
}
LIF = {"block1": "lif1", "block2": "lif2", "block3": "lif3", "block_out": "lif_out"}   # block3: NMNIST
BLOCK = {v: k for k, v in LIF.items()}
DELAY_PAD = 12          # (KERNEL - 1) // 2: each layer adds 12 output steps


class HardwareTargetNotImplemented(NotImplementedError):
    pass


def persistence_mode(spec="mixed") -> dict:
    """Preset name or {class: persistence} (missing classes default to the `mixed` preset)."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown persistence preset {spec!r}; presets: {sorted(PRESETS)}")
        return dict(PRESETS[spec])
    mode = dict(PRESETS["mixed"])
    for k, v in spec.items():
        if k not in CLASSES or v not in ("permanent", "transient"):
            raise ValueError(f"invalid persistence entry {k}={v}")
        mode[k] = v
    return mode


def check_hw_target(hw_target: str):
    if hw_target == "gpu":
        raise HardwareTargetNotImplemented(
            "hardware target 'gpu' (FP32, ECC-protected GPU memory) is not implemented yet; "
            "only 'neuromorphic' is available in this version")
    if hw_target not in HW_TARGETS:
        raise ValueError(f"unknown hardware target {hw_target!r}; choose from {HW_TARGETS}")


@dataclass(frozen=True)
class Site:
    cls: str            # M1..M4
    layer: str          # block1 | block2 | block_out
    field: str          # weight | delay | bias | threshold | leak | membrane | spike
    shape: tuple        # cell shape (no batch / time)
    fmt: C.Format | None
    bits: int           # bits per cell (1 event bit for spikes)
    persistence: str
    steps: int | None   # timesteps per sample (M3/M4) for the reference input length

    @property
    def cells(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def fault_types(self):
        return FAULT_TYPES[(self.cls, self.persistence)]

    @property
    def lif(self) -> str:
        return LIF[self.layer]

    @property
    def written(self) -> str:
        return WRITTEN[self.cls]

    def n_faults(self, transient_steps=True) -> int:
        """Size of the single-fault space of this site: cells x bits x types (x timesteps for
        transient M3/M4; transient M1/M2 onsets are sampled, not enumerated)."""
        per = self.cells * (self.bits * len(self.fault_types) if self.cls != "M4" else len(self.fault_types))
        if self.cls in ("M3", "M4") and self.persistence == "transient" and transient_steps:
            per *= self.steps
        return per


# SpikingJET injection points of SHD_DELAY -> where they live in the deployment form.
SPIKINGJET_POINTS = {
    "{b}.delay.syn.weight": ("mapped", "M1 weight"),
    "{b}.bn.weight": ("mapped", "folded into M1 weight (scale) and M2 bias"),
    "{b}.bn.bias": ("mapped", "folded into M2 bias"),
    "{b}.lif.beta": ("mapped", "M2 leak"),
    "{b}.lif.threshold": ("mapped", "M2 threshold"),
    "{l}.potential": ("mapped", "M3 membrane"),
    "{l}.spike": ("mapped", "M4 spike"),
}
EXCLUDED = {
    "block_out.lif.threshold": "the readout is a non-spiking accumulator (threshold 1e9): not a stored quantity",
    "lif_out.spike": "the readout never spikes: its spike output is constant zero and not routed",
}
ADDED = {"{b}.delay.k_idx": "M1 delay (not injected by SpikingJET)",
         "{b}.bn.running_mean/var": "folded into M1 weight and M2 bias"}


class MemoryMap:
    """Sites of a DeployedSHD under a hardware target and persistence mode."""

    def __init__(self, net, hw_target: str = "neuromorphic", mode="mixed", T: int | None = None):
        """T: input timesteps (default: the network's T_STEPS if it defines layer_steps, else 150).
        A network that defines LAYERS, layer_steps(i, T) and state_cells(layer) (DeployedNMNIST)
        gives its own layers, timesteps and state sizes; otherwise the SHD_DELAY layout is used."""
        check_hw_target(hw_target)
        self.hw_target = hw_target
        self.mode = persistence_mode(mode)
        self.net = net
        generic = hasattr(net, "layer_steps")
        if T is None:
            T = getattr(net, "T_STEPS", None) or (10 if generic else 150)
        self.T = T
        self.sites = []
        layers = tuple(net.LAYERS) if generic else ("block1", "block2", "block_out")
        for i, layer in enumerate(layers):
            steps = net.layer_steps(i, T) if generic else T + DELAY_PAD * (i + 1)
            n = net.state_cells(layer) if generic else net.widths[i + 1]
            for field in ("weight", "delay", "bias", "threshold", "leak"):
                if field in net.fmt[layer]:
                    fmt = net.fmt[layer][field]
                    shape = tuple(net.code(layer, field).shape)
                    self.sites.append(Site(CLASS_OF[field], layer, field, shape, fmt, fmt.bits,
                                           self.mode[CLASS_OF[field]], None))
            fm = net.fmt[layer]["membrane"]
            self.sites.append(Site("M3", layer, "membrane", (n,), fm, fm.bits, self.mode["M3"], steps))
            if layer != "block_out":
                self.sites.append(Site("M4", layer, "spike", (n,), None, 1, self.mode["M4"], steps))

    def site(self, layer: str, field: str) -> Site:
        for s in self.sites:
            if s.layer == layer and s.field == field:
                return s
        raise KeyError((layer, field))

    def by_class(self, cls: str):
        return [s for s in self.sites if s.cls == cls]

    def total_bits(self, cls: str) -> int:
        """Stored bits of a class (M3: neurons x word; M4: neurons x 1 event bit)."""
        return sum(s.cells * s.bits for s in self.by_class(cls))

    def spikingjet_coverage(self, sj_points) -> dict:
        """sj_points: names of SpikingJET injection points of SHD_DELAY (parameters and
        '<lif>.potential' / '<lif>.spike'). Returns name -> (status, where/reason)."""
        out = {}
        for p in sj_points:
            if p in EXCLUDED:
                out[p] = ("excluded", EXCLUDED[p])
                continue
            hit = None
            for pat, val in SPIKINGJET_POINTS.items():
                for b, l in LIF.items():
                    if pat.format(b=b, l=l) == p:
                        hit = val
            out[p] = hit or ("unmapped", "no memory-map entry")
        return out

    def to_dict(self) -> dict:
        return {"hw_target": self.hw_target, "persistence": self.mode,
                "sites": [{"class": s.cls, "layer": s.layer, "field": s.field, "shape": list(s.shape),
                           "cells": s.cells, "format": s.fmt.describe() if s.fmt else "event",
                           "bits": s.bits, "persistence": s.persistence, "written": s.written,
                           "fault_types": list(s.fault_types), "steps": s.steps} for s in self.sites]}

    def table(self) -> str:
        rows = [("class", "layer", "site", "cells", "format", "bits", "written", "persistence", "types")]
        for s in self.sites:
            rows.append((s.cls, s.layer, s.field, f"{s.cells:,}", s.fmt.describe() if s.fmt else "event",
                         str(s.bits), s.written, s.persistence, ",".join(s.fault_types)))
        w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
        lines.insert(1, "  ".join("-" * x for x in w))
        lines.append("")
        lines.append(f"hardware target: {self.hw_target}; persistence: " +
                     ", ".join(f"{k}={v}" for k, v in self.mode.items()))
        lines.append("excluded: " + "; ".join(f"{k} ({v})" for k, v in EXCLUDED.items()))
        return "\n".join(lines)
