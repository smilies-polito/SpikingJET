"""Fault sites: everything in a model that can hold a fault, found automatically.

    kind        name                    tensor
    param       fc1.weight              a parameter (no batch dimension)
    neuron      lif1.potential          membrane potential of a spiking layer, [B, ...] at every step
    neuron      lif1.spike              its output spikes
    activation  features.2.output       output of any module (ReLU, Conv2d, Linear, ...), [B, ...]

Neuron sites come from either
  * the explicit hook the model calls after every neuron update
        self.FaultInjector.injection(spk, mem, "lif1", t)      (models/snn_sota.py, README)
  * or, if the model never calls it, forward hooks on every snntorch neuron (snn.Leaky,
    snn.Synaptic, ...), named after the module; no change to the model is needed.
Activation sites are forward hooks on modules of the configured types. Gradients and optimizer
state (training only) mirror the parameter sites and are handled by spikingjet.training.

Shapes of neuron / activation sites are found by one probe forward pass on a real batch.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn

try:
    import snntorch as snn
    _SNN_NEURON = tuple(c for c in (getattr(snn, "SpikingNeuron", None),) if c is not None) or \
        tuple(getattr(snn, n) for n in ("Leaky", "Synaptic", "RLeaky", "RSynaptic", "Alpha", "Lapicque")
              if hasattr(snn, n))
    _SNN_WEIGHTED = (snn.Leaky, snn.Synaptic)
except ImportError:                                   # pragma: no cover - snntorch is a requirement
    snn = None
    _SNN_NEURON, _SNN_WEIGHTED = (), ()

# module types whose output is an activation site when target "activations" is used
DEFAULT_ACTIVATION_TYPES = ("ReLU", "ReLU6", "LeakyReLU", "GELU", "SiLU", "ELU", "Sigmoid", "Tanh",
                            "Hardtanh", "Conv1d", "Conv2d", "Conv3d", "Linear")


@dataclass
class Site:
    kind: str                       # param | neuron | activation
    name: str                       # unique, e.g. "fc1.weight", "lif1.potential", "conv1.output"
    shape: tuple                    # per-sample shape (no batch dimension) for neuron / activation
    layer: str = ""                 # module / hook layer the site belongs to
    tensor: Optional[torch.Tensor] = field(default=None, repr=False)   # the parameter (kind param)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


def _weighted_top_modules(model: nn.Module) -> set:
    """Top-level children that are, or contain, Conv / Linear / snntorch weighted neurons: the
    parameters SpikingJET has always injected (scope 'spikingjet')."""
    classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear) + _SNN_WEIGHTED
    return {name.split(".")[0] for name, m in model.named_modules() if name and isinstance(m, classes)}


def param_sites(model: nn.Module, scope: str = "spikingjet") -> list:
    """Float32 parameters in named_parameters() order. scope 'spikingjet' keeps the parameters of
    top-level modules that are or contain Conv/Linear/LIF (historical default), 'all' keeps all."""
    if scope not in ("spikingjet", "all"):
        raise ValueError(f"param scope must be 'spikingjet' or 'all', not {scope!r}")
    keep = _weighted_top_modules(model) if scope == "spikingjet" else None
    out = []
    for name, p in model.named_parameters():
        if keep is not None and name.split(".")[0] not in keep:
            continue
        if p.dtype != torch.float32:
            continue
        out.append(Site("param", name, tuple(p.shape), name.rsplit(".", 1)[0], p))
    return out


def select(sites: Iterable[Site], patterns=("*",), exclude=()) -> list:
    """Sites whose name matches any glob pattern and no exclude pattern."""
    pats, excl = list(patterns or ["*"]), list(exclude or [])
    return [s for s in sites if any(fnmatch.fnmatchcase(s.name, p) for p in pats)
            and not any(fnmatch.fnmatchcase(s.name, p) for p in excl)]


class _Probe:
    """Stands in for the injector during the probe pass and records every neuron / activation
    tensor shape in call order."""

    def __init__(self):
        self.shapes = {}                  # name -> per-sample shape, insertion = first-call order
        self.explicit = False

    def injection(self, spike, pot, layer, t):
        self.explicit = True
        self.shapes.setdefault(f"{layer}.potential", tuple(pot.shape[1:]))
        self.shapes.setdefault(f"{layer}.spike", tuple(spike.shape[1:]))


def _split_neuron_output(out):
    """snntorch neurons return (spk, mem), (spk, syn, mem) or, with init_hidden and output=False,
    just spk. Returns (spike tensor, membrane tensor or None)."""
    if isinstance(out, tuple):
        return out[0], out[-1]
    return out, None


def discover(model: nn.Module, example_batch: torch.Tensor, param_scope: str = "spikingjet",
             activation_types=DEFAULT_ACTIVATION_TYPES, with_activations: bool = True) -> "SiteTable":
    """Find every site of `model` with one forward pass on `example_batch` (already on the device)."""
    probe = _Probe()
    had_attr = hasattr(model, "FaultInjector")
    old = getattr(model, "FaultInjector", None)
    if had_attr:
        model.FaultInjector = probe
    auto_neurons, activations, handles = {}, {}, []
    types = tuple(getattr(nn, t) for t in activation_types if hasattr(nn, t))

    for name, m in model.named_modules():
        if not name:
            continue
        if _SNN_NEURON and isinstance(m, _SNN_NEURON):
            def hook(mod, inp, out, name=name):
                spk, mem = _split_neuron_output(out)
                if mem is not None:
                    auto_neurons.setdefault(f"{name}.potential", tuple(mem.shape[1:]))
                auto_neurons.setdefault(f"{name}.spike", tuple(spk.shape[1:]))
            handles.append(m.register_forward_hook(hook))
        elif with_activations and types and isinstance(m, types) and not list(m.children()):
            def hook(mod, inp, out, name=name):
                if isinstance(out, torch.Tensor) and out.dtype == torch.float32 and out.dim() >= 1:
                    activations.setdefault(f"{name}.output", tuple(out.shape[1:]))
            handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            was_training = model.training
            model.eval()
            model(example_batch)
            model.train(was_training)
    finally:
        for h in handles:
            h.remove()
        if had_attr:
            model.FaultInjector = old

    neuron_shapes = probe.shapes if probe.explicit else auto_neurons
    # historical order of SpikingJET fault lists: every potential first, then every spike
    neurons = [Site("neuron", n, s, n.rsplit(".", 1)[0]) for n, s in neuron_shapes.items() if n.endswith(".potential")]
    neurons += [Site("neuron", n, s, n.rsplit(".", 1)[0]) for n, s in neuron_shapes.items() if n.endswith(".spike")]
    acts = [Site("activation", n, s, n.rsplit(".", 1)[0]) for n, s in activations.items()]
    return SiteTable(param_sites(model, param_scope), neurons, acts, explicit_hook=probe.explicit)


@dataclass
class SiteTable:
    params: list
    neurons: list
    activations: list
    explicit_hook: bool = False       # neuron sites come from model.FaultInjector.injection(...)

    def by_target(self, target: str) -> list:
        return {"params": self.params, "neurons": self.neurons, "activations": self.activations}[target]

    def all(self) -> list:
        return self.params + self.neurons + self.activations

    def get(self, name: str) -> Site:
        for s in self.all():
            if s.name == name:
                return s
        raise KeyError(f"no fault site {name!r}")

    def describe(self) -> str:
        rows = [f"{'kind':<11}{'site':<40}{'shape':<22}{'elements':>12}{'bits':>14}"]
        for s in self.all():
            rows.append(f"{s.kind:<11}{s.name:<40}{str(s.shape):<22}{s.numel:>12,}{32 * s.numel:>14,}")
        hook = ("explicit FaultInjector.injection hook" if self.explicit_hook else
                "forward hooks on snntorch neurons" if self.neurons else "none (no spiking neurons found)")
        rows.append(f"\nneuron sites via: {hook}")
        rows.append(f"totals: params {sum(s.numel for s in self.params):,}, neuron state "
                    f"{sum(s.numel for s in self.neurons):,} per sample, activations "
                    f"{sum(s.numel for s in self.activations):,} per sample")
        return "\n".join(rows)
