"""The injection engine: puts faults into a model's parameters and into its neuron / activation
tensors while it runs, for any model (SNN, CNN or mixed), at inference and during training.

    inj = Injector(model, sites)          # sites = spikingjet.sites.discover(model, batch)
    inj.set_param_faults([...])           # permanent faults in parameters (restore_params() undoes them)
    inj.set_state_faults([...])           # faults in neuron / activation tensors at every forward
    inj.add_state_ber(...)                # random faults at a bit error rate in those tensors
    with inj.enabled(): model(x)
    inj.detach()

State faults are applied
  * through model.FaultInjector.injection(spk, mem, layer, t) when the model calls it, in place
    (during training through .data: the faulty value is used, the gradient passes unchanged);
  * otherwise through forward hooks, which return a faulty copy (straight-through gradient).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .bitops import apply_faults, sample_ber
from .sites import DEFAULT_ACTIVATION_TYPES, SiteTable, _SNN_NEURON, _split_neuron_output


@dataclass
class Fault:
    """One fault: element `index` (per-sample for neuron / activation sites) of `site`, `bit`,
    `value` = -1 bit-flip, 0 / 1 stuck-at. `step`: None = at every timestep / call, else only at
    that timestep (transient neuron fault)."""
    site: str
    index: tuple
    bit: int
    value: int = -1
    step: Optional[int] = None


class _StateBER:
    """Random faults at rate `ber` in a group of neuron / activation sites.
    transient: fresh draw at every call of every site (over the whole [B, ...] tensor);
    permanent: one map per site over the per-sample shape, applied to every sample at every call."""

    def __init__(self, sites, bits, ber, stuck, rng, persistence):
        self.sites = {s.name: s for s in sites}
        self.bits, self.ber, self.stuck, self.rng = tuple(bits), float(ber), bool(stuck), rng
        self.persistence = persistence
        self.fixed = {}
        if persistence == "permanent":
            for s in sites:
                self.fixed[s.name] = sample_ber(s.numel, self.bits, self.ber, rng, self.stuck)
        self.count = 0

    def faults_for(self, name, tensor):
        if name not in self.sites:
            return None
        if self.persistence == "permanent":
            e, b, v = self.fixed[name]
            if e.size == 0:
                return None
            per = self.sites[name].numel
            B = tensor.shape[0]
            e = (np.arange(B, dtype=np.int64)[:, None] * per + e[None, :]).ravel()
            b, v = np.tile(b, B), np.tile(v, B)
        else:
            e, b, v = sample_ber(tensor.numel(), self.bits, self.ber, self.rng, self.stuck)
            if e.size == 0:
                return None
        self.count += e.size
        return e, b, v


class Injector:
    def __init__(self, model: nn.Module, sites: SiteTable,
                 activation_types=DEFAULT_ACTIVATION_TYPES, hook_activations: Optional[bool] = None):
        self.model, self.sites = model, sites
        self.active = False
        self.param_faults = []               # [(site, flat idx, bits, values)]
        self._saved = []                     # [(int32 view, idx, original ints)]
        self.state_faults = {}               # site name -> [(per-sample flat idx, bit, value, step)]
        self.state_ber = []                  # [_StateBER]
        self._calls = {}                     # site layer -> call count in this forward (timestep)
        self._handles = []
        self._prev_injector = getattr(model, "FaultInjector", None)
        self._has_attr = hasattr(model, "FaultInjector")
        self._names = {s.name for s in sites.all()}
        if self._has_attr:
            model.FaultInjector = self
        self._handles.append(model.register_forward_pre_hook(self._reset_steps))
        # automatic neuron hooks (models that do not call the explicit hook)
        if sites.neurons and not sites.explicit_hook:
            for name, m in model.named_modules():
                if name and _SNN_NEURON and isinstance(m, _SNN_NEURON):
                    self._handles.append(m.register_forward_hook(self._neuron_hook(name)))
        if hook_activations is None:
            hook_activations = bool(sites.activations)
        if hook_activations:
            act_layers = {s.layer for s in sites.activations}
            for name, m in model.named_modules():
                if name in act_layers:
                    self._handles.append(m.register_forward_hook(self._activation_hook(name)))

    # ------------------------------------------------------------------ lifecycle
    def detach(self):
        self.restore_params()
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._has_attr:
            self.model.FaultInjector = self._prev_injector

    @contextlib.contextmanager
    def enabled(self, on: bool = True):
        prev, self.active = self.active, on
        try:
            yield self
        finally:
            self.active = prev

    def clear_states(self):
        self.state_faults, self.state_ber = {}, []

    # ------------------------------------------------------------------ parameters
    @torch.no_grad()
    def set_param_faults(self, faults):
        """Apply a list of Fault on parameter sites (in place), saving the golden values."""
        groups = {}
        for f in faults:
            groups.setdefault(f.site, []).append(f)
        out = []
        for name, fs in groups.items():
            site = self.sites.get(name)
            idx = np.array([np.ravel_multi_index(f.index, site.shape) if site.shape else 0 for f in fs], np.int64)
            out.append((site, idx, np.array([f.bit for f in fs], np.int64), np.array([f.value for f in fs], np.int64)))
        self.apply_param_arrays(out)

    @torch.no_grad()
    def apply_param_arrays(self, arrays, save=True):
        """arrays: [(site, flat idx, bits, values)] numpy; applied in place; saved for restore."""
        for site, idx, bits, vals in arrays:
            if len(idx) == 0:
                continue
            p = site.tensor
            ints = p.data.view(-1).view(torch.int32)
            t_idx = torch.as_tensor(np.unique(idx), device=p.device)
            if save:
                self._saved.append((ints, t_idx, ints[t_idx].clone()))
            apply_faults(p.data.view(-1), idx, bits, vals)
        self.param_faults.extend(arrays)

    @torch.no_grad()
    def restore_params(self):
        for ints, idx, orig in reversed(self._saved):
            ints[idx] = orig
        self._saved, self.param_faults = [], []

    # ------------------------------------------------------------------ neuron / activation states
    def set_state_faults(self, faults):
        self.state_faults = {}
        for f in faults:
            site = self.sites.get(f.site)
            flat = int(np.ravel_multi_index(f.index, site.shape)) if site.shape else 0
            self.state_faults.setdefault(f.site, []).append((flat, f.bit, f.value, f.step))

    def add_state_ber(self, sites, bits, ber, stuck=False, rng=None, persistence="transient"):
        if persistence not in ("transient", "permanent"):
            raise ValueError("state BER persistence must be 'transient' or 'permanent'")
        g = _StateBER(sites, bits, ber, stuck, rng if rng is not None else np.random.default_rng(), persistence)
        self.state_ber.append(g)
        return g

    def _faults_for(self, name, tensor, step):
        """(flat idx over the whole [B, ...] tensor, bits, values) to apply now, or None."""
        parts = []
        for flat, bit, value, at in self.state_faults.get(name, ()):
            if at is None or at == step:
                per = tensor[0].numel()
                parts.append((np.arange(tensor.shape[0], dtype=np.int64) * per + flat,
                              np.full(tensor.shape[0], bit, np.int64), np.full(tensor.shape[0], value, np.int64)))
        for g in self.state_ber:
            r = g.faults_for(name, tensor)
            if r is not None:
                parts.append(r)
        if not parts:
            return None
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    def _corrupt_inplace(self, tensor, name, step):
        f = self._faults_for(name, tensor, step)
        if f is not None:
            data = tensor.data
            if not data.is_contiguous():
                raise RuntimeError(f"{name}: cannot inject into a non-contiguous tensor in place")
            apply_faults(data.view(-1), *f)

    def _corrupt_copy(self, tensor, name, step):
        f = self._faults_for(name, tensor, step)
        if f is None:
            return tensor
        faulty = tensor.detach().clone().contiguous()
        apply_faults(faulty.view(-1), *f)
        if tensor.requires_grad:                      # straight-through: faulty value, clean gradient
            return tensor + (faulty - tensor).detach()
        return faulty

    def _reset_steps(self, module, inp):
        self._calls = {}

    def _step(self, layer):
        t = self._calls.get(layer, 0)
        self._calls[layer] = t + 1
        return t

    # explicit hook API used by the models: FaultInjector.injection(spk, mem, layer, t)
    def injection(self, spike, pot, layer, curr_step):
        if not self.active:
            return
        self._corrupt_inplace(pot, f"{layer}.potential", curr_step)
        self._corrupt_inplace(spike, f"{layer}.spike", curr_step)

    def _neuron_hook(self, layer):
        def hook(mod, inp, out):
            if not self.active:
                return None
            t = self._step(layer)
            spk, mem = _split_neuron_output(out)
            new_spk = self._corrupt_copy(spk, f"{layer}.spike", t)
            new_mem = self._corrupt_copy(mem, f"{layer}.potential", t) if mem is not None else None
            if not isinstance(out, tuple):
                return new_spk
            return (new_spk,) + tuple(out[1:-1]) + (new_mem,)
        return hook

    def _activation_hook(self, layer):
        def hook(mod, inp, out):
            if not self.active or not isinstance(out, torch.Tensor):
                return None
            return self._corrupt_copy(out, f"{layer}.output", self._step(layer))
        return hook
