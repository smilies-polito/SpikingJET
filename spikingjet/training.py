"""Faults during training, for fault-aware training and for studying faults in training hardware.

Add four calls to any training loop:

    tf = TrainingFaults(cfg_or_groups, model, optimizer(s), example_batch, steps_per_epoch, seed)
    # or, from an experiment file:  tf = maybe(from_file(path, model, optimizer, example_batch, steps_per_epoch))
    for epoch ...:
        for x, y in loader:
            tf.before_forward(epoch)            # faults for this step
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward()
            tf.after_backward()                 # transient weight faults removed, gradient faults
            optimizer.step()
            tf.after_step()                     # permanent weight faults re-applied, optimizer-state faults
    tf.close()

Persistence of a fault group (config key `persistence`):

  target       transient                          upset                          permanent
  params       faults in the forward/backward of  bits flip in the stored weight  stuck cells: the map is
               one step, then removed: the update  and stay (until the optimizer   re-applied after every
               is applied to the clean weights     rewrites them)                  update
               (fault-aware training, RandBET)
  neurons /    fresh faults at every timestep of  -                              one map, the same positions
  activations  one training step                                                 at every step
  grads        faults in the gradients of a step (transient and upset are the    fixed positions, every step
               same: gradients are recomputed every step)
  optimizer    -                                  bits flip in the optimizer      stuck cells, re-applied
                                                  state (momentum, Adam moments)  after every step

Schedule of a group: epochs [start_epoch, stop_epoch), every `every`-th step, with probability
`prob` (permanent groups ignore every / prob). Evaluation (model.eval()) runs without transient
faults; permanent and upset faults are in the weights, so they stay.
"""
from __future__ import annotations

import numpy as np
import torch

from .bitops import apply_faults, resolve_bits, sample_ber
from .config import FAULT_GROUP, _fault_group
from .faults import ParamBER
from .injector import Injector
from .sites import Site, discover, select

_OPT_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer", "square_avg",
                   "acc_delta", "sum")


class _TensorPopulation:
    """BER sampling over a list of (name, tensor getter) as one memory."""

    def __init__(self, names, getters, shapes):
        self.names, self.getters = names, getters
        self.sizes = np.array([int(np.prod(s)) if len(s) else 1 for s in shapes], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def sample(self, ber, bits, rng, stuck):
        elem, bit, vals = sample_ber(int(self.offsets[-1]), bits, ber, rng, stuck)
        out = []
        if elem.size:
            which = np.searchsorted(self.offsets, elem, side="right") - 1
            for k in np.unique(which):
                sel = which == k
                out.append((k, elem[sel] - self.offsets[k], bit[sel], vals[sel]))
        return out

    @torch.no_grad()
    def apply(self, arrays):
        n = 0
        for k, idx, bit, vals in arrays:
            t = self.getters[k]()
            if t is None:
                continue
            apply_faults(t.view(-1), idx, bit, vals)
            n += len(idx)
        return n


class _Group:
    def __init__(self, i, g, rng):
        self.i, self.g, self.rng = i, g, rng
        self.target, self.pers = g["target"], g["persistence"]
        self.bits = resolve_bits(g["bits"])
        self.stuck = g["fault_model"] == "stuck-at"
        self.ber = float(g["ber"])
        self.count = 0
        self.fixed = None             # permanent map
        self.pop = None               # _TensorPopulation (grads / optimizer) or ParamBER (params)
        self.sites = []
        self.state = None             # _StateBER handle (neurons / activations)

    def in_window(self, epoch):
        stop = self.g["stop_epoch"]
        return epoch >= self.g["start_epoch"] and (stop < 0 or epoch < stop)

    def fires(self, epoch, step):
        if not self.in_window(epoch):
            return False
        if self.pers == "permanent":
            return True
        return step % max(1, int(self.g["every"])) == 0 and self.rng.random() < float(self.g["prob"])


def from_file(path, model, optimizer, example_batch, steps_per_epoch=None, seed=0):
    """TrainingFaults from the [[train.faults]] groups (and [sites]) of an experiment file; None if
    path is empty. Used by the --faults option of train_cnn.py / train_snn.py / train_shd_delays.py."""
    if not path:
        return None
    from .config import load_config
    cfg = load_config(path)
    cfg["seed"] = seed
    tf = TrainingFaults(cfg, model, optimizer, example_batch, steps_per_epoch)
    print(f"faults during training ({path}):\n{tf.describe()}")
    return tf


class _NoFaults:
    """Stand-in with the same calls when no faults are configured."""
    def before_forward(self, epoch=None): pass
    def after_backward(self): pass
    def after_step(self): pass
    def summary(self): return {}
    def close(self): pass


def maybe(tf):
    return tf if tf is not None else _NoFaults()


class TrainingFaults:
    def __init__(self, groups, model, optimizer, example_batch, steps_per_epoch=None, seed=0,
                 param_scope="spikingjet", activation_types=None):
        if isinstance(groups, dict):                      # a whole config: use its train.faults
            cfg = groups
            groups = cfg["train"]["faults"]
            param_scope = cfg["sites"]["param_scope"]
            activation_types = cfg["sites"]["activation_types"]
            seed = cfg.get("seed", seed)
        groups = [g if set(g) == set(FAULT_GROUP) else _fault_group(g, f"faults[{i}]") for i, g in enumerate(groups)]
        self.model = model
        self.opts = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        self.steps_per_epoch = steps_per_epoch
        self.step = 0
        self.epoch = 0
        kw = {} if activation_types is None else {"activation_types": activation_types}
        need_acts = any(g["target"] == "activations" for g in groups)
        self.sites = discover(model, example_batch, param_scope=param_scope, with_activations=need_acts, **kw)
        self.inj = Injector(model, self.sites, hook_activations=need_acts)
        self.groups = []
        for i, g in enumerate(groups):
            G = _Group(i, g, np.random.default_rng(np.random.SeedSequence([int(seed), 1000 + i])))
            if G.target in ("params", "grads", "optimizer"):
                G.sites = select(self.sites.params, g["select"], g["exclude"])
            else:
                G.sites = select(self.sites.by_target(G.target), g["select"], g["exclude"])
            if not G.sites:
                raise ValueError(f"train.faults[{i}]: no {G.target} sites match {g['select']} "
                                 f"(python spikingjet.py sites -n NETWORK lists them)")
            if G.target in ("neurons", "activations") and G.pers == "upset":
                raise ValueError(f"train.faults[{i}]: {G.target} have no stored state to upset: use transient or permanent")
            if G.target == "optimizer" and G.pers == "transient":
                raise ValueError(f"train.faults[{i}]: optimizer state is stored memory: use upset or permanent")
            self._prepare(G)
            self.groups.append(G)
        self._transient_param = False
        self.log = []                                     # (epoch, step, group, n faults)

    # ------------------------------------------------------------------ setup per target
    def _prepare(self, G):
        if G.target == "params":
            G.pop = ParamBER(G.sites)
            if G.pers == "permanent":
                G.fixed = G.pop.sample(G.ber, G.bits, G.rng, G.stuck)
        elif G.target == "grads":
            G.pop = _TensorPopulation([s.name + ".grad" for s in G.sites],
                                      [(lambda p=s.tensor: p.grad) for s in G.sites], [s.shape for s in G.sites])
            if G.pers == "permanent":
                G.fixed = G.pop.sample(G.ber, G.bits, G.rng, G.stuck)
        elif G.target == "optimizer":
            G.pop = None                                   # optimizer state exists after the first step
        else:                                              # neurons / activations
            if G.pers == "permanent":
                G.state = self.inj.add_state_ber(G.sites, G.bits, G.ber, G.stuck, G.rng, "permanent")
                self.inj.state_ber.remove(G.state)         # attached only while in its window

    def _optimizer_population(self, G):
        names, getters, shapes = [], [], []
        for s in G.sites:
            st = next((o.state[s.tensor] for o in self.opts if s.tensor in o.state), {})
            for k in _OPT_STATE_KEYS:
                v = st.get(k)
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.shape == s.tensor.shape:
                    names.append(f"{s.name}.{k}")
                    getters.append(lambda st=st, k=k: st.get(k))
                    shapes.append(tuple(v.shape))
        if not names:
            return None
        pop = _TensorPopulation(names, getters, shapes)
        if G.pers == "permanent":
            G.fixed = pop.sample(G.ber, G.bits, G.rng, G.stuck)
        return pop

    def _record(self, G, n):
        if n:
            G.count += n
            self.log.append((self.epoch, self.step, G.i, n))

    # ------------------------------------------------------------------ hooks
    def before_forward(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        elif self.steps_per_epoch:
            self.epoch = self.step // self.steps_per_epoch
        self.inj.clear_states()
        for G in self.groups:
            if not G.fires(self.epoch, self.step):
                continue
            if G.target == "params":
                if G.pers == "transient":
                    arrays = G.pop.sample(G.ber, G.bits, G.rng, G.stuck)
                    self.inj.apply_param_arrays(arrays, save=True)
                    self._transient_param = True
                    self._record(G, sum(len(a[1]) for a in arrays))
                elif G.pers == "upset":
                    arrays = G.pop.sample(G.ber, G.bits, G.rng, G.stuck)
                    self.inj.apply_param_arrays(arrays, save=False)
                    self.inj.param_faults = []
                    self._record(G, sum(len(a[1]) for a in arrays))
                elif not getattr(G, "_applied", False):          # permanent: first step of its window
                    self.inj.apply_param_arrays(G.fixed, save=False)
                    self.inj.param_faults = []
                    G._applied = True
                    self._record(G, sum(len(a[1]) for a in G.fixed))
            elif G.target in ("neurons", "activations"):
                if G.pers == "transient":
                    G.state = self.inj.add_state_ber(G.sites, G.bits, G.ber, G.stuck, G.rng, "transient")
                else:
                    self.inj.state_ber.append(G.state)
        self.inj.active = True

    def after_backward(self):
        self.inj.active = False
        for G in self.groups:
            if G.target in ("neurons", "activations") and G.state is not None and G.state.count:
                self._record(G, G.state.count)
                G.state.count = 0
        if self._transient_param:
            self.inj.restore_params()
            self._transient_param = False
        for G in self.groups:
            if G.target == "grads" and G.fires(self.epoch, self.step):
                arrays = G.fixed if G.pers == "permanent" else G.pop.sample(G.ber, G.bits, G.rng, G.stuck)
                self._record(G, G.pop.apply(arrays))

    def after_step(self):
        for G in self.groups:
            if not G.in_window(self.epoch):
                continue
            if G.target == "params" and G.pers == "permanent":
                self.inj.apply_param_arrays(G.fixed, save=False)       # stuck cells hold after the update
                self.inj.param_faults = []
            elif G.target == "optimizer":
                if G.pop is None:
                    G.pop = self._optimizer_population(G)
                    if G.pop is None:
                        continue
                if G.pers == "permanent":
                    n = G.pop.apply(G.fixed)
                    if not getattr(G, "_applied", False):
                        G._applied = True
                        self._record(G, n)
                elif G.fires(self.epoch, self.step):
                    self._record(G, G.pop.apply(G.pop.sample(G.ber, G.bits, G.rng, G.stuck)))
        self.step += 1

    # ------------------------------------------------------------------ reporting
    def epoch_counts(self, epoch):
        c = {}
        for e, _, gi, n in self.log:
            if e == epoch:
                c[gi] = c.get(gi, 0) + n
        return c

    def describe(self):
        rows = []
        for G in self.groups:
            g = G.g
            rows.append(f"  [{G.i}] {G.target:<11} {g['fault_model']:<8} {g['persistence']:<9} BER {G.ber:.1e} "
                        f"bits {g['bits']} epochs [{g['start_epoch']}, {'end' if g['stop_epoch'] < 0 else g['stop_epoch']}) "
                        f"every {g['every']} prob {g['prob']} | {len(G.sites)} sites: "
                        f"{', '.join(s.name for s in G.sites[:4])}{' ...' if len(G.sites) > 4 else ''}")
        return "\n".join(rows) if rows else "  (no faults)"

    def summary(self):
        return {f"group{G.i}_{G.target}_{G.pers}": G.count for G in self.groups}

    def close(self):
        self.inj.active = False
        if self._transient_param:
            self.inj.restore_params()
        self.inj.detach()
