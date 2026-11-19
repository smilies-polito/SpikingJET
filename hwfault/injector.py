"""
Fault injection on the deployment form (models.deployed_shd.DeployedSHD), spec hardware-fault-model.

A Fault is identified by (class, layer, field, cell index, bit, fault type, persistence,
timestep/onset, sample). Semantics:

M1/M2 (stored once: weight, delay, bias, threshold, leak)
    permanent  the stored code is modified before the run and restored afterwards
               (flip / sa0 / sa1 on the code bit pattern)
    transient  an upset of the code during ONE sample (or every sample if `sample` is None, as in
               the single-fault catalog), active from the onset timestep t0 to the end of that
               sample. Realized without touching the codes: the delayed-synapse layer is linear,
               so the faulty current is the clean current plus the contribution of the faulty
               cell from t0 on; bias / threshold / leak become time-varying from t0. Several
               upsets of one cell in one sample are applied cumulatively, in onset order.
M3 (membrane, rewritten every timestep) and M4 (spike routing)
    applied in the LIF hook injection(spike, mem, name, t), after the neuron update:
    permanent  re-applied after EVERY write, every timestep and sample (as SpikingJET's
               non-transient mode): stuck-at forces the bit, flip inverts the bit of each newly
               written value; M4 dead (spike forced 0) / saturated (spike forced 1)
    transient  only the value written at timestep t (M4 drop 1->0 / spurious 0->1)

Usage:
    inj = HWInjector(net, faults)            # or DrawnInjector(net, drawer) for BER realizations
    with inj:                                # applies permanent M1/M2 codes, installs the hook
        for start, (x, y) in batches:
            inj.set_batch(start, x.shape[0])
            out = net(x)
    # codes restored, hook removed
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict

import torch

from . import codec as C
from .memory_map import CLASS_OF, FAULT_TYPES, LIF, BLOCK

STATIC_FIELDS = ("weight", "delay", "bias", "threshold", "leak")
TYPE_ID = {"flip": 0, "sa0": 1, "sa1": 2}
CHUNK = 65536                          # faults per vectorized chunk (memory bound)


@dataclass(frozen=True)
class Fault:
    cls: str                 # M1..M4
    layer: str               # block1 | block2 | block_out
    field: str               # weight | delay | bias | threshold | leak | membrane | spike
    index: tuple             # cell index within the stored tensor (neuron index for M3/M4)
    bit: int                 # bit of the stored word (0 for M4 events)
    ftype: str               # flip | sa0 | sa1 | drop | spurious | dead | saturated
    persistence: str         # permanent | transient
    t: int | None = None     # transient: timestep (M3/M4) or onset t0 (M1/M2)
    sample: int | None = None  # transient: global sample index; None = every sample

    def __post_init__(self):
        if CLASS_OF.get(self.field) != self.cls:
            raise ValueError(f"field {self.field} is not in class {self.cls}")
        if self.ftype not in FAULT_TYPES[(self.cls, self.persistence)]:
            raise ValueError(f"fault type {self.ftype} not allowed for {self.cls} {self.persistence}")
        if self.persistence == "transient" and self.t is None:
            raise ValueError("transient faults need a timestep / onset t")

    @property
    def site_key(self):
        """Stable cell+bit key (pairing across widths, overlap checks)."""
        return (self.cls, self.layer, self.field, tuple(self.index), self.bit)

    def to_row(self) -> dict:
        d = asdict(self)
        d["index"] = "/".join(str(i) for i in self.index)
        return d

    @staticmethod
    def from_row(r: dict) -> "Fault":
        def opt(v):
            return None if v in (None, "", "None", "nan") else int(float(v))
        return Fault(r["cls"], r["layer"], r["field"], tuple(int(i) for i in str(r["index"]).split("/") if i != ""),
                     int(r["bit"]), r["ftype"], r["persistence"], opt(r.get("t")), opt(r.get("sample")))


def apply_op(code: torch.Tensor, fmt: C.Format, ftype: str, bit: int) -> torch.Tensor:
    if ftype == "flip":
        return C.flip(code, fmt, bit)
    if ftype in ("sa0", "sa1"):
        return C.stuck(code, fmt, bit, 1 if ftype == "sa1" else 0)
    raise ValueError(ftype)


def apply_ops(code: torch.Tensor, fmt: C.Format, bits: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Vectorized flip (type 0) / sa0 (1) / sa1 (2) on code tensors."""
    m = torch.ones_like(bits) << bits
    p = C.to_pattern(code, fmt)
    p = torch.where(types == 0, p ^ m, torch.where(types == 1, p & ~m, p | m))
    return C.from_pattern(p, fmt)


def static_fault_groups(faults) -> dict:
    """Permanent M1/M2 faults grouped per (layer, field): {key: (index [K, d], bits [K], types [K], pos [K])}, pos =
    position in `faults` (the order in which the sequential loop applied them). CPU int64 tensors."""
    acc = defaultdict(lambda: ([], [], [], []))
    for k, f in enumerate(faults):
        a = acc[(f.layer, f.field)]
        a[0].append(tuple(f.index))
        a[1].append(f.bit)
        a[2].append(TYPE_ID[f.ftype])
        a[3].append(k)
    out = {}
    for key, (idx, bits, types, pos) in acc.items():
        d = len(idx[0])
        it = torch.tensor(idx, dtype=torch.int64).reshape(len(idx), d)
        out[key] = (it, torch.tensor(bits, dtype=torch.int64), torch.tensor(types, dtype=torch.int64),
                    torch.tensor(pos, dtype=torch.int64))
    return out


def apply_static_group(code: torch.Tensor, fmt: C.Format, index, bits, types, pos) -> torch.Tensor:
    """Vectorized equivalent of `for f in faults (in pos order): code[f.index] = apply_op(code[f.index], ...)`.
    Returns the new code tensor (same shape / dtype); `code` is not modified. Faults on the same (cell, bit) compose
    in pos order: the bit ends at (value of the last stuck-at, else its clean value) xor (parity of the flips after
    that stuck-at, or of all flips)."""
    dev = code.device
    index, bits, types, pos = index.to(dev), bits.to(dev), types.to(dev), pos.to(dev)
    shape = code.shape
    flat = torch.zeros(index.shape[0], dtype=torch.int64, device=dev)
    for k, n in enumerate(shape):
        flat = flat * n + index[:, k]
    key = flat * fmt.bits + bits
    uk, inv = torch.unique(key, return_inverse=True)
    n = uk.numel()
    stuck = types != 0
    last = torch.full((n,), -1, dtype=torch.int64, device=dev).scatter_reduce(
        0, inv, torch.where(stuck, pos, torch.full_like(pos, -1)), reduce="amax")
    forced = last >= 0
    is_last = stuck & (pos == last[inv])
    fval = torch.zeros(n, dtype=torch.int64, device=dev).index_add_(0, inv, (is_last & (types == 2)).long())
    flips = (types == 0) & (pos > last[inv])
    par = torch.zeros(n, dtype=torch.int64, device=dev).index_add_(0, inv, flips.long()) % 2
    cell, b = uk // fmt.bits, uk % fmt.bits
    one = torch.ones_like(b) << b
    numel = max(1, code.numel())
    fmask = torch.zeros(numel, dtype=torch.int64, device=dev).index_add_(0, cell, torch.where(forced, one, 0))
    fv = torch.zeros(numel, dtype=torch.int64, device=dev).index_add_(0, cell, torch.where(forced, (fval ^ par) * one, 0))
    xm = torch.zeros(numel, dtype=torch.int64, device=dev).index_add_(0, cell, torch.where(forced, 0, par * one))
    p = C.to_pattern(code.reshape(-1), fmt)
    p = ((p & ~fmask) | fv) ^ xm
    return C.from_pattern(p, fmt).to(code.dtype).reshape(shape)


def unravel(flat: torch.Tensor, shape) -> torch.Tensor:
    """Flat indices -> [K, len(shape)] multi-indices (torch.unravel_index is not in torch 2.0)."""
    out = []
    for n in reversed(shape):
        out.append(flat % n)
        flat = torch.div(flat, n, rounding_mode="floor")
    return torch.stack(out[::-1], 1)


def cell_scale(net, layer, field, o=None):
    sc = net.scale(layer, field)
    if sc is not None and sc.dim() > 0:
        sc = sc.reshape(-1)[o] if o is not None else sc
    return sc


def cell_value(net, f: Fault, code: torch.Tensor) -> torch.Tensor:
    """Decoded value of one M1/M2 cell holding `code` (uses that cell's scale)."""
    sc = cell_scale(net, f.layer, f.field, f.index[0])
    return C.decode(code.reshape(()), net.fmt[f.layer][f.field], sc)


def faulty_code(net, f: Fault) -> torch.Tensor:
    """Code of the faulted M1/M2 cell (a 0-dim tensor)."""
    c = net.code(f.layer, f.field)[f.index]
    return apply_op(c, net.fmt[f.layer][f.field], f.ftype, f.bit)


def masked_by_construction(net, f: Fault) -> bool:
    """A stuck-at on a stored-once cell whose bit already holds the stuck value changes nothing."""
    if f.field not in STATIC_FIELDS or f.ftype not in ("sa0", "sa1"):
        return False
    return int(faulty_code(net, f)) == int(net.code(f.layer, f.field)[f.index])


class HWInjector:
    """Injects a list of faults into a DeployedSHD; see the module docstring."""

    def __init__(self, net, faults=()):
        self.net = net
        self.faults = list(faults)
        self.batch_start, self.batch_size = 0, None
        self._saved = {}
        self._prev_injector = None
        self.active = False
        self._perm_masks = {}
        self._index()

    # ------------------------------------------------------------------------------------------
    def _index(self):
        self.perm_static = [f for f in self.faults if f.field in STATIC_FIELDS and f.persistence == "permanent"]
        self.trans_static = defaultdict(list)                       # layer -> faults
        self.perm_state = defaultdict(list)                         # lif -> faults
        self.trans_state = defaultdict(lambda: defaultdict(list))   # lif -> t -> faults
        for f in self.faults:
            if f.field in STATIC_FIELDS:
                if f.persistence == "transient":
                    self.trans_static[f.layer].append(f)
            elif f.persistence == "permanent":
                self.perm_state[LIF[f.layer]].append(f)
            else:
                self.trans_state[LIF[f.layer]][f.t].append(f)
        self._perm_masks = {}
        self._static_groups = None

    def set_batch(self, start: int, size: int):
        """Global index of the first sample of the next forward pass, and the batch size."""
        self.batch_start, self.batch_size = int(start), int(size)

    def _rows(self, f: Fault, B: int):
        """Batch rows hit by a transient fault (all rows if sample is None), or None."""
        if f.sample is None:
            return slice(None)
        r = f.sample - self.batch_start
        return r if 0 <= r < B else None

    # ------------------------------------------------------------------------------------------
    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *exc):
        self.restore()
        return False

    def permanent_static_faults(self):
        return self.perm_static

    def static_groups(self):
        """Permanent M1/M2 faults grouped per (layer, field) as index tensors (cached; see static_fault_groups)."""
        if getattr(self, "_static_groups", None) is None:
            self._static_groups = static_fault_groups(self.perm_static)
        return self._static_groups

    @torch.no_grad()
    def apply(self):
        if self.active:
            return
        self._saved = {}
        for (layer, field), g in self.static_groups().items():      # vectorized per (layer, field), same result
            code = self.net.code(layer, field)                        # as the per-fault loop in fault order
            self._saved[(layer, field)] = code.clone()
            code.copy_(apply_static_group(code, self.net.fmt[layer][field], *g))
        self._prev_injector = self.net.FaultInjector
        self.net.FaultInjector = self
        self.active = True

    @torch.no_grad()
    def restore(self):
        for (layer, field), orig in self._saved.items():
            self.net.code(layer, field).copy_(orig)
        self._saved = {}
        if self.active:
            self.net.FaultInjector = self._prev_injector
        self.active = False

    # ------------------------------------------------------------------------------------------
    # transient M1/M2, called by DeployedSHD.layer_forward
    def transient_static_items(self, name: str, B: int, device):
        """{field: (rows [K], cell [K, d], bits [K], types [K], t0 [K])} for this batch."""
        acc = defaultdict(lambda: ([], [], [], [], []))
        for f in self.trans_static.get(name, ()):
            r = self._rows(f, B)
            if r is None:
                continue
            rows = list(range(B)) if isinstance(r, slice) else [r]
            a = acc[f.field]
            a[0].extend(rows)
            a[1].extend([list(f.index)] * len(rows))
            a[2].extend([f.bit] * len(rows))
            a[3].extend([TYPE_ID[f.ftype]] * len(rows))
            a[4].extend([f.t] * len(rows))
        return {fld: tuple(torch.tensor(v, dtype=torch.int64, device=device) for v in a) for fld, a in acc.items()}

    def layer_hook(self, net, name, x, counts, bias, leak, th):
        items = self.transient_static_items(name, counts.shape[0], counts.device)
        if not items:
            return counts, bias, leak, th
        return apply_transient_static(net, name, x, counts, bias, leak, th, items)

    # ------------------------------------------------------------------------------------------
    # M3 / M4: SpikingJET hook interface
    def _permanent_state_masks(self, name, n, device):
        if name not in self._perm_masks:
            fmt = self.net.fmt[BLOCK[name]]["membrane"]
            xor = torch.zeros(n, dtype=torch.int64, device=device)
            setm, clr = torch.zeros_like(xor), torch.zeros_like(xor)
            dead = torch.zeros(n, dtype=torch.bool, device=device)
            sat = torch.zeros_like(dead)
            for f in self.perm_state.get(name, ()):
                j = f.index[0]
                if f.field == "membrane":
                    m = 1 << f.bit
                    if f.ftype == "flip":
                        xor[j] ^= m
                    elif f.ftype == "sa1":
                        setm[j] |= m
                        clr[j] &= ~m
                    else:
                        clr[j] |= m
                        setm[j] &= ~m
                elif f.ftype == "dead":
                    dead[j], sat[j] = True, False
                else:
                    sat[j], dead[j] = True, False
            has_mem = bool((xor | setm | clr).any())
            self._perm_masks[name] = (fmt, has_mem, xor, setm, clr, bool(dead.any()) or bool(sat.any()), dead, sat)
        return self._perm_masks[name]

    @torch.no_grad()
    def injection(self, spike, mem, name, t):
        if self.perm_state.get(name):
            fmt, has_mem, xor, setm, clr, has_spk, dead, sat = self._permanent_state_masks(name, mem.shape[1], mem.device)
            if has_mem:
                apply_membrane_masks(mem, fmt, xor, setm, clr)
            if has_spk:
                spike.masked_fill_(dead.expand_as(spike), 0.0)
                spike.masked_fill_(sat.expand_as(spike), 1.0)
        trans = self.trans_state[name].get(t) if name in self.trans_state else None
        if trans:
            B = mem.shape[0]
            fmt = self.net.fmt[BLOCK[name]]["membrane"]
            for f in trans:
                rows = self._rows(f, B)
                if rows is None:
                    continue
                j = f.index[0]
                if f.field == "membrane":
                    code = C.encode(mem[rows, j], fmt, None)
                    mem[rows, j] = C.decode(apply_op(code, fmt, f.ftype, f.bit), fmt, None).to(mem.dtype)
                else:
                    spike[rows, j] = 0.0 if f.ftype == "drop" else 1.0


class DrawnInjector(HWInjector):
    T_CHUNK = 32                       # timesteps of transient M3/M4 faults drawn at once
    """Injector of one BER realization drawn by hwfault.draw.FaultDrawer: permanent faults are
    drawn once (at construction), transient faults lazily per batch (M1/M2, per sample) and per
    timestep (M3/M4, per sample and step)."""

    def __init__(self, net, drawer, device=None, extra_faults=()):
        self.drawer = drawer
        dev = device or net.code("block1", "weight").device
        self.permanent = drawer.permanent_faults(dev)
        self.transient_counts = defaultdict(int)
        self._state_draw, self._state_ctx, self._state_lo = {}, {}, {}
        super().__init__(net, list(self.permanent) + list(extra_faults))

    def _samples(self, B, device):
        return torch.arange(self.batch_start, self.batch_start + B, device=device)

    def layer_hook(self, net, name, x, counts, bias, leak, th):
        B, Tp, n = counts.shape
        samples = self._samples(B, counts.device)
        self._state_ctx[LIF[name]] = (samples, Tp, n)                  # M3/M4 drawn lazily per chunk
        self._state_draw[LIF[name]] = {}
        self._state_lo.pop(LIF[name], None)
        items = self.transient_static_items(name, B, counts.device)
        drawn = self.drawer.transient_static(name, samples, Tp)
        for site, rows, cell, bits, t0, types in drawn:
            self.transient_counts[site.cls] += int(rows.numel())
            if site.field in items:
                old = items[site.field]
                items[site.field] = tuple(torch.cat([a, b]) for a, b in zip(old, (rows, cell, bits, types, t0)))
            else:
                items[site.field] = (rows, cell, bits, types, t0)
        if not items:
            return counts, bias, leak, th
        return apply_transient_static(net, name, x, counts, bias, leak, th, items)

    @staticmethod
    def _index_by_step(draw):
        """{'membrane': {t: (row, col, xor, set, clr)}, 'spike': {t: (row, col)}}."""
        out = {}
        for key, val in draw.items():
            if val is None:
                continue
            t = val[0]
            uniq, counts = torch.unique_consecutive(t, return_counts=True)
            ends = torch.cumsum(counts, 0).tolist()
            starts = [0] + ends[:-1]
            out[key] = {int(u): tuple(v[s:e] for v in val[1:]) for u, s, e in zip(uniq.tolist(), starts, ends)}
        return out

    def _state_chunk(self, name, t):
        """Draw the transient M3/M4 faults of steps [t, t + T_CHUNK) of layer `name` (once)."""
        samples, Tp, n = self._state_ctx[name]
        lo = (t // self.T_CHUNK) * self.T_CHUNK
        if self._state_lo.get(name) == lo:
            return self._state_draw[name]
        full = self.drawer.transient_state_layer(name, samples, min(Tp, lo + self.T_CHUNK), n,
                                                 t_chunk=self.T_CHUNK, t_start=lo)
        self._state_draw[name] = self._index_by_step(full)
        self._state_lo[name] = lo
        return self._state_draw[name]

    @torch.no_grad()
    def injection(self, spike, mem, name, t):
        super().injection(spike, mem, name, t)
        if name not in self._state_ctx:
            return
        d = self._state_chunk(name, t)
        if not d:
            return
        m = d.get("membrane", {}).get(t)
        if m is not None:
            r, c, xor, setm, clr = m
            self.transient_counts["M3"] += int(r.numel())
            fmt = self.net.fmt[BLOCK[name]]["membrane"]
            v = mem[r, c]
            p = C.to_pattern(C.encode(v, fmt, None), fmt)
            # several faulty bits of one word: combine them on the stored pattern
            if r.numel() > 1:
                key = r * mem.shape[1] + c
                uk, inv = torch.unique(key, return_inverse=True)
                if uk.numel() != key.numel():
                    agg = [torch.zeros(uk.numel(), dtype=torch.int64, device=mem.device).index_add_(0, inv, q)
                           for q in (xor, setm, clr)]
                    rr, cc = uk // mem.shape[1], uk % mem.shape[1]
                    v = mem[rr, cc]
                    p = C.to_pattern(C.encode(v, fmt, None), fmt)
                    r, c, (xor, setm, clr) = rr, cc, agg
            p = ((p ^ xor) | setm) & ~clr
            mem[r, c] = C.decode(C.from_pattern(p, fmt), fmt, None).to(mem.dtype)
        sp = d.get("spike", {}).get(t)
        if sp is not None:
            r, c = sp
            self.transient_counts["M4"] += int(r.numel())
            spike[r, c] = 1.0 - spike[r, c]


# ----------------------------------------------------------------------------------------------
def apply_membrane_masks(mem, fmt, xor, setm, clr):
    """mem <- decode(((pattern ^ xor) | set) & ~clear), in place."""
    p = C.to_pattern(C.encode(mem, fmt, None), fmt)
    p = ((p ^ xor) | setm) & ~clr
    mem.copy_(C.decode(C.from_pattern(p, fmt), fmt, None).to(mem.dtype))


def _sequential_codes(clean, keys, t0, bits, types, fmt):
    """prev/new code of every upset; upsets of the same (row, cell) are applied cumulatively in
    onset order, so the code from upset k on includes upsets 1..k."""
    order = torch.argsort(t0, stable=True)
    order = order[torch.argsort(keys[order], stable=True)]
    k_s = keys[order]
    prev = clean[order].clone()
    new = apply_ops(prev, fmt, bits[order], types[order])
    dup = torch.zeros_like(k_s, dtype=torch.bool)
    dup[1:] = k_s[1:] == k_s[:-1]
    if bool(dup.any()):                          # rare: chain within groups
        idx = torch.nonzero(dup).squeeze(1).tolist()
        new_l, prev_l = new.tolist(), prev.tolist()
        b_l, ty_l = bits[order].tolist(), types[order].tolist()
        for j in idx:
            prev_l[j] = new_l[j - 1]
            new_l[j] = int(apply_ops(torch.tensor([prev_l[j]]), fmt, torch.tensor([b_l[j]]),
                                     torch.tensor([ty_l[j]]))[0])
        new = torch.tensor(new_l, dtype=torch.int64, device=clean.device)
        prev = torch.tensor(prev_l, dtype=torch.int64, device=clean.device)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return prev[inv], new[inv]


def apply_transient_static(net, name, x, counts, bias, leak, th, items):
    """Apply transient M1/M2 upsets (items: field -> (rows, cell, bits, types, t0)) of layer
    `name` for the batch: weight/delay upsets change the synaptic accumulation in the code domain
    (exact), bias upsets replace the bias of the row from t0 on, threshold/leak become
    time-varying (see module docstring). Returns (counts, bias, leak, th)."""
    from models.deployed_shd import delay_used
    B, Tp, n_out = counts.shape
    counts = counts.clone()
    dev = counts.device
    tt = torch.arange(Tp, device=dev)
    leak_changes, th_changes = [], []
    w_op, _ = net.weight_operand(name)
    d_clean = delay_used(net.value(name, "delay"))
    for field, (rows, cell, bits, types, t0) in items.items():
        if rows.numel() == 0:
            continue
        code = net.code(name, field)
        fmt = net.fmt[name][field]
        shape = tuple(code.shape)
        flat = cell[:, 0] * shape[1] + cell[:, 1] if len(shape) == 2 else cell[:, 0]
        clean = code.reshape(-1)[flat]
        prev, new = _sequential_codes(clean, rows * code.numel() + flat, t0, bits, types, fmt)
        if field in ("weight", "delay"):
            o, i = cell[:, 0], cell[:, 1]
            if field == "weight":
                if fmt.kind in ("int", "uint"):
                    w_prev, w_new = prev.float(), new.float()            # code domain
                else:
                    w_prev, w_new = C.decode(prev, fmt), C.decode(new, fmt)
                d_prev = d_new = d_clean[o, i]
            else:
                w_prev = w_new = w_op[o, i]
                d_prev, d_new = delay_used(C.decode(prev, fmt)), delay_used(C.decode(new, fmt))
            _add_synapse_deltas(counts, x, rows, o, i, w_prev, d_prev, w_new, d_new, t0, tt)
        elif field == "bias":
            o = cell[:, 0]
            vals = C.decode(new, fmt)
            if bias.dim() == 1:
                bias = bias.reshape(1, 1, -1).expand(B, Tp, n_out).clone()
            order = torch.argsort(t0, stable=True)              # later onsets overwrite earlier ones
            for k in order.tolist():
                bias[rows[k], int(t0[k]):, o[k]] = vals[k]
        else:
            vals = C.decode(new, fmt)
            lst = leak_changes if field == "leak" else th_changes
            for r, c0, v, s0 in zip(rows.tolist(), cell[:, 0].tolist(), vals.tolist(), t0.tolist()):
                lst.append((s0, r, c0, v))
    if leak_changes:
        leak = _schedule(leak, leak_changes, B, n_out)
    if th_changes:
        th = _schedule(th, th_changes, B, n_out)
    return counts, bias, leak, th


def _add_synapse_deltas(cur, x, rows, o, i, w_prev, d_prev, w_new, d_new, t0, tt):
    """cur[r, t, o] += w_new x[r, t - d_new, i] - w_prev x[r, t - d_prev, i] for t >= t0."""
    Tp = cur.shape[1]
    T = x.shape[1]
    xp = torch.nn.functional.pad(x.float(), (0, 0, 24, max(0, Tp - T)))   # time index t - d + 24
    for s in range(0, rows.numel(), CHUNK):
        sl = slice(s, s + CHUNK)
        r, ii = rows[sl, None], i[sl, None]
        x_new = xp[r, tt[None, :] - d_new[sl, None] + 24, ii]
        x_old = xp[r, tt[None, :] - d_prev[sl, None] + 24, ii]
        val = (w_new[sl, None] * x_new - w_prev[sl, None] * x_old) * (tt[None, :] >= t0[sl, None])
        cur.index_put_((r.expand_as(val), tt[None, :].expand_as(val), o[sl, None].expand_as(val)),
                       val.to(cur.dtype), accumulate=True)


def _schedule(base, changes, B, n):
    """Time-varying parameter: base (tensor [1] or [n]) with per-row values from their onset on.
    changes: [(t0, row, cell, value)], applied in onset order (the latest active one wins)."""
    changes = sorted(changes)
    base_t = base.reshape(1, -1).expand(B, base.numel())

    def at(t):
        active = [c for c in changes if c[0] <= t]
        if not active:
            return base
        out = base_t.clone()
        for _, r, c0, v in active:
            out[r, c0] = v
        return out
    return at
