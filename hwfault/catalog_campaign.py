"""
Single-fault damage catalog, fault-parallel.

Each measurement has exactly ONE active fault. To use the GPU efficiently, K faults of the same
layer are evaluated in one pass, one per network replica, the replicas laid out along the batch
axis ([K*B, ...]):
  * the layers before the faulty one are computed once per data batch and cached;
  * M1 faults (weight / delay) change one output column of the code-domain synaptic
    accumulation of the faulty layer: counts_k = counts + delta_k (exact integer arithmetic);
  * M2 faults (bias / threshold / leak) give each replica its own configuration values;
  * M3 / M4 faults are applied in the LIF hook to the rows of their replica only;
  * the layers after the faulty one run on the clean codes over all K*B rows.
Because the integer accumulations are exact in FP32 and every other operation is elementwise per
row, the replica results equal sequential single-fault runs (tested), for integer weight formats.

Records hold the fault identity, golden / faulty stored codes (M1/M2), and the metrics of
hwfault.metrics (dacc, dconf, D, flips, kl, ...). Writing is incremental and resumable.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import time

import torch

from . import codec as C
from .injector import (Fault, HWInjector, apply_membrane_masks, apply_op, cell_value, faulty_code,
                       masked_by_construction, STATIC_FIELDS, TYPE_ID, _sequential_codes)
from .memory_map import BLOCK, LIF
from .metrics import Accumulator

LAYERS = ("block1", "block2", "block_out")
RECORD_FIELDS = ("fault_id", "cls", "layer", "field", "index", "bit", "ftype", "persistence", "t",
                 "golden_code", "faulty_code", "golden_value", "faulty_value", "masked",
                 "n", "clean_acc", "acc", "dacc", "dconf", "D", "flips", "kl", "nonfinite", "seconds")


def fault_id(f: Fault) -> str:
    """Stable identity string (catalog key, resume, de-duplication)."""
    idx = "/".join(str(i) for i in f.index)
    return f"{f.cls}:{f.layer}:{f.field}:{idx}:{f.bit}:{f.ftype}:{f.persistence}:{'' if f.t is None else f.t}"


# ----------------------------------------------------------------------------------------------
class DataCache:
    """Inputs of every layer and clean outputs for the whole evaluation split, on the device."""

    def __init__(self, net, loader, device, chunk: int = 256):
        self.x, self.s1, self.s2, self.out, self.y = [], [], [], [], []
        self.c2, self.c3 = [], []                       # clean counts of block2 / block_out
        self.v1, self.v2, self.v3 = [], [], []          # clean membranes per layer
        saved = net.FaultInjector
        net.FaultInjector = None
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device).float()
                for s in range(0, x.shape[0], chunk):
                    xb = x[s:s + chunk]
                    s1, v1 = net.layer_forward("block1", xb)
                    s2, v2 = net.layer_forward("block2", s1)
                    _, v = net.layer_forward("block_out", s2)
                    self.v1.append(v1)
                    self.v2.append(v2)
                    self.v3.append(v)
                    self.x.append(xb)
                    self.s1.append(s1.to(torch.uint8))
                    self.s2.append(s2.to(torch.uint8))
                    self.c2.append(net.layer_counts("block2", s1))
                    self.c3.append(net.layer_counts("block_out", s2))
                    self.out.append(torch.softmax(v, 2).sum(1))
                    self.y.append(y[s:s + chunk].to(device))
        net.FaultInjector = saved
        self.n = sum(t.shape[0] for t in self.y)
        self.clean_acc = sum(int((o.argmax(1) == y).sum()) for o, y in zip(self.out, self.y)) / max(self.n, 1)

    def batches(self):
        return range(len(self.y))

    def layer_input(self, layer: str, b: int) -> torch.Tensor:
        return {"block1": self.x, "block2": self.s1, "block_out": self.s2}[layer][b].float()


class ReplicaStateInjector:
    """M3/M4 single faults, one per replica (rows k*B .. (k+1)*B - 1), in the LIF hook. Only the
    (rows, neuron) entries of the faulty replicas are touched."""

    def __init__(self, net, lif, faults, B, n, device):
        self.lif = lif
        self.B = B
        self.fmt = net.fmt[BLOCK[lif]]["membrane"]
        ar = torch.arange(B, device=device)
        mem_r, mem_c, xor, setm, clr = [], [], [], [], []
        dead_r, dead_c, sat_r, sat_c = [], [], [], []
        self.trans = {}
        for k, f in enumerate(faults):
            if f is None or f.field not in ("membrane", "spike"):
                continue
            j = f.index[0]
            rows = ar + k * B
            if f.persistence == "permanent":
                if f.field == "membrane":
                    m = 1 << f.bit
                    mem_r.append(rows)
                    mem_c.append(torch.full_like(rows, j))
                    for lst, on in ((xor, f.ftype == "flip"), (setm, f.ftype == "sa1"), (clr, f.ftype == "sa0")):
                        lst.append(torch.full_like(rows, m if on else 0))
                elif f.ftype == "dead":
                    dead_r.append(rows)
                    dead_c.append(torch.full_like(rows, j))
                else:
                    sat_r.append(rows)
                    sat_c.append(torch.full_like(rows, j))
            else:
                self.trans.setdefault(f.t, []).append((k, f))
        cat = lambda l: torch.cat(l) if l else None        # noqa: E731
        self.mem_idx = (cat(mem_r), cat(mem_c)) if mem_r else None
        self.xor, self.setm, self.clr = cat(xor), cat(setm), cat(clr)
        self.dead_idx = (cat(dead_r), cat(dead_c)) if dead_r else None
        self.sat_idx = (cat(sat_r), cat(sat_c)) if sat_r else None

    @torch.no_grad()
    def injection(self, spike, mem, name, t):
        if name != self.lif:
            return
        if self.mem_idx is not None:
            r, c = self.mem_idx
            v = mem[r, c]
            p = C.to_pattern(C.encode(v, self.fmt, None), self.fmt)
            p = ((p ^ self.xor) | self.setm) & ~self.clr
            mem[r, c] = C.decode(C.from_pattern(p, self.fmt), self.fmt, None).to(mem.dtype)
        if self.dead_idx is not None:
            spike[self.dead_idx] = 0.0
        if self.sat_idx is not None:
            spike[self.sat_idx] = 1.0
        for k, f in self.trans.get(t, ()):
            rows = slice(k * self.B, (k + 1) * self.B)
            j = f.index[0]
            if f.field == "membrane":
                code = C.encode(mem[rows, j], self.fmt, None)
                mem[rows, j] = C.decode(apply_op(code, self.fmt, f.ftype, f.bit), self.fmt, None).to(mem.dtype)
            else:
                spike[rows, j] = 0.0 if f.ftype == "drop" else 1.0


class CatalogRunner:
    """Measures single faults on a DeployedSHD over a DataCache, K faults per pass."""

    def __init__(self, net, cache: DataCache, replicas: int = 64, max_rows: int = 8192,
                 sparse: bool = True, dense_switch: float = 0.25):
        self.net = net
        self.cache = cache
        self.K = replicas
        self.max_rows = max_rows
        self.sparse = sparse                   # propagate only changed rows / spikes downstream
        self.dense_switch = dense_switch       # use a dense conv when the update would be larger

    # ------------------------------------------------------------------------------------------
    def _replica_layer(self, layer, faults, x):
        """Faulty layer for K replicas: returns spikes [K*B, T', n] (or readout membrane)."""
        from models.deployed_shd import delay_used, lif_scan
        net = self.net
        K, (B, _, _) = len(faults), x.shape
        counts = net.layer_counts(layer, x)                               # [B, T', n]
        Tp, n = counts.shape[1], counts.shape[2]
        _, sc = net.weight_operand(layer)
        readout = layer == "block_out"
        rows_per = torch.arange(B, device=x.device)
        reps = counts.unsqueeze(0).repeat(K, 1, 1, 1)                     # [K, B, T', n]
        bias = None if readout else net.value(layer, "bias")
        bias_rep = None if readout else bias.reshape(1, 1, 1, n).repeat(K, 1, Tp, 1)
        leak = net.value(layer, "leak")
        th = None if readout else net.value(layer, "threshold")
        leak_rep = leak.reshape(1, -1).repeat(K, 1)                       # [K, 1|n]
        th_rep = None if readout else th.reshape(1, -1).repeat(K, 1)
        leak_on, th_on = {}, {}                                           # k -> onset
        tt = torch.arange(Tp, device=x.device)
        w_op, _ = net.weight_operand(layer)
        d_clean = delay_used(net.value(layer, "delay"))
        xp = torch.nn.functional.pad(x, (0, 0, 24, max(0, Tp - x.shape[1])))
        INF = 10 ** 9
        t_first, t_last = [INF] * K, [-1] * K                              # perturbation window
        for k, f in enumerate(faults):
            if f is None:
                continue
            if f.field not in STATIC_FIELDS:                               # M3 / M4
                t_first[k], t_last[k] = (0, INF) if f.persistence == "permanent" else (f.t, f.t)
                continue
            t0 = 0 if f.persistence == "permanent" else f.t
            if f.field not in ("weight", "delay"):                         # M2: from t0 to the end
                t_first[k], t_last[k] = t0, INF
            fmt = net.fmt[layer][f.field]
            new = faulty_code(net, f)
            if f.field in ("weight", "delay"):
                o, i = f.index
                if f.field == "weight":
                    w0 = w_op[o, i]
                    w1 = new.float() if fmt.kind in ("int", "uint") else C.decode(new, fmt)
                    d0 = d1 = int(d_clean[o, i])
                else:
                    w0 = w1 = w_op[o, i]
                    d0, d1 = int(d_clean[o, i]), int(delay_used(C.decode(new.reshape(1), fmt))[0])
                delta = w1 * xp[:, tt - d1 + 24, i] - w0 * xp[:, tt - d0 + 24, i]      # [B, T']
                if t0:
                    delta = delta * (tt >= t0)
                active = torch.nonzero(delta.any(0)).flatten()
                if active.numel():
                    t_first[k], t_last[k] = int(active[0]), int(active[-1])
                    reps[k, :, :, o] += delta
            elif f.field == "bias":
                bias_rep[k, 0, t0:, f.index[0]] = cell_value(net, f, new)
            elif f.field == "leak":
                leak_rep[k, f.index[0]] = cell_value(net, f, new)
                leak_on[k] = t0
            else:
                th_rep[k, f.index[0]] = cell_value(net, f, new)
                th_on[k] = t0
        cur = net.assemble_current(reps.reshape(K * B, Tp, n), sc,
                                   None if readout else bias_rep.expand(K, B, Tp, n).reshape(K * B, Tp, n))

        def param(base, rep, onsets):
            per_row = rep.repeat_interleave(B, 0)                         # [K*B, 1|n]
            if not onsets or all(v == 0 for v in onsets.values()):
                return per_row if (rep != base.reshape(1, -1)).any() else base
            clean_rows = base.reshape(1, -1).expand(K * B, base.numel())
            late = {k: v for k, v in onsets.items() if v > 0}

            def at(t):
                out = per_row.clone()
                for k, v in late.items():
                    if t < v:
                        out[k * B:(k + 1) * B] = clean_rows[k * B:(k + 1) * B]
                return out
            return at
        leak_p = param(leak, leak_rep, leak_on)
        th_p = None if readout else param(th, th_rep, th_on)
        hook = ReplicaStateInjector(net, LIF[layer], faults, B, n, x.device)
        clean_s = {"block1": self.cache.s1, "block2": self.cache.s2}.get(layer)
        clean_v = {"block1": self.cache.v1, "block2": self.cache.v2, "block_out": self.cache.v3}[layer][self._b]
        s_rows = None if clean_s is None else clean_s[self._b].float().repeat(K, 1, 1)
        return self._scan(cur, leak_p, th_p, net.qmem_fn(layer), hook.injection, LIF[layer], readout,
                          s_rows, clean_v.repeat(K, 1, 1), min(t_first), max(t_last))

    @staticmethod
    def _scan(cur, leak, th, qmem, inj, name, readout, s_clean, v_clean, t_start, t_last):
        """lif_scan restricted to the perturbed window: starts from the clean membrane of each
        row at t_start - 1 and stops as soon as every row is back on its clean trajectory after
        the last perturbation (the remaining steps are then the clean ones). s_clean / v_clean
        [R, T', n] are the clean spikes / membranes of the rows (consumed: returned filled in).
        Same operations as models.deployed_shd.lif_scan, so the results are identical."""
        Tp = cur.shape[1]
        v_all = v_clean
        s_all = torch.zeros_like(v_all) if readout else s_clean
        if t_start >= Tp:
            return s_all, v_all
        leak_t = leak if callable(leak) else (lambda t: leak)
        th_t = th if callable(th) else (lambda t: th)
        ref_v = v_all.clone() if t_last < Tp else None                    # clean trajectory
        ref_s = None if (readout or ref_v is None) else s_all.clone()
        v = torch.zeros_like(v_all[:, 0]) if t_start == 0 else v_all[:, t_start - 1].clone()
        for t in range(t_start, Tp):
            h = qmem(leak_t(t) * v + cur[:, t])
            if readout:
                s = torch.zeros_like(h)
                v = h
            else:
                s = (h - th_t(t) > 0).to(h.dtype)
                v = h * (1.0 - s.detach())
            inj(s, v, name, t)
            if not readout:
                s_all[:, t] = s
            v_all[:, t] = v
            if t >= t_last and t + 1 < Tp:
                # converged: the next step only depends on v (and the unchanged input)
                if torch.equal(v, ref_v[:, t]) and (readout or torch.equal(s, ref_s[:, t])):
                    break
        return s_all, v_all

    @torch.no_grad()
    def _pass(self, layer, faults, b):
        """Outputs [K, B, C] of the K replicas for cached batch b."""
        net = self.net
        x = self.cache.layer_input(layer, b)
        K, B = len(faults), x.shape[0]
        self._b = b
        s, v = self._replica_layer(layer, faults, x)
        if not self.sparse:
            saved, net.FaultInjector = net.FaultInjector, None
            try:
                for nxt in LAYERS[LAYERS.index(layer) + 1:]:
                    s, v = net.layer_forward(nxt, s)
            finally:
                net.FaultInjector = saved
            return torch.softmax(v, 2).sum(1).reshape(K, B, -1)
        if layer == "block_out":
            return torch.softmax(v, 2).sum(1).reshape(K, B, -1)
        return self._propagate(layer, s, b, K, B)

    def _propagate(self, layer, s, b, K, B):
        """Downstream layers on the rows whose spikes changed only; the clean counts of the next
        layer are updated by the changed spikes (exact integer arithmetic), unchanged rows reuse
        the cached clean results."""
        from models.deployed_shd import delay_used, lif_scan
        net, cache = self.net, self.cache
        out = cache.out[b].unsqueeze(0).repeat(K, 1, 1).reshape(K * B, -1)
        clean_s = {"block1": cache.s1, "block2": cache.s2}
        clean_c = {"block2": cache.c2, "block_out": cache.c3}
        clean_v = {"block2": cache.v2, "block_out": cache.v3}
        rows = torch.arange(K * B, device=s.device)               # global replica rows still tracked
        cur_layer = layer
        while True:
            nxt = LAYERS[LAYERS.index(cur_layer) + 1]
            base = clean_s[cur_layer][b].float()                  # [B, T, n]
            delta = s - base.repeat(K, 1, 1)[rows] if rows.numel() != K * B else s - base.repeat(K, 1, 1)
            changed = delta.reshape(delta.shape[0], -1).any(1)
            if not bool(changed.any()):
                break
            delta, s_ch, rows = delta[changed], s[changed], rows[changed]
            counts = clean_c[nxt][b][rows % B].clone()                 # [R, T', n_next]
            nz = torch.nonzero(delta)                                 # (r, t, i)
            n_next = counts.shape[2]
            if nz.shape[0] * n_next <= self.dense_switch * counts.numel():
                w_op, _ = net.weight_operand(nxt)
                d = delay_used(net.value(nxt, "delay"))               # [n_next, n_in]
                r, t, i = nz[:, 0], nz[:, 1], nz[:, 2]
                sign = delta[r, t, i]
                for c0 in range(0, nz.shape[0], 4096):
                    sl = slice(c0, c0 + 4096)
                    tt = t[sl, None] + d[:, i[sl]].T                  # [m, n_next]
                    val = sign[sl, None] * w_op[:, i[sl]].T
                    ok = tt < counts.shape[1]
                    oo = torch.arange(n_next, device=s.device).expand_as(tt)
                    counts.index_put_((r[sl, None].expand_as(tt)[ok], tt[ok], oo[ok]), val[ok], accumulate=True)
            else:
                counts = net.layer_counts(nxt, s_ch)
            readout = nxt == "block_out"
            _, sc = net.weight_operand(nxt)
            dc = (counts != clean_c[nxt][b][rows % B]).any(2).any(0)          # [T'] times with changes
            tch = torch.nonzero(dc).flatten()
            if tch.numel() == 0:                                             # spikes changed but cancel
                break
            cur = net.assemble_current(counts, sc, None if readout else net.value(nxt, "bias"))
            v_rows = clean_v[nxt][b][rows % B].clone()
            s_rows = None if readout else clean_s[nxt][b][rows % B].float()
            s, v = self._scan(cur, net.value(nxt, "leak"), None if readout else net.value(nxt, "threshold"),
                              net.qmem_fn(nxt), lambda *a: None, LIF[nxt], readout, s_rows, v_rows,
                              int(tch[0]), int(tch[-1]))
            if readout:
                out[rows] = torch.softmax(v, 2).sum(1)
                break
            cur_layer = nxt
        return out.reshape(K, B, -1)

    @torch.no_grad()
    def measure(self, faults, lam: float = 1.0) -> list:
        """Metrics of every fault (input order). Faults masked by construction are not run."""
        res = [None] * len(faults)
        by_layer = {}
        for idx, f in enumerate(faults):
            if masked_by_construction(self.net, f):
                ca = self.cache.clean_acc
                res[idx] = {"n": self.cache.n, "clean_acc": ca, "acc": ca, "dacc": 0.0, "dconf": 0.0,
                            "D": 0.0, "flips": 0, "kl": 0.0, "nonfinite": 0, "masked": True}
            else:
                by_layer.setdefault(f.layer, []).append(idx)
        for layer, idxs in by_layer.items():
            idxs.sort(key=lambda i: (faults[i].field, faults[i].persistence, faults[i].t or 0))
            for s in range(0, len(idxs), self.K):
                chunk = idxs[s:s + self.K]
                acc = Accumulator(len(chunk), device=self.cache.y[0].device)
                for b in self.cache.batches():
                    B = self.cache.y[b].shape[0]
                    step = max(1, self.max_rows // max(B, 1))
                    outs = []
                    for r in range(0, len(chunk), step):
                        outs.append(self._pass(layer, [faults[i] for i in chunk[r:r + step]], b))
                    acc.add(self.cache.out[b], torch.cat(outs), self.cache.y[b])
                for i, r in zip(chunk, acc.result(lam)):
                    r["masked"] = False
                    res[i] = r
        return res


@torch.no_grad()
def measure_sequential(net, faults, cache: DataCache, lam: float = 1.0) -> list:
    """Reference: one fault at a time through HWInjector and the normal forward pass."""
    out = []
    for f in faults:
        acc = Accumulator(1, device=cache.y[0].device)
        inj = HWInjector(net, [f])
        start = 0
        with inj:
            for b in cache.batches():
                B = cache.y[b].shape[0]
                inj.set_batch(start, B)
                acc.add(cache.out[b], net(cache.x[b]), cache.y[b])
                start += B
        r = acc.result(lam)[0]
        r["masked"] = masked_by_construction(net, f)
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------
# Records, incremental writing
# ----------------------------------------------------------------------------------------------
def stored_values(net, f: Fault):
    if f.field not in STATIC_FIELDS:
        return "", "", "", ""
    g = net.code(f.layer, f.field)[f.index]
    fc = faulty_code(net, f)
    return int(g), int(fc), float(cell_value(net, f, g)), float(cell_value(net, f, fc))


def read_done(path: str) -> set:
    """fault_ids already written (a truncated last line is ignored)."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]                       # interrupted write
    for r in csv.DictReader(lines):
        if r.get("seconds") not in (None, ""):
            done.add(r["fault_id"])
    return done


def _repair_tail(path: str):
    """Drop a partial last line left by an interrupted run, so appending stays valid CSV."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as fh:
        data = fh.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            fh.seek(cut)
            fh.truncate()


def run_catalog(net, faults, cache: DataCache, path: str, replicas=64, max_rows=8192, lam=1.0,
                flush_every: int = 256, log=print, extra: dict | None = None) -> int:
    """Measure `faults` not yet in `path` (CSV, appended incrementally); returns how many ran."""
    _repair_tail(path)
    done = read_done(path)
    todo = []
    seen = set(done)
    for f in faults:
        fid = fault_id(f)
        if fid not in seen:
            seen.add(fid)
            todo.append(f)
    log(f"catalog: {len(faults)} faults requested, {len(done)} already measured, {len(todo)} to run")
    if not todo:
        return 0
    runner = CatalogRunner(net, cache, replicas, max_rows)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    extra = extra or {}
    fields = list(RECORD_FIELDS) + list(extra)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        t_start = time.time()
        for s in range(0, len(todo), flush_every):
            chunk = todo[s:s + flush_every]
            t0 = time.time()
            res = runner.measure(chunk, lam)
            dt = (time.time() - t0) / len(chunk)
            for f, r in zip(chunk, res):
                g, fc, gv, fv = stored_values(net, f)
                row = {**f.to_row(), "fault_id": fault_id(f), "golden_code": g, "faulty_code": fc,
                       "golden_value": gv, "faulty_value": fv, **{k: r[k] for k in r if k in RECORD_FIELDS},
                       "seconds": f"{dt:.4f}", **extra}
                row.pop("sample", None)
                w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
            el = time.time() - t_start
            rate = (s + len(chunk)) / max(el, 1e-9)
            log(f"catalog: {s + len(chunk)}/{len(todo)} faults, {rate:.1f} faults/s, "
                f"eta {(len(todo) - s - len(chunk)) / max(rate, 1e-9) / 3600:.2f} h")
    return len(todo)


def compress(path: str) -> str:
    gz = path + ".gz"
    with open(path, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    return gz


def main_catalog(args, net) -> int:
    """hw_campaign.py catalog: measure the faults of a gzip-CSV fault list on a data split."""
    from .draw import read_faults
    from models.snn_sota import get_loader
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev).eval()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    loader = get_loader("SHD_DELAY", args.split, args.batch_size, data_root=data_root)
    cache = DataCache(net, loader, dev)
    faults = read_faults(args.faults)
    n = run_catalog(net, faults, cache, os.path.join(args.out, "records.csv"), args.replicas)
    with open(os.path.join(args.out, "catalog_meta.json"), "w") as fh:
        json.dump({"faults": args.faults, "split": args.split, "measured_now": n,
                   "profile": net.profile.to_dict(), "checkpoint": args.checkpoint}, fh, indent=1)
    return 0
