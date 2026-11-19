"""
Damage metrics against cached clean outputs.

The network output is out = sum_t softmax(v_t) over the T_out readout steps, so
p = out / T_out is a probability vector and p_true = p[y].
Per run (single fault or BER realization), over the evaluated samples:
    acc       fraction of correct predictions (a non-finite output is never correct)
    dacc      clean acc - faulty acc                     (fraction, not percent)
    dconf     mean_i (p_true,clean - p_true,faulty)
    D         dacc + lam * dconf                          (lam = 1 by default)
    flips     number of samples whose predicted class changed
    kl        mean_i KL(p_clean || p_faulty)
All are exactly 0 when the faulty outputs equal the clean ones.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def probs(out: torch.Tensor) -> torch.Tensor:
    """[B, C] summed softmax -> probabilities (non-finite rows -> uniform)."""
    out = out.double()
    p = out / out.sum(1, keepdim=True)
    bad = ~torch.isfinite(p).all(1)
    if bool(bad.any()):
        p[bad] = 1.0 / p.shape[1]
    return p


class Accumulator:
    """Running sums for K runs at once (K = 1 for a single run); add() per batch."""

    def __init__(self, k: int = 1, device="cpu"):
        def z(dt=torch.float64):
            return torch.zeros(k, dtype=dt, device=device)
        self.n = 0
        self.correct_c, self.correct_f = z(torch.int64), z(torch.int64)
        self.pt_c, self.pt_f = z(), z()
        self.flips, self.nonfinite = z(torch.int64), z(torch.int64)
        self.kl = z()

    @torch.no_grad()
    def add(self, clean_out: torch.Tensor, faulty_out: torch.Tensor, y: torch.Tensor):
        """clean_out [B, C]; faulty_out [K, B, C] (or [B, C] for K = 1); y [B]."""
        if faulty_out.dim() == 2:
            faulty_out = faulty_out.unsqueeze(0)
        K, B, _ = faulty_out.shape
        pc = probs(clean_out)                                    # [B, C]
        pf = probs(faulty_out.reshape(K * B, -1)).reshape(K, B, -1)
        fin = torch.isfinite(faulty_out).all(2)                  # [K, B]
        pred_c = clean_out.argmax(1)
        pred_f = torch.where(fin, faulty_out.nan_to_num(0.0).argmax(2), torch.full_like(fin, -1, dtype=torch.long))
        self.n += B
        self.correct_c += int((pred_c == y).sum())
        self.correct_f += (pred_f == y).sum(1)
        self.pt_c += float(pc.gather(1, y[:, None]).sum())
        self.pt_f += pf.gather(2, y[None, :, None].expand(K, B, 1)).squeeze(2).sum(1)
        self.flips += (pred_f != pred_c).sum(1)
        self.nonfinite += (~fin).sum(1)
        # KL(pc || pf); identical outputs give exactly 0
        diff = (faulty_out != clean_out.unsqueeze(0)).any(2)     # [K, B]
        kl = (pc.unsqueeze(0) * (torch.log(pc.unsqueeze(0) + EPS) - torch.log(pf + EPS))).sum(2)
        self.kl += torch.where(diff, kl, torch.zeros_like(kl)).sum(1)

    def result(self, lam: float = 1.0) -> list:
        """One dict per run."""
        n = max(self.n, 1)
        acc_c = self.correct_c.double() / n
        acc_f = self.correct_f.double() / n
        dacc = (self.correct_c - self.correct_f).double() / n
        dconf = (self.pt_c - self.pt_f) / n
        out = []
        for k in range(acc_f.numel()):
            out.append({"n": self.n, "clean_acc": float(acc_c[k]), "acc": float(acc_f[k]),
                        "clean_conf": float(self.pt_c[k] / n), "conf": float(self.pt_f[k] / n),
                        "dacc": float(dacc[k]), "dconf": float(dconf[k]),
                        "D": float(dacc[k] + lam * dconf[k]), "flips": int(self.flips[k]),
                        "kl": float(self.kl[k] / n), "nonfinite": int(self.nonfinite[k])})
        return out
