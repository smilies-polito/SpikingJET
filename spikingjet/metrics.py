"""Effect of a fault on the network outputs (single-fault campaign classes)."""
from __future__ import annotations

import torch

COLUMNS = ["# of faults", "# of injections", "fully_masked", "SDC_1", "SDC_0_5%", "SDC_5_10%",
           "SDC_10_20%", "SDC_20", "facc", "acc"]


def sdc_interval(faulty: torch.Tensor, clean: torch.Tensor):
    """Fractions of the batch: fully masked (same top-1 and same top score), SDC_1 (top-1 changed),
    and, with the same top-1, the relative change of the top score in (0,5) / (5,10) / (10,20) /
    >20 %."""
    clean_v, clean_i = torch.max(clean, dim=1)
    faulty_v, faulty_i = torch.max(faulty, dim=1)
    same_i = torch.eq(clean_i, faulty_i)
    same_v = torch.eq(clean_v, faulty_v)
    diff = 100 * torch.abs(clean_v - faulty_v) / clean_v
    fully_masked = (same_i & same_v).float().mean()
    sdc1 = 1 - same_i.float().mean()
    s0_5 = (same_i & (diff < 5) & ~same_v).float().mean()
    s5_10 = (same_i & (diff < 10) & (diff > 5)).float().mean()
    s10_20 = (same_i & (diff < 20) & (diff > 10)).float().mean()
    s20 = (same_i & (diff > 20)).float().mean()
    return tuple(float(v) for v in (fully_masked, sdc1, s0_5, s5_10, s10_20, s20))


def accuracy_of(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(1) == labels).float().mean())


class LayerSummary:
    """Per-layer (site) accumulation, weighted by batch size: the table analyse.py writes."""

    def __init__(self):
        self.rows = {}

    def add(self, site, fault_id, faulty, clean, labels):
        r = self.rows.setdefault(site, {"faults": set(), "inj": 0, "n": 0, "sums": [0.0] * 8})
        r["faults"].add(fault_id)
        r["inj"] += 1
        n = labels.shape[0]
        r["n"] += n
        vals = sdc_interval(faulty, clean) + (accuracy_of(faulty, labels), accuracy_of(clean, labels))
        for i, v in enumerate(vals):
            r["sums"][i] += v * n

    def table(self):
        import pandas as pd
        data = {site: [len(r["faults"]), r["inj"]] + [s / r["n"] for s in r["sums"]]
                for site, r in self.rows.items() if r["n"]}
        return pd.DataFrame(data=data, index=COLUMNS)
