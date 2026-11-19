"""Task 5.1: damage metrics."""
import torch

from hwfault.metrics import Accumulator


def _out(B=50, C=20, T=186, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, C, generator=g)
    return torch.softmax(logits, 2).sum(1)


def test_zero_fault_gives_exact_zero():
    c, y = _out(), torch.randint(0, 20, (50,))
    a = Accumulator(3)
    a.add(c, torch.stack([c, c, c]), y)
    a.add(c[:7], torch.stack([c[:7]] * 3), y[:7])
    for r in a.result():
        assert r["dacc"] == 0.0 and r["dconf"] == 0.0 and r["D"] == 0.0 and r["kl"] == 0.0 and r["flips"] == 0


def test_metrics_values():
    c = torch.tensor([[150.0, 30.0, 6.0], [10.0, 170.0, 6.0]])          # T_out = 186
    f = torch.tensor([[20.0, 160.0, 6.0], [10.0, 170.0, 6.0]])
    y = torch.tensor([0, 1])
    a = Accumulator()
    a.add(c, f, y)
    r = a.result()[0]
    assert r["clean_acc"] == 1.0 and r["acc"] == 0.5 and r["dacc"] == 0.5 and r["flips"] == 1
    assert abs(r["dconf"] - ((150 - 20) / 186) / 2) < 1e-12
    assert abs(r["D"] - (r["dacc"] + r["dconf"])) < 1e-15 and r["kl"] > 0


def test_nonfinite_output_is_wrong():
    c = torch.tensor([[150.0, 36.0]])
    f = torch.tensor([[float("nan"), 1.0]])
    a = Accumulator()
    a.add(c, f, torch.tensor([0]))
    r = a.result()[0]
    assert r["acc"] == 0.0 and r["nonfinite"] == 1 and r["flips"] == 1
