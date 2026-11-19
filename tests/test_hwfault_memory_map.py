"""Task 4.1: memory map, profile-dependent fault space, hardware-target switch."""
import os
import subprocess
import sys

import pytest
import torch

from hwfault import codec as C
from hwfault.memory_map import MemoryMap, HardwareTargetNotImplemented, persistence_mode
from models.deployed_shd import DeployedSHD

SJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _net(profile="default", hidden=256):
    fr = [10, 10, 9] if C.load_profile(profile)["membrane"].kind == "fixed" else None
    return DeployedSHD(profile, (140, hidden, hidden, 20), fr)


def spikingjet_points():
    """Injection points SpikingJET's FaultListGenerator + hook offer for SHD_DELAY."""
    from models.snn_sota import SHD_DELAY
    net = SHD_DELAY()
    params = [n for n, _ in net.named_parameters()]
    states = [f"{l}.{k}" for l in ("lif1", "lif2", "lif_out") for k in ("potential", "spike")]
    return params + states


def test_every_spikingjet_point_mapped_or_excluded():
    cov = MemoryMap(_net()).spikingjet_coverage(spikingjet_points())
    assert len(cov) == 3 * 1 + 2 * 2 + 3 * 2 + 6        # weights, bn w/b, beta+threshold, states
    unmapped = {k: v for k, v in cov.items() if v[0] == "unmapped"}
    assert not unmapped, unmapped
    excluded = {k for k, v in cov.items() if v[0] == "excluded"}
    assert excluded == {"block_out.lif.threshold", "lif_out.spike"}


def test_sites_per_layer_and_classes():
    mm = MemoryMap(_net())
    fields = {(s.layer, s.field): s for s in mm.sites}
    for b in ("block1", "block2"):
        for f in ("weight", "delay", "bias", "threshold", "leak", "membrane", "spike"):
            assert (b, f) in fields
    assert {f for (l, f) in fields if l == "block_out"} == {"weight", "delay", "leak", "membrane"}
    assert fields[("block1", "weight")].cls == "M1" and fields[("block2", "leak")].cls == "M2"
    assert fields[("block_out", "membrane")].cls == "M3" and fields[("block1", "spike")].cls == "M4"
    assert fields[("block1", "weight")].cells == 35840 and fields[("block1", "threshold")].cells == 1
    assert fields[("block1", "membrane")].steps == 162 and fields[("block_out", "membrane")].steps == 186


@pytest.mark.parametrize("profile,bits", [("default", 8), ("int4w", 4), ("fp32", 32)])
def test_weight_bits_follow_profile(profile, bits):
    mm = MemoryMap(_net(profile))
    assert mm.site("block2", "weight").bits == bits
    if profile == "fp32":
        assert all(s.bits == 32 for s in mm.sites if s.field != "spike")
    if profile == "int4w":
        assert mm.site("block2", "delay").bits == 5 and mm.site("block1", "membrane").bits == 16


def test_fp32_bit30_is_exponent():
    f = C.Format("float", 32)
    v = torch.tensor([0.0123])
    out = C.decode(C.flip(C.encode(v, f), f, 30), f)
    assert abs(float(out)) > 1e30                    # exponent MSB set


def test_persistence_presets_and_types():
    assert persistence_mode("mixed") == {"M1": "permanent", "M2": "permanent", "M3": "transient", "M4": "transient"}
    mm = MemoryMap(_net(), mode="permanent")
    assert all(s.persistence == "permanent" for s in mm.sites)
    assert mm.site("block1", "spike").fault_types == ("dead", "saturated")
    assert MemoryMap(_net()).site("block1", "spike").fault_types == ("drop", "spurious")
    with pytest.raises(ValueError):
        persistence_mode("sometimes")


def test_gpu_target_not_implemented():
    with pytest.raises(HardwareTargetNotImplemented):
        MemoryMap(_net(), hw_target="gpu")


def test_cli_gpu_target_exits_without_results(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "hw_campaign.py", "catalog", "--checkpoint", "missing.th",
                        "--hw-target", "gpu", "--out", str(out), "--faults", "x.csv.gz"],
                       cwd=SJ, capture_output=True, text=True)
    assert r.returncode != 0 and "not implemented" in r.stderr
    assert not out.exists()


def test_cli_map_lists_sites(tmp_path):
    p = tmp_path / "d.th"
    _net().save(str(p))
    r = subprocess.run([sys.executable, "hw_campaign.py", "map", "--checkpoint", str(p)], cwd=SJ,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for word in ("M1", "M2", "M3", "M4", "membrane", "spike", "int8", "uint5", "persistence"):
        assert word in r.stdout
