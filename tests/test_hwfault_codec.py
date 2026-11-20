"""Task 3.2: format codecs and precision profiles."""
import pytest
import torch

from hwfault import codec as C


@pytest.mark.parametrize("fmt", [
    C.Format("int", 8), C.Format("int", 4), C.Format("uint", 5), C.Format("uint", 8),
    C.Format("fixed", 16, signed=True, frac=10), C.Format("fixed", 12, signed=False, frac=12),
])
def test_integer_round_trip(fmt):
    lo = -fmt.qmax if fmt.kind == "int" else fmt.qmin
    codes = torch.arange(lo, fmt.qmax + 1, dtype=torch.int64)
    scale = torch.tensor([0.37]) if fmt.kind in ("int", "uint") else None
    vals = C.decode(codes, fmt, scale)
    assert torch.equal(C.encode(vals, fmt, scale), codes)
    # pattern view is a bijection onto [0, 2**bits)
    pat = C.to_pattern(codes, fmt)
    assert int(pat.min()) >= 0 and int(pat.max()) < (1 << fmt.bits)
    assert torch.equal(C.from_pattern(pat, fmt), codes)


def test_float_round_trip_is_bit_exact():
    fmt = C.Format("float", 32)
    x = torch.tensor([0.0, -0.0, 1.0, -3.25e-7, 1e30, float("inf"), 0.004975124])
    c = C.encode(x, fmt)
    assert int(c.min()) >= 0 and int(c.max()) < (1 << 32)
    assert torch.equal(C.decode(c, fmt).view(torch.int32), x.view(torch.int32))


def test_float_flip_matches_spikingjet():
    from spikingjet.bitops import float32_bit_flip
    fmt = C.Format("float", 32)
    for v in (0.731, -1.0, 2.5e-3):
        for b in (0, 13, 23, 30, 31):
            got = float(C.decode(C.flip(C.encode(torch.tensor([v]), fmt), fmt, b), fmt))
            assert got == pytest.approx(float32_bit_flip(v, b), nan_ok=True, rel=0, abs=0)


def test_signed_flip_and_stuck():
    fmt = C.Format("int", 8)
    c = torch.tensor([5])
    assert int(C.flip(c, fmt, 7)) == 5 - 128          # sign bit
    assert int(C.flip(c, fmt, 1)) == 7
    assert int(C.stuck(c, fmt, 0, 1)) == 5            # already 1: masked
    assert int(C.stuck(c, fmt, 0, 0)) == 4
    with pytest.raises(ValueError):
        C.flip(c, fmt, 8)


def test_symmetric_int_encoding_and_saturation():
    fmt = C.Format("int", 8)
    s = C.compute_scale(torch.tensor([[-2.0, 1.0], [0.5, 0.1]]), fmt)
    assert float(s) == pytest.approx(2.0 / 127)
    assert C.encode(torch.tensor([-5.0, 5.0]), fmt, s).tolist() == [-127, 127]


def test_per_channel_scale():
    fmt = C.Format("int", 8, granularity="channel")
    s = C.compute_scale(torch.tensor([[-2.0, 1.0], [0.5, 0.1]]), fmt)
    assert s.shape == (2, 1) and s[1, 0] == pytest.approx(0.5 / 127)


def test_default_profile():
    p = C.load_profile("default")
    assert p["weight"].describe() == "int8" and p["delay"].describe() == "uint5"
    assert p["leak"].describe() == "ufix12.12" and p["threshold"].storage == "shared"
    assert p["bias"].kind == p["membrane"].kind == "fixed" and p["membrane"].bits == 16


def test_int4w_changes_only_weights():
    d, i = C.load_profile("default"), C.load_profile("int4w")
    assert i["weight"].bits == 4 and i["weight"].kind == "int"
    for q in C.QUANTITIES:
        if q != "weight":
            assert i[q] == d[q], q


def test_fp32_profile_is_float():
    p = C.load_profile("fp32")
    assert all(p[q].kind == "float" for q in C.QUANTITIES)


def test_narrow_delay_rejected():
    with pytest.raises(ValueError, match="delay format uint4"):
        C.load_profile("default").replace(delay=C.Format("uint", 4))
    C.load_profile("default").replace(delay=C.Format("uint", 6))      # wider is fine


def test_unsigned_membrane_rejected():
    with pytest.raises(ValueError, match="membrane must be signed"):
        C.load_profile("default").replace(membrane=C.Format("fixed", 16, signed=False))
