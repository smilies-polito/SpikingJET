"""Bit-level faults on float32 tensors, vectorized (no Python loop over elements).

A fault is (flat element index, bit, stuck value). The tensor is viewed as int32, so the fault
changes exactly one stored bit, as in a memory cell; bit 31 is the sign, 23-30 the exponent,
0-22 the mantissa.
"""
from __future__ import annotations

import struct
from typing import Iterable, Sequence, Union

import numpy as np
import torch

# named bit groups of an IEEE-754 float32
BIT_GROUPS = {
    "all": tuple(range(0, 32)),
    "mantissa": tuple(range(0, 23)),
    "exponent": tuple(range(23, 31)),
    "sign": (31,),
    "low-mantissa": tuple(range(0, 16)),
}

BitsSpec = Union[str, int, Sequence[int]]


def resolve_bits(spec: BitsSpec) -> tuple:
    """'mantissa' | 30 | [0, 22] (inclusive range) | [1, 5, 30] (explicit list) -> sorted tuple of bits."""
    if isinstance(spec, str):
        if spec not in BIT_GROUPS:
            raise ValueError(f"unknown bit group {spec!r}; use one of {', '.join(BIT_GROUPS)} or a list")
        return BIT_GROUPS[spec]
    if isinstance(spec, int):
        bits = (spec,)
    else:
        spec = list(spec)
        bits = tuple(range(spec[0], spec[1] + 1)) if len(spec) == 2 and spec[0] < spec[1] else tuple(spec)
    if not bits or not all(0 <= b <= 31 for b in bits):
        raise ValueError(f"bits must be in 0..31, got {spec!r}")
    return tuple(sorted(set(bits)))


def _as_int32(mask: torch.Tensor) -> torch.Tensor:
    """int64 holding an unsigned 32-bit pattern -> int32 with the same bits."""
    return torch.where(mask >= 2 ** 31, mask - 2 ** 32, mask).to(torch.int32)


def bit_masks(flat_idx: torch.Tensor, bits: torch.Tensor, values: torch.Tensor, numel: int, device):
    """Combine faults into per-element masks: (unique element indices, xor mask, set mask, clear mask).
    `values` is the stuck value per fault (0 / 1), or -1 for a bit-flip."""
    flat_idx = flat_idx.to(device=device, dtype=torch.int64)
    m = torch.ones_like(flat_idx) << bits.to(device=device, dtype=torch.int64)
    v = values.to(device=device, dtype=torch.int64)
    uniq, inv = torch.unique(flat_idx, return_inverse=True)
    z = torch.zeros(uniq.numel(), dtype=torch.int64, device=device)
    xor = z.clone().index_add_(0, inv, m * (v < 0))
    set_ = z.clone().index_add_(0, inv, m * (v == 1))
    clr = z.clone().index_add_(0, inv, m * (v == 0))
    return uniq, _as_int32(xor), _as_int32(set_), _as_int32(clr)


def apply_faults(t: torch.Tensor, flat_idx, bits, values) -> None:
    """In place on t (float32, contiguous): flip (value -1) or stick (value 0/1) the given bits.
    Several faults on one element must hit different bits."""
    if t.dtype != torch.float32:
        raise TypeError(f"only float32 tensors can be faulted, got {t.dtype}")
    if len(flat_idx) == 0:
        return
    flat_idx = torch.as_tensor(flat_idx)
    bits = torch.as_tensor(bits)
    values = torch.as_tensor(values)
    uniq, xor, set_, clr = bit_masks(flat_idx, bits, values, t.numel(), t.device)
    ints = t.view(-1).view(torch.int32)
    ints[uniq] = ((ints[uniq] ^ xor) | set_) & ~clr


def faulty_copy(t: torch.Tensor, flat_idx, bits, values) -> torch.Tensor:
    """Out-of-place version of apply_faults (autograd-safe: the result is a new tensor)."""
    out = t.detach().clone().contiguous()
    apply_faults(out, flat_idx, bits, values)
    return out


def sample_ber(numel: int, bits: Sequence[int], ber: float, rng: np.random.Generator, stuck: bool = False):
    """Every (element, bit) position, bit in `bits`, fails independently with probability `ber`.
    Returns (flat element index, bit, value) int64 numpy arrays; value is -1 (flip) or the stuck value.
    Same draws as ber_eval.py has always made, so a seed gives the same fault map as before."""
    bits = np.asarray(bits, dtype=np.int64)
    nb = bits.size
    total = int(numel) * nb
    n = int(rng.binomial(total, ber)) if ber > 0 and total > 0 else 0
    if n == 0:
        e = np.empty(0, np.int64)
        return e, e, e
    flat = np.unique(rng.integers(0, total, size=n, dtype=np.int64))   # collisions: negligible
    elem = flat // nb
    bit = bits[flat % nb]
    values = rng.integers(0, 2, size=elem.size, dtype=np.int64)         # drawn in both modes (pairing)
    return elem, bit, (values if stuck else np.full(elem.size, -1, np.int64))


# scalar reference implementations (used by the tests and by legacy code)
def float32_bit_flip(golden_value: float, bit: int) -> float:
    a = struct.pack("!f", golden_value)
    b = struct.pack("!I", int(2 ** bit))
    return struct.unpack("!f", bytes(x ^ y for x, y in zip(a, b)))[0]


def float32_stuck_at(golden_value: float, bit: int, value: int) -> float:
    a = struct.pack("!f", golden_value)
    b = struct.pack("!I", int(2 ** bit))
    out = bytes((x | y) if value == 1 else (x & (255 - y)) for x, y in zip(a, b))
    return struct.unpack("!f", out)[0]


def to_tuple(x: Iterable) -> tuple:
    return tuple(int(v) for v in x)
