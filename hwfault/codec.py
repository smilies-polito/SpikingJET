"""
Storage formats of the hardware fault model, and precision profiles.

A Format says how one stored quantity is held in memory:
    kind   "float"  IEEE-754 binary32 bit pattern (bits must be 32)
           "int"    signed two's complement integer code, value = code * scale
           "uint"   unsigned integer code, value = code * scale
           "fixed"  signed or unsigned fixed point, value = code * 2**-frac
    bits   width of the stored word; faults address bits 0 (LSB) .. bits-1 (MSB / sign)

Codes are int64 tensors holding the code value (for "float": the unsigned 32-bit pattern), so
every format shares the same bit-level fault operations (flip, stuck-at) through
to_pattern/from_pattern. A profile (profiles/<name>.toml) gives one Format per quantity:
weight, delay, bias, threshold, leak, membrane.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict

import torch

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
QUANTITIES = ("weight", "delay", "bias", "threshold", "leak", "membrane")
MAX_DELAY = 24                       # SHD_DELAY: delays 0..24 steps


@dataclass(frozen=True)
class Format:
    kind: str
    bits: int
    signed: bool = True
    granularity: str = "layer"      # scale granularity of "int"/"uint": "layer" | "channel"
    frac: int | None = None         # "fixed": fraction bits; None = chosen at deployment (calibration)
    scale: str | None = None        # "membrane": share the layer's membrane fixed-point scale
    storage: str = "per-neuron"     # threshold/leak: "shared" (one word per layer) | "per-neuron"
    saturate: bool = True           # membrane: saturate on overflow (else wrap)
    calibration_percentile: float = 99.99

    def __post_init__(self):
        if self.kind not in ("float", "int", "uint", "fixed"):
            raise ValueError(f"unknown format kind {self.kind!r}")
        if self.kind == "float" and self.bits != 32:
            raise ValueError(f"float formats must be 32-bit IEEE-754, got {self.bits}")
        if not 1 <= self.bits <= 32:
            raise ValueError(f"format width must be 1..32 bits, got {self.bits}")
        if self.storage not in ("shared", "per-neuron"):
            raise ValueError(f"storage must be 'shared' or 'per-neuron', got {self.storage!r}")

    @property
    def is_signed(self) -> bool:
        return self.kind == "int" or (self.kind == "fixed" and self.signed)

    @property
    def qmin(self) -> int:
        if self.kind == "float":
            return 0
        return -(1 << (self.bits - 1)) if self.is_signed else 0

    @property
    def qmax(self) -> int:
        if self.kind == "float":
            return (1 << 32) - 1
        return (1 << (self.bits - 1)) - 1 if self.is_signed else (1 << self.bits) - 1

    def describe(self) -> str:
        if self.kind == "float":
            return "fp32"
        base = {"int": "int", "uint": "uint"}.get(self.kind, "sfix" if self.signed else "ufix")
        extra = f".{self.frac}" if self.kind == "fixed" and self.frac is not None else ""
        return f"{base}{self.bits}{extra}"

    def to_dict(self) -> dict:
        return asdict(self)


def with_frac(fmt: Format, frac: int) -> Format:
    d = fmt.to_dict()
    d["frac"] = int(frac)
    return Format(**d)


# ----------------------------------------------------------------------------------------------
# Values <-> codes
# ----------------------------------------------------------------------------------------------
def step(fmt: Format, scale=None):
    """Value of one LSB (tensor or float)."""
    if fmt.kind == "fixed":
        if fmt.frac is None:
            raise ValueError("fixed-point format without frac (not calibrated yet)")
        return 2.0 ** -fmt.frac
    if fmt.kind in ("int", "uint"):
        return 1.0 if scale is None else scale
    raise ValueError("float formats have no fixed step")


def compute_scale(x: torch.Tensor, fmt: Format) -> torch.Tensor:
    """Symmetric scale for "int"/"uint": max|x| / qmax per layer, or per output channel (dim 0)."""
    if fmt.kind not in ("int", "uint"):
        raise ValueError(f"compute_scale is for int/uint formats, not {fmt.kind}")
    a = x.detach().abs().float()
    m = a.amax(dim=tuple(range(1, a.dim())), keepdim=True) if fmt.granularity == "channel" and a.dim() > 1 \
        else a.max()                                     # per layer: a 0-dim tensor
    return torch.where(m > 0, m / fmt.qmax, torch.ones_like(m))


def encode(x: torch.Tensor, fmt: Format, scale=None) -> torch.Tensor:
    """Float tensor -> int64 code tensor (round to nearest, saturate; symmetric for "int")."""
    x = x.detach().float()
    if fmt.kind == "float":
        return x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    q = torch.round(x / step(fmt, scale))
    lo = -fmt.qmax if fmt.kind == "int" else fmt.qmin           # symmetric int range
    return q.clamp(lo, fmt.qmax).to(torch.int64)


def decode(code: torch.Tensor, fmt: Format, scale=None) -> torch.Tensor:
    """int64 code tensor -> float32 values."""
    if fmt.kind == "float":
        c = code.to(torch.int64) & 0xFFFFFFFF
        c = torch.where(c >= (1 << 31), c - (1 << 32), c).to(torch.int32)
        return c.view(torch.float32)
    return (code.to(torch.float32) * step(fmt, scale)).to(torch.float32)


def quantize(x: torch.Tensor, fmt: Format, scale=None) -> torch.Tensor:
    """decode(encode(x)): the value actually stored."""
    if fmt.kind == "float":
        return x.float()
    s = step(fmt, scale)
    lo = -fmt.qmax if fmt.kind == "int" else fmt.qmin
    return torch.round(x / s).clamp(lo, fmt.qmax) * s


# ----------------------------------------------------------------------------------------------
# Bit-level view and fault operations
# ----------------------------------------------------------------------------------------------
def to_pattern(code: torch.Tensor, fmt: Format) -> torch.Tensor:
    """Code -> unsigned stored bit pattern in [0, 2**bits)."""
    return code.to(torch.int64) & ((1 << fmt.bits) - 1)


def from_pattern(pattern: torch.Tensor, fmt: Format) -> torch.Tensor:
    """Unsigned stored bit pattern -> code (two's complement for signed formats)."""
    p = pattern.to(torch.int64) & ((1 << fmt.bits) - 1)
    if fmt.kind != "float" and fmt.is_signed:
        p = torch.where(p >= (1 << (fmt.bits - 1)), p - (1 << fmt.bits), p)
    return p


def flip(code: torch.Tensor, fmt: Format, bit) -> torch.Tensor:
    _check_bit(fmt, bit)
    return from_pattern(to_pattern(code, fmt) ^ (torch.as_tensor(1, dtype=torch.int64) << torch.as_tensor(bit)), fmt)


def stuck(code: torch.Tensor, fmt: Format, bit, value) -> torch.Tensor:
    _check_bit(fmt, bit)
    m = torch.as_tensor(1, dtype=torch.int64) << torch.as_tensor(bit)
    p = to_pattern(code, fmt)
    v = torch.as_tensor(value, dtype=torch.int64)
    return from_pattern(torch.where(v.bool(), p | m, p & ~m), fmt)


def bit_value(code: torch.Tensor, fmt: Format, bit) -> torch.Tensor:
    return (to_pattern(code, fmt) >> torch.as_tensor(bit)) & 1


def _check_bit(fmt, bit):
    b = torch.as_tensor(bit)
    if bool((b < 0).any()) or bool((b >= fmt.bits).any()):
        raise ValueError(f"bit {bit} outside 0..{fmt.bits - 1} of format {fmt.describe()}")


# ----------------------------------------------------------------------------------------------
# Precision profiles
# ----------------------------------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    formats: dict = field(default_factory=dict)        # quantity -> Format
    source: dict = field(default_factory=dict)         # raw TOML content (recorded in run metadata)

    def __getitem__(self, q) -> Format:
        return self.formats[q]

    def validate(self):
        missing = [q for q in QUANTITIES if q not in self.formats]
        if missing:
            raise ValueError(f"profile {self.name!r}: missing quantities {missing}")
        d = self.formats["delay"]
        if d.kind != "float":
            can = d.qmax if d.kind != "fixed" else d.qmax * 2.0 ** -(d.frac or 0)
            if d.kind == "fixed" and (d.frac or 0) != 0:
                raise ValueError(f"profile {self.name!r}: delay format {d.describe()} must be an integer format")
            if can < MAX_DELAY:
                raise ValueError(f"profile {self.name!r}: delay format {d.describe()} cannot hold delays "
                                 f"0..{MAX_DELAY} (max code {can})")
        for q in ("bias", "threshold", "membrane"):
            f = self.formats[q]
            if f.kind == "uint" or (f.kind == "fixed" and not f.signed):
                raise ValueError(f"profile {self.name!r}: {q} must be signed or float, got {f.describe()}")
        return self

    def to_dict(self) -> dict:
        """Flat form {name, <quantity>: format fields}; profile_from_dict(to_dict()) rebuilds it."""
        return {"name": self.name, **{q: f.to_dict() for q, f in self.formats.items()}}

    def replace(self, name=None, **formats) -> "Profile":
        p = copy.deepcopy(self)
        p.name = name or p.name
        for q, f in formats.items():
            p.formats[q] = f if isinstance(f, Format) else Format(**f)
        return p.validate()


def profile_from_dict(d: dict, name: str | None = None) -> Profile:
    fm = {q: Format(**d[q]) for q in QUANTITIES if q in d}
    return Profile(name=name or d.get("name", "custom"), formats=fm, source=copy.deepcopy(d)).validate()


def load_profile(name_or_path: str = "default") -> Profile:
    """Load profiles/<name>.toml (or a path). Profiles may set `base = "<other>"` and override
    only some quantities."""
    import tomli
    path = name_or_path if name_or_path.endswith(".toml") else os.path.join(PROFILE_DIR, name_or_path + ".toml")
    with open(path, "rb") as f:
        d = tomli.load(f)
    if "base" in d:
        base = load_profile(d["base"]).source
        merged = {**base, **{k: v for k, v in d.items() if k != "base"}}
        for q in QUANTITIES:
            if q in base and q in d:
                merged[q] = {**base[q], **d[q]}
        d = merged
    return profile_from_dict(d, name=d.get("name", os.path.splitext(os.path.basename(path))[0]))
