"""Experiment configuration: a TOML file + command-line overrides, checked against the defaults.

    cfg = load_config("experiments/shd_campaign.toml", overrides=["campaign.max_faults=50", "network=CSNN"])

Every key has a default (DEFAULTS below, also printed by `python spikingjet.py config`), so a file
only lists what differs. Unknown keys are errors (typos do not pass silently). Override values are
parsed as TOML literals: 1e-5, true, "text", [1, 2] (a bare word is taken as a string).
"""
from __future__ import annotations

import copy
import difflib
import json

try:
    import tomllib as _toml                         # Python >= 3.11
except ImportError:                                 # pragma: no cover
    import tomli as _toml

# One fault group (lists: [[ber.faults]], [[train.faults]]).
FAULT_GROUP = {
    "target": "params",          # params | neurons | activations | grads | optimizer (last two: training only)
    "select": ["*"],             # glob patterns on site names (python spikingjet.py sites -n NET lists them)
    "exclude": [],
    "fault_model": "bit-flip",   # bit-flip | stuck-at
    "bits": "all",               # all | mantissa | exponent | sign | low-mantissa | 30 | [0, 22] | [1, 5, 30]
    "ber": 1e-5,                 # bit error rate: every selected bit fails with this probability
    "persistence": "transient",  # transient | upset | permanent (see README, 'Fault persistence')
    # training schedule (ignored outside training); epochs count from 0, stop_epoch -1 = until the end
    "start_epoch": 0,
    "stop_epoch": -1,
    "every": 1,                  # every n-th training step
    "prob": 1.0,                 # probability that the group fires at an eligible step
}

DEFAULTS = {
    "network": "",               # a name of networks.py
    "seed": 0,
    "device": "auto",            # auto | cuda | cpu
    "tf32": False,               # allow TF32 (faster on Ampere+, masks low mantissa bit-flips)
    "batch_size": 64,
    "out": "",                   # output folder / file (each command has its own default)
    "checkpoint": "",            # weights to load instead of the registered checkpoint (e.g. a fault-aware one)
    "sites": {
        "param_scope": "spikingjet",      # spikingjet (Conv/Linear/LIF layers) | all (every parameter)
        "activation_types": ["ReLU", "ReLU6", "LeakyReLU", "GELU", "SiLU", "ELU", "Sigmoid", "Tanh",
                             "Hardtanh", "Conv1d", "Conv2d", "Conv3d", "Linear"],
    },
    "campaign": {                # single-fault campaign (DATE'09 statistical sample)
        "fault_model": "stuck-at",        # stuck-at | bit-flip
        "targets": ["params", "neurons"], # params | neurons | activations
        "select": ["*"],
        "exclude": [],
        "bits": "all",
        "seed": 51196,                    # seed of the fault sample
        "error_margin": 0.03,
        "confidence": 2.58,               # t of the confidence level (2.58 = 99 %)
        "p": 0.5,
        "max_faults": 0,                  # 0 = the full DATE'09 sample; else a cap (quick checks)
        "neuron_persistence": "permanent",# permanent: every timestep | transient: one random timestep
        "fault_list": "",                 # rerun an existing fault list CSV instead of sampling
        "save_outputs": True,             # save every output score (.pt) for analyse.py / pt_to_csv.py
    },
    "ber": {                     # bit-error-rate sweep: [[ber.faults]] groups, the BER of each is swept
        "bers": [1e-7, 1e-6, 1e-5, 1e-4],
        "realizations": 10,
        "faults": [],
    },
    "train": {
        "init": "scratch",               # scratch | checkpoint (fine-tune the registered or `checkpoint` weights)
        "epochs": 10,
        "optimizer": "adamw",            # sgd | adam | adamw
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler": "cosine",           # cosine | onecycle | none
        "grad_clip": 0.0,                # max gradient norm, 0 = off
        "eval_batch_size": 256,
        "faults": [],                    # [[train.faults]] groups active during training
        "eval_bers": [],                 # after training: accuracy at these weight BERs (robustness check)
        "eval_realizations": 3,
    },
}

_SECTIONS_WITH_GROUPS = ("ber", "train")


def _suggest(key, options):
    m = difflib.get_close_matches(key, options, n=1)
    return f" (did you mean {m[0]!r}?)" if m else ""


def _merge(base, new, path=""):
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key {path + k!r}{_suggest(k, list(base))}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"{path + k} must be a table")
            _merge(base[k], v, path + k + ".")
        elif k == "faults":
            base[k] = [_fault_group(g, f"{path}faults[{i}]") for i, g in enumerate(v)]
        else:
            base[k] = v


def _fault_group(g, where):
    out = copy.deepcopy(FAULT_GROUP)
    for k, v in g.items():
        if k not in out:
            raise KeyError(f"unknown key {k!r} in {where}{_suggest(k, list(out))}")
        out[k] = v
    if out["target"] not in ("params", "neurons", "activations", "grads", "optimizer"):
        raise ValueError(f"{where}.target: {out['target']!r} is not params|neurons|activations|grads|optimizer")
    if out["persistence"] not in ("transient", "upset", "permanent"):
        raise ValueError(f"{where}.persistence: {out['persistence']!r} is not transient|upset|permanent")
    if out["fault_model"] not in ("bit-flip", "stuck-at"):
        raise ValueError(f"{where}.fault_model: {out['fault_model']!r} is not bit-flip|stuck-at")
    return out


def _parse_value(text):
    try:
        return _toml.loads(f"v = {text}")["v"]
    except Exception:
        return text


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for i, k in enumerate(keys[:-1]):
        if k.endswith("]"):                                   # faults[0].ber
            name, idx = k[:-1].split("[")
            node = node[name][int(idx)]
        else:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"unknown config section {'.'.join(keys[:i + 1])!r}")
            node = node[k]
    last = keys[-1]
    template = FAULT_GROUP if "faults[" in dotted else node
    if last not in template:
        raise KeyError(f"unknown config key {dotted!r}{_suggest(last, list(template))}")
    node[last] = _parse_value(value) if isinstance(value, str) else value


def load_config(path=None, overrides=(), **direct):
    """Defaults <- TOML file <- overrides ('a.b=value' strings) <- direct keyword values."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "rb") as f:
            _merge(cfg, _toml.load(f))
    for o in overrides or ():
        if "=" not in o:
            raise ValueError(f"override {o!r} must be key=value")
        k, v = o.split("=", 1)
        _set(cfg, k.strip(), v.strip())
    for k, v in direct.items():
        if v is not None:
            _set(cfg, k.replace("__", "."), v)
    return cfg


def dump(cfg) -> str:
    return json.dumps(cfg, indent=2, default=str)
