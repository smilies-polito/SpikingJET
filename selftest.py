#!/usr/bin/env python3
"""
Check that every tool of the repository works, from the repository root.

    python3 selftest.py            quick (~3 min): tests and smoke runs on synthetic data, no download
    python3 selftest.py --full     every tool on every included network; downloads the test sets
                                   (~1.5 GB) and the SHD / CIFAR-10 training sets; ~15 min on a GPU after the downloads (CIFAR-10 alone can take 30 min)

In the container:  singularity run --nv faulty_snn.sif selftest.py [--full]

Each step's output goes to output/selftest/<step>.log; a PASS / FAIL table is printed at the end
and the exit status is non-zero if a step failed. Results of real campaigns are not touched:
the single-fault check uses its own seed and batch size and is deleted afterwards.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("output", "selftest")
PY = sys.executable
SNNS = ["SHD_DELAY", "NMNIST_CSNN", "NMNIST", "CSNN"]      # SNNs shipped with a checkpoint
SHIPPED = SNNS + ["NMNIST_HYBRID"]                          # + the mixed network
SEED, BS = 777, 50                                         # single-fault check (own output folder)


def has_dcls():
    try:
        import DCLS  # noqa: F401
        return True
    except ImportError:
        return False


def steps(full, cuda):
    o = OUT
    dep = f"{o}/SHD_DELAY_deployed.th"
    gpu = ["--use-cuda"] if cuda else []
    s = [
        ("pytest", [PY, "-m", "pytest", "-q", "tests"]),
        ("train_snn SHD_RLIF (synthetic)",
         [PY, "train_snn.py", "--network", "SHD_RLIF", "--synthetic", "--epochs", "1", "--out", f"{o}/rlif.th"]),
        ("train_snn NMNIST_CSNN (synthetic)",
         [PY, "train_snn.py", "--network", "NMNIST_CSNN", "--synthetic", "--epochs", "1", "--out", f"{o}/csnn.th"]),
        ("train_shd_delays (synthetic)" + ("" if has_dcls() else " [needs DCLS]"),
         [PY, "train_shd_delays.py", "--synthetic", "--epochs", "1", "--out", f"{o}/delay.th"]),
        ("ber_eval + ber_report (synthetic)",
         ["sh", "-c", f"{PY} ber_eval.py --network SHD_DELAY NMNIST_CSNN --synthetic --bers 1e-5 1e-3 "
                      f"--realizations 1 --out {o}/ber_syn.csv && {PY} ber_report.py {o}/ber_syn.csv --plot {o}/ber_syn.png"]),
        ("spikingjet sites (every shipped network)",
         ["sh", "-c", " && ".join(f"{PY} spikingjet.py sites -n {n} --set device=cpu > /dev/null" for n in SHIPPED)]),
        ("training faults: fault-aware, mixed net (synthetic)",
         [PY, "spikingjet.py", "train", "-c", "experiments/train_fault_aware.toml", "--synthetic",
          "--set", "train.epochs=2", "--set", "batch_size=16", "--set", f"out={o}/fat_syn.th"]),
        ("training faults: all targets, mixed net (synthetic)",
         [PY, "spikingjet.py", "train", "-c", "experiments/train_resilience.toml", "--synthetic",
          "--set", "train.epochs=2", "--set", "batch_size=16", "--set", f"out={o}/res_syn.th"]),
        ("training faults: train_snn.py --faults (synthetic)",
         [PY, "train_snn.py", "--network", "NMNIST_CSNN", "--synthetic", "--epochs", "1", "--out", f"{o}/csnn_f.th",
          "--faults", "experiments/train_resilience.toml"]),
    ]
    if not full:
        return s
    s += [("download_datasets", [PY, "download_datasets.py"])]
    s += [(f"verify_snn {n}", [PY, "verify_snn.py", "--network", n]) for n in SNNS]
    s += [
        ("main.py bit-flip (SHD_DELAY, 30 faults)",
         [PY, "main.py", "-n", "SHD_DELAY", *gpu, "--fault-mode", "bit-flip", "--seed", str(SEED),
          "-b", str(BS), "--max-faults", "30"]),
        ("analyse.py", [PY, "analyse.py", "-n", "SHD_DELAY", "-b", str(BS), "-fl", f"{SEED}_fault_list.csv"]),
        ("pt_to_csv.py", [PY, "pt_to_csv.py", "-n", "SHD_DELAY", "-b", str(BS)]),
        ("main.py stuck-at (NMNIST_CSNN, 10 faults)",
         [PY, "main.py", "-n", "NMNIST_CSNN", *gpu, "--fault-mode", "stuck-at", "--seed", str(SEED),
          "-b", str(BS), "--max-faults", "10"]),
        ("campaign: activations + neurons (NMNIST_HYBRID)",
         [PY, "spikingjet.py", "campaign", "-c", "experiments/campaign_activations.toml",
          "--set", "campaign.max_faults=20", "--set", f"out={o}/campaign"]),
        ("campaign: transient neuron faults (SHD_DELAY)",
         [PY, "spikingjet.py", "campaign", "-c", "experiments/campaign_snn.toml", "--set", "campaign.max_faults=20",
          "--set", "campaign.targets=['neurons']", "--set", "campaign.save_outputs=false", "--set", f"out={o}/campaign"]),
        ("ber: weights + neuron states (NMNIST_HYBRID)",
         ["sh", "-c", f"rm -f {o}/ber_mixed.csv && {PY} spikingjet.py ber -c experiments/ber_mixed.toml "
                      f"--set ber.realizations=1 --set 'ber.bers=[1e-5, 1e-3]' --set out={o}/ber_mixed.csv"]),
        ("verify_snn NMNIST_HYBRID", [PY, "verify_snn.py", "--network", "NMNIST_HYBRID"]),
        ("fault-aware fine-tuning (NMNIST_HYBRID, 1 epoch)",
         [PY, "spikingjet.py", "train", "-c", "experiments/train_fault_aware.toml", "--set", "train.epochs=1",
          "--set", "train.init='checkpoint'", "--set", "train.eval_realizations=1", "--set", f"out={o}/hybrid_fat.th"]),
        ("ber_eval + ber_report (all SNNs, test sets)",
         ["sh", "-c", f"{PY} ber_eval.py --network {' '.join(SNNS)} --bers 1e-6 1e-4 --bits all mantissa "
                      f"--realizations 1 --out {o}/ber.csv && {PY} ber_report.py {o}/ber.csv --plot {o}/ber.png"]),
        ("deploy_shd (default, int4w, fp32)",
         ["sh", "-c", " && ".join(f"{PY} deploy_shd.py --profile {p} --out {o}/dep_{p}.th"
                                  for p in ("int4w", "fp32", "default")) + f" && cp {o}/dep_default.th {dep}"]),
        ("hw_campaign map", [PY, "hw_campaign.py", "map", "--checkpoint", dep]),
        ("hw_campaign ber",
         ["sh", "-c", f"rm -rf {o}/hw_ber && echo '{{\"base_seed\": 1, \"bers\": [1e-4, 1e-2], \"realizations\": 2, "
                      f"\"sweeps\": [\"joint\", \"M1\", \"M2\", \"M3\", \"M4\"]}}' > {o}/ber.json && "
                      f"{PY} hw_campaign.py ber --checkpoint {dep} --config {o}/ber.json --out {o}/hw_ber"]),
        ("hw_campaign faults + catalog (resume)",
         # M2 is always sampled exhaustively (~8k faults): the catalog check uses 200 random faults of the list
         ["sh", "-c", f"rm -rf {o}/catalog && "
                      f"{PY} hw_campaign.py faults --checkpoint {dep} --fraction 0.001 --out {o}/faults_all.csv.gz && "
                      f"{PY} -c \"import pandas as pd; pd.read_csv('{o}/faults_all.csv.gz').sample(200, random_state=0)"
                      f".to_csv('{o}/faults.csv.gz', index=False)\" && "
                      f"{PY} hw_campaign.py catalog --checkpoint {dep} --faults {o}/faults.csv.gz --out {o}/catalog "
                      f"--replicas 16 && {PY} hw_campaign.py catalog --checkpoint {dep} --faults {o}/faults.csv.gz "
                      f"--out {o}/catalog --replicas 16 | grep ', 0 to run'"]),
        ("train_cnn ResNet20 (1 epoch, CIFAR-10)",
         [PY, "train_cnn.py", "-n", "ResNet20", "--epochs", "1", "--out", f"{o}/ResNet20.th"]),
        ("CNN: campaign on activations + BER (ResNet20)",
         ["sh", "-c", f"rm -f {o}/ber_cnn.csv && {PY} spikingjet.py campaign -n ResNet20 --set checkpoint={o}/ResNet20.th "
                      f"--set \"campaign.targets=['params', 'activations']\" --set campaign.max_faults=10 "
                      f"--set campaign.save_outputs=false --set out={o}/campaign && "
                      f"{PY} spikingjet.py ber -n ResNet20 --set checkpoint={o}/ResNet20.th --set ber.realizations=1 "
                      f"--set 'ber.bers=[1e-6]' --set out={o}/ber_cnn.csv"]),
    ]
    return s


def cleanup():
    """Remove the single-fault check's results (own seed / batch size), keep real campaigns."""
    for net in ("SHD_DELAY", "NMNIST_CSNN"):
        for kind in ("label", "clean", "faulty"):
            for fmt in ("pt", "csv"):
                shutil.rmtree(f"output/{net}/{fmt}/{kind}/batch_size_{BS}", ignore_errors=True)
        for f in (f"{SEED}_fault_list.csv", f"{SEED}_weights.npz", f"{SEED}_fault_list_result_batch_size_{BS}.csv"):
            path = f"output/fault_list/{net}/{f}"
            if os.path.exists(path):
                os.remove(path)
    for n in SNNS:
        shutil.rmtree(f"output/{n}_verify", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="every tool on every network (downloads datasets)")
    ap.add_argument("--keep", action="store_true", help="keep the single-fault check results")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    import torch
    cuda = torch.cuda.is_available()
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
               MPLBACKEND="Agg", PYTHONNOUSERSITE="1")
    print(f"selftest ({'full' if a.full else 'quick'}), device {'cuda' if cuda else 'cpu'}, logs in {OUT}/")
    results = []
    for name, cmd in steps(a.full, cuda):
        log = os.path.join(OUT, "".join(c if c.isalnum() else "_" for c in name).strip("_") + ".log")
        print(f"  {name:<48}", end="", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        ok = rc == 0
        results.append((name, ok, log))
        print(f"{'PASS' if ok else 'FAIL'}  {time.time() - t0:6.0f} s" + ("" if ok else f"   see {log}"))
    if a.full and not a.keep:
        cleanup()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} steps passed")
    for name, _, log in failed:
        print(f"FAILED: {name}\n  last lines of {log}:")
        with open(log) as f:
            print("    " + "    ".join(f.readlines()[-6:]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
