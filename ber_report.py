#!/usr/bin/env python3
"""
Summarise ber_eval.py results: accuracy vs BER per network and bit range, the tolerable
BER at a given accuracy loss, and paired differences between networks.

    python3 ber_report.py ber_results/*.csv                     # table to stdout
    python3 ber_report.py ber_results/*.csv --plot ber.pdf      # + figure
    python3 ber_report.py ber_results/*.csv --ref ResNet18      # paired diff vs a reference

Tolerable BER (as in EDEN / Stutz et al.): the largest BER at which the MEAN accuracy over
realizations stays within --loss pp of the clean accuracy, interpolated linearly in
log10(BER) between the two sweep points that bracket the crossing. '>max' means the
network never crossed within the sweep; '<min' means it was already below at the first point.

Paired difference: realizations with the same (bits, BER, realization) index use the same fault
map for networks with identical parameter shapes, so the per-realization difference
(net - ref) is paired; the report gives its mean and a 95% t-interval.
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
       10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df):
    keys = sorted(T95)
    return T95[min(keys, key=lambda k: abs(k - df))] if df < 30 else 1.96


def tolerable_ber(g, loss):
    g = g.sort_values("ber")
    clean = g["clean_acc"].iloc[0]
    target = clean - loss
    bers, accs = g["ber"].values, g["mean"].values
    if accs[0] < target:
        return f"<{bers[0]:.0e}"
    for i in range(1, len(bers)):
        if accs[i] < target:
            x0, x1 = math.log10(bers[i - 1]), math.log10(bers[i])
            y0, y1 = accs[i - 1], accs[i]
            x = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return f"{10 ** x:.1e}"
    return f">{bers[-1]:.0e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--loss", type=float, default=1.0, help="accuracy loss (pp) for tolerable BER")
    ap.add_argument("--ref", default=None, help="reference network for paired differences")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in a.csv], ignore_index=True)
    df = df.drop_duplicates(subset=["network", "mode", "scope", "bits", "ber", "realization"], keep="last")
    agg = (df.groupby(["mode", "scope", "bits", "network", "ber"])
             .agg(clean_acc=("clean_acc", "first"), mean=("faulty_acc", "mean"),
                  std=("faulty_acc", "std"), min=("faulty_acc", "min"),
                  nonfinite=("nonfinite_pct", "mean"), n=("faulty_acc", "size"),
                  faults=("n_faults", "mean"))
             .reset_index())

    pd.set_option("display.width", 200)
    for (mode, scope, bits), block in agg.groupby(["mode", "scope", "bits"]):
        print(f"\n=== {mode} | scope={scope} | bits={bits} ===")
        wide = block.pivot(index="ber", columns="network", values="mean").round(2)
        print("mean accuracy (%) over realizations:")
        print(wide.to_string())
        print(f"\ntolerable BER at {a.loss:g} pp loss:")
        for net, g in block.groupby("network"):
            print(f"  {net:45s} clean {g['clean_acc'].iloc[0]:6.2f}%   "
                  f"tolerable BER {tolerable_ber(g, a.loss):>8s}   "
                  f"worst realization at max BER {g.sort_values('ber')['min'].iloc[-1]:6.2f}%")

        if a.ref:
            sub = df[(df["mode"] == mode) & (df["scope"] == scope) & (df["bits"] == bits)]
            ref = sub[sub["network"] == a.ref].set_index(["ber", "realization"])
            if ref.empty:
                continue
            print(f"\npaired difference vs {a.ref} (pp, mean [95% CI]):")
            for net, g in sub[sub["network"] != a.ref].groupby("network"):
                g = g.set_index(["ber", "realization"])
                common = g.index.intersection(ref.index)
                if g.loc[common, "n_params"].iloc[0] != ref.loc[common, "n_params"].iloc[0]:
                    print(f"  {net}: different parameter count, NOT paired - skipped")
                    continue
                d = (g.loc[common, "faulty_acc"] - ref.loc[common, "faulty_acc"]).groupby(level=0)
                cells = []
                for ber, s in d:
                    n = len(s)
                    hw = t95(n - 1) * s.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
                    cells.append(f"{ber:.0e}: {s.mean():+6.2f} [±{hw:.2f}]")
                print(f"  {net}:\n    " + "\n    ".join(cells))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = list(agg.groupby(["mode", "scope", "bits"]))
        fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), squeeze=False)
        for ax, ((mode, scope, bits), block) in zip(axes[0], groups):
            for net, g in block.groupby("network"):
                g = g.sort_values("ber")
                ax.plot(g["ber"], g["mean"], marker="o", ms=3, label=net)
                ax.fill_between(g["ber"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                                alpha=0.15)
            ax.set_xscale("log")
            ax.set_xlabel("bit error rate")
            ax.set_title(f"{bits} bits ({mode})", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("test accuracy (%)")
        axes[0][-1].legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(a.plot)
        print(f"\nfigure written to {a.plot}")


if __name__ == "__main__":
    sys.exit(main())
