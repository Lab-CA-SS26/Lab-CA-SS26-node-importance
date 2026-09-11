#!/usr/bin/env python3
"""Does the burn-in normalisation bias account for KADABRA's absolute error exceeding eps?

The reference discards the burn-in counts (`Probabilistic.cpp:363-368`) but then adds the
burn-in back to the divisor (`n_pairs += tau`, line 405, read by `get_centrality` in
`Probabilistic.h:36`), so every score is `count_main / (N_main + tau)` -- low by the
factor `1 - tau/N`. Our port and the Graphs.jl fork reproduce it (`final_n_pairs =
... + tau`). NetworKit divides *before* adding tau (`KadabraBetweenness.cpp:430` vs
`:442`) and is unaffected.

For each unstripped C++ run this reports the max absolute error and the number of
vertices over eps three ways: as output; rescaled by N/(N - tau) with tau taken from the
matching Julia run (independent of the ground truth); and rescaled by a least-squares fit
on the top-1000 vertices (uses the ground truth -- a cross-check on the first).

Usage
-----
    python3 check_burnin_bias.py <gt_dir> <julia_tau.json> <eps> <cpp_stats.json> [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_cpp_centralities import load_gt, place


def err_stats(approx, exact, eps):
    e = np.abs(approx - exact)
    i = int(e.argmax())
    return {"max_ae": float(e.max()), "over_eps": int((e > eps).sum()),
            "argmax_rank": int((exact > exact[i]).sum()) + 1, "mae": float(e.mean())}


def main():
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    gt_dir, taus, eps = sys.argv[1], json.load(open(sys.argv[2])), float(sys.argv[3])
    for path in sys.argv[4:]:
        run = json.load(open(path))
        g = os.path.basename(run["parameters"]["input_file"]).replace(".txt", "")
        exact, shift = load_gt(os.path.join(gt_dir, f"{g}_bet.csv"))
        approx = place(run["centralities"], shift, len(exact))
        n, tau = run["num_samples"], taus[g]
        top = np.argsort(-exact, kind="stable")[:1000]
        fit = float((approx[top] @ exact[top]) / (approx[top] @ approx[top]))
        print(json.dumps({
            "graph": g, "set": path.split("/")[-2], "N": n, "tau": tau, "b_max": float(exact.max()),
            "raw": err_stats(approx, exact, eps),
            "debiased_tau": err_stats(approx * n / (n - tau), exact, eps),
            "debiased_fit": err_stats(approx * fit, exact, eps), "fit_scale": 1 / fit,
        }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
