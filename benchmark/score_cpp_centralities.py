#!/usr/bin/env python3
"""Score the C++ reference's per-node output with the Julia runner's exact metrics.

Section 6.1 compares the C++ reference and the Julia port on time and sample count
only. The C++ runs of that section were made before `run_experiments.cpp` scored
itself, so their JSONs carry a `centralities` blob and no accuracy fields; the
archived copies in `results/report_runs/` then had the blob stripped. This rebuilds
the missing accuracy columns from the unstripped originals.

Why not trust the C++ runner's own scoring block: it fills `bt_approx[v]` from the
reference's internal id `v` but `bt_exact[u + shift]` from the ground-truth id `u`,
so with a 0-based ground-truth file the two vectors are offset by one. This script
instead reproduces `run_experiments.jl` line for line:

  * ground truth `node,betweenness`; shift = 1 if its first node is 0; vector of
    length max(node) + shift, entry `u + shift`
  * approximation placed at `raw_id + shift` -- the reference keys vertices by their
    raw edge-list id, and the Julia loader maps raw id `u` to vertex `u + shift`
  * tau_overall = Kendall tau-b over all vertices (StatsBase.corkendall)
  * top-k (k = 100 at k = 0) by a *stable* descending sort, ties by index
  * overlap, tau_topk on the exact top-k, nDCG, MAE, max AE as in the runner

Mapping check: the same score is also computed with the approximation shifted by
one position. A correct mapping wins by a wide margin; if it does not, the script
says so rather than printing a plausible-looking number.

Usage
-----
    python3 score_cpp_centralities.py <gt_dir> <cpp_stats.json> [...]

Prints one JSON object per input on stdout.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau


def load_gt(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    ids = data[:, 0].astype(np.int64)
    shift = 1 if ids[0] == 0 else 0
    n = int(ids.max()) + shift
    exact = np.zeros(n + 1)          # index 0 unused when shift = 1, as in Julia's 1-based vector
    exact[ids + shift] = data[:, 1]
    return exact[1:], shift          # drop to Julia's 1..n, now 0-based in numpy


def place(cent, shift, n, offset=0):
    approx = np.zeros(n)
    for k, v in cent.items():
        i = int(k) + shift - 1 + offset   # Julia index raw+shift, minus 1 for numpy
        if 0 <= i < n:
            approx[i] = v
    return approx


def metrics(exact, approx, k=100):
    n = len(exact)
    k = min(k, n)
    tau = kendalltau(exact, approx, variant="b").statistic
    ae = np.abs(exact - approx)
    p_exact = np.argsort(-exact, kind="stable")[:k]
    p_approx = np.argsort(-approx, kind="stable")[:k]
    tau_topk = kendalltau(exact[p_exact], approx[p_exact], variant="b").statistic
    overlap = len(np.intersect1d(p_exact, p_approx))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = float((exact[p_exact] * disc).sum())
    ndcg = float((exact[p_approx] * disc).sum()) / idcg if idcg > 0 else 1.0
    return {"tau_overall": float(tau), "tau_topk": float(tau_topk), "overlap_topk": int(overlap),
            "ndcg_topk": ndcg, "mae": float(ae.mean()), "max_ae": float(ae.max())}


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    gt_dir = sys.argv[1]
    for path in sys.argv[2:]:
        run = json.load(open(path))
        cent = run.get("centralities")
        if not isinstance(cent, dict):
            print(json.dumps({"file": path, "error": "no centralities blob"}))
            continue
        graph = os.path.basename(run["parameters"]["input_file"]).replace(".txt", "")
        exact, shift = load_gt(os.path.join(gt_dir, f"{graph}_bet.csv"))
        approx = place(cent, shift, len(exact))
        m = metrics(exact, approx)
        wrong = kendalltau(exact, place(cent, shift, len(exact), offset=1), variant="b").statistic
        # A best-fit scale between the two, to catch a normalisation mismatch that
        # would leave tau and overlap intact but make MAE incomparable.
        top = exact > 0
        scale = float(np.median(approx[top & (approx > 0)] / exact[top & (approx > 0)]))
        out = {"file": path, "graph": graph, "num_samples": run["num_samples"],
               "execution_time_seconds": run["execution_time_seconds"],
               "directed": run["parameters"].get("directed"),
               "threads": run["parameters"].get("threads"),
               "nonzero_entries": len(cent), "n": len(exact), **m,
               "tau_if_offset_by_one": float(wrong), "median_scale_vs_gt": scale}
        if not m["tau_overall"] > wrong + 0.1:
            out["warning"] = "mapping check failed: the offset-by-one placement scores as well"
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
