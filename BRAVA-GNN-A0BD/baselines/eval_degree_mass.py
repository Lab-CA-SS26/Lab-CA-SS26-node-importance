"""Zero-learning degree-mass baseline.

Ranks nodes by cumulative multi-hop degree mass d^(m) = (sum_{k=0}^{m} A^k) d
(Li et al. 2014, Section 3.3). Matches `degree_mix_mass_k` in layer.py.

Uses the same adjacency matrices as BRAVA-GNN (including leaf/clique
preprocessing). Directed graphs fuse d^(m)_out * d^(m)_in, mirroring BRAVA's
y_in * y_out fusion, so the gap isolates exactly what the learning adds.
"""
import argparse
import os
import sys
from contextlib import redirect_stdout

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_GRAPHS_ALL
from graph_regimes import DIRECTED_SKIP
from data import load_test_adj_or_build


def degree_masses(A, max_order):
    """Returns (N, max_order+1) float64 array; column m holds d^(m)."""
    deg = A @ np.ones(A.shape[0], dtype=np.float64)
    masses = [deg.copy()]
    cur = deg
    cumulative = deg.copy()
    for _ in range(max_order):
        cur = A @ cur
        cumulative = cumulative + cur
        masses.append(cumulative.copy())
    return np.column_stack(masses)


def evaluate(name, max_order, preprocessing):
    """KT array of length max_order+1, or None if pickle missing."""
    with redirect_stdout(sys.stderr):                 # keep stdout clean for the table
        loaded = load_test_adj_or_build(name, preprocessing=preprocessing)
    if loaded is None:
        return None
    adj_list, adj_t_list, _, bc_list = loaded
    A, A_t = adj_list[0], adj_t_list[0]
    bc = np.asarray(bc_list[0], dtype=np.float64)

    if name in DIRECTED_SKIP:
        scores = degree_masses(A, max_order) * degree_masses(A_t, max_order)
    else:
        scores = degree_masses(A, max_order)  # undirected: A = A^T, product redundant

    return np.array([kendalltau(scores[:, m], bc)[0] for m in range(max_order + 1)])


def _fmt_row(label, values):
    cells = "  ".join(f"{v * 100:6.1f}" if v == v else "     -" for v in values)
    return f"  {label:<22} {cells}"


def _print_table(by_graph, test_graphs, orders):
    header = "  ".join(f"d^({m})" for m in orders)
    print(f"  {'Graph':<22} {header}")
    for name in test_graphs:
        if name in by_graph:
            print(_fmt_row(name, by_graph[name]))
        else:
            print(_fmt_row(name + " (missing)", [float("nan")] * len(orders)))

    def avg_block(graphs, label):
        present = [g for g in graphs if g in by_graph]
        if present:
            print(_fmt_row(label, np.mean([by_graph[g] for g in present], axis=0)))

    print()
    avg_block([g for g in test_graphs if g not in DIRECTED_SKIP], "Undirected AVG")
    avg_block([g for g in test_graphs if g in DIRECTED_SKIP], "Directed AVG")
    avg_block(test_graphs, "AVG (all)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_order", type=int, default=6,
                    help="highest degree-mass order m (default 6: d^(0)..d^(6))")
    ap.add_argument("--test_graphs", nargs="+", default=list(TEST_GRAPHS_ALL),
                    help="subset of graphs to evaluate (default: all 14)")
    ap.add_argument("--no_preprocessing", action="store_true",
                    help="evaluate on the full graph instead of BRAVA's leaf/clique-pruned graph")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (long format: graph,order,kt)")
    args = ap.parse_args()

    preprocessing = not args.no_preprocessing
    prep_tag = "noprep" if args.no_preprocessing else "prep"
    if args.csv is None:
        args.csv = f"results/betweenness/analysis/degree_mass_kt_{prep_tag}.csv"

    print(f"Zero-learning degree-mass baseline  (preprocessing={preprocessing}, "
          f"prep={prep_tag})", file=sys.stderr)

    orders = list(range(args.max_order + 1))
    rows, by_graph = [], {}
    for name in args.test_graphs:
        print(f"  evaluating {name} ...", file=sys.stderr, flush=True)
        kt = evaluate(name, args.max_order, preprocessing)
        if kt is None:
            print(f"  {name}: pickle not found, skipped", file=sys.stderr)
            continue
        by_graph[name] = kt
        for m in orders:
            rows.append((name, m, kt[m]))

    print(f"Degree-mass ranking baseline  (preprocessing={preprocessing})  "
          f"Kendall tau-b over all nodes (x100)\n")
    _print_table(by_graph, args.test_graphs, orders)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w") as f:
        f.write("graph,order,kt\n")
        for name, m, v in rows:
            f.write(f"{name},{m},{v:.4f}\n")
    print(f"\nWrote per-graph results to {args.csv}")


if __name__ == "__main__":
    main()
