"""Shared harness for C++ sampling baselines (KADABRA, SILVAN, Bavarian, ...).

Each caller provides binary_path, build_cmd, and parse_output; everything else
(edge-list writing, subprocess timing, CSV append) is handled here.
"""
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_GRAPHS_ALL
from graph_regimes import DIRECTED_SKIP, assert_regime
from results import append_csv


def get_graph_data(name):
    if name in ["SF", "HY", "ER", "GRP"] or name.startswith("SF_") or name.startswith("HY_"):
        path = f"./datasets/data_splits/{name}/betweenness/test.pickle"
    else:
        path = f"./datasets/data_splits/{name}_bet.pickle"
    if not os.path.exists(path):
        return None, None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    graphs, sequences, bc_mats = data[0], data[1], data[3]
    G, node_seq = graphs[0], sequences[0]
    bc_scores = bc_mats[:, 0] if isinstance(bc_mats, np.ndarray) else bc_mats[0]
    num_nodes = len(node_seq)
    if len(bc_scores) > num_nodes:
        bc_scores = bc_scores[:num_nodes]
    return G, node_seq, np.asarray(bc_scores, dtype=np.float64)


def calculate_topk(pred, label, p):
    k = max(1, int(len(label) * p))
    top_pred = np.argpartition(pred, -k)[-k:]
    top_true = np.argpartition(label, -k)[-k:]
    return len(np.intersect1d(top_pred, top_true)) / k


def write_edge_list(G, node_seq, path, directed, separator=" "):
    """Relabel nodes to 0..N-1 and write edges. `separator` is space (KADABRA/SILVAN)
    or tab (Bavarian EdgeListReader). Undirected graphs are written once per undirected edge."""
    mapping = {original: i for i, original in enumerate(node_seq)}
    H = nx.relabel_nodes(G, mapping, copy=True)
    if not directed and isinstance(H, nx.DiGraph):
        H = H.to_undirected()
    with open(path, "w") as f:
        for u, v in H.edges():
            f.write(f"{u}{separator}{v}\n")
    return len(node_seq)


def run_sampling_baseline(*, algorithm_name, binary_path, build_cmd, parse_output,
                          parse_edges_visited=None, test_graphs=None, seed=None,
                          edge_separator=" ", stdout_is_output=False,
                          log_stderr=False):
    """Run a C++ sampling binary over test_graphs and append results to the CSV files.

    seed seeds the Python/numpy RNG only (C++ internal randomness is not externally
    seedable; it only affects the algorithm name suffix).
    stdout_is_output: binary writes scores to stdout rather than a file.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if not os.path.exists(binary_path):
        print(f"Binary not found at {binary_path}. Run 'make' in its source folder first.")
        sys.exit(1)

    test_graphs = test_graphs or TEST_GRAPHS_ALL

    kt_raw, kt_filt = [], []
    t1_r, t5_r, t10_r = [], [], []
    t1_f, t5_f, t10_f = [], [], []
    times_list = []
    flops_list = []
    all_lists = (kt_raw, kt_filt, t1_r, t5_r, t10_r, t1_f, t5_f, t10_f, times_list)

    for name in test_graphs:
        print(f"Processing {name}...", end=" ", flush=True)
        G, node_seq, bc_scores = get_graph_data(name)
        if G is None:
            for lst in all_lists:
                lst.append("")
            flops_list.append("-")
            print("Skipped (not found)")
            continue
        assert_regime(name, G)
        directed = name in DIRECTED_SKIP

        with tempfile.TemporaryDirectory() as tmp:
            edges_path = os.path.join(tmp, "edges.txt")
            output_path = os.path.join(tmp, "out.txt")
            num_nodes = write_edge_list(G, node_seq, edges_path, directed, edge_separator)
            cmd = build_cmd(edges_path, output_path, directed)
            t0 = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: binary failed ({e})")
                if log_stderr and e.stderr:
                    print("stderr:\n" + e.stderr.rstrip())
                for lst in all_lists:
                    lst.append("")
                flops_list.append("-")
                continue
            elapsed = time.time() - t0
            if log_stderr and result.stderr:
                print("stderr:\n" + result.stderr.rstrip())
            edges_visited = parse_edges_visited(result.stdout) if parse_edges_visited else None
            if stdout_is_output:
                with open(output_path, "w") as f:
                    f.write(result.stdout)
            pred = parse_output(output_path, num_nodes)

        kt_r, _ = kendalltau(pred, bc_scores)
        p1r = calculate_topk(pred, bc_scores, 0.01)
        p5r = calculate_topk(pred, bc_scores, 0.05)
        p10r = calculate_topk(pred, bc_scores, 0.10)

        keep = bc_scores > 0
        if int(keep.sum()) >= 2:
            kt_f, _ = kendalltau(pred[keep], bc_scores[keep])
            p1f = calculate_topk(pred[keep], bc_scores[keep], 0.01)
            p5f = calculate_topk(pred[keep], bc_scores[keep], 0.05)
            p10f = calculate_topk(pred[keep], bc_scores[keep], 0.10)
        else:
            kt_f, p1f, p5f, p10f = float("nan"), float("nan"), float("nan"), float("nan")

        print(f"KT raw/filt: {kt_r:.4f}/{kt_f:.4f} | T1 r/f: {p1r:.4f}/{p1f:.4f} | Time: {elapsed:.4f}s")
        kt_raw.append(f"{kt_r:.4f}")
        kt_filt.append(f"{kt_f:.4f}" if not np.isnan(kt_f) else "")
        t1_r.append(f"{p1r:.4f}"); t5_r.append(f"{p5r:.4f}"); t10_r.append(f"{p10r:.4f}")
        t1_f.append(f"{p1f:.4f}" if not np.isnan(p1f) else "")
        t5_f.append(f"{p5f:.4f}" if not np.isnan(p5f) else "")
        t10_f.append(f"{p10f:.4f}" if not np.isnan(p10f) else "")
        times_list.append(f"{elapsed:.6f}")
        flops_list.append(f"{edges_visited/1e9:.6f}" if edges_visited is not None else "-")

    alg = algorithm_name
    header = "Algorithm," + ",".join(TEST_GRAPHS_ALL)

    def _pad(values):
        by_name = dict(zip(test_graphs, values))
        return [by_name.get(n, "-") or "-" for n in TEST_GRAPHS_ALL]

    append_csv("all_results.csv", header, f"{alg}," + ",".join(_pad(kt_raw)), results_dir="results/betweenness")
    append_csv("all_results.csv", header, f"{alg}_filtered," + ",".join(_pad(kt_filt)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top1%," + ",".join(_pad(t1_r)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top5%," + ",".join(_pad(t5_r)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top10%," + ",".join(_pad(t10_r)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top1%," + ",".join(_pad(t1_f)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top5%," + ",".join(_pad(t5_f)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top10%," + ",".join(_pad(t10_f)), results_dir="results/betweenness")
    append_csv("all_results_wallclock.csv", header, f"{alg}," + ",".join(_pad(times_list)), results_dir="results/betweenness")
    append_csv("all_results_flops.csv", header, f"{alg}," + ",".join(_pad(flops_list)), results_dir="results/betweenness")
