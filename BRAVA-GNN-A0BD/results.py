import fcntl
import os

from config import TEST_GRAPHS_ALL

def append_csv(filename, header, row, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(header + "\n")
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

def save_results(algo_name, test_graphs, scores, times, topk_scores, training_time,
                 top_k=False, results_dir="results",
                 scores_filtered=None, topk_scores_filtered=None, flops=None):
    """Append result rows. scores_filtered/topk_scores_filtered emit a `_filtered` row
    (bc>0 nodes only). Wallclock/training_time are always single-row."""
    # Always emit TEST_GRAPHS_ALL columns so rows stay aligned across runs with different --test_graphs subsets.
    header = "Algorithm," + ",".join(TEST_GRAPHS_ALL)

    def _fmt(d, name, decimals=4):
        return f"{d[name][0]:.{decimals}f}" if name in d and d[name] else "-"

    append_csv("all_results.csv", header, algo_name + "," + ",".join(_fmt(scores, n) for n in TEST_GRAPHS_ALL), results_dir)
    if scores_filtered is not None:
        append_csv("all_results.csv", header,
                   f"{algo_name}_filtered," + ",".join(_fmt(scores_filtered, n) for n in TEST_GRAPHS_ALL),
                   results_dir)

    if top_k:
        def _topk_row(name, src):
            parts = [name]
            for n in TEST_GRAPHS_ALL:
                v = src.get(n, {}).get(pval)
                parts.append(f"{v:.4f}" if v is not None else "-")
            return ",".join(parts)
        for label, pval in [("Top1%", 0.01), ("Top5%", 0.05), ("Top10%", 0.1)]:
            append_csv("all_results_topk.csv", header,
                       _topk_row(f"{algo_name}_{label}", topk_scores), results_dir)
            if topk_scores_filtered is not None:
                append_csv("all_results_topk.csv", header,
                           _topk_row(f"{algo_name}_filtered_{label}", topk_scores_filtered),
                           results_dir)

    time_row = algo_name + "," + ",".join(_fmt(times, n, 6) for n in TEST_GRAPHS_ALL)
    append_csv("all_results_wallclock.csv", header, time_row, results_dir)

    append_csv("all_results_training_time.csv", "Algorithm,Time", f"{algo_name},{training_time:.4f}", results_dir)

    if flops is not None:
        def _fmt_flops(d, name):
            return f"{d[name][0]:.4f}" if name in d and d[name] else "-"
        append_csv("all_results_flops.csv", header,
                   algo_name + "," + ",".join(_fmt_flops(flops, n) for n in TEST_GRAPHS_ALL),
                   results_dir)