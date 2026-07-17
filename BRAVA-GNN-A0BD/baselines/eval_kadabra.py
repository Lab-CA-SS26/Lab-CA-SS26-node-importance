import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from _sampling_baseline import run_sampling_baseline

KADABRA_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "baselines", "kadabra", "kadabra",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--err", type=float, default=0.01,
                   help="KADABRA error tolerance (epsilon/lambda)")
    p.add_argument("--delta", type=float, default=0.1,
                   help="KADABRA failure probability (delta)")
    p.add_argument("--algorithm_name", default=None)
    p.add_argument("--test_graphs", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def parse_output(path, num_nodes):
    pred = np.zeros(num_nodes, dtype=np.float64)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            i = int(parts[0])
            v = float(parts[1])
            if 0 <= i < num_nodes:
                pred[i] = v
    return pred


def parse_edges_visited(stdout):
    # Proxy for compute cost stored in all_results_flops.csv.
    m = re.search(r"Edges visited:\s+(\d+)", stdout)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    name = args.algorithm_name or (
        f"KADABRA_NK_err{args.err}_delta{args.delta}"
        + (f"_S{args.seed}" if args.seed is not None else "")
    )

    def build_cmd(edges_path, output_path, directed):
        cmd = [KADABRA_BIN, "-o", output_path]
        if directed:
            cmd.append("-d")
        cmd += [str(args.err), str(args.delta), edges_path]
        return cmd

    run_sampling_baseline(
        algorithm_name=name,
        binary_path=KADABRA_BIN,
        build_cmd=build_cmd,
        parse_output=parse_output,
        parse_edges_visited=parse_edges_visited,
        test_graphs=args.test_graphs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
