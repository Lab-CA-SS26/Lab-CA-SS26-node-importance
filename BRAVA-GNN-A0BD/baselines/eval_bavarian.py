import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from _sampling_baseline import run_sampling_baseline

# Built by baselines/Bavarian/compile.sh (install target -> bin/).
BAVARIAN_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "baselines", "Bavarian", "bin", "progrbavarian",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--err", type=float, default=0.01,
                   help="Bavarian target accuracy (epsilon)")
    p.add_argument("--delta", type=float, default=0.1,
                   help="Bavarian failure probability (delta)")
    p.add_argument("--method", default="rk", choices=["ab", "bp", "rk"],
                   help="estimator wrapped by Bavarian: ab=ABRA, "
                        "bp=Borassi-Natale, rk=Riondato-Kornaropoulos")
    p.add_argument("--mctrials", type=int, default=100,
                   help="Monte-Carlo trials for the Rademacher-average bound")
    p.add_argument("--multiplier", type=float, default=2.0,
                   help="progressive sample-size scaling factor (must be > 1)")
    p.add_argument("--algorithm_name", default=None)
    p.add_argument("--test_graphs", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def parse_output(path, num_nodes):
    """progrbavarian writes one `node_id : score` line per node to stdout."""
    pred = np.zeros(num_nodes, dtype=np.float64)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 or parts[1] != ":":
                continue
            i = int(parts[0])
            v = float(parts[2])
            if 0 <= i < num_nodes:
                pred[i] = v
    return pred


def main():
    args = parse_args()
    name = args.algorithm_name or (
        f"Bavarian_{args.method}_eps{args.err}_delta{args.delta}"
        + (f"_S{args.seed}" if args.seed is not None else "")
    )

    def build_cmd(edges_path, output_path, directed):
        # progrbavarian prints scores to stdout, so output_path is unused.
        # `-m` runs progressively until epsilon is met; the trailing `1`
        # (iteration count) is then an ignored placeholder positional.
        cmd = [BAVARIAN_BIN]
        if directed:
            cmd.append("-d")
        cmd += ["-m", str(args.multiplier), "1", str(args.err),
                str(args.mctrials), args.method, str(args.delta), edges_path]
        return cmd

    run_sampling_baseline(
        algorithm_name=name,
        binary_path=BAVARIAN_BIN,
        build_cmd=build_cmd,
        parse_output=parse_output,
        test_graphs=args.test_graphs,
        seed=args.seed,
        edge_separator="\t",
        stdout_is_output=True,
    )


if __name__ == "__main__":
    main()
