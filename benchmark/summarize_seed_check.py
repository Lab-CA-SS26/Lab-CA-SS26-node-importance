#!/usr/bin/env python3
"""
summarize_seed_check.py --- report the seed experiment (TODO.md).

Reads the raw per-run JSON written by run_seed_check.sh (sc_<arm>_<graph>_s<seed>.json,
arm in {cpp, julia, ci11, sib, seq}) and prints, per graph:

  * mean +- std of num_samples, runtime, tau_b (tau_overall), top-100 overlap, n_checks,
    for each arm;
  * the Julia / C++ sample ratio per seed (default Julia, ci11, seq vs cpp at the same seed);
  * Julia's sample counts expressed in units of its default check_interval, and its excess
    over C++ in the same units.

Usage:  python3 summarize_seed_check.py [results/seed_check]
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

ARMS = ["cpp", "julia", "ci11", "sib", "seq"]
ARM_LABEL = {
    "cpp": "C++ (arm 2)",
    "julia": "Julia default (arm 1)",
    "ci11": "Julia ci=11 (arm 3)",
    "seq": "Julia sequential (arm 4)",
    "sib": "Julia stop-in-batch (arm 5)",
}
FNAME = re.compile(r"sc_(cpp|julia|ci11|sib|seq)_(.+)_s(\d+)\.json$")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(m, s, nd=1):
    if isinstance(m, float) and math.isnan(m):
        return "  --   "
    return f"{m:,.{nd}f} +- {s:,.{nd}f}"


def load(d):
    # runs[graph][arm][seed] = dict of fields
    runs = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(d, "sc_*.json")):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        arm, graph, seed = m.group(1), m.group(2), int(m.group(3))
        with open(path) as f:
            j = json.load(f)
        p = j.get("parameters", {})
        runs[graph][arm][seed] = {
            "num_samples": j.get("num_samples"),
            "phase2_pairs": j.get("phase2_pairs"),
            "n_checks": j.get("n_checks"),
            "runtime": j.get("execution_time_seconds"),
            "tau_b": j.get("tau_overall"),
            "overlap": j.get("overlap_topk"),
            "tau": j.get("kadabra_tau"),
            "threads": p.get("threads"),
            "directed": p.get("directed"),
            "parallel": p.get("parallel"),
            "ci_eff": p.get("check_interval_effective"),
        }
    return runs


def sanity(runs):
    """Loudly flag runs that were not made at 8 threads / correct directedness."""
    problems = []
    directed_graphs = {"soc-Epinions1", "soc-Slashdot0902", "email-EuAll"}
    for graph, arms in runs.items():
        for arm, seeds in arms.items():
            for seed, r in seeds.items():
                tag = f"sc_{arm}_{graph}_s{seed}"
                if r["threads"] not in (8, None) and r["threads"] != 8:
                    # sequential arm still records threads=8 (JULIA_NUM_THREADS); C++ records 8.
                    problems.append(f"{tag}: threads={r['threads']} (expected 8)")
                if graph in directed_graphs and r["directed"] not in (True, 1):
                    problems.append(f"{tag}: directed={r['directed']} (expected true)")
    if problems:
        print("!! SANITY WARNINGS")
        for p in problems:
            print("   " + p)
        print()


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/seed_check"
    if not os.path.isdir(d):
        sys.exit(f"no such directory: {d}")
    runs = load(d)
    if not runs:
        sys.exit(f"no sc_*.json files in {d}")

    sanity(runs)

    order = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",
             "amazon", "dblp"]
    graphs = [g for g in order if g in runs] + [g for g in runs if g not in order]

    for graph in graphs:
        arms = runs[graph]
        print("=" * 78)
        print(graph)
        print("=" * 78)

        # default Julia check_interval unit (max(1000, tau/10)); take it from a julia run.
        interval = None
        for arm in ("julia", "ci11", "seq"):
            for r in arms.get(arm, {}).values():
                if r["tau"] is not None:
                    interval = max(1000, r["tau"] // 10)
                    break
            if interval:
                break

        # per-arm aggregate table
        hdr = f"{'arm':<26}{'num_samples':>26}{'runtime s':>16}{'tau_b':>14}{'overlap':>12}{'n_checks':>16}"
        print(hdr)
        print("-" * len(hdr))
        for arm in ARMS:
            seeds = arms.get(arm, {})
            if not seeds:
                continue
            ns = [r["num_samples"] for r in seeds.values()]
            rt = [r["runtime"] for r in seeds.values()]
            tb = [r["tau_b"] for r in seeds.values()]
            ov = [r["overlap"] for r in seeds.values()]
            nc = [r["n_checks"] for r in seeds.values()]
            # The C++ binary's own ground-truth scoring maps nodes by a numbering that does
            # not line up with the CSV, so its internal tau_b/overlap are meaningless (the
            # report scores C++ centralities through a separate pipeline). Suppress them;
            # accuracy-invariance is assessed across the three Julia arms, which share the
            # runner's node mapping.
            tb_cell = " n/a*  " if arm == "cpp" else fmt(mean(tb), std(tb), 4)
            ov_cell = " n/a*  " if arm == "cpp" else fmt(mean(ov), std(ov), 1)
            print(f"{ARM_LABEL[arm]:<26}"
                  f"{fmt(mean(ns), std(ns), 0):>26}"
                  f"{fmt(mean(rt), std(rt), 2):>16}"
                  f"{tb_cell:>14}"
                  f"{ov_cell:>12}"
                  f"{('  --   ' if all(x is None for x in nc) else fmt(mean(nc), std(nc), 0)):>16}")
        print("  * C++ internal tau_b/overlap use an incompatible node numbering; ignore them.")
        print()

        # per-seed Julia/C++ ratio and interval accounting
        cpp = arms.get("cpp", {})
        if cpp and interval:
            print(f"check_interval (default Julia) = {interval:,}   "
                  f"[= max(1000, tau/10), tau ~ omega/100]")
            print(f"{'seed':<6}{'C++ samples':>16}{'Julia samples':>16}{'J/C ratio':>11}"
                  f"{'J-C':>14}{'J-C /interval':>15}{'ci11 J/C':>11}{'sib J/C':>11}{'seq J/C':>11}")
            print("-" * 111)
            for seed in sorted(cpp):
                c = cpp[seed]["num_samples"]
                row = f"{seed:<6}{c:>16,}"
                j = arms.get("julia", {}).get(seed, {}).get("num_samples")
                if j is not None:
                    row += f"{j:>16,}{j / c:>11.4f}{j - c:>14,}{(j - c) / interval:>15.2f}"
                else:
                    row += f"{'--':>16}{'--':>11}{'--':>14}{'--':>15}"
                ci = arms.get("ci11", {}).get(seed, {}).get("num_samples")
                sb = arms.get("sib", {}).get(seed, {}).get("num_samples")
                sq = arms.get("seq", {}).get(seed, {}).get("num_samples")
                row += f"{ci / c:>11.4f}" if ci else f"{'--':>11}"
                row += f"{sb / c:>11.4f}" if sb else f"{'--':>11}"
                row += f"{sq / c:>11.4f}" if sq else f"{'--':>11}"
                print(row)
            print()

        # headline ratios
        def ratio_over_cpp(arm):
            rs = []
            for seed in cpp:
                a = arms.get(arm, {}).get(seed, {}).get("num_samples")
                c = cpp[seed]["num_samples"]
                if a is not None and c:
                    rs.append(a / c)
            return rs

        for arm in ("julia", "ci11", "sib", "seq"):
            rs = ratio_over_cpp(arm)
            if rs:
                print(f"  {ARM_LABEL[arm]:<26} / C++ samples: "
                      f"{mean(rs):.4f} +- {std(rs):.4f}  (n={len(rs)})")
        print()


if __name__ == "__main__":
    main()
