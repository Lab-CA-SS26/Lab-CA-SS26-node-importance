#!/usr/bin/env python3
"""
summarize_topk.py --- summarise stage 'kx'/'kxl' of reproduce_report.sh.

Reads the kx_<graph>_k<k>_<variant>_s<seed>.json runs in a directory and answers
the two questions that stage was written for:

  (a) Is the top-k restriction cheaper than k=0, and is the "sometimes it is
      dearer" effect larger than the seed-to-seed spread?  Ratios are formed
      PER SEED against that seed's own k=0 run and only then averaged, so the
      shared noise of the k=0 denominator cancels.

  (b) Does the paper's confidence-budget allocation differ from the C++
      reference's?  Compared per seed as well, for the same reason.

Usage:  python3 summarize_topk.py <dir>
"""
import json, sys, glob, os, re
from statistics import mean, pstdev

VARIANT_ORDER = ["code", "paper", "cpp", "paper_bd", "paper_ex"]
PAT = re.compile(r"kx_(?P<graph>.+)_k(?P<k>\d+)_(?P<variant>code|paper_bd|paper_ex|paper|cpp)_s(?P<seed>\d+)\.json$")


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "kx_*.json"))):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        try:
            j = json.load(open(f))
        except json.JSONDecodeError:
            print(f"  !! unreadable (still being written?): {os.path.basename(f)}")
            continue
        g, k, v, s = m["graph"], int(m["k"]), m["variant"], int(m["seed"])
        runs[(g, k, v, s)] = {
            "samples": j["num_samples"],
            "time": j["execution_time_seconds"],
            "tau_b": j.get("tau_overall"),
            "overlap": j.get("overlap_topk"),
            "threads": j.get("parameters", {}).get("threads"),
        }
    return runs


def ms(xs):
    """mean and (sample) std as a compact string."""
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{mean(xs):.3f} +- {pstdev(xs) * (len(xs) / (len(xs) - 1)) ** 0.5:.3f}"


def main(d):
    runs = load(d)
    if not runs:
        print(f"no kx_*.json runs in {d}")
        return 1

    graphs = sorted({g for g, _, _, _ in runs})
    ks = sorted({k for _, k, _, _ in runs if k > 0})
    seeds = sorted({s for _, _, _, s in runs})

    threads = {r["threads"] for r in runs.values()}
    print(f"runs: {len(runs)}   graphs: {len(graphs)}   seeds: {seeds}   threads: {sorted(threads)}")
    if len(threads) > 1:
        print("  !! MIXED THREAD COUNTS --- these runs are not comparable")
    print()

    # ---------------------------------------------------------------- raw table
    print("=== raw (mean +- std over seeds) ===")
    hdr = f"{'graph':<17}{'k':>5} {'variant':<7}{'samples/1e6':>18}{'time [s]':>18}{'tau_b':>10}"
    print(hdr)
    print("-" * len(hdr))
    for g in graphs:
        base = [runs[(g, 0, "code", s)]["samples"] / 1e6 for s in seeds if (g, 0, "code", s) in runs]
        bt = [runs[(g, 0, "code", s)]["time"] for s in seeds if (g, 0, "code", s) in runs]
        btau = [runs[(g, 0, "code", s)]["tau_b"] for s in seeds if (g, 0, "code", s) in runs and runs[(g, 0, "code", s)]["tau_b"] is not None]
        print(f"{g:<17}{0:>5} {'--':<7}{ms(base):>18}{ms(bt):>18}{ms(btau):>10}")
        for k in ks:
            for v in VARIANT_ORDER:
                key = [(g, k, v, s) for s in seeds if (g, k, v, s) in runs]
                if not key:
                    continue
                print(f"{'':<17}{k:>5} {v:<7}"
                      f"{ms([runs[x]['samples'] / 1e6 for x in key]):>18}"
                      f"{ms([runs[x]['time'] for x in key]):>18}"
                      f"{ms([runs[x]['tau_b'] for x in key if runs[x]['tau_b'] is not None]):>10}")
        print()

    # ------------------------------------------------- (a) top-k vs k=0, paired
    print("=== (a) samples relative to k=0, paired per seed ===")
    print("    ratio > 1 means the top-k run drew MORE samples than computing all centralities.")
    hdr = f"{'graph':<17}{'k':>5} {'variant':<7}{'sample ratio':>20}{'time ratio':>20}  verdict"
    print(hdr)
    print("-" * (len(hdr) + 10))
    anomalies = []
    for g in graphs:
        for k in ks:
            for v in VARIANT_ORDER:
                sr, tr = [], []
                for s in seeds:
                    a, b = (g, k, v, s), (g, 0, "code", s)
                    if a in runs and b in runs:
                        sr.append(runs[a]["samples"] / runs[b]["samples"])
                        tr.append(runs[a]["time"] / runs[b]["time"])
                if not sr:
                    continue
                lo = mean(sr) - (pstdev(sr) * (len(sr) / (len(sr) - 1)) ** 0.5 if len(sr) > 1 else 0)
                verdict = ""
                if len(sr) > 1 and lo > 1.0:
                    verdict = "<< DEARER than k=0 (>1 std)"
                    anomalies.append((g, k, v, mean(sr)))
                elif mean(sr) > 1.0:
                    verdict = "dearer on average, within 1 std of 1"
                print(f"{g:<17}{k:>5} {v:<7}{ms(sr):>20}{ms(tr):>20}  {verdict}")
        print()
    if anomalies:
        print("  top-k costs MORE than k=0, beyond one std, on:")
        for g, k, v, r in anomalies:
            print(f"    {g} k={k} ({v}): {r:.3f}x")
    else:
        print("  no configuration is dearer than k=0 by more than one std.")
    print()

    # ------------------------------------------- (b) paper vs code vs cpp, paired
    print("=== (b) budget-allocation variant, paired per seed against 'code' ===")
    print("    ratio < 1 means that variant needed FEWER samples than the C++ reference's allocation.")
    hdr = f"{'graph':<17}{'k':>5} {'variant':<7}{'samples vs code':>20}{'time vs code':>20}{'tau_b delta':>14}"
    print(hdr)
    print("-" * len(hdr))
    pooled = {v: [] for v in VARIANT_ORDER if v != "code"}
    for g in graphs:
        for k in ks:
            for v in VARIANT_ORDER:
                if v == "code":
                    continue
                sr, tr, dt = [], [], []
                for s in seeds:
                    a, b = (g, k, v, s), (g, k, "code", s)
                    if a in runs and b in runs:
                        sr.append(runs[a]["samples"] / runs[b]["samples"])
                        tr.append(runs[a]["time"] / runs[b]["time"])
                        if runs[a]["tau_b"] is not None and runs[b]["tau_b"] is not None:
                            dt.append(runs[a]["tau_b"] - runs[b]["tau_b"])
                if not sr:
                    continue
                pooled[v].extend(sr)
                print(f"{g:<17}{k:>5} {v:<7}{ms(sr):>20}{ms(tr):>20}{ms(dt):>14}")
        print()
    for v, xs in pooled.items():
        if xs:
            print(f"  pooled over all (graph, k): {v}/code samples = {ms(xs)}   "
                  f"[min {min(xs):.3f}, max {max(xs):.3f}, n={len(xs)}]")
    print()

    # ---------------------------------------------------------------- LaTeX body
    # Report table "topk_allocation": one row per (graph, k), the two allocations'
    # sample counts relative to that seed's own k=0 run, and their ratio.
    print("=== LaTeX table body (tables/topk_allocation_table.tex) ===")

    def cell(xs):
        if not xs:
            return "--"
        if len(xs) == 1:
            return f"${xs[0]:.2f}$"
        sd = pstdev(xs) * (len(xs) / (len(xs) - 1)) ** 0.5
        return f"${mean(xs):.2f} \\pm {sd:.2f}$"

    for g in graphs:
        for j, k in enumerate(ks):
            rel = {}
            for v in ("code", "paper", "paper_bd"):
                rel[v] = [runs[(g, k, v, s)]["samples"] / runs[(g, 0, "code", s)]["samples"]
                          for s in seeds if (g, k, v, s) in runs and (g, 0, "code", s) in runs]
            name = f"\\texttt{{{g}}}" if j == 0 else ""
            print(f"            {name} & ${k}$ & {cell(rel['code'])} & {cell(rel['paper'])} "
                  f"& {cell(rel['paper_bd'])} \\\\")
        if g != graphs[-1]:
            print(r"            \midrule")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/reproduce"))
