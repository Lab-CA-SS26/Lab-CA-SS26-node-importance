#!/usr/bin/env python3
"""Turn reproduce_report.sh's raw JSON output into the report's LaTeX tables.

Usage:  summarize_reproduce.py {tight|bvk} <result-dir>

Emits, for the requested stage, the LaTeX table body that belongs in
Report/tables/ plus the summary statistics quoted in the report prose.
Missing runs are reported rather than silently skipped.
"""
import json
import os
import statistics
import sys

TIGHT_GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902",
                "email-EuAll", "amazon", "dblp"]
BVK_GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",
              "com-youtube", "amazon", "dblp", "cit-Patents", "com-lj"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def sec(x):
    """Match the report's formatting: 2 decimals under 100s, 1 above, thin-space thousands."""
    if x < 100:
        return f"{x:.2f}\\,s"
    if x < 1000:
        return f"{x:.1f}\\,s"
    return f"{int(x // 1000)}\\,{x % 1000:05.1f}\\,s"


def num(n):
    return f"{n:,}".replace(",", "\\,")


def stats(label, values, unit="x"):
    return (f"  {label}: mean {statistics.mean(values):.3f}{unit}, "
            f"median {statistics.median(values):.3f}{unit}, "
            f"range {min(values):.3f}--{max(values):.3f}{unit}")


def tight(d):
    rows, missing = [], []
    for g in TIGHT_GRAPHS:
        c, j = load(f"{d}/tight_cpp_{g}.json"), load(f"{d}/tight_julia_{g}.json")
        if not (c and j):
            missing.append(g)
            continue
        rows.append((g, c["execution_time_seconds"], j["execution_time_seconds"],
                     c["num_samples"], j["num_samples"], j.get("samples_over_omega"),
                     j.get("tau_overall"), j.get("overlap_topk")))
    if missing:
        print(f"!! missing runs: {', '.join(missing)}", file=sys.stderr)
    if not rows:
        return

    print("% ---- Report/tables/cpp_vs_julia_tight_table.tex (body) ----")
    for g, ct, jt, cs, js, _, _, _ in rows:
        print(f"            {g:17s} & {sec(ct)} & {sec(jt)} & ${jt/ct:.2f}\\times$ "
              f"& ${num(cs)}$ & ${num(js)}$ & ${js/cs:.2f}\\times$ \\\\")

    t_ratios = [jt / ct for _, ct, jt, _, _, _, _, _ in rows]
    s_ratios = [js / cs for _, _, _, cs, js, _, _, _ in rows]
    sow = [r[5] for r in rows if r[5] is not None]
    taus = [r[6] for r in rows if r[6] is not None]
    ovl = [r[7] for r in rows if r[7] is not None]

    print("\n% ---- figures quoted in Section 6.1 / Appendix A.2 ----")
    print(stats("time ratio (Julia/C++)", t_ratios))
    print(stats("sample ratio (Julia/C++)", s_ratios))
    if sow:
        print(f"  samples_over_omega: range {min(sow):.2f}--{max(sow):.2f}")
    if taus:
        print(f"  Kendall tau_b vs ground truth: range {min(taus):.3f}--{max(taus):.3f}")
    if ovl:
        print(f"  top-100 overlap: range {min(ovl)}--{max(ovl)} out of 100")


def bvk(d):
    rows, missing = [], []
    for g in BVK_GRAPHS:
        k = load(f"{d}/bvk_kadabra_{g}.json")
        bc = load(f"{d}/bvk_brava_cpu_{g}.json")
        bg = load(f"{d}/bvk_brava_gpu_{g}.json")
        if not (k and bc and bg):
            missing.append(g)
            continue
        rows.append((g, bc["execution_time_seconds"], bg["execution_time_seconds"],
                     k["execution_time_seconds"], bc.get("tau_overall"),
                     k.get("tau_overall"), bc.get("overlap_topk"), k.get("overlap_topk")))
    if missing:
        print(f"!! missing runs: {', '.join(missing)}", file=sys.stderr)
    if not rows:
        return

    f3 = lambda v: "--" if v is None else f"{v:.3f}"
    ov = lambda v: "--" if v is None else f"{v}/100"

    print("% ---- Report/tables/brava_vs_kadabra_table.tex (body) ----")
    for g, bcpu, bgpu, kt, tb, tk, ob, ok in rows:
        print(f"            {g:17s} & {sec(bcpu)} & {sec(bgpu)} & {sec(kt)} "
              f"& {f3(tb)} & {f3(tk)} & {ov(ob)} & {ov(ok)} \\\\")

    print("\n% ---- figures quoted in Section 6.2 ----")
    gpu = [bcpu / bgpu for _, bcpu, bgpu, _, _, _, _, _ in rows]
    print(stats("BRAVA GPU speedup over CPU", gpu))
    faster_b = [g for g, bcpu, _, kt, *_ in rows if bcpu < kt]
    faster_k = [g for g, _, bgpu, kt, *_ in rows if kt < bgpu]
    print(f"  BRAVA-CPU faster than KADABRA on: {', '.join(faster_b) or 'none'}")
    print(f"  KADABRA faster than BRAVA-GPU on: {', '.join(faster_k) or 'none'}")
    tb_wins = sum(1 for r in rows if r[4] is not None and r[5] is not None and r[4] > r[5])
    ok_wins = sum(1 for r in rows if r[6] is not None and r[7] is not None and r[7] > r[6])
    print(f"  BRAVA higher overall tau_b on {tb_wins}/{len(rows)} graphs")
    print(f"  KADABRA higher top-100 overlap on {ok_wins}/{len(rows)} graphs")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("tight", "bvk"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    (tight if sys.argv[1] == "tight" else bvk)(sys.argv[2].rstrip("/"))
