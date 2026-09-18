#!/usr/bin/env python3
"""
summarize_threads.py --- the figures Section 6.1's thread-scaling paragraphs quote.

Reads `{julia,cpp}_<graph>_t<T>_s<seed>.json` (the runs make_plots.py threads plots) and prints,
per graph and implementation, the median runtime and sample count over seeds at every thread
count, then the quoted quantities:

  * optimum thread count (lowest median runtime);
  * slowdown at t=48 against t=8;
  * peak speedup = median runtime at t=1 / median runtime at the optimum;
  * Julia/C++ median runtime ratio at every t (how far C++ leads);
  * samples at t=48 against t=1 (stopping overshoot growing with threads).

Usage:  python3 summarize_threads.py <dir>
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

PAT = re.compile(r"(julia|cpp)_(.+)_t(\d+)_s(\d+)\.json$")
GRAPHS = ["soc-Slashdot0902", "amazon", "cit-Patents", "com-lj"]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/report_runs"
    rt = defaultdict(list)   # (impl, graph, t) -> runtimes
    ns = defaultdict(list)   # (impl, graph, t) -> sample counts
    for f in glob.glob(os.path.join(d, "*_t*_s*.json")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        impl, g, t = m.group(1), m.group(2), int(m.group(3))
        j = json.load(open(f))
        if impl == "julia" and j["parameters"].get("threads") != t:
            sys.exit(f"!! {f}: parameters.threads={j['parameters'].get('threads')} but filename says t={t}")
        rt[(impl, g, t)].append(j["execution_time_seconds"])
        ns[(impl, g, t)].append(j["num_samples"])

    ts = sorted({k[2] for k in rt})
    summary = defaultdict(dict)
    for g in GRAPHS:
        print(f"== {g}")
        for impl in ("cpp", "julia"):
            med = {t: st.median(rt[(impl, g, t)]) for t in ts if rt[(impl, g, t)]}
            smp = {t: st.median(ns[(impl, g, t)]) for t in ts if ns[(impl, g, t)]}
            opt = min(med, key=med.get)
            row = "  ".join(f"t{t}:{med[t]:.2f}s" for t in ts if t in med)
            print(f"  {impl:5} {row}")
            print(f"  {'':5} samples " + "  ".join(f"t{t}:{smp[t]:.0f}" for t in ts if t in smp))
            summary[impl][g] = dict(opt=opt, slow48=med[48] / med[8], peak=med[1] / med[opt],
                                    t1=med[1], topt=med[opt], samp48=smp[48] / smp[1])
            s = summary[impl][g]
            print(f"  {'':5} optimum t={opt}  t48/t8 {s['slow48']:.2f}x  peak speedup {s['peak']:.2f}x"
                  f"  samples t48/t1 {s['samp48']:.2f}x")
        ratio = [st.median(rt[("julia", g, t)]) / st.median(rt[("cpp", g, t)]) for t in ts]
        print("  julia/cpp " + "  ".join(f"t{t}:{r:.2f}" for t, r in zip(ts, ratio)))
        summary["ratio"][g] = ratio

    print("\n== quoted in Section 6.1")
    for impl in ("cpp", "julia"):
        S = summary[impl]
        print(f"  {impl:5} optima {sorted({S[g]['opt'] for g in GRAPHS})}"
              f"  t48/t8 {min(S[g]['slow48'] for g in GRAPHS):.2f}--{max(S[g]['slow48'] for g in GRAPHS):.2f}x"
              f"  peak speedup {min(S[g]['peak'] for g in GRAPHS):.2f}--{max(S[g]['peak'] for g in GRAPHS):.2f}x"
              f"  samples t48/t1 {min(S[g]['samp48'] for g in GRAPHS):.2f}--{max(S[g]['samp48'] for g in GRAPHS):.2f}x")
    s = summary["julia"]["soc-Slashdot0902"]
    r = summary["ratio"]["soc-Slashdot0902"]
    print(f"  soc-Slashdot0902 julia: {s['t1']:.2f}s at t=1 vs {s['topt']:.2f}s at optimum t={s['opt']};"
          f"  julia/cpp {min(r):.1f}--{max(r):.1f}x")


if __name__ == "__main__":
    main()
