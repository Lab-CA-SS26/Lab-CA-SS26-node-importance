#!/usr/bin/env python3
"""Appendix table: accuracy of the C++ reference against the Julia port, and the burn-in bias.

Emits the table body for `Report/tables/cpp_julia_quality_table.tex` and every range the
surrounding prose quotes. Reads, all relative to `benchmark/results/`:

  cpp_quality/set_report.jsonl, set_aug10.jsonl   C++ runs scored by score_cpp_centralities.py
  cpp_quality/debias.jsonl                        check_burnin_bias.py on set_report
  report_runs/tight_julia_<g>.json                Julia, the Section 6.1 run
  topk_variant/measured/kx_<g>_k0_code_s{1,2,3}   Julia, three seeded repeats (k = 0 never
                                                  reaches the top-k allocation)
  bias_fix/fix_<g>_k0_s1.json                     Julia after the normalisation fix

The pre-fix Julia runs are used for tau_b and overlap, which are scale-invariant and so
unaffected by the fix; the absolute-error column for Julia comes from the fixed runs only.

With `--julia-from rerun_fix`, the Julia side comes from the 2026-09-16 rerun with the
stopping-coordination fix instead: rerun_fix/tight/tight_julia_<g>.json and
rerun_fix/topk/kx_<g>_k0_code_s{1,2,3}.json for tau_b and overlap, and the seed-1 k = 0 run for
the absolute error (every rerun run also carries the normalisation fix). The C++ side is the
same either way.

Usage
-----
    python3 summarize_cpp_quality.py [results_dir] [--julia-from rerun_fix]
"""
import json
import os
import statistics as st
import sys

GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll", "amazon", "dblp"]
EPS = 1e-4


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def span(xs):
    lo, hi = min(xs), max(xs)
    return f"${lo}$" if lo == hi else f"${lo}$--${hi}$"


def main():
    args = sys.argv[1:]
    rerun = None
    if "--julia-from" in args:
        i = args.index("--julia-from")
        rerun = args[i + 1]
        del args[i:i + 2]
    d = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    def julia_runs(g):
        if rerun:
            return [load(f"{d}/{rerun}/tight/tight_julia_{g}.json")] + \
                   [load(f"{d}/{rerun}/topk/kx_{g}_k0_code_s{s}.json") for s in (1, 2, 3)]
        return [load(f"{d}/report_runs/tight_julia_{g}.json")] + \
               [load(f"{d}/topk_variant/measured/kx_{g}_k0_code_s{s}.json") for s in (1, 2, 3)]

    def fixed_run(g):
        if rerun:
            return load(f"{d}/{rerun}/topk/kx_{g}_k0_code_s1.json")
        return load(f"{d}/bias_fix/fix_{g}_k0_s1.json")
    cpp = {}
    for s in ("set_report", "set_aug10"):
        for r in jsonl(f"{d}/cpp_quality/{s}.jsonl"):
            cpp.setdefault(r["graph"], []).append(r)
    report_set = {r["graph"]: r for r in jsonl(f"{d}/cpp_quality/set_report.jsonl")}
    debias = {r["graph"]: r for r in jsonl(f"{d}/cpp_quality/debias.jsonl")}

    rows, q = [], {"dtau": [], "tau": [], "ovl": [], "bias": [], "raw": [], "over": [],
                   "resc": [], "fixed": [], "jhigher": 0}
    for g in GRAPHS:
        jl = julia_runs(g)
        jl = [r for r in jl if r and r.get("tau_overall") is not None]
        c = cpp[g]
        ct = st.mean(r["tau_overall"] for r in c)
        jt = st.mean(r["tau_overall"] for r in jl)
        co = [r["overlap_topk"] for r in c]
        jo = [r["overlap_topk"] for r in jl]
        raw = report_set[g]["max_ae"] / EPS
        resc = debias[g]["debiased_tau"]["max_ae"] / EPS
        fx = fixed_run(g)
        fixed = fx["max_ae"] / EPS if fx else None
        q["dtau"].append(jt - ct); q["tau"] += [ct, jt]; q["ovl"] += co + jo
        q["bias"].append(debias[g]["tau"] / debias[g]["N"]); q["raw"].append(raw)
        q["over"].append(debias[g]["raw"]["over_eps"]); q["resc"].append(resc)
        q["jhigher"] += jt > ct
        if fixed is not None:
            q["fixed"].append(fixed)
        rows.append(
            f"            {g:<17} & ${ct:.3f}$ & ${jt:.3f}$ & {span(co)} & {span(jo)} & "
            f"${raw:.1f}$ & ${resc:.2f}$ & {'$%.2f$' % fixed if fixed is not None else '--'} \\\\"
        )

    print(f"% C++: {len(cpp[GRAPHS[0]])} runs/graph; Julia: up to 4 runs/graph for tau_b and overlap"
          f" ({'rerun ' + rerun if rerun else 'pre-fix runs'})")
    print("% ---- Report/tables/cpp_julia_quality_table.tex (body) ----")
    print("\n".join(rows))
    print("\n% ---- ranges quoted in the text ----")
    print(f"  tau_b, both implementations: {min(q['tau']):.3f}--{max(q['tau']):.3f}")
    print(f"  Julia - C++ tau_b: {min(q['dtau']):+.4f} .. {max(q['dtau']):+.4f}; Julia higher on {q['jhigher']}/{len(GRAPHS)}")
    print(f"  top-100 overlap, all runs: {min(q['ovl'])}--{max(q['ovl'])}")
    print(f"  burn-in fraction tau/N: {100*min(q['bias']):.1f}%--{100*max(q['bias']):.1f}%")
    print(f"  reference max error as output: {min(q['raw']):.1f}--{max(q['raw']):.1f} x eps; "
          f"{min(q['over'])}--{max(q['over'])} vertices over eps")
    print(f"  reference rescaled by N/(N-tau): {min(q['resc']):.2f}--{max(q['resc']):.2f} x eps")
    if q["fixed"]:
        print(f"  Julia after the fix: {min(q['fixed']):.2f}--{max(q['fixed']):.2f} x eps "
              f"({len(q['fixed'])}/{len(GRAPHS)} graphs measured)")
    missing = [g for g in GRAPHS if not fixed_run(g)]
    if missing:
        print(f"  !! no post-fix run yet for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
