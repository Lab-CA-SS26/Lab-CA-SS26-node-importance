#!/usr/bin/env python3
"""Turn reproduce_report.sh's raw JSON output into the report's LaTeX tables.

Usage:  summarize_reproduce.py {tight|bvk|tightx|k} <result-dir>

Emits, for the requested stage, the LaTeX table body that belongs in
Report/tables/ plus the summary statistics quoted in the report prose.
Missing runs are reported rather than silently skipped.
"""
import json
import os
import statistics
import sys

# The six graphs of the C++ vs Julia comparison (Section 6.1).
TIGHT_GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902",
                "email-EuAll", "amazon", "dblp"]
# The three further graphs that carry an eps=1e-4 Julia run but no C++ counterpart;
# they exist only to complete the tight-epsilon column of Section 6.4.
TIGHT_EXTRA_GRAPHS = ["com-youtube", "cit-Patents", "com-lj"]
BVK_GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",
              "com-youtube", "amazon", "dblp", "cit-Patents", "com-lj"]


def _secs(x):
    """Seconds as LaTeX, thousands separated by a thin space to match the other tables."""
    return f"{x:,.1f}".replace(",", "\\,") + "\\,s"


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


def tightx(d):
    """Section 6.4's tight-epsilon claim: KADABRA at eps=1e-4 against BRAVA-GNN.

    Cross-references the eps=1e-4 Julia runs with the eps=1e-2 BRAVA-GNN runs of
    stage 'bvk'. The comparison is deliberately against BRAVA on *CPU*: that is
    its slower device, so it is the conservative choice for the runtime ratio.
    Accuracy is device-independent.

    The per-graph verdict is what the report's "on every instance" wording rests
    on, so it is computed here rather than read off a table by hand.
    """
    rows, missing = [], []
    for g in BVK_GRAPHS:
        k = load(f"{d}/tight_julia_{g}.json")
        b = load(f"{d}/bvk_brava_cpu_{g}.json")
        if not (k and b):
            missing.append(f"{g}({'tight' if not k else 'brava'})")
            continue
        rows.append((g, k["execution_time_seconds"], b["execution_time_seconds"],
                     k.get("tau_overall"), b.get("tau_overall"),
                     k.get("overlap_topk"), b.get("overlap_topk")))
    if missing:
        print(f"!! missing runs: {', '.join(missing)}", file=sys.stderr)
    if not rows:
        return

    print("% ---- Section 6.4: KADABRA eps=1e-4 vs BRAVA-GNN eps=1e-2 ----")
    print(f"{'graph':17s} {'K tau_b':>9s} {'B tau_b':>9s} {'K ovl':>6s} {'B ovl':>6s} "
          f"{'K time':>11s} {'B cpu':>9s} {'ratio':>9s}  verdict")
    both, tau_losses, ovl_losses = 0, [], []
    for g, kt, bt, ktau, btau, kov, bov in rows:
        wins_tau = ktau > btau
        wins_ovl = kov > bov
        if wins_tau and wins_ovl:
            both += 1
            verdict = "KADABRA both"
        else:
            verdict = "!! " + ", ".join(
                x for x, w in (("tau_b", wins_tau), ("overlap", wins_ovl)) if not w
            ) + " NOT won"
        if not wins_tau:
            tau_losses.append(g)
        if not wins_ovl:
            ovl_losses.append(g)
        print(f"{g:17s} {ktau:9.3f} {btau:9.3f} {kov:6d} {bov:6d} "
              f"{kt:10.1f}s {bt:8.2f}s {kt/bt:8.1f}x  {verdict}")

    ktaus = [r[3] for r in rows]
    btaus = [r[4] for r in rows]
    kovs = [r[5] for r in rows]
    bovs = [r[6] for r in rows]
    ratios = [(r[0], r[1] / r[2]) for r in rows]
    slowest = max(ratios, key=lambda t: t[1])

    # Section 6.4 and Appendix A.2 quote only ranges; the per-graph values live here so
    # the claim can be checked instance by instance. BRAVA-GNN's accuracy is deliberately
    # NOT repeated: Table `tab:brava_vs_kadabra` carries it as a mean over three seeds,
    # and a single-run value beside it would contradict that table. Wall-clock is the
    # same single timed pass in both, so the ratio is safe to give here.
    print("\n% ---- Report/tables/tight_accuracy_table.tex (body) ----")
    for g, kt, bt, ktau, btau, kov, bov in rows:
        r = load(f"{d}/tight_julia_{g}.json")
        sow = r.get("samples_over_omega") if r else None
        ratio = f"{kt/bt:,.0f}".replace(",", "\\,")
        print(f"            {g:<17} & ${ktau:.3f}$ & ${kov}$ & "
              f"${sow:.2f}$ & {_secs(kt)} & ${ratio}\\times$ \\\\"
              + ("  % KADABRA loses tau_b to BRAVA-GNN here" if ktau <= btau else ""))

    print("\n% ---- figures quoted in Section 6.4 and the Conclusion ----")
    print(f"  instances compared: {len(rows)} of {len(BVK_GRAPHS)}")
    print(f"  KADABRA tau_b:  {min(ktaus):.3f}--{max(ktaus):.3f}")
    print(f"  BRAVA   tau_b:  {min(btaus):.3f}--{max(btaus):.3f}")
    print(f"  KADABRA overlap: {min(kovs)}--{max(kovs)} out of 100")
    print(f"  BRAVA   overlap: {min(bovs)}--{max(bovs)} out of 100")
    print(f"  runtime ratio (KADABRA 1e-4 / BRAVA CPU): "
          f"{min(r for _, r in ratios):.0f}--{max(r for _, r in ratios):.0f}x")
    print(f"  slowest-case anchor: {slowest[0]} at {slowest[1]:.0f}x")
    print(f"  KADABRA wins BOTH metrics on {both}/{len(rows)} graphs")
    if tau_losses or ovl_losses:
        print(f"  !! CLAIM DOES NOT HOLD --- tau_b lost on: {', '.join(tau_losses) or 'none'}"
              f"; overlap lost on: {', '.join(ovl_losses) or 'none'}")
    elif len(rows) == len(BVK_GRAPHS):
        print("  => 'on every instance tested' holds across all 9 graphs")
    else:
        print(f"  => holds on the {len(rows)} graphs measured so far (INCOMPLETE)")


def k_sweep(d):
    """Effect of the top-k restriction on runtime and sample count."""
    ks = [0, 10, 100]
    rows, missing = [], []
    for g in TIGHT_GRAPHS:
        runs = {k: load(f"{d}/k_julia_{g}_k{k}.json") for k in ks}
        if not all(runs.values()):
            missing.append(f"{g}({','.join(str(k) for k in ks if not runs[k])})")
            continue
        rows.append((g,
                     [runs[k]["execution_time_seconds"] for k in ks],
                     [runs[k]["num_samples"] for k in ks]))
    if missing:
        print(f"!! missing runs: {', '.join(missing)}", file=sys.stderr)
    if not rows:
        return

    print("% ---- k-sweep: runtime (s) and samples at k = 0, 10, 100 ----")
    for g, times, samples in rows:
        rel = " & ".join(f"${t/times[0]:.2f}\\times$" for t in times[1:])
        print(f"            {g:17s} & {sec(times[0])} & {sec(times[1])} & {sec(times[2])} "
              f"& {rel} & ${num(samples[0])}$ \\\\")

    print("\n% ---- figures quoted in the k-sweep section ----")
    for idx, k in ((1, 10), (2, 100)):
        r = [times[idx] / times[0] for _, times, _ in rows]
        srel = [s[idx] / s[0] for _, _, s in rows]
        print(stats(f"runtime k={k} / k=0", r))
        print(stats(f"samples k={k} / k=0", srel))
    unchanged = [g for g, _, s in rows if abs(s[1] - s[0]) / s[0] < 0.01
                 and abs(s[2] - s[0]) / s[0] < 0.01]
    print(f"  sample count essentially unchanged (<1%) on: {', '.join(unchanged) or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("tight", "bvk", "tightx", "k"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    {"tight": tight, "bvk": bvk, "tightx": tightx,
     "k": k_sweep}[sys.argv[1]](sys.argv[2].rstrip("/"))
