#!/usr/bin/env python3
"""
summarize_topk_claims.py --- the Section 6.3 figures that summarize_topk.py does not print.

  1. C++ binary vs our port configured like it (`:cpp`), per (graph, k >= 3, seed): the
     "draws X +- Y times the samples of our port configured to match it" sentence.
  2. The C++ binary against its own k = 0 runs, per seed: its dearest configurations and how
     many of the 16 (4 graphs x k in {3,5,10,100}) lie above k = 0 by more than one std.
  3. The same count for the reference allocation inside our port (`:code`).
  4. Mean sample ratio to k = 0 over the 16 small-graph configurations, `:paper_bd` vs `:code`.
  5. Whether `:paper_bd` returns the same top-k answer as `:code`: mean top-100/top-k overlap and
     top-k Kendall tau_b per configuration, compared at three decimals.
  6. `:paper_bd` over all its configurations (small graphs + amazon/dblp): the largest mean
     ratio and whether any lies above 1 by more than one std.
  7. How many runs Figure `fig:topk_allocation` is drawn from (the arms make_plots.py topk plots).

"Beyond one std" uses the sample std over the three seeds, as summarize_topk.py and the figure's
whiskers do. Item 1 is our port divided by the C++ binary, the same direction as Section 6.1's
Julia/C++ ratio.

Usage:  python3 summarize_topk_claims.py <kx_dir> <kxc_dir>
"""
import glob
import json
import os
import re
import statistics as st
import sys

KX = re.compile(r"kx_(?P<g>.+)_k(?P<k>\d+)_(?P<v>code|paper_bd|paper_ex|paper|cpp)_s(?P<s>\d+)\.json$")
KXC = re.compile(r"kxc_(?P<g>.+)_k(?P<k>\d+)_s(?P<s>\d+)\.json$")
SMALL = ["email-EuAll", "p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902"]
KS = [3, 5, 10, 100]
SEEDS = [1, 2, 3]


def load(d, pat):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        j = json.load(open(f))
        key = (m["g"], int(m["k"]), m.groupdict().get("v"), int(m["s"]))
        out[key] = dict(n=j["num_samples"], ovl=j.get("overlap_topk"), tt=j.get("tau_topk"))
    return out


def ratios(runs, g, k, v, base_v):
    return [runs[(g, k, v, s)]["n"] / runs[(g, 0, base_v, s)]["n"]
            for s in SEEDS if (g, k, v, s) in runs and (g, 0, base_v, s) in runs]


def dearer(r):
    return len(r) > 1 and st.mean(r) - st.stdev(r) > 1


def main():
    kx_dir, kxc_dir = sys.argv[1], sys.argv[2]
    kx, kxc = load(kx_dir, KX), load(kxc_dir, KXC)

    # 1. C++ binary / port(:cpp)
    r = [kx[(g, k, "cpp", s)]["n"] / kxc[(g, k, None, s)]["n"]
         for g in SMALL for k in KS for s in SEEDS
         if (g, k, None, s) in kxc and (g, k, "cpp", s) in kx]
    print(f"1. port(:cpp) / C++ binary, k>=3: {st.mean(r):.3f} +- {st.stdev(r):.3f}, n={len(r)}, "
          f"range {min(r):.3f}--{max(r):.3f}")
    r0 = [kx[(g, 0, "code", s)]["n"] / kxc[(g, 0, None, s)]["n"]
          for g in SMALL for s in SEEDS if (g, 0, None, s) in kxc and (g, 0, "code", s) in kx]
    print(f"   same at k=0 (port / C++ binary): {st.mean(r0):.3f} +- {st.stdev(r0):.3f}, n={len(r0)}")

    # 2. C++ binary against its own k=0
    cpp_dear = []
    for g in SMALL:
        for k in KS:
            rr = [kxc[(g, k, None, s)]["n"] / kxc[(g, 0, None, s)]["n"] for s in SEEDS
                  if (g, k, None, s) in kxc and (g, 0, None, s) in kxc]
            if dearer(rr):
                cpp_dear.append((g, k, st.mean(rr)))
    print(f"2. C++ binary above its own k=0 beyond 1 std: {len(cpp_dear)} of {len(SMALL) * len(KS)}: "
          + ", ".join(f"{g} k={k} {m:.2f}x" for g, k, m in cpp_dear))

    # 3. port, reference allocation (:code)
    code_dear = [(g, k, st.mean(ratios(kx, g, k, "code", "code"))) for g in SMALL for k in KS
                 if dearer(ratios(kx, g, k, "code", "code"))]
    print(f"3. port :code above k=0 beyond 1 std: {len(code_dear)} of 16: "
          + ", ".join(f"{g} k={k} {m:.2f}x" for g, k, m in code_dear))

    # 4. mean ratio over the 16 small-graph configurations
    for v in ("paper_bd", "code", "cpp"):
        ms = [st.mean(ratios(kx, g, k, v, "code")) for g in SMALL for k in KS]
        print(f"4. mean ratio to k=0 over 16 configs, {v:8}: {st.mean(ms):.3f}")

    # 5. same top-k answer?
    same, diffs = 0, []
    for g in SMALL:
        for k in KS:
            a = [kx[(g, k, "paper_bd", s)] for s in SEEDS if (g, k, "paper_bd", s) in kx]
            b = [kx[(g, k, "code", s)] for s in SEEDS if (g, k, "code", s) in kx]
            oa, ob = st.mean(x["ovl"] for x in a), st.mean(x["ovl"] for x in b)
            ta, tb = round(st.mean(x["tt"] for x in a), 3), round(st.mean(x["tt"] for x in b), 3)
            maxd = max(globals().get("maxd", 0), abs(ta - tb))
            globals()["maxd"] = maxd
            if round(oa, 3) == round(ob, 3) and ta == tb:
                same += 1
            else:
                diffs.append(f"{g} k={k}: overlap {oa:.2f} vs {ob:.2f}, tau_topk {ta:.3f} vs {tb:.3f}")
    print(f"5. :paper_bd gives the same top-k answer as :code in {same} of 16 (largest top-k tau_b "
          f"difference {globals().get('maxd', 0):.3f}); differ: " + "; ".join(diffs))

    # 6. :paper_bd everywhere
    cfg = sorted({(g, k) for (g, k, v, s) in kx if v == "paper_bd"})
    worst = max(((g, k, st.mean(ratios(kx, g, k, "paper_bd", "code"))) for g, k in cfg), key=lambda x: x[2])
    beyond = [(g, k) for g, k in cfg if dearer(ratios(kx, g, k, "paper_bd", "code"))]
    above = [(g, k) for g, k in cfg if st.mean(ratios(kx, g, k, "paper_bd", "code")) > 1]
    print(f"6. :paper_bd configurations: {len(cfg)}; mean above 1 in {len(above)} {above}; beyond 1 std in "
          f"{len(beyond)}; largest mean {worst[2]:.3f} ({worst[0]} k={worst[1]})")
    dblp = {k: ratios(kx, "dblp", k, "paper_bd", "code") for k in (10, 100)}
    print("   dblp :paper_bd " + ", ".join(f"k={k} {st.mean(r):.2f} +- {st.stdev(r):.2f}" for k, r in dblp.items() if r))

    # 7. runs behind the figure: arms cpp and paper_bd at k in KS, plus each graph's k=0 :code runs
    fig = [key for key in kx if key[2] in ("cpp", "paper_bd") and key[1] in KS]
    graphs = {key[0] for key in fig}
    base = [key for key in kx if key[1] == 0 and key[2] == "code" and key[0] in graphs]
    print(f"7. Figure fig:topk_allocation is drawn from {len(fig) + len(base)} runs "
          f"({len(fig)} top-k + {len(base)} k=0)")


if __name__ == "__main__":
    main()
