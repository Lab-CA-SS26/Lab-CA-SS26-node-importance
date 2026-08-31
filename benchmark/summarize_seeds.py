#!/usr/bin/env python3
"""Build the Section 6.4 table body with mean +/- std over three seeds.

Both algorithms in that table are stochastic, but not in the same way, and the table
reports a single draw for each:

  * KADABRA samples shortest paths, so repeated runs of one binary differ. The `bvk`
    stage passes `-s 0` (no seed), so its numbers are one unseeded draw.
  * BRAVA-GNN inference is deterministic given a checkpoint; its spread comes from the
    *training* seed, i.e. from three separately trained models.

Quoting +/- for one and not the other would imply the other is deterministic, so this
computes both. The two spreads mean different things and the caption must say so.

Inputs
------
kad_dir     directory of `bvk_kadabra_<graph>_s<seed>.json` (see benchmark/kad_seeds.sh)
brava_log   output of `eval_brava_paper.jl` run once per checkpoint, concatenated --
            the per-graph line is `<graph> <paper> <tau*100> <delta> <reported> <ovl>/100`
            and the active seed is announced by a `seed : <n>` line
bvk_dir     directory of the single-seed `bvk_*` runs, for the wall-clock columns

Usage
-----
    python3 summarize_seeds.py <kad_dir> <brava_log> <bvk_dir>
"""
import glob
import json
import os
import re
import statistics as st
import sys

GRAPHS = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",
          "com-youtube", "amazon", "dblp", "cit-Patents", "com-lj"]

ROW = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+\s+[\d.]+\s+(\d+)/100\s*$")
SEED = re.compile(r"^seed\s*:\s*(\d+)\s*$")


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def kadabra_seeds(d):
    """{graph: {"tau": [...], "ovl": [...], "time": [...]}} from seeded KADABRA runs."""
    out = {}
    for path in sorted(glob.glob(f"{d}/bvk_kadabra_*_s*.json")):
        m = re.match(r"bvk_kadabra_(.+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        r = load(path)
        if not r or r.get("tau_overall") is None:
            continue
        e = out.setdefault(m.group(1), {"tau": [], "ovl": [], "time": []})
        e["tau"].append(r["tau_overall"])
        e["ovl"].append(r["overlap_topk"])
        e["time"].append(r["execution_time_seconds"])
    return out


def brava_seeds(path):
    """{graph: {"tau": [...], "ovl": [...]}} parsed from the concatenated eval logs."""
    out, seen = {}, set()
    seed = None
    with open(path) as fh:
        for line in fh:
            s = SEED.match(line.strip())
            if s:
                seed = int(s.group(1))
                continue
            m = ROW.match(line.rstrip())
            if not m or seed is None:
                continue
            g = m.group(1)
            if (g, seed) in seen:          # a checkpoint scored twice; keep the first
                continue
            seen.add((g, seed))
            e = out.setdefault(g, {"tau": [], "ovl": []})
            e["tau"].append(float(m.group(3)) / 100.0)
            e["ovl"].append(int(m.group(4)))
    return out


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    kad_dir, brava_log, bvk_dir = sys.argv[1:4]
    kad, bra = kadabra_seeds(kad_dir), brava_seeds(brava_log)

    missing = [g for g in GRAPHS if g not in kad or g not in bra]
    if missing:
        print(f"!! no seed data for: {', '.join(missing)}", file=sys.stderr)

    nk = {len(kad[g]["tau"]) for g in kad}
    nb = {len(bra[g]["tau"]) for g in bra}
    print(f"% KADABRA: {sorted(nk)} sampling runs/graph; "
          f"BRAVA-GNN: {sorted(nb)} training seeds/graph")
    print("% ---- Report/tables/brava_vs_kadabra_table.tex (body) ----")

    rows = []
    for g in GRAPHS:
        if g not in kad or g not in bra:
            continue
        cpu = load(f"{bvk_dir}/bvk_brava_cpu_{g}.json")
        gpu = load(f"{bvk_dir}/bvk_brava_gpu_{g}.json")
        if not (cpu and gpu):
            print(f"!! missing bvk timing for {g}", file=sys.stderr)
            continue
        bt, bts = ms(bra[g]["tau"]); bo, bos = ms(bra[g]["ovl"])
        kt, kts = ms(kad[g]["tau"]); ko, kos = ms(kad[g]["ovl"])
        # All three wall-clock columns come from the single `bvk` pass, which had the
        # machine to itself. The repeat runs exist to bound accuracy, and were not
        # guaranteed exclusive use of the box, so their timings are not quoted.
        kad_run = load(f"{bvk_dir}/bvk_kadabra_{g}.json")
        if not kad_run:
            print(f"!! missing bvk KADABRA timing for {g}", file=sys.stderr)
            continue
        rows.append((g, bt, bts, bo, bos, kt, kts, ko, kos))
        print(f"            {g:<17} & {cpu['execution_time_seconds']:.2f}\\,s "
              f"& {gpu['execution_time_seconds']:.2f}\\,s "
              f"& {kad_run['execution_time_seconds']:.2f}\\,s "
              f"& ${bt:.3f} \\pm {bts:.3f}$ & ${kt:.3f} \\pm {kts:.3f}$ "
              f"& ${bo:.0f} \\pm {bos:.0f}$ & ${ko:.0f} \\pm {kos:.0f}$ \\\\")

    if not rows:
        return 1
    print("\n% ---- figures quoted in Section 6.4 ----")
    print(f"  BRAVA   tau_b:   {min(r[1] for r in rows):.3f}--{max(r[1] for r in rows):.3f}"
          f"   (seed std {min(r[2] for r in rows):.3f}--{max(r[2] for r in rows):.3f})")
    print(f"  KADABRA tau_b:   {min(r[5] for r in rows):.3f}--{max(r[5] for r in rows):.3f}"
          f"   (seed std {min(r[6] for r in rows):.3f}--{max(r[6] for r in rows):.3f})")
    print(f"  BRAVA   overlap: {min(r[3] for r in rows):.0f}--{max(r[3] for r in rows):.0f}"
          f"   (seed std {min(r[4] for r in rows):.1f}--{max(r[4] for r in rows):.1f})")
    print(f"  KADABRA overlap: {min(r[7] for r in rows):.0f}--{max(r[7] for r in rows):.0f}"
          f"   (seed std {min(r[8] for r in rows):.1f}--{max(r[8] for r in rows):.1f})")

    bwin_t = sum(1 for r in rows if r[1] > r[5])
    kwin_o = sum(1 for r in rows if r[7] > r[3])
    print(f"\n  BRAVA higher tau_b on {bwin_t}/{len(rows)} graphs")
    print(f"  KADABRA higher overlap on {kwin_o}/{len(rows)} graphs")

    # A win inside one standard deviation is not a result worth asserting.
    close = [(r[0], r[3], r[4], r[7], r[8]) for r in rows
             if abs(r[3] - r[7]) < max(r[4], r[8])]
    if close:
        print("\n  !! overlap gap is within one seed std on:")
        for g, bo, bos, ko, kos in close:
            print(f"       {g}: BRAVA {bo:.0f} +/- {bos:.1f} vs KADABRA {ko:.0f} +/- {kos:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
