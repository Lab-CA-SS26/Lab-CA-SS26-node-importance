#!/usr/bin/env python3
"""Collect the report's benchmark runs from the compute server's scratch dirs
into one durable, committable directory.

Run this ON the server, then scp the output directory into
`benchmark/results/report_runs/`:

    ssh coan-wrk-01 'python3 ~/Lab-CA-SS26-node-importance/benchmark/archive_runs.py'
    scp 'coan-wrk-01:~/report_runs/*.json' benchmark/results/report_runs/

The runs are produced in assorted scratch locations (`/tmp`, `~/bvk_results`,
`~/ts_results`, ...) under inconsistent names. This normalises them to the
filenames `reproduce_report.sh` writes --- and therefore that
`summarize_reproduce.py` and `make_plots.py` read --- so the archive can be
passed straight to either as a result directory.

The C++ runner embeds a per-node `centralities` object which is ~99.7% of the
raw bytes and backs no reported number (accuracy metrics are stored as scalar
fields beside it; the C++ runs are read only for time and sample count). It is
dropped. Every other field is passed through untouched.

See `benchmark/results/report_runs/README.md` for the resulting layout.
"""
import glob
import json
import os
import shutil
import sys

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "report_runs")

TIGHT6 = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902",
          "email-EuAll", "amazon", "dblp"]
# Tight runs that exist only to complete Section 6.4; no C++ counterpart.
EXTRA3 = ["com-youtube", "cit-Patents", "com-lj"]
BVK = ["p2p-Gnutella31", "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",
       "com-youtube", "amazon", "dblp", "cit-Patents", "com-lj"]

missing, stripped = [], 0


def put(src, dst):
    """Copy one run into the archive, dropping the per-node centralities blob."""
    global stripped
    if not os.path.exists(src):
        missing.append(src)
        return
    with open(src) as fh:
        d = json.load(fh)
    if d.pop("centralities", None) is not None:
        stripped += 1
    with open(os.path.join(DST, dst), "w") as fh:
        json.dump(d, fh, indent=4, sort_keys=True)


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    # Section 6.1 (C++ vs Julia) and the tight column of Section 6.4.
    for g in TIGHT6:
        put(f"/tmp/cpp_{g}.stats.json", f"tight_cpp_{g}.json")
        put(f"{HOME}/tight_all/julia_{g}.stats.json", f"tight_julia_{g}.json")
    for g in EXTRA3:
        put(f"{HOME}/tight_extra/julia_{g}.stats.json", f"tight_julia_{g}.json")

    # Section 6.4 at eps=1e-2.
    for g in BVK:
        for src, dst in (("kadabra", "bvk_kadabra"),
                         ("brava_cpu", "bvk_brava_cpu"),
                         ("brava_gpu", "bvk_brava_gpu")):
            put(f"{HOME}/bvk_results/{src}_{g}.stats.json", f"{dst}_{g}.json")

    # Section 6.3 top-k sweep. k=0 is the tight run under identical parameters,
    # which is how reproduce_report.sh treats it too.
    for g in TIGHT6:
        for k in (10, 100):
            put(f"{HOME}/k_results/julia_{g}_k{k}.json", f"k_julia_{g}_k{k}.json")
        k0 = f"{DST}/tight_julia_{g}.json"
        if os.path.exists(k0):
            shutil.copy(k0, f"{DST}/k_julia_{g}_k0.json")

    # Section 6.2 thread scaling; already uniquely named by impl/graph/threads/seed.
    for src in sorted(glob.glob(f"{HOME}/ts_results/*.json")):
        put(src, os.path.basename(src))

    n = len(os.listdir(DST))
    print(f"wrote {n} files to {DST} (centralities stripped from {stripped})")
    if missing:
        print(f"\n!! {len(missing)} expected run(s) not found:", file=sys.stderr)
        for m in missing:
            print(f"   {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
