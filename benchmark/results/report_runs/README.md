# `report_runs/` --- the raw measurements behind every number in the report

This directory is the durable copy of the benchmark runs that the lab report
quotes. Every figure, table and in-text number in `Report/` can be regenerated
from these files alone, with no access to the compute server.

It exists because the original runs lived in `coan-wrk-01:/tmp` and in
scratch directories under `~`, both of which are volatile --- the six
`\epsilon=10^{-4}` runs of Section 6.1 were one `/tmp` cleanup away from being
lost, and reproducing them costs several hours of compute each.

## Provenance

All runs were measured on `coan-wrk-01.informatik.uni-bonn.de`
(48 threads, 125 GiB RAM, 2x RTX 2080 Ti), on 8 threads unless the filename says
otherwise, with `\delta = 0.1`, `k = 0` and seed `0` unless the filename says
otherwise. Collected 2026-08-11 through 2026-08-15.

Gathered from the server's `/tmp/{cpp,julia}_*.stats.json`, `~/tight_all/`,
`~/tight_extra/`, `~/bvk_results/`, `~/k_results/` and `~/ts_results/` by
`benchmark/archive_runs.py`.

## What was removed

The C++ runner writes a `centralities` object holding one score per node; the
Julia runner does not. That blob accounts for essentially the entire raw size
(362 MB -> 1.0 MB here) and **no reported number is computed from it** --- the
accuracy metrics (`tau_overall`, `overlap_topk`, `ndcg_topk`, `mae`, `max_ae`)
are precomputed scalar fields stored alongside it, and the C++ runs are used only
for their `execution_time_seconds` and `num_samples`. It was dropped from the 102
C++ files; every other field of every run is byte-for-byte as the runner emitted
it.

If you need per-node scores, re-run the measurement --- they are not recoverable
from this directory.

## Layout

One flat directory, using exactly the filenames `reproduce_report.sh` writes and
`summarize_reproduce.py` / `make_plots.py` read, so it can be passed to any of
them directly as a result directory.

| pattern | count | report section |
|---|---|---|
| `tight_cpp_<graph>.json` | 6 | 6.1 --- C++ at `\epsilon=1e-4` |
| `tight_julia_<graph>.json` | 9 | 6.1 (six) + 6.4's tight column (all nine) |
| `bvk_{kadabra,brava_cpu,brava_gpu}_<graph>.json` | 27 | 6.4 --- at `\epsilon=1e-2` |
| `k_julia_<graph>_k{0,10,100}.json` | 18 | 6.3 --- top-`k` sweep |
| `{cpp,julia}_<graph>_t<threads>_s<seed>.json` | 192 | 6.2 --- thread scaling |

`k_julia_<graph>_k0.json` is a copy of the corresponding `tight_julia_` run: k=0
under identical parameters, exactly as `reproduce_report.sh` treats it.

Note the asymmetry in `tight_julia_`: nine graphs are present, but only six have
a `tight_cpp_` counterpart. `com-youtube`, `cit-Patents` and `com-lj` were run at
the tight bound solely to complete Section 6.4's accuracy comparison against
BRAVA-GNN, not to compare implementations.

## Regenerating the report's numbers

```bash
cd benchmark
python3 summarize_reproduce.py tight  results/report_runs   # Section 6.1, Appendix A.2
python3 summarize_reproduce.py bvk    results/report_runs   # Section 6.4, eps=1e-2
python3 summarize_reproduce.py tightx results/report_runs   # Section 6.4, eps=1e-4 + Conclusion
python3 summarize_reproduce.py k      results/report_runs   # Section 6.3
```

and the three figures (verified to reproduce the PDFs in `Report/figures/`
byte-for-byte in size):

```bash
python3 make_plots.py threads results/report_runs ../Report/figures
python3 make_plots.py bvk     results/report_runs ../Report/figures
python3 make_plots.py k       results/report_runs ../Report/figures
```

The `tightx` stage is the one that checks the report's headline claim --- that at
`\epsilon = 10^{-4}` KADABRA beats BRAVA-GNN on *both* metrics on every instance.
It prints a `!! CLAIM DOES NOT HOLD` line naming the offending graphs if that ever
stops being true, so the wording is never validated by hand. As of 2026-08-15 it
holds 9/9.

## Re-measuring from scratch

`./reproduce_report.sh --outdir <dir>` regenerates everything except thread
scaling (`~/run_thread_scaling.sh` on the server). Pointed at *this* directory it
will skip every stage, since it treats an existing output file as done --- copy
to a fresh directory, or delete the specific runs you want recomputed.
