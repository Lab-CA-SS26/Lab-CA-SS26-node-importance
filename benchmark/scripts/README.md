# Benchmark Scripts

- `BenchmarkUtils.jl`: reusable graph-loading and instance-parsing helpers. Imported by
  `../simexpal_runners/run_experiments.jl`; not meant to be run directly.

To reproduce the measurements in the report, use `../reproduce_report.sh` (see
`../README.md`). It drives `../simexpal_runners/run_experiments.jl` directly and
post-processes the results with `../summarize_reproduce.py`.

## Removed scripts

An earlier generation of ad-hoc benchmark scripts lived here
(`run_benchmarks.jl`, `generate_benchmarks.jl`, `benchmark_error_bounds.jl`,
`compare_topk.jl`, `run_thread_scaling.jl`, `worker_thread_scaling.jl`, and the
`kadabra/` evaluation suite). They produced the thread-scaling, top-$k$ and
error-bound analyses that are no longer part of the report, and they predate the
sampling fix in `src/kadabra.jl`, so their numbers are not comparable with
current runs. They were removed rather than left to rot; recover any of them with:

```
git log --diff-filter=D --oneline -- benchmark/scripts
git checkout <commit>^ -- benchmark/scripts/<file>
```
