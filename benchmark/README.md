# Benchmarks

Everything that produces, archives and summarizes the measurements in the report. To
reproduce the report, start with `reproduce_all.sh` (see the repository's
[README](../README.md#reproducing-the-results)); this page lists what it is built from.

## Runners

| file | what |
| --- | --- |
| `simexpal_runners/run_experiments.jl` | one Julia run (KADABRA, BRAVA-GNN or exact Brandes) → one JSON with runtime, samples and accuracy against the ground truth |
| `simexpal_runners/run_experiments.cpp` | the same for the authors' C++ KADABRA (`make -C simexpal_runners build`); `-centralities 1` also writes the per-vertex scores |
| `scripts/BenchmarkUtils.jl` | graph loading shared by the runners |

Every run writes `execution_time_seconds`, `io_time_seconds`, `num_samples`, `tau_overall`,
`tau_topk`, `overlap_topk`, `max_ae`, `mae`, `ndcg_topk` and `parameters` (graph, epsilon,
delta, k, seed, and the **actual** thread count).

## Running measurements

| script | what |
| --- | --- |
| `reproduce_all.sh` | every measurement, table and figure of the report, in order; `--help` for the steps |
| `reproduce_report.sh --stage …` | one measurement stage (`tight`, `tightx`, `bvk`, `kx`, `kxs`, `kxb`, `kxc`, `kxl`) |
| `rerun_after_fix.sh` | the sequence that produced `results/rerun_fix/` (2026-09-16 to 09-19) |
| `run_seed_check.sh` | the experiment behind the stopping-coordination fix (`results/seed_check/`) |
| `archive_runs.py` | collects runs from the server into a committable directory |

Run on the benchmark server, inside `tmux`, with nothing else on the machine during timed
stages.

## From runs to the report

| script | emits |
| --- | --- |
| `update_report_from_rerun.py` | all table bodies and figures, written into `../Report/` or `--report DIR` |
| `summarize_reproduce.py {tight,tightx,bvk}` | Tables 1, 5 and the quoted ranges; `tightx` checks Section 6.4's claim |
| `summarize_seeds.py` | Table 2 (mean ± std over seeds) |
| `summarize_threads.py` | the thread-scaling numbers of Section 6.1 |
| `summarize_topk.py`, `summarize_topk_claims.py` | Table 6 and the numbers of Section 6.3 |
| `summarize_cpp_quality.py`, `score_cpp_centralities.py`, `check_burnin_bias.py` | Table 8 (Appendix A.7) |
| `summarize_seed_check.py` | the seed experiment |
| `make_plots.py {threads,bvk,topk}` | Figures 1–3 |

## Diagnostics

| script | what |
| --- | --- |
| `eval_brava_paper.jl [checkpoint]` | a BRAVA-GNN checkpoint against the paper's Table 2 |
| `diagnose_brava_mask.jl`, `diagnose_brava_nopr.jl` | the 2×2 diagnostic behind the BRAVA-GNN fix |
| `diagnose_topk_delta.jl` | per-vertex view of why the top-$k$ allocations differ |
| `diagnose_topk_oracle.jl` | how much an exact ranking could save top-$k$ mode (`results/topk_oracle/`) |

## Data

`results/` holds every run the report uses; see [`results/README.md`](results/README.md).
`cache/` holds exact-betweenness caches and, once installed, the BRAVA-GNN checkpoints
(both gitignored).

`experiments.yml` is the original `simexpal` configuration. The report's runs were made with
the shell scripts above, which call the runners directly.

Scripts and results from before August 2026 were removed because they predate the sampling
fixes in `src/kadabra.jl` and the retrained BRAVA-GNN. Recover any of them from git:

```bash
git log --diff-filter=D --oneline -- benchmark
git checkout <commit>^ -- <path>
```
