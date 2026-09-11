# Accuracy of the C++ reference in Section 6.1 (scored 2026-09-11)

Section 6.1 compares the C++ reference and the Julia port on time and sample count only;
no accuracy figure existed for the reference. Its runs predate `run_experiments.cpp`
scoring itself, and `../report_runs/tight_cpp_*.json` had the per-node blob stripped.
The unstripped originals survive on the server:

| set | server path | notes |
| --- | --- | --- |
| `set_report` | `/tmp/cpp_*.stats.json`, copied to `~/cpp_quality/raw_report_set/` | **the report's runs** -- `num_samples` matches Table 1 exactly |
| `set_aug10` | `~/Lab-CA-SS26-node-importance/benchmark/output/cpp_vs_julia_tight/` | an independent second draw, one day earlier |

`/tmp` is volatile; the copy under `~` is the durable one. Neither is in git (150 MB).

## Files

| file | produced by |
| --- | --- |
| `set_report.jsonl`, `set_aug10.jsonl` | `score_cpp_centralities.py <gt_dir> <runs>` |
| `debias.jsonl` | `check_burnin_bias.py <gt_dir> julia_tau.json 1e-4 <set_report runs>` |
| `julia_tau.json` | `kadabra_tau` from `../report_runs/tight_julia_*.json` |

Julia is compared over four draws per graph: `../report_runs/tight_julia_<g>.json` and
`../topk_variant/measured/kx_<g>_k0_code_s{1,2,3}.json` (k = 0 never reaches the top-k
allocation, so the variant is irrelevant there).

## Result

**Ranking quality is equivalent.** Top-100 overlap 99-100 for both on all six graphs;
tau_top100 within +-0.004 either way. Julia's Kendall tau_b is higher on all six by
+0.0001 to +0.0049 -- beyond both implementations' seed spread on five, third decimal
only. Not isolated; consistent with Julia's 3-6% larger sample count.

**Absolute error is not, and for the same reason in both.** Max absolute error is
3.5-12.3x eps (40-89 vertices over eps per graph, always worst at rank 1) in both
implementations. The reference drops the burn-in counts but divides by `N_main + tau`;
our port and the Graphs.jl fork copy this. The measured scale against the ground truth
equals `1 - tau/N` to 3-4 decimals on all six graphs. Rescaled by `N/(N - tau)`, max
error falls to **0.15-0.31x eps with zero vertices over eps on every graph.** Julia's
3-6% lower absolute error is the same bias diluted by its larger N (p2p: 4.86e-4 x
0.0602/0.0637 = 4.59e-4, measured 4.60e-4).

Every ranking metric the report quotes (tau_b, overlap, the top-k answer) is
scale-invariant and unaffected. The stopping rule uses phase-2 counts over phase-2 pairs
and is also unaffected; only the returned scores and bounds are scaled.
