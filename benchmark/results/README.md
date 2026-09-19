# Benchmark Results

One JSON per run (per-vertex centralities stripped; the metrics are kept). Directories with
their own README describe their contents in detail.

| directory | runs | what | used by the report |
| --- | --- | --- | --- |
| `rerun_fix/` | 531 | every KADABRA measurement after the stopping-coordination fix, plus the reused C++ and BRAVA-GNN runs | **yes — everything in Sections 6.1, 6.3, 6.4 and Appendix A.4, A.5, A.7** |
| `brava_retrained/` | 27 | the retrained BRAVA-GNN: runs, the 3 checkpoints, evaluation logs (`logs/eval_seeds.log` feeds Table 2) | yes |
| `cpp_quality/` | | the C++ reference's per-vertex output, scored against the ground truth | yes, Appendix A.7 |
| `topk_variant/measured/` | 333 | top-$k$ runs from before the stopping fix; its `kxc_*` runs (the C++ binary in top-$k$ mode) are still used | yes, Section 6.3 |
| `topk_variant/predicted/` | | `diagnose_topk_delta.jl` output | background |
| `seed_check/` | 120 | the experiment that located the stopping-coordination overshoot | background |
| `bias_fix/` | 6 | first runs after the burn-in normalisation fix | superseded by `rerun_fix/` |
| `topk_oracle/` | | `diagnose_topk_oracle.jl` output | background |
| `report_runs/` | 252 | the original runs, before the fixes | only as history (Section 6.3's single-seed `dblp` value) |
| `exact_brandes_timings.csv` | | exact-betweenness recomputation times | Appendix A.2 |
