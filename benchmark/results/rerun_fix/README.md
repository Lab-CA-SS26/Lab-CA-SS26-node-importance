# `rerun_fix/` --- every KADABRA measurement, re-run with the stopping-coordination fix

Produced by `benchmark/rerun_after_fix.sh` on `coan-wrk-01` (2026-09-16 22:44 onwards), after
the seed experiment (`../seed_check/`, `TODO.md`) showed that the parallel stopping check made
Julia draw 3-6% more samples than the C++ reference. Every Julia run here carries
`parameters.stop_in_batch = true` and `parameters.consistent_pairs = true` (the script checks).

| dir | stage | replaces | feeds |
| --- | --- | --- | --- |
| `tight/` | A tight, E tightx | `report_runs/tight_julia_*` | Section 6.1 table, 6.4 eps=1e-4 column, A.7 |
| `ts/` | B thread scaling | `report_runs/julia_*_t*_s*` | Section 6.1 thread-scaling figure |
| `bvk/`, `bvk/kadabra_seeds/` | C | `brava_retrained/bvk_kadabra_*`, `kadabra_seeds/` | Section 6.4 table, Figure 3 |
| `topk/` | D kx/kxs/kxb, F kxl | `topk_variant/measured/` | Section 6.3, A.5 |

**Not re-measured** (the fix does not touch them), copied in unchanged so the stage scripts skip
them: `tight/tight_cpp_*` and `ts/cpp_*` from `report_runs/`, `bvk/bvk_brava_*` from
`brava_retrained/`. The C++ timings are therefore the August measurements.
