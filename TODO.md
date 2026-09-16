# TODO

## Seed experiment: why Julia draws more samples than the C++ reference

**Question.** At ε = 0.0001, k = 0, 8 threads, Julia draws 3–6% more samples than the C++
reference (Section 6.1: mean 1.04×). Is that a systematic implementation effect, or run-to-run
randomness? And how much do sample counts vary between seeds, per implementation?

### What the existing runs already show

Julia: `report_runs/tight_julia_<g>.json` + `topk_variant/measured/kx_<g>_k0_code_s{1,2,3}.json`
(4 runs). C++: `cpp_quality/set_report.jsonl`, `set_aug10.jsonl` (2 runs).

| graph | C++ runs | Julia distinct counts | Julia steps | Julia − C++ |
| --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 51 145 729 / 51 081 214 | 54 085 460, 54 411 276 | 325 816 | 9.0 intervals, 5.7% |
| soc-Epinions1 | 82 899 916 / 82 908 650 | 86 286 728 (all 4) | — | 8.7 intervals, 4.1% |
| soc-Slashdot0902 | 118 156 834 / 118 165 293 | 121 718 143 … 122 403 879 | 342 868 | 10.4 intervals, 3.0% |
| email-EuAll | 54 335 686 / 54 296 603 | 56 607 657, 57 285 593 | 677 936 | 6.7 intervals, 4.2% |
| amazon | 109 428 589 / 109 475 944 | 112 377 935 … 113 078 109 | 350 087 | 8.3 intervals, 2.7% |
| dblp | 44 507 467 / 44 447 902 | 47 315 787 (all 4) | — | 8.2 intervals, 6.3% |

"Interval" = Julia's `check_interval = max(1000, tau ÷ 10)` with tau = ω/100, i.e. ω/1000.

- Julia's counts across seeds differ **only in exact multiples of `check_interval`**.
- Julia's excess over C++ is **6.7–10.4 intervals** on every graph — roughly one per thread
  (8 threads).
- The two C++ runs per graph differ by < 0.13%.

### Hypothesis (from the code, not yet tested)

- **C++** (`cpp_reference/src/Probabilistic.cpp`, main loop): every thread draws **11 samples**,
  then calls `compute_finished`. Stopping granularity ≈ 11 × threads samples.
- **Julia** (`src/kadabra.jl`, parallel Phase 2): every thread draws **`check_interval`**
  (≈ ω/1000, here ~340 000) samples before checking; when one thread sets the stop flag, the
  others still finish their current batch. Expected overshoot ≈ up to one interval per thread.
- So the ~4% excess would be systematic overshoot from the coarse check, and seed randomness
  would only move Julia's stop by one or two intervals.
- **Consequence:** Section 6.1's "both test the stopping condition only every few thousand
  samples and overshoot by up to one batch" is wrong for C++ (11 samples) and understates Julia
  (one batch *per thread*). Fix the sentence once the experiment confirms it.

### Experiment

- **Graphs:** p2p-Gnutella31, soc-Epinions1, soc-Slashdot0902, email-EuAll (1–5 min per run
  at ε = 0.0001); amazon and dblp only if time allows (2–3 h per run).
- **Settings:** ε = 0.0001, δ = 0.1, k = 0, 8 threads (`export JULIA_NUM_THREADS=8`; check
  `parameters.threads` in the JSON), C++ with `-d` for the directed graphs (check
  `parameters.directed`).
- **Seeds:** at least 5 per implementation and graph, same seeds for both.
- **Arms:**
  1. Julia as is.
  2. C++ as is.
  3. Julia with `check_interval = 11` (or a small value) — if the excess disappears and the
     counts match C++, the hypothesis holds.
  4. Julia with `parallel = false` (sequential, bit-reproducible) — separates the threading
     effect from the check granularity.
- **Record per run:** `num_samples`, Phase 2 pair count, number of stopping checks, runtime,
  Kendall τ_b and top-100 overlap (to confirm accuracy does not change).
- **Report:** per graph, mean ± std of samples per arm; Julia/C++ ratio per seed; histogram of
  Julia counts in units of `check_interval`.
- **Where:** on the server in tmux, like every other measurement.
- **Also check:** whether arm 3 changes Julia's runtime (a check costs a lock and a partial
  sort of the tracking set — more checks may make Julia slower).
