# KADABRA "delta problem" — handoff notes

Context for a fresh session: we found and fixed one confirmed bug in the Julia
KADABRA port's sample-complexity formula, but that fix only closed part of an
observed ~2x sample-count gap between the Julia port and the C++ reference
implementation. What's left is a bigger, structural gap: the Julia port is
missing an entire adaptive calibration phase that the reference algorithm
performs. This file is the handoff for finishing that investigation/fix.

## Where things live

- **Local project repo**: `/Users/martinschlaier/Documents/10_Universitaet/Master/Lab-CA/Lab-CA-SS26-node-importance`
  (git remote `origin` → `git@github.com:Lab-CA-SS26/Lab-CA-SS26-node-importance.git`,
  currently on branch `Martin`)
  - Julia KADABRA port: `src/kadabra.jl`
  - C++ reference implementation (vendored, used for comparison, NOT the file to edit): `cpp_reference/src/Probabilistic.cpp` and `cpp_reference/include/Probabilistic.h`
  - Benchmark runner (calls both): `benchmark/simexpal_runners/run_experiments.jl` (Julia) and `benchmark/simexpal_runners/run_experiments_cpp` (compiled C++ binary, already built)
  - Report (LaTeX): `Report/main.tex` + `Report/sections/*.tex`; the C++-vs-Julia timing discussion and the sample-count discrepancy are in `Report/sections/results.tex` (§6.1) and `Report/sections/appendix.tex` (§A.4, `tables/cpp_vs_julia_tight_table.tex`)

- **Graphs.jl fork** (separate local clone, has an open PR upstream adding KADABRA to the real `Graphs.jl` package): `/Users/martinschlaier/Documents/10_Universitaet/Master/Lab-CA/Graphs.jl`
  (git remote `origin` → `git@github.com:DerSchmachtin/Graphs.jl.git`, `upstream` → `https://github.com/JuliaGraphs/Graphs.jl.git`, on branch `master`)
  - Same algorithm, same bug, ported separately: `src/centrality/kadabra.jl`
  - **Both copies need to stay in sync** — any fix here should also go there.

- **Compute server**: `ssh coan-wrk-01` (already configured, no special flags needed;
  full hostname is `coan-wrk-01.informatik.uni-bonn.de` if the short name doesn't
  resolve). Project lives at `~/Lab-CA-SS26-node-importance` on the server (same
  repo, same branch `Martin`, kept in sync by `scp`-ing changed files up — there is
  no `git pull` workflow set up on the server side, so **push local commits AND
  scp the changed file(s) to the server** when testing there).
  - 48 threads (24 core/48 thread AMD Ryzen Threadripper 3960X), 125GiB RAM, 2x RTX 2080 Ti (not needed for KADABRA, only for BRAVA-GNN).
  - `JULIA_NUM_THREADS` is **unset by default** — Julia runs single-threaded unless
    you explicitly set it (`export JULIA_NUM_THREADS=N`) or pass `julia --threads=N`.
    The Julia script's own `-t`/`--threads` CLI flag is **log-only**, it does NOT
    configure real Julia threading — this bit us once already, don't repeat it.
  - SSH sessions to this server drop after some idle period (seen `exit code 255`
    "Operation timed out" repeatedly this session) — this is a transient network
    thing, not a sign anything on the server died. Just reconnect and check
    `ps aux | grep julia` / tmux sessions; long jobs kept running fine through it.
  - For anything that takes more than ~2 minutes, launch it inside `tmux` on the
    server (`tmux new-session -d -s <name> '<command> 2>&1 | tee ~/<name>.log'`)
    so it survives SSH drops, then poll with `tmux capture-pane -t <name> -p` or
    `tail ~/<name>.log`.

## What's already confirmed and fixed (commit `023a4a0` in the project repo,
## `32386889` in the Graphs.jl fork — both pushed)

`src/kadabra.jl` computes the KADABRA sample-complexity bound as:

```julia
omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
```

The C++ reference (`cpp_reference/src/Probabilistic.cpp:322`) computes it as:

```cpp
this->omega = 0.5 / err / err * (log2(estimate_diameter()-1) + 1 + log(0.5 / delta));
```

The delta term differs: Julia used `log2(1/delta)` (base-2 log of `1/delta`),
the reference uses `log(0.5/delta)` (**natural** log of `0.5/delta`, since C++'s
`log()` from `<math.h>` is natural log — confirmed by checking the include and
that `log2()` is used explicitly elsewhere in the same file for the diameter
term, so this isn't just a naming-convention thing). At delta=0.1 these differ
by a factor of `log2(10)/ln(5) ≈ 2.064`. This looked at first like it would
explain almost the entire observed sample-count gap (predicted 2.064x vs.
observed mean 2.087x across 6 graphs) — **that turned out to be misleading**,
see "What's still wrong" below. The fix itself is correct and worth keeping
regardless: it's what the reference implementation actually computes.

Fix applied (both copies): change `log2(1.0 / delta)` → `log(0.5 / delta)`.

Also carried along in the same commit: `kadabra_centrality` now returns
`omega` and `tau` in its result named-tuple (was only `centralities`,
`lower_bounds`, `upper_bounds`, `n_samples` before), and
`run_experiments.jl` logs `kadabra_omega`, `kadabra_tau`,
`samples_over_omega` in its output JSON. This was added earlier in the
session for a different (already-finished) investigation into top-k timing
behavior, unrelated to the delta problem, but it's what makes it easy to
inspect `omega` per-run now.

### Validation done so far

Ran `p2p-Gnutella31` at ε=0.0001, δ=0.1, k=0, seed=0, t=8 (matching the C++
comparison methodology already used in the report), before and after the fix:

| | samples | ratio vs C++ |
|---|---|---|
| C++ (reference) | 51,081,214 | — |
| Julia, before fix | 104,917,216 | 2.05x |
| Julia, after fix | 88,947,776 | **1.74x** |

Command used (from `~/Lab-CA-SS26-node-importance/benchmark` on the server):

```bash
JULIA_NUM_THREADS=8 julia --project=.. simexpal_runners/run_experiments.jl \
  -i ../Instances/TestInstances/p2p-Gnutella31.txt \
  -o /tmp/kadabra_fixed_test.stats.json \
  -a kadabra -v julia --epsilon 0.0001 --delta 0.1 -k 0 -s 0 -t 8
```

C++ equivalent for comparison:

```bash
simexpal_runners/run_experiments_cpp \
  -i ../Instances/TestInstances/p2p-Gnutella31.txt \
  -o /tmp/cpp_smoke.stats.json \
  -threads 8 -k 0 -delta 0.1 -epsilon 0.0001 -a kadabra -v cpp -s 0
```

(Both write a `.stats.json` with a `num_samples` field; C++'s doesn't have
`kadabra_omega`/`kadabra_tau` since that instrumentation was only added to the
Julia side.)

## What's still wrong: the actual "delta problem"

The fix above only closed part of the gap (2.05x → 1.74x, not → ~1x). The
remaining gap is structural, not a formula typo. Reading
`Probabilistic::run()` in `cpp_reference/src/Probabilistic.cpp` (lines
317–403) closely:

```cpp
void Probabilistic::run(uint32_t k, double delta, double err, uint32_t union_sample, uint32_t start_factor) {
    ...
    this->omega = 0.5 / err / err * (log2(estimate_diameter()-1) + 1 + log(0.5 / delta));
    uint32_t tau = omega / start_factor;
    ...
    // PHASE 1: burn-in. Just sample tau pairs, no stopping check.
    #pragma omp parallel
    {
        while (n_pairs <= tau) { one_round(sp_sampler); ... }
    }

    // Between phases: use what Phase 1 observed to calibrate delta allocation.
    compute_delta_guess();

    // Phase 1's samples/estimates are then THROWN AWAY:
    n_pairs = 0;
    for (uint32_t i = 0; i < get_nn(); i++) { approx[i] = 0; }

    // PHASE 2: a FRESH, independent sampling round, using the calibrated deltas,
    // checking the stopping condition against n_pairs < omega (Phase-2-only count).
    #pragma omp parallel
    {
        Status status(union_sample);
        status.n_pairs = 0;
        while (!stop && status.n_pairs < omega) {
            for (i = 0; i <= 10; i++) one_round(sp_sampler);
            get_status(&status);
            stop = compute_finished(&status);
        }
    }
    ...
    n_pairs += tau;  // tau added back only for REPORTING the total sample count
}
```

`compute_delta_guess()` (lines 260–309) is a bisection search: it uses Phase
1's observed betweenness estimates to find, per node, a tightened
`delta_l_guess[v]`/`delta_u_guess[v]` (via a Chernoff-bound-style formula),
instead of the uniform `delta/(4n)` every node gets by default. This is the
"adaptive" part of the KADABRA paper (Borassi, Natale 2016) that makes the
stopping condition converge with fewer samples than a naively uniform delta
allocation would need.

**`src/kadabra.jl` does neither of these things.** It has no
`compute_delta_guess` equivalent — `delta_l_guess`/`delta_u_guess` are set
once to `fill(delta / (4 * n), n)` and never change (see around line 197 in
the current file, right after the `omega`/`tau` computation). And it never
resets between "phases" — Phase 1's `tau` samples stay in the same running
`global_approx`/`n_pairs` counter that Phase 2 continues, so there's no
separate calibration step and no fresh Phase 2 sampling round at all; it's
really just one continuous sampling loop with a burn-in-sized head start
before the first stopping check.

This is presumably why the fix above didn't fully close the gap: Julia's
stopping condition, using the uniform (non-recalibrated) delta and never
restarting, needs more samples to satisfy the same delta guarantee than the
reference's two-phase, recalibrated approach does.

## What to do

1. **Re-derive/confirm this is really the (main) remaining cause** before
   writing a lot of code. Options, cheapest first:
   - Re-run the same `p2p-Gnutella31` comparison (or a couple more graphs) and
     check whether the *remaining* 1.74x gap is roughly graph-independent
     (like the delta-term bug's 2.06x was) — if it varies a lot by graph,
     the calibration-phase theory might not be the whole story either, and
     it's worth checking `compute_bet_err`/`compute_finished` (lines
     92–170 of `Probabilistic.cpp`) for other discrepancies against Julia's
     `check_finished` before committing to porting the calibration phase.
   - It would also be worth checking whether Julia's `estimate_diameter(g)`
     (top of `src/kadabra.jl`) produces the same diameter estimate as C++'s
     `estimate_diameter()` (`cpp_reference/src/Probabilistic.cpp`, search for
     `estimate_diameter` — not yet read this session) on the same graph. A
     silent diameter-estimate difference would also shift `omega` and hasn't
     been ruled out — the `log2(D-1)` term was *assumed* equal between
     implementations when predicting the 2.064x ratio, never actually checked.
     Easy check: add a temporary `println("diam_est = ", diam_est)` in
     `kadabra.jl` right after it's computed, and compare against C++'s
     equivalent (may need a similar temporary `cout` in `Probabilistic.cpp`,
     then rebuild — see `benchmark/simexpal_runners/` or wherever the cpp
     binary's build script lives, not yet located this session).

2. **If confirmed**, port `compute_delta_guess()` into `src/kadabra.jl`:
   - It's a bisection search over a single scalar `c` (see lines 260–292 of
     `Probabilistic.cpp`), using per-candidate `bet[i]`, `err_l[i]`, `err_u[i]`
     computed by `compute_bet_err` (lines 216–248) from Phase 1's observed
     estimates (`status->approx_top_k[i] / status->n_pairs`).
   - Needs `compute_bet_err`'s logic too — note it branches on `absolute`
     (i.e. `k==0`) vs. top-k mode, with different `err_l`/`err_u` initialization
     for each (lines 224–239 shown above; there's more below line 248 not yet
     read this session — read the rest of that function before porting).
   - Restructure `kadabra_centrality`'s main loop (both the parallel and
     sequential branches — currently at roughly lines 220–355 depending on
     how much this file has changed since; grep for `PHASE 1`/`PHASE 2`
     comments) to: burn-in → calibrate → **reset `global_approx`/`n_pairs` to
     zero** → fresh Phase 2 loop checking `n_pairs < omega` (Phase-2-only
     count) → report `final_n_pairs = phase2_samples + tau` for the total.
   - This touches both the `parallel=true` (multi-threaded, `Threads.@spawn`)
     and `parallel=false` (sequential) code paths in `kadabra_centrality` —
     don't fix only one and forget the other again (see the earlier
     `run_experiments.jl` JIT-warmup bug this session, which was exactly this
     kind of "only handled one branch" mistake).
   - Test the same way as above: `p2p-Gnutella31` at ε=0.0001 first (small,
     fast, ~90s for C++/~2-3 min for Julia even before this fix), compare
     `num_samples` against C++'s 51,081,214. Then re-run the full 6-graph
     comparison (`amazon`, `dblp`, `email-EuAll`, `p2p-Gnutella31`,
     `soc-Epinions1`, `soc-Slashdot0902` — same 6 as the existing report table)
     to get a clean, updated `Report/tables/cpp_vs_julia_tight_table.tex`.
   - **Also re-check accuracy, not just sample count**: confirm
     `tau_overall`/`overlap_topk` against a known-correct ground truth (we have
     one for `soc-Slashdot0902`: `Instances/ground_truth/test_instances/soc-Slashdot0902_bet.csv`,
     regenerated as part of an earlier, separate investigation this session —
     see git history / the report's Appendix A.1 for how) don't regress after
     restructuring the sampling loop. A bug in the reset/restart logic could
     silently break correctness even if it "fixes" the sample count.

3. **Push both copies again** (project repo + Graphs.jl fork,
   `src/kadabra.jl` and `src/centrality/kadabra.jl` respectively) once
   validated, same as this session's fix.

4. **Update the report** (`Report/sections/results.tex` §6.1 and
   `Report/sections/appendix.tex` §A.4 / `Report/tables/cpp_vs_julia_tight_table.tex`)
   with corrected numbers and an honest account of what the "~2x sample gap"
   actually was (formula bug + missing calibration phase, not sampling noise
   as originally guessed). Recompile with `pdflatex -interaction=nonstopmode
   -halt-on-error main.tex` (run twice, then `biber main` + one more pdflatex
   pass if any citations/labels changed) from `Report/`, check
   `grep -iE "overfull|underfull|error" main.log` for new issues, and
   visually check the affected pages via `pdftoppm` before considering it
   done — this session had multiple rounds of overfull-hbox fixes
   (`\begin{sloppypar}...\end{sloppypar}` around paragraphs dense with
   `\texttt{}` graph names) that are easy to reintroduce.

## Open loose end from this session, unrelated to the delta problem but worth
## knowing about if you go looking at `run_experiments.jl`/`kadabra.jl`

There's an unrelated, already-fixed bug in `benchmark/simexpal_runners/run_experiments.jl`'s
JIT-warmup section that always built an *undirected* dummy graph and then
unconditionally tried `StaticDiGraph(...)` on it whenever `--directed` was
passed, crashing before reaching real work. Already fixed this session
(dummy graph is now `is_directed ? SimpleDiGraph(...) : SimpleGraph(...)`) —
mentioning it only so it's not mistaken for part of the delta problem if you
see it in `git log` or git blame.
