# Handoff — completing the tight-ε column in the BRAVA-vs-KADABRA comparison

Written 2026-08-14 ~08:15 UTC. The only thing in flight is a set of three long
KADABRA runs on the compute server; everything else in the report is finished and
committed.

> ## ✅ CLOSED 2026-08-15. All three runs finished; the report is updated.
>
> All three completed cleanly (zero `FAILED`, `TIGHT EXTRA DONE` at 02:56 UTC on
> 08-15). They ran ~2× longer than the estimates below — `com-youtube` 8 042 s
> (2 h 14 m), `com-lj` 38 735 s (10 h 46 m), `cit-Patents` 20 904 s (5 h 48 m).
>
> **The §6.4 headline claim HOLDS on all nine instances — no softening needed.**
> KADABRA at ε=1e-4 beats BRAVA-GNN on both metrics on every graph, and the two
> feared cases were not close: `com-lj` τ_b 0.918 vs 0.749 (overlap 100 vs 53),
> `cit-Patents` 0.945 vs 0.738 (99 vs 34).
>
> Final ranges now in the report:
>
> | figure | was (6 graphs) | now (9 graphs) |
> |---|---|---|
> | KADABRA τ_b | 0.898–0.995 | 0.898–0.995 (unchanged) |
> | KADABRA overlap | 99–100 | 99–100 (unchanged) |
> | BRAVA τ_b | 0.459–0.826 | 0.459–0.826 (unchanged) |
> | BRAVA overlap | 46–85 | **34–85** |
> | runtime ratio | 40–550× | **40–1 412×** |
> | anchor example | `amazon`, 7 306 s vs 13.3 s | **`com-youtube`, 8 042 s vs 5.70 s** |
>
> The ratio ceiling jumped because BRAVA is unusually *fast* on `com-youtube`
> (5.70 s), not because KADABRA got slower.
>
> **Also fixed, and worth knowing about:** Appendix A.2 generalised
> `samples_over_omega` ∈ [0.14, 0.36] into a claim that at ε=1e-4 the stopping
> condition uses "between a seventh and a third" of the budget. `com-youtube`
> consumes **0.73** of its ceiling, so that generalisation was contradicted by the
> report's own new data. A.2 now gives both the six-graph range and the nine-graph
> range (0.13–0.73) and states that the margin varies widely.
>
> Build verified: no LaTeX errors, no undefined refs, largest overfull box
> unchanged at 3.75 pt, underfull count unchanged at 11, main content still 16 pages.
>
> **Page limit: decided.** The user chose to stay at **16 pages** and submit one
> over Lab.pdf's 15-page limit. Do not re-raise it or cut anything. (For the
> record, moving the thread-scaling figure to the appendix was measured to land
> on exactly 15 pages with the overfull box unchanged.)
>
> **Raw data is now durable.** Every run backing the report lives in
> `benchmark/results/report_runs/` (252 files, 1.0 MB, tracked by git), gathered
> by `benchmark/archive_runs.py`. This replaces the old situation where the six
> §6.1 tight runs sat in the server's `/tmp`, one cleanup from being lost. The C++
> runs' per-node `centralities` blob was stripped (362 MB → 1.0 MB); it backs no
> reported number. Verified: all four `summarize_reproduce.py` stages and all
> three `make_plots.py` figures regenerate from that directory alone, and the
> figures come out byte-identical in size to those in `Report/figures/`.
>
> Nothing is committed — the three `sections/*.tex` edits, the two `benchmark/`
> script changes, and the new `report_runs/` + `archive_runs.py` are all still in
> the working tree.

---

## 1. What is running right now

**Server**: `ssh coan-wrk-01` (full name `coan-wrk-01.informatik.uni-bonn.de`).
48 threads, 125 GiB RAM, 2× RTX 2080 Ti. Project lives at
`~/Lab-CA-SS26-node-importance`, kept in sync by `scp` — there is **no `git pull`
workflow on the server**, so push locally *and* `scp` changed files up.

**tmux session `tx`**, log at `~/tx.log`, script `~/run_tight_extra.sh`,
output JSON into `~/tight_extra/julia_<graph>.stats.json`.

It runs KADABRA-Julia at **ε=0.0001, δ=0.1, k=0, t=8, seed 0** (identical to the
six existing tight runs) on the three instances that lacked one:

| order | graph | rough estimate |
|---|---|---|
| 1 | `com-youtube` | ~1 h |
| 2 | `com-lj` | ~5 h |
| 3 | `cit-Patents` | ~5.5 h |

The script skips any output file that already exists, so it is safe to re-launch
if something dies:

```bash
ssh coan-wrk-01 "tmux new-session -d -s tx '~/run_tight_extra.sh 2>&1 | tee -a ~/tx.log'"
```

Check progress with:

```bash
ssh coan-wrk-01 "tail -5 ~/tx.log; ls ~/tight_extra/; grep -c FAILED ~/tx.log"
```

`grep -q 'TIGHT EXTRA DONE' ~/tx.log` is the completion test. `SSH sessions to
this box drop after some idle period (exit code 255) — that is transient, just
reconnect; tmux jobs survive it.`

---

## 2. What to do when they finish

### 2.1 Pull and inspect

```bash
mkdir -p /tmp/tight_extra && scp "coan-wrk-01:~/tight_extra/*.json" /tmp/tight_extra/
```

Each JSON has `execution_time_seconds`, `num_samples`, `tau_overall`,
`overlap_topk`, `kadabra_omega`, `samples_over_omega`.

### 2.2 Check whether the headline claim still holds — **do not assume it does**

`Report/sections/results.tex` §6.4 currently claims that at ε=0.0001 KADABRA beats
BRAVA-GNN **on both metrics on every instance tested**. That is verified for the
six graphs we have. The three now finishing are the largest in the set, and
`com-lj`/`cit-Patents` are exactly where BRAVA-GNN's τ_b is strongest relative to
KADABRA's at loose ε. **If any of the three lands below BRAVA-GNN's τ_b, the
"every one of them" wording must be softened, not just extended.**

BRAVA-GNN's numbers to compare against (ε=0.01 run, CPU; accuracy is
device-independent) — from `/tmp/bvk_results/brava_cpu_<graph>.stats.json`, and
already in Report Table 4:

| graph | BRAVA τ_b | BRAVA overlap | BRAVA CPU time | KADABRA ε=1e-2 time |
|---|---|---|---|---|
| `com-youtube` | 0.736 | 64/100 | 5.70 s | 3.90 s |
| `cit-Patents` | 0.738 | 34/100 | 39.22 s | 19.34 s |
| `com-lj` | 0.749 | 53/100 | 46.89 s | 17.47 s |

For reference, the six already-measured tight runs gave τ_b **0.898–0.995** and
overlap **99–100**, against BRAVA-GNN's τ_b 0.459–0.826 and overlap 46–85, at
**40–550×** the runtime (7 306 s vs 13.3 s on `amazon`; ratio is vs BRAVA **CPU**,
the conservative choice — vs GPU it reaches ~910×).

### 2.3 Update the report

Edit `Report/sections/results.tex` §6.4 (`\paragraph{\ldots but the opposition is
an artefact of the budget.}`):

- replace "On the six instances for which we have tight-bound runs
  (Section~\ref{sec:kadabra-cpp-vs-julia})" with all nine,
- update the τ_b range, overlap range, and the 40–550× runtime-ratio range,
- re-check the `amazon` anchor example is still the most illustrative.

Also check `Report/sections/conclusion.tex`, which repeats "at ε = 0.0001 KADABRA
beats BRAVA-GNN on **both** metrics on every instance we could test" and the
"40–550×" figure.

Then rebuild and verify (from `Report/`):

```bash
biber main && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex; done
grep -iE "^! |undefined ref|undefined cit" main.log | grep -v "Package.*Info"   # expect empty
grep -oE "Overfull \\\\hbox \(([0-9.]+)pt" main.log | grep -oE "[0-9.]+" | sort -rn | head -2
```

Largest overfull box is currently **3.75 pt**; anything appreciably above that is
new and should be fixed (usually by wrapping the paragraph in
`\begin{sloppypar}…\end{sloppypar}` — the report already does this in several
places for paragraphs dense with `\texttt{}` graph names).

### 2.4 Fold the three runs into the reproducible pipeline

**Done (2026-08-14).** `benchmark/reproduce_report.sh` gained a `tightx` stage
(the three graphs, Julia only, no C++ counterpart) and
`benchmark/summarize_reproduce.py` gained a matching `tightx` summarizer that
computes the §6.4 comparison — per-graph win/loss on both metrics, the τ_b,
overlap and runtime-ratio ranges, and the slowest-case anchor. It prints an
explicit `!! CLAIM DOES NOT HOLD` line if any graph fails, so the "on every
instance" wording is never checked by hand. Both files are `scp`'d to the server.

It was validated against the six known runs: it reproduces the previously
published ranges exactly (τ_b 0.898–0.995, overlap 99–100, 40–550× with `amazon`
at 549×).

To re-run the check as the remaining JSONs land, stage the raw results into the
naming `reproduce_report.sh` uses (`tight_julia_<g>.json`,
`bvk_brava_cpu_<g>.json`, …) and call:

```bash
python3 benchmark/summarize_reproduce.py tightx <result-dir>
```

The six §6.1 tight runs live in the server's **`/tmp`** (volatile); they have
been copied to `~/tight_all/` on the server and `/tmp/tight6/` locally.

---

## 3. State of everything else (all finished, all committed)

### Repos

| repo | path | branch | HEAD |
|---|---|---|---|
| project | `.../Lab-CA-SS26-node-importance` | `Martin` | `3fd55e5` (pushed) |
| report | `.../Lab-CA-SS26-node-importance/Report` | `main` | `15b36a9` (local only, no remote) |
| Graphs.jl fork | `.../Graphs.jl` | `master` | `e7d10c9f` |

`Report/` is **gitignored by the project repo** but is its own git repo. The
Graphs.jl fork was squashed onto upstream into a single "Add KADABRA betweenness
centrality algorithm" commit; both `src/kadabra.jl` (project) and
`src/centrality/kadabra.jl` (fork) currently carry every fix and **must stay in
sync**. The server copy is byte-identical to local (`md5 09e7c5ec…`).

### Report structure (main content = pages 5–20, i.e. **16 pages**)

1 Introduction · 2 Preliminaries · 3 KADABRA (3.4 = our implementation) ·
4 BRAVA-GNN · 5 Experimental Setup · 6 Results (6.1 C++ vs Julia, 6.2 Thread
scaling, 6.3 Top-k, 6.4 BRAVA vs KADABRA) · 7 Conclusion · A Appendix
(A.1 ground-truth correction, A.2 ω/τ budget).

Figures: `thread_scaling`, `brava_vs_kadabra`, `k_sweep`.
Tables: `instance_table_new`, `cpp_vs_julia_tight_table`,
`brava_vs_kadabra_table`, `ground_truth_recompute_table`.

**Page limit — RESOLVED, do not reopen.** `Lab.pdf` allows **10–15 pages of main
content** (+ bibliography + appendix). We are at **16**. On 2026-08-14 the user
was shown both fixes with measured page counts and chose to **stay at 16 and
submit one page over**. Nothing is to be cut or moved.

### Established findings (do not re-derive)

- **KADABRA C++ vs Julia at ε=1e-4, matched t=8**: sample ratio mean 1.04×,
  time ratio mean/median 0.98×. Diameter estimates agree exactly on all six graphs.
- **Thread scaling** (4 graphs × 8 thread counts × 3 seeds × 2 impls, all clean):
  C++ optimum `t=8` on all four; Julia `t=8`–`16`. Both get *slower* past the
  optimum (C++ up to 3.6× its own optimum at t=48). Peak speedup only 3.1–4.6×
  (C++) / 2.0–2.8× (Julia) — memory-bound.
- **Top-k**: saving is modest (k=100 → 0.90× samples; k=10 → 0.96×) and on 3 of 6
  graphs top-k drew *more* samples than k=0 (`soc-Epinions1` 1.38× at k=10).
  Cause: KADABRA re-derives the confidence budgets per k, so the runs are not
  nested. §3.3.2 says this.
- **Code-vs-paper discrepancy** (footnote in §3.3.2): the reference C++
  `compute_bet_err` assigns the rank gaps to λ_L/λ_U in the *opposite* order from
  Borassi & Natale. We follow the code so §6.1 compares implementations. Tested:
  the paper's order *moves* which configurations exceed k=0 rather than removing
  the effect — so it does **not** explain the top-k anomaly. Do not "fix" this.
- **`parallel=true` is not bit-reproducible** even with a fixed `rng` (worker
  seeds are deterministic, work split is not); `parallel=false` is. Documented.

### Things deliberately *not* in the report

- The delta-calibration fix narrative (user's own porting bug; §3.2.2 describes
  the algorithm, not the fix).
- Any pre-fix measurement — every number in the report was measured this session
  with the corrected code.
- The tight-ε ground-truth approximation for `web-Google`, `soc-Pokec`,
  `wiki-topcats`, `wiki-Talk`: those five instances are **never used**, so all
  experiments run on the nine with an exact/provided reference.

### Working conventions the user expects

- **Every run with numbers goes on `coan-wrk-01` inside `tmux`**, not locally.
- `JULIA_NUM_THREADS` is what configures Julia's threads; the runner's `-t` flag
  is **log-only**.
- Verify claims against real runs before trusting arithmetic; test small/fast
  graphs before committing to expensive ones.
- Don't fix only one of the parallel/sequential code paths.
- Regenerate report tables/figures with `benchmark/summarize_reproduce.py` and
  `benchmark/make_plots.py` rather than typing numbers by hand — this has already
  caught two transcription errors.
