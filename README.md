# Node Importance — KADABRA and BRAVA-GNN in Julia

Lab course project (SS26, Computational Analytics, University of Bonn). We reimplemented two
approximate betweenness-centrality algorithms in Julia and evaluated them against each other,
against the authors' C++ reference, and against exact betweenness:

- **KADABRA** — adaptive sampling with an $(\varepsilon, \delta)$ guarantee
  (Borassi & Natale, *ESA 2016 / JEA 2019*).
- **BRAVA-GNN** — a graph neural network that learns betweenness *rankings*
  (Dachille, Rossi et al., *CIKM '26*).

This README covers [what we did](#what-we-did), [what you need](#prerequisites) to
reproduce it, and [how to reproduce it](#reproducing-the-results). Every number in the
report comes from the output of the scripts in `benchmark/`: tables and figures are written
by them directly, and the numbers in the prose are taken from what they print.

---

## What we did

### Implementations

- **KADABRA** (`src/kadabra.jl`): a multithreaded port of the authors' C++ reference, with
  both modes — additive error $\varepsilon$ on every vertex ($k = 0$) and a correct top-$k$
  ranking ($k > 0$). It follows the `Graphs.jl` conventions and is proposed upstream as
  [JuliaGraphs/Graphs.jl#518](https://github.com/JuliaGraphs/Graphs.jl/pull/518), from the
  fork at [DerSchmachtin/Graphs.jl](https://github.com/DerSchmachtin/Graphs.jl).
- **BRAVA-GNN** (`src/BRAVAGNN.jl`, `src/train_bravagnn.jl`): the architecture, inference
  with the leaf/clique pruning mask, and GPU training with the paper's pairwise ranking loss,
  ported from the authors' Python code.
- **Benchmark harness** (`benchmark/`): per-run drivers for Julia and for the C++ reference
  (one JSON per run), stage scripts, summarizers that emit the LaTeX table bodies, and the
  plotting code.

### Experiments

| report | experiment | instances |
| --- | --- | --- |
| 6.1, Table 1 | C++ reference vs. Julia port at $\varepsilon = 10^{-4}$: runtime and samples | 6 graphs |
| 6.1, Figure 1 | thread scaling, 1–48 threads, both implementations | 4 graphs × 8 thread counts × 3 seeds |
| 6.3, Figure 2, Table 6 | cost of top-$k$ mode relative to $k = 0$, under five budget allocations | 4 graphs × $k \in \{3,5,10,100\}$, plus `amazon` and `dblp` at $k \in \{10,100\}$; 3 seeds |
| 6.4, Table 2, Figure 3 | BRAVA-GNN vs. KADABRA at $\varepsilon = 10^{-2}$: Kendall $\tau_b$, top-100 overlap, runtime | 9 graphs × 3 seeds |
| 6.4, Table 5 | KADABRA at $\varepsilon = 10^{-4}$ against BRAVA-GNN | 9 graphs |
| A.7, Table 8 | per-vertex accuracy of the C++ reference vs. the port | 6 graphs |

### Results in brief

- **The Julia port matches the C++ reference.** At $\varepsilon = 10^{-4}$ it draws 1.003×
  the reference's samples (range 0.996–1.012) and takes 0.90× its time (0.75–1.04), and it
  ranks equally well (top-100 overlap 98–100, $\tau_b$ within 0.003).
- **BRAVA-GNN reproduces the paper** after two fixes (finding 1): mean $\tau_b$ 87.3 against
  the paper's 87.7 over three training seeds. Against KADABRA at $\varepsilon = 10^{-2}$,
  BRAVA-GNN has the higher $\tau_b$ on all 9 graphs, KADABRA the higher top-100 overlap on
  all 9.

### Findings

1. **BRAVA-GNN: two bugs that only mattered together.** Our first version scored mean
   $\tau_b$ 73.5. It had a PageRank input channel the paper does not have, and it masked the
   adjacency matrix differently at inference ($A'D$) than in training ($DA'$). Either fix
   alone leaves `email-EuAll` at 0.460; both together give 0.991. Details:
   `benchmark/results/brava_retrained/README.md`.
2. **KADABRA's top-$k$ budget allocation: the paper, the C++ reference and NetworKit
   disagree.** The reference swaps the lower and upper confidence budgets relative to the
   paper, and NetworKit copies the reference. The tie-collapse guard also skips the one rank
   pair $(v_k, v_{k+1})$ that the stopping test depends on. With both repaired (our default,
   `topk_variant = :paper_bd`), top-$k$ mode needs on average 0.69× the samples of $k = 0$,
   against 0.97–0.98× with the reference's allocation, and returns the same top-$k$ answer.
   The $(\varepsilon,\delta)$ guarantee holds either way. Details: `KADABRA_TOPK_FINDINGS.md`.
3. **A burn-in normalisation bias in the reference.** The reference discards the burn-in
   samples but still divides by them, so every score comes out too low by the burn-in
   fraction (2.9–7.7%). The maximum absolute error was 3.5–12.3 $\varepsilon$; after the fix
   it is 0.19–0.44 $\varepsilon$. Rankings and stopping are unchanged.
4. **Stopping coordination in the parallel port.** Threads kept sampling after the stop, and
   checks ignored unfinished batches, costing 3–6% extra samples. Both are fixed, and every
   KADABRA number in the report was re-measured afterwards.

---

## Prerequisites

### Hardware

- **Full reproduction:** a multi-core Linux machine with a CUDA GPU. Ours had 48 hardware
  threads, 125 GB RAM and 2× RTX 2080 Ti (11 GB). The thread-scaling experiment goes up to
  48 threads.
- **Time:** about 4 days of exclusive use. The timed steps (`tight`, `threads`, `bvk`)
  must have the machine to themselves.
- **Quick check** (tables and figures from the committed runs): any machine with Python;
  it takes seconds.

### Software

| what | version we used | for |
| --- | --- | --- |
| Julia | 1.12 (`Manifest.toml` pins 1.12.5) | `src/` and the Julia runner |
| g++ with OpenMP | 11.4 on Linux; clang + Homebrew `libomp` on macOS | the C++ reference runner |
| Python 3 with `numpy`, `scipy`, `matplotlib` | 3.10 | summarizers and plots |
| CUDA + cuDNN | via `CUDA.jl` | BRAVA-GNN GPU runs and training (inference falls back to CPU) |

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia packages, pinned by Manifest.toml
make -C benchmark/simexpal_runners build              # C++ runner; the reference and nlohmann/json are vendored in cpp_reference/
```

For the dataset and training-data scripts, a Python environment with the authors'
requirements:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r BRAVA-GNN-A0BD/requirements.txt
```

### Data

`Instances/` is not in git.

- **Graphs.** `python3 scripts/download_datasets.py --calibration` fetches the SNAP test
  graphs into `Instances/TestInstances/` and the five ABCDE graphs (`amazon`, `dblp`,
  `com-youtube`, `com-lj`, `cit-Patents`) into `Instances/ABCDE/`.
- **Exact ground truth**, `Instances/ground_truth/test_instances/<graph>_bet.csv` with
  columns `node,betweenness`, needed for every accuracy number. For the five ABCDE graphs it
  was provided by the BRAVA-GNN authors. For `p2p-Gnutella31`, `soc-Epinions1`,
  `soc-Slashdot0902` and `email-EuAll` we computed it with `Graphs.jl`'s exact
  `betweenness_centrality`, which takes hours to days per graph (report, Appendix A.2).
  These files are not distributed with the repository.
- **BRAVA-GNN training graphs**, only for retraining:
  `python3 scripts/generate_training_data.py --datasets SF_10_Dir SF_10_Sym --num_nodes 100000`.

The three trained BRAVA-GNN checkpoints are committed in
`benchmark/results/brava_retrained/weights/`, so retraining is optional.

---

## Reproducing the results

Everything goes through one script, `benchmark/reproduce_all.sh`.

### Quick check: tables and figures from the committed runs

```bash
./benchmark/reproduce_all.sh --from-archive --out /tmp/repro_check
```

This rebuilds every table and figure, and every number quoted in the text, from the runs
committed under `benchmark/results/`. With the Report repository checked out as `Report/`,
the regenerated tables are identical to the report's
(`diff -r /tmp/repro_check/report/tables Report/tables`), and so are the figures; the PDFs
differ only in their embedded creation date.

### Full reproduction: every measurement from scratch

On the benchmark machine, inside `tmux`:

```bash
tmux new-session -d -s repro './benchmark/reproduce_all.sh 2>&1 | tee -a ~/reproduce_all.log'
```

| step | what | report | time |
| --- | --- | --- | --- |
| `build` | C++ runner, Julia environment | | minutes |
| `weights` | install the committed BRAVA-GNN checkpoints (`--retrain` trains them) | 6.4 | 0 / ~50 min |
| `tight` | C++ vs Julia at $\varepsilon = 10^{-4}$ — **timed** | Table 1 | ~6 h |
| `threads` | thread scaling, C++ and Julia — **timed** | Figure 1 | ~4 h |
| `bvk` | BRAVA-GNN vs KADABRA at $\varepsilon = 10^{-2}$ — **timed** | Table 2, Figure 3 | ~1 h |
| `brava-seeds` | the three checkpoints against the paper's Table 2 | Table 2 | ~30 min |
| `topk` | top-$k$ sweeps, and the C++ binary in top-$k$ mode | Figure 2, Table 6 | ~10 h |
| `cpp-quality` | C++ per-vertex output scored against the ground truth | Table 8 | ~6 h |
| `tightx` | KADABRA at $\varepsilon = 10^{-4}$ on the three largest graphs | Table 5 | ~18 h |
| `kxl` | top-$k$ on `amazon` and `dblp` | Figure 2 | ~40 h |
| `report` | tables, figures and quoted numbers | all | seconds |

Options: `--out DIR` (default `~/reproduce_all`), `--from STEP` to resume from a step,
`--only STEP` to run one step, `--retrain` to train BRAVA-GNN instead of using the committed
checkpoints, `--dry-run` to print every command. Every step is incremental: existing run
files are skipped, so an interrupted run resumes when relaunched.

**Output.** One JSON per run under `DIR/rerun_fix/` (the layout of
`benchmark/results/rerun_fix/`), tables and figures in `DIR/report/`, and the numbers the
prose quotes in `DIR/claims/`, one file per report section. The script ends by listing every
claim the fresh data no longer supports. One is expected: `6.4_tight_accuracy.txt` flags
`email-EuAll`, where BRAVA-GNN beats KADABRA on $\tau_b$, which the report states.

**What to expect.** KADABRA's multithreaded sampling and BRAVA-GNN's GPU training are not
deterministic, so fresh numbers differ from the report's within the spread over seeds it
gives. Timings depend on the machine.

**Not re-run:** the exact ground truth (an input, see [Data](#data)), and the runs the report
cites only as history (the single-seed top-$k$ sweep and the runs from before the fixes).

### Individual stages and scripts

`reproduce_all.sh` chains these, all in `benchmark/`; each can also be run on its own.

| script | purpose |
| --- | --- |
| `reproduce_report.sh --stage {tight,tightx,bvk,kx,kxs,kxb,kxc,kxl}` | one measurement stage |
| `update_report_from_rerun.py [--results DIR] [--report DIR]` | writes the table bodies and figures |
| `summarize_reproduce.py {tight,tightx,bvk} <dir>` | table bodies and quoted ranges; `tightx` checks Section 6.4's claim |
| `summarize_seeds.py`, `summarize_topk.py`, `summarize_topk_claims.py`, `summarize_threads.py`, `summarize_cpp_quality.py` | the other sections' tables and quoted numbers |
| `make_plots.py {threads,bvk,topk} <dir> <outdir>` | the figures |
| `eval_brava_paper.jl [checkpoint]` | scores a BRAVA-GNN checkpoint against the paper's Table 2 |
| `diagnose_brava_mask.jl`, `diagnose_brava_nopr.jl` | the 2×2 diagnostic behind finding 1 |
| `diagnose_topk_delta.jl <graph>` | why the top-$k$ allocations differ, per vertex |

---

## Using the algorithms

### KADABRA

```julia
include("src/kadabra.jl")
using Graphs

g = loadgraph("...")

res = kadabra_centrality(g, 0, 1e-2, 0.1)    # additive error 1e-2 on every vertex
res = kadabra_centrality(g, 10, 1e-2, 0.1)   # rank the top 10 correctly
res.centralities, res.lower_bounds, res.upper_bounds, res.n_samples, res.omega, res.tau
```

`kadabra_centrality(g, k, err, delta; kwargs...)`; `kadabra_top_k(g, k, err, delta)`
returns just the ranked vertices.

| keyword | default | meaning |
| --- | --- | --- |
| `start_factor` | `100` | burn-in draws `omega / start_factor` samples |
| `endpoints` | `false` | count path endpoints, as in `Graphs.jl` |
| `normalize` | `:graphs` | `:graphs`, `:kadabra` (raw paper scale), or `:none` |
| `parallel` | `true` | sample with `Threads.nthreads()` tasks |
| `rng` | `nothing` | seed source; bit-identical output only with `parallel=false` |
| `topk_variant` | `:paper_bd` | top-$k$ budget allocation: `:paper_bd`, `:paper`, `:paper_ex`, `:cpp` (the reference verbatim), `:code` (the port's original hybrid) |

Weighted graphs are rejected. Output follows `Graphs.jl`'s `betweenness_centrality`
scaling by default.

### BRAVA-GNN

```julia
include("src/BRAVAGNN.jl")
using .BRAVAGNN

scores, nsamples = brava_centrality(g, 10, 1e-2, 0.1)
```

> **Weights.** `brava_centrality` loads `benchmark/cache/bravagnn_weights.jld2` and, if that
> file is missing, **falls back to an untrained model with only a warning**. Copy a
> checkpoint from `benchmark/results/brava_retrained/weights/` or pass `weight_path=`;
> `reproduce_all.sh` does this for you.

Training (`--seed=1` is the canonical model; the defaults are the paper's configuration):

```bash
julia --project=. --threads=auto src/train_bravagnn.jl --seed=1
```

### Tests

```bash
julia --project=. test/test_kadabra.jl
```

`test/test_kadabra_graphs_style.jl` is the upstream-style suite; it needs the `Graphs.jl`
fork checked out next to this repository.

---

## Repository layout

| path | what |
| --- | --- |
| `src/` | `kadabra.jl`, `BRAVAGNN.jl`, `train_bravagnn.jl` |
| `benchmark/` | runners, stage scripts, summarizers, plots, `reproduce_all.sh` |
| `benchmark/results/` | the committed run data behind the report (below) |
| `cpp_reference/` | Borassi & Natale's C++ KADABRA, the baseline |
| `BRAVA-GNN-A0BD/` | the authors' Python BRAVA-GNN, the reference for the port |
| `scripts/` | dataset download, training-data generation, instance tables |
| `test/` | unit tests and the `Graphs.jl`-style suite |
| `Instances/` | graphs and ground truth — not in git |
| `KADABRA_TOPK_FINDINGS.md` | standalone write-up of finding 2 |
| `presentation/` | the slides of the final talk (16 September 2026, without speaker notes) and the interactive demo shown in it; open `presentation_demo/betweenness_demo.html` in a browser, or use the slides' demo buttons with the folder kept next to the PDF |

### Committed run data

| path | runs | what |
| --- | --- | --- |
| `benchmark/results/rerun_fix/` | 531 | **what the report uses**: every KADABRA measurement after the fixes, plus the reused C++ and BRAVA-GNN runs |
| `benchmark/results/brava_retrained/` | 27 | the retrained BRAVA-GNN: runs, 3 checkpoints, evaluation logs |
| `benchmark/results/cpp_quality/` | | the C++ reference's scored per-vertex output (Appendix A.7) |
| `benchmark/results/topk_variant/measured/` | 333 | top-$k$ runs before the stopping fix, including the C++ binary in top-$k$ mode |
| `benchmark/results/seed_check/` | 120 | the experiment that located the stopping overshoot (finding 4) |
| `benchmark/results/report_runs/` | 252 | the original runs from before the fixes, cited as history |

Per-vertex centralities are stripped from the stored JSON; the metrics are kept.

---

## Pitfalls

- **Thread count.** `-t N` on `run_experiments.jl` is only recorded, not applied; Julia's
  threads come from `JULIA_NUM_THREADS`, which the scripts set. A single-threaded run is
  valid JSON with ~30% fewer samples, so check `parameters.threads` in the output.
- **Directed graphs in the C++ runner.** Only `-d` works; `--directed` is silently ignored
  and the graph is loaded undirected. The scripts translate this.
- **Missing BRAVA-GNN weights** fall back to an untrained model (see above);
  its `weights` step refuses to continue without them.
- **Stopping runs over SSH.** `pkill -f run_experiments.jl` also kills your own SSH
  command; anchor the pattern: `pkill -f "^julia --project"`.

---

## References

- M. Borassi, E. Natale. *KADABRA is an ADaptive Algorithm for Betweenness via Random
  Approximation.* ESA 2016 / ACM JEA 24, 2019.
- Dachille, Rossi et al. *BRAVA-GNN.* CIKM 2026.
- M. Borassi, P. Crescenzi, M. Habib, W. Kosters, A. Marino, F. Takes. *Fast diameter and
  radius BFS-based computation in (weakly connected) real-world graphs.* TCS 586, 2015.
