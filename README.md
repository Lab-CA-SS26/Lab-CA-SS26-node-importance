# Node Importance — KADABRA and BRAVA-GNN in Julia

Lab course project (SS26, Computational Analytics). Two approximate betweenness-centrality
algorithms reimplemented in Julia from their papers, plus the benchmark harness that
measures them against each other, against the authors' C++ reference, and against exact
Brandes:

- **KADABRA** — adaptive sampling with an $(\varepsilon, \delta)$ guarantee
  (Borassi & Natale, *ESA 2016 / JEA 2019*).
- **BRAVA-GNN** — a graph neural network that regresses betweenness rankings
  (Dachille, Rossi et al., *CIKM '26*).

The deliverable is the LaTeX report in `Report/`. Everything in this repository exists to
produce a number that goes into it.

Two findings came out of the work and are written up below: a **train/inference mismatch
plus a spurious PageRank channel** that together cost BRAVA-GNN 14 points of Kendall
$\tau_b$, and a **three-way disagreement between the KADABRA paper, the authors' C++
reference, and NetworKit** over the top-$k$ confidence-budget allocation.

---

## Layout

| path | what |
| --- | --- |
| `src/kadabra.jl` | the KADABRA port — diameter bound, adaptive stopping rule, threaded sampler |
| `src/BRAVAGNN.jl` | the BRAVA-GNN architecture and inference wrapper (Flux, sparse) |
| `src/train_bravagnn.jl` | training loop: pairwise margin-ranking loss over exact-BC labels |
| `benchmark/` | the measurement harness — see [`benchmark/README.md`](benchmark/README.md) |
| `benchmark/simexpal_runners/` | the per-run drivers (`run_experiments.jl`, `run_experiments_cpp`) |
| `benchmark/results/` | committed run data backing the report |
| `test/` | unit tests plus a Graphs.jl-style suite run against the fork |
| `scripts/` | dataset download, training-data generation, instance tables |
| `cpp_reference/` | Borassi & Natale's C++ KADABRA, used as the baseline |
| `BRAVA-GNN-A0BD/` | the authors' Python BRAVA-GNN, used as ground truth for the port |
| `Instances/` | graph data — **not in git**, fetch it (see below) |
| `Report/` | the LaTeX report — **its own separate git repository**, ignored here |
| `docs/`, `state.md`, `training.md` | working notes from the port |
| `KADABRA_TOPK_FINDINGS.md` | the write-up of the allocation discrepancy, for the authors |
| `KADABRA_DELTA_PROBLEM.md`, `HANDOFF_TIGHT_EXTRA.md` | closed investigations |

The KADABRA implementation is also maintained as a Graphs.jl fork
(`../Graphs.jl`, `src/centrality/kadabra.jl`), proposed upstream as
[JuliaGraphs/Graphs.jl#518](https://github.com/JuliaGraphs/Graphs.jl/pull/518).

---

## Setup

Julia dependencies (CUDA and cuDNN are optional at runtime — BRAVA falls back to CPU):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Python, for the dataset and training-data scripts:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r BRAVA-GNN-A0BD/requirements.txt
```

The C++ reference runner, needed for the `tight` benchmark stage:

```bash
make -C benchmark/simexpal_runners
```

### Getting the graphs

`Instances/` is gitignored; all of it is reproducible.

```bash
source venv/bin/activate
python3 scripts/download_datasets.py --calibration
```

fetches the SNAP test and calibration graphs into `Instances/TestInstances/`. The ABCDE
graphs (`amazon`, `dblp`, `com-youtube`, `com-lj`, `cit-Patents`) land in
`Instances/ABCDE/` with their `-score.txt` ground truth.

Synthetic BRAVA-GNN training graphs — scale-free only, matching the paper's `-HY`
configuration:

```bash
python3 scripts/generate_training_data.py --datasets SF_10_Dir SF_10_Sym --num_nodes 100000
```

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

Signature: `kadabra_centrality(g, k, err, delta; kwargs...)`, with `k = 0` requesting the
absolute-error guarantee and `k > 0` the top-$k$ ranking guarantee.

| keyword | default | meaning |
| --- | --- | --- |
| `start_factor` | `100` | burn-in draws `omega / start_factor` samples |
| `endpoints` | `false` | count path endpoints, as in `Graphs.jl` |
| `normalize` | `:graphs` | `:graphs`, `:kadabra` (raw paper scale), or `:none` |
| `parallel` | `true` | sample with `Threads.nthreads()` tasks |
| `rng` | `nothing` | seed source; bit-identical output only with `parallel=false` |
| `topk_variant` | `:paper_bd` | which reading of the top-$k$ budget allocation to use (below) |

The output matches `Graphs.jl`'s `betweenness_centrality` conventions by default, not the
C++ reference's raw expected values. Weighted graphs are rejected.

`kadabra_top_k(g, k, err, delta; kwargs...)` returns just the ranked vertices.

### BRAVA-GNN

```julia
include("src/BRAVAGNN.jl")
using .BRAVAGNN

scores, nsamples = brava_centrality(g, 10, 1e-2, 0.1)
```

> **Weights.** `brava_centrality` defaults to `benchmark/cache/bravagnn_weights.jld2`, and
> if that file is missing it **falls back to a randomly initialised model with only a
> warning** — a silently meaningless run. That path is gitignored; the trained checkpoints
> live in `benchmark/results/brava_retrained/weights/` (three seeds, each with a
> `.config.txt` recording how it was trained). Copy one in, or pass `weight_path=`.

Training:

```bash
julia --project=. --threads=auto src/train_bravagnn.jl --seed=1
```

Defaults reproduce the paper's configuration: `degree_mix_mass_6` features, `nhid=12`,
`L=2`, dropout 0.3, 10 epochs, Adam at 5e-3, over all 30 training graphs. `--pr` re-enables
the PageRank channel (don't — see below), `--loss=drop-ties` swaps the pair sampling.
Training is seeded but **not bit-reproducible**: cuDNN and the sparse matmul reduce in
nondeterministic order.

### Tests

```bash
julia --project=. test/test_kadabra.jl
```

`test/test_kadabra_graphs_style.jl` is the upstream-style suite and additionally needs the
Graphs.jl clone checked out next to this repository, for its `testdata/` and
`GenericGraph` helpers.

---

## Reproducing the report

Every number in the report comes from `benchmark/reproduce_report.sh`. Nothing is
hand-typed into the LaTeX.

```bash
cd benchmark
./reproduce_report.sh --stage bvk          # BRAVA-GNN vs KADABRA, eps=1e-2, 9 graphs (~20 min)
./reproduce_report.sh --stage tight        # C++ vs Julia KADABRA, eps=1e-4 (~6 h)
./reproduce_report.sh --stage tightx       # eps=1e-4 on the three largest bvk graphs (~23 h)
./reproduce_report.sh --stage k            # top-k runtime sweep, single seed (hours)
./reproduce_report.sh --stage kx           # top-k x allocation variant, 3 seeds (~3 h)
./reproduce_report.sh --stage kxs          # the same at k=3 and k=5 (~3 h)
./reproduce_report.sh --stage kxl          # repaired allocation on amazon/dblp (~40 h)
```

One JSON per run lands in `results/reproduce/`. Re-runs are incremental — existing files
are skipped, so delete one to recompute just that measurement.

Turning runs into report artifacts:

| script | purpose |
| --- | --- |
| `summarize_reproduce.py {tight,tightx,bvk,k} <dir>` | LaTeX table bodies plus the ranges quoted in the text |
| `summarize_seeds.py <kad_dir> <brava_log> <bvk_dir>` | mean ± std bodies; flags gaps inside one std |
| `summarize_topk.py <dir>` | pairs `kx`/`kxs`/`kxl` ratios per seed; flags top-$k$ runs dearer than $k=0$ |
| `make_plots.py {threads,bvk,k,topk} <dir> <outdir>` | the report's figures |
| `eval_brava_paper.jl [checkpoint]` | scores a checkpoint against the paper's Table 2 |
| `diagnose_brava_mask.jl`, `diagnose_brava_nopr.jl` | the 2×2 diagnostic behind the BRAVA finding |
| `diagnose_topk_delta.jl <graph> [-k …] [--epsilon E]` | why the allocations differ, deterministically, at ~1% of a real run's cost |
| `archive_runs.py` | gathers scratch runs off the server into a committable directory |

`summarize_reproduce.py tightx` doubles as the **claim checker** for the report's
accuracy comparison; give it a directory holding both `tight_julia_*.json` and
`bvk_*.json`. It currently reports KADABRA winning both metrics on 8 of 9 graphs, losing
$\tau_b$ on `email-EuAll` — which is exactly what the report claims. **Re-run it before
trusting that section after any BRAVA-GNN change.**

### Committed run data

| path | what |
| --- | --- |
| `benchmark/results/report_runs/` | 252 runs backing the main results sections |
| `benchmark/results/brava_retrained/` | the retrained model: 27 `bvk` runs, 27 KADABRA repeats, 3 checkpoints, logs |
| `benchmark/results/topk_variant/measured/` | 252 runs: 4 graphs × k ∈ {0,3,5,10,100} × 3 variants × 3 seeds at ε=1e-4 |
| `benchmark/results/topk_variant/predicted/` | `diagnose_topk_delta.jl` output, 4 graphs × 3 seeds |

Centrality arrays are stripped from the JSON on purpose; the metrics
(`execution_time_seconds`, `num_samples`, `tau_overall`, `tau_topk`, `overlap_topk`,
`mae`, `max_ae`, `ndcg_topk`, `parameters`) are what the summarizers read.

---

## The two findings

### 1. BRAVA-GNN: two bugs that only mattered together

The port scored mean Kendall $\tau_b$ **73.5 against the paper's 87.7**. It was not a bad
implementation — it faithfully reproduced an ablation the authors ran once and abandoned,
matching their `_pr` results to within 1.7 points.

Two causes, and **neither alone moves `email-EuAll` at all**:

| configuration | `email-EuAll` $\tau_b$ |
| --- | --- |
| PageRank on, `A_t` masked wrongly | 0.459 |
| PageRank on, `A_t` fixed | 0.460 |
| PageRank off, `A_t` masked wrongly | 0.460 |
| **PageRank off, `A_t` fixed** | **0.991** |

1. **A PageRank input channel the paper does not have.** `use_pr` defaulted to `true`; the
   paper never mentions PageRank, and exactly 1 of the 846 configurations in the authors'
   `all_results.csv` carries `_pr`. Costs ~12.6 points.
2. **A train/inference mismatch in the clique mask.** Training and upstream build
   $A_t = DA'$ (rows of $A'$ zeroed); inference built $(DA)' = A'D$ (columns zeroed).

They interact because $\tau_b$ here is dominated by tie structure. The mask prunes 18–96%
of vertices, all with true betweenness exactly zero. Fixed *and* without PageRank they get
identically zero features and therefore one shared score, tying exactly where the ground
truth ties them. PageRank is not zeroed by the mask, so it shatters that block.

Retrained to the paper's configuration: **mean $\tau_b$ 87.3 against the paper's 87.7**
over three seeds. Full write-up in `benchmark/results/brava_retrained/README.md`.

> $\tau_b$ is stable across seeds (per-graph std 0.75); **top-100 overlap is not** — std
> 13.2 on `p2p-Gnutella31`, which gave 43 / 52 / 26 on three seeds. Never quote an overlap
> figure from a single seed.

### 2. KADABRA: the paper, the reference, and NetworKit disagree on the top-$k$ budget

In top-$k$ mode KADABRA sizes each vertex's confidence interval from its rank gaps *before*
sampling. Paper Section 5.2 sets $\lambda_L(v_i)$ from the gap *below* $v_i$ and
$\lambda_U(v_i)$ from the gap *above* — which is what Algorithm 2 goes on to consult. The
authors' `Probabilistic.cpp` transposes them, leaves $\lambda_L(v_1)$ unconstrained (the one
bound the top vertex's own test needs), and *adds* $\lambda_L(v_k)$ where the paper
subtracts it outside the top $k$. NetworKit's `KadabraBetweenness.cpp` copies all of it
verbatim.

None of this breaks correctness — the allocation is a heuristic for *where* to spend the
confidence budget, not part of the $(\varepsilon,\delta)$ guarantee, and $k = 0$ never
enters the code path. It costs samples. Over 4 graphs × k ∈ {3,5,10,100} × 3 seeds at
ε=1e-4, paired per seed, the paper's allocation needs **0.80 ± 0.17** as many samples as
the reference's, at **identical top-$k$ accuracy** (same overlap, same `tau_topk` to three
decimals).

A third off-by-one turned up alongside it: the tie-collapse guard sweeps the pairs
$(1,2)\ldots(k-1,k)$ and $(k+1,i)$ for $i \ge k+2$, so the boundary pair $(k, k+1)$ — the one
gap the external exclusion test depends on — is never checked, in the reference, in
NetworKit, or originally here. Collapsing it removes an anomaly where **asking for the top
$k$ costs more samples than computing every centrality** (`email-EuAll` k=3: 1.83× → 0.81×;
`soc-Epinions1` k=5: 1.85× → 0.92×), with the top-$k$ answer unchanged in 15 of 16 cells.

Variants selectable via `topk_variant`:

| value | what it is |
| --- | --- |
| `:paper_bd` | **shipped default** — the paper's allocation plus the boundary-pair collapse |
| `:paper` | the paper's allocation, reference collapse loops |
| `:paper_ex` | alternative repair, re-anchoring the guard on $v_k$; matches on the mean but unstable |
| `:cpp` | the C++ reference verbatim |
| `:code` | the hybrid this port carried before the comparison — reference allocation, paper external test |

`:cpp` and `:code` exist only to reproduce the report's comparison. `KADABRA_TOPK_FINDINGS.md`
is the standalone note; every figure in it was re-verified against the raw JSON.

---

## Traps that have already cost real time

- **`-t N` on `run_experiments.jl` configures nothing.** It is recorded into the JSON for
  logging only. Julia's thread pool comes from `JULIA_NUM_THREADS`, which
  `reproduce_report.sh` exports. A run started with only `-t 8` is single-threaded, emits
  perfectly valid JSON, and draws ~30% fewer samples at 0.02–0.05 lower $\tau_b$ — which
  reads exactly like a real effect. Check `parameters.threads`, which records the actual
  count. One full set of repeats was discarded to this.
- **Benchmark on the server, never the laptop**, and always inside `tmux`. Timed stages
  need the machine to themselves — never run anything alongside `bvk`.
- **`pkill -f run_experiments.jl` over SSH kills your own command.** `pkill -f` matches full
  command lines and the remote `bash -c` wrapper contains the pattern, so the connection
  dies with exit 255 and you cannot tell whether the kill worked. Anchor it:
  `pkill -f "^julia --project"`.
- **`grep` silently bails on TeX logs.** They contain binary bytes, so `grep -c` returns
  *nothing* rather than 0 and every count reads as empty. Use `grep -a`, and measure from
  `main.log`, not from redirected stdout.
- **`pdftotext` scrambles stacked fractions.** Verify maths by rasterising:
  `pdftoppm -f N -l N -r 100 -png main.pdf /tmp/pg`.
- **Report underfull-vbox warnings are fixed by float placement, not by shortening prose.**
  Trimming text has repeatedly made them worse. Sweep `[h] [ht] [htbp] [tbp] [!ht]` on the
  floats near the offending page.
- **`reproduce_report.sh`'s `k` stage is pinned to `--topk-variant code`** on purpose, so
  the single-seed sweep still reproduces after the default moved. Not a leftover.

---

## References

- M. Borassi, E. Natale. *KADABRA is an ADaptive Algorithm for Betweenness via Random
  Approximation.* ESA 2016 / ACM JEA 24, 2019.
- Dachille, Rossi et al. *BRAVA-GNN.* CIKM 2026.
- M. Borassi, P. Crescenzi, M. Habib, W. Kosters, A. Marino, F. Takes. *Fast diameter and
  radius BFS-based computation in (weakly connected) real-world graphs.* TCS 586, 2015.
