# BRAVA-GNN retrained to the paper's configuration (2026-08-15)

These runs supersede the BRAVA-GNN columns in `../report_runs/bvk_brava_*.json`, which
were produced by a misconfigured model. **The report now quotes these** -- Section 6.4's
prose was updated 2026-08-31, and its table and Figure 3 on 2026-09-03 (they had been
missed: the table was reverted to a pre-retraining commit by a file-sync event, and the
figure had never been regenerated). The old runs are kept for provenance only.

## What was wrong

Our BRAVA-GNN scored a mean Kendall tau_b of 73.5 against the paper's 87.7. Two causes,
and *both* were needed to produce the collapse -- neither alone moves `email-EuAll` at all:

| configuration | `email-EuAll` tau_b |
| --- | --- |
| PageRank on, `A_t` masked wrongly (what the report currently quotes) | 0.459 |
| PageRank on, `A_t` fixed | 0.460 |
| PageRank off, `A_t` masked wrongly | 0.460 |
| **PageRank off, `A_t` fixed** | **0.991** |

1. **A PageRank input channel that the paper does not have.** `use_pr` defaulted to
   `true` in `brava_centrality` and `train_bravagnn.jl`. The BRAVA-GNN paper never
   mentions PageRank, and exactly 1 of the 846 configurations in the authors'
   `BRAVA-GNN-A0BD/results/betweenness/all_results.csv` carries the `_pr` suffix. Our
   old numbers reproduce that one abandoned ablation row to within 1.7 points mean
   absolute error, and `email-EuAll` to within 0.2 -- the implementation was faithful,
   it was just reproducing the wrong configuration. Their own A/B prices the channel at
   -12.6 tau_b points on average.

2. **A train/inference mismatch in the clique mask.** `train_bravagnn.jl` and upstream
   `utils.graph_to_adj_bet` transpose *before* masking, giving `A_t = D A'` (rows of `A'`
   zeroed). `brava_centrality` transposed *after*, giving `(D A)' = A' D` (columns
   zeroed), so pruned vertices kept live in-features at inference.

Why the interaction matters: tau_b on these graphs is dominated by tie structure. The
clique mask prunes 18--96% of vertices, all with true betweenness exactly 0. With the
mask applied correctly and no PageRank channel, every pruned vertex gets identically
zero features and therefore **one shared score** -- verified: `1 distinct / 254838
pruned` on `email-EuAll` -- tying them exactly where the ground truth ties them. The
PageRank channel is not zeroed by the mask, so it hands each pruned vertex a distinct
value and shatters that block into an arbitrary order.

## Result

Mean tau_b 87.3 against the paper's 87.7 (mean delta -0.4, mean |delta| 1.69) over three
seeds. See `logs/eval_seeds.log`.

**tau_b is stable across seeds (mean per-graph std 0.75, the paper's is 0.52). Top-100
overlap is not** -- mean std 4.2, and 13.2 on `p2p-Gnutella31`, which gave 43 / 52 / 26.
Any overlap figure quoted from a single seed should be treated as one draw, not a
reproducible measurement.

## Layout

| path | contents |
| --- | --- |
| `bvk_{kadabra,brava_cpu,brava_gpu}_<graph>.json` | 27 runs, eps=1e-2, delta=0.1, k=0, t=8, seed-1 model |
| `kadabra_seeds/bvk_kadabra_<graph>_s{1,2,3}.json` | 27 KADABRA repeats at eps=1e-2, for the +/- on its columns |
| `weights/bravagnn_weights.jld2` | canonical seed-1 checkpoint (+ `_S2`, `_S3`) |
| `weights/*.config.txt` | training configuration per checkpoint |
| `logs/eval_seeds.log` | three-seed evaluation against the paper's Table 2 |
| `logs/diag_mask.log`, `logs/diag_nopr.log` | the 2x2 diagnostic above |
| `logs/rerun_brava.log` | retrain + bvk + seed sweep transcript |

Only the `bvk_*` runs at the top level carry quotable wall-clock times: that stage had the
machine to itself, whereas the repeats did not. The repeats bound accuracy only.

## Both columns need error bars, and they do not mean the same thing

KADABRA samples shortest paths, so repeated runs of one binary differ; BRAVA-GNN
inference is deterministic given a checkpoint, and its spread comes from the *training*
seed. Reporting `+/-` for one and not the other would imply the other is deterministic.
`summarize_seeds.py` computes both, and the caption has to say which is which.

With three runs each, **KADABRA wins top-100 overlap on 9 of 9 graphs and BRAVA-GNN wins
tau_b on 9 of 9** -- the two fail in opposite directions, cleanly. A single draw had
suggested BRAVA-GNN took the overlap on `soc-Slashdot0902` (92 against 88); across three
runs it is BRAVA-GNN 90 +/- 2.6 against KADABRA 92 +/- 1.7, i.e. KADABRA wins and the gap
is inside one standard deviation either way. `summarize_seeds.py` flags such pairs.

The error bars matter in exactly one place: BRAVA-GNN's top-100 overlap on
`p2p-Gnutella31` is `40 +/- 13` (43 / 52 / 26 across seeds). Quoting that cell as a bare
integer claims precision the model does not have. Every other cell is stable.

### A trap that cost one full run of this data

`-t N` on `run_experiments.jl` is recorded for logging and sets nothing; the thread count
is whatever Julia started with, which `reproduce_report.sh` controls by exporting
`JULIA_NUM_THREADS`. Invoking the runner directly with only `-t 8` yields a
single-threaded run whose JSON is perfectly valid. It is not merely slower: fewer workers
overshoot the shared stopping condition by less, so KADABRA draws ~30% fewer samples and
lands 0.02--0.05 lower in tau_b -- which reads exactly like a real effect. The first set
of repeats was discarded for this reason. `parameters.threads` records the *actual*
count, so it is the field to check; the runner now also warns when `-t` exceeds it.

The `bvk_*.json` files sit at the top level so this directory can be passed straight to
`summarize_reproduce.py`. KADABRA's runs were re-executed unchanged and agree with
`../report_runs` to within run-to-run timing noise.

## Regenerating

Training is not bit-reproducible (cuDNN and the sparse matmul reduce nondeterministically)
but seeding fixes initialisation and pair sampling.

```bash
julia --project=. src/train_bravagnn.jl --seed=1     # ~16 min on an RTX 2080 Ti
cd benchmark && ./reproduce_report.sh --stage bvk --threads 8 --outdir <dir>
python3 summarize_reproduce.py bvk <dir>

julia --project=. benchmark/eval_brava_paper.jl benchmark/cache/bravagnn_weights.jld2
```

The tight-epsilon claim of Section 6.4 is checked mechanically by placing
`../report_runs/tight_julia_*.json` and these `bvk_*.json` in one directory and running
`summarize_reproduce.py tightx <dir>`. It reports:

```
KADABRA wins BOTH metrics on 8/9 graphs
!! CLAIM DOES NOT HOLD --- tau_b lost on: email-EuAll; overlap lost on: none
```

`email-EuAll` at eps=1e-4: KADABRA 0.898 against BRAVA-GNN 0.991. The claim in Section
6.4 and the Conclusion that KADABRA beats BRAVA-GNN on *both* metrics on *every*
instance held only while BRAVA-GNN was broken on that graph, and must be softened.
