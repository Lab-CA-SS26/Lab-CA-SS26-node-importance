#!/usr/bin/env bash
#
# reproduce_report.sh --- regenerate every measurement reported in the lab report.
#
# Produces the two result tables:
#   Table "cpp_vs_julia_tight"  (Report section 6.1)  -- C++ vs Julia KADABRA at eps=1e-4
#   Table "brava_vs_kadabra"    (Report section 6.2)  -- BRAVA-GNN vs KADABRA at eps=1e-2
#
# and the tight-epsilon accuracy comparison of Section 6.4 (stage 'tightx'),
#
# and prints, for each, the LaTeX table body plus the summary statistics quoted
# in the report text.
#
# Usage:
#   ./reproduce_report.sh [--stage tight|tightx|bvk|k|kx|kxs|kxb|kxl|all] [--threads N] [--outdir DIR]
#
#   --stage tight   only the eps=1e-4 C++ vs Julia comparison  (~6h, dominated by amazon/dblp)
#   --stage tightx  only the eps=1e-4 Julia runs on the three largest bvk graphs (~23h)
#   --stage bvk     only the eps=1e-2 BRAVA vs KADABRA comparison (~20 min)
#   --stage k       only the top-k runtime sweep, Julia at eps=1e-4 (hours; reuses tight's k=0)
#   --stage kx      multi-seed top-k sweep x budget-allocation variant, 4 small graphs (~3h)
#   --stage kxs     the same variants at k=3 and k=5, where they actually differ (~3h)
#   --stage kxl     the repaired allocation on amazon/dblp, k in {10,100} (~40h)
#   --stage all     all of the above except kxl (default)
#
# Notes:
#   * Run this from the `benchmark/` directory of a checkout that has the
#     Instances/ tree populated (see ../README.md).
#   * The tight stage takes hours. Launch it inside tmux/screen so it survives
#     a dropped SSH session, e.g.
#         tmux new-session -d -s repro './reproduce_report.sh 2>&1 | tee ~/repro.log'
#   * Julia threading is controlled by JULIA_NUM_THREADS, which this script sets
#     explicitly. The runner's own -t flag is only recorded in the output JSON;
#     it does NOT configure Julia's thread pool.
#   * Both implementations are pinned to the SAME thread count on purpose: the
#     size of KADABRA's internal tracking set is derived from the thread count,
#     so an asymmetric comparison would conflate throughput with workload.
#
set -euo pipefail

THREADS=8
OUTDIR="$(pwd)/results/reproduce"
STAGE=all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)   STAGE="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"
JL_RUNNER=simexpal_runners/run_experiments.jl
CPP_RUNNER=simexpal_runners/run_experiments_cpp
INST=../Instances

for f in "$JL_RUNNER" "$CPP_RUNNER"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f (build the C++ runner with 'make -C simexpal_runners')" >&2; exit 1; }
done
[[ -x "$CPP_RUNNER" ]] || { echo "ERROR: $CPP_RUNNER is not executable" >&2; exit 1; }

mkdir -p "$OUTDIR"
export JULIA_NUM_THREADS="$THREADS"

echo "reproduce_report.sh: threads=$THREADS outdir=$OUTDIR stage=$STAGE"
echo

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# julia_run <out.json> <graph> <directed-flag> <algo> <epsilon> [extra args...]
julia_run() {
    local out="$1" graph="$2" dflag="$3" algo="$4" eps="$5"; shift 5
    [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; return 0; }
    julia --project=.. "$JL_RUNNER" $dflag -i "$graph" -o "$out" \
        -a "$algo" -v julia --epsilon "$eps" --delta 0.1 -k 0 -s 0 -t "$THREADS" "$@" >/dev/null
}

# cpp_run <out.json> <graph> <directed-flag> <epsilon>
cpp_run() {
    local out="$1" graph="$2" dflag="$3" eps="$4"
    [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; return 0; }
    "$CPP_RUNNER" $dflag -i "$graph" -o "$out" \
        -threads "$THREADS" -k 0 -delta 0.1 -epsilon "$eps" -a kadabra -v cpp -s 0 >/dev/null
}

# kx_run <out.json> <graph> <directed-flag> <k> <variant> <seed>
kx_run() {
    local out="$1" graph="$2" dflag="$3" k="$4" variant="$5" seed="$6"
    [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; return 0; }
    echo "[$(date +%H:%M:%S)] $(basename "$out")"
    julia --project=.. "$JL_RUNNER" $dflag -i "$graph" -o "$out" \
        -a kadabra -v julia --epsilon 0.0001 --delta 0.1 -k "$k" -s "$seed" \
        -t "$THREADS" --topk-variant "$variant" >/dev/null
}

# ---------------------------------------------------------------------------
# Stage 1: C++ vs Julia KADABRA at eps = 1e-4  (report Section 6.1)
# ---------------------------------------------------------------------------
# Fields consumed: execution_time_seconds, num_samples, samples_over_omega,
#                  tau_overall, overlap_topk.
run_tight() {
    echo "=== Stage 'tight': C++ vs Julia KADABRA, eps=1e-4 ==="
    local specs=(
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "amazon           $INST/ABCDE/amazon.txt                   "
        "dblp             $INST/ABCDE/dblp.txt                     "
    )
    for spec in "${specs[@]}"; do
        read -r name path dflag <<<"$spec"
        echo "[$(date +%H:%M:%S)] $name (cpp)"
        cpp_run   "$OUTDIR/tight_cpp_$name.json"   "$path" "${dflag:-}" 0.0001
        echo "[$(date +%H:%M:%S)] $name (julia)"
        julia_run "$OUTDIR/tight_julia_$name.json" "$path" "${dflag:-}" kadabra 0.0001
    done
    echo
    python3 summarize_reproduce.py tight "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 1b: KADABRA at eps = 1e-4 on the three remaining 'bvk' graphs
#           (report Section 6.4's tight-epsilon column)
# ---------------------------------------------------------------------------
# Julia only, and no C++ counterpart: these runs exist to complete the accuracy
# comparison against BRAVA-GNN, not the implementation comparison of Section 6.1.
# They are by far the longest measurements in the report --- roughly 2h, 10h and
# 11h respectively on 8 threads --- so they are a separate stage from 'tight'.
run_tightx() {
    echo "=== Stage 'tightx': KADABRA eps=1e-4 on the three largest bvk graphs ==="
    local specs=(
        "com-youtube      $INST/ABCDE/com-youtube.txt              "
        "com-lj           $INST/ABCDE/com-lj.txt                   "
        "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
    )
    for spec in "${specs[@]}"; do
        read -r name path dflag <<<"$spec"
        echo "[$(date +%H:%M:%S)] $name (julia)"
        julia_run "$OUTDIR/tight_julia_$name.json" "$path" "${dflag:-}" kadabra 0.0001
    done
    echo
    # Needs stage 'bvk' for BRAVA-GNN's side of the comparison.
    python3 summarize_reproduce.py tightx "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 2: BRAVA-GNN vs KADABRA at eps = 1e-2  (report Section 6.2)
# ---------------------------------------------------------------------------
# Restricted to the 9 graphs with a validated ground truth (report Appendix A.1).
run_bvk() {
    echo "=== Stage 'bvk': BRAVA-GNN vs KADABRA, eps=1e-2 ==="
    local specs=(
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "com-youtube      $INST/ABCDE/com-youtube.txt              "
        "amazon           $INST/ABCDE/amazon.txt                   "
        "dblp             $INST/ABCDE/dblp.txt                     "
        "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
        "com-lj           $INST/ABCDE/com-lj.txt                   "
    )
    for spec in "${specs[@]}"; do
        read -r name path dflag <<<"$spec"
        echo "[$(date +%H:%M:%S)] $name"
        julia_run "$OUTDIR/bvk_kadabra_$name.json"   "$path" "${dflag:-}" kadabra 0.01
        julia_run "$OUTDIR/bvk_brava_cpu_$name.json" "$path" "${dflag:-}" brava   0.01
        julia_run "$OUTDIR/bvk_brava_gpu_$name.json" "$path" "${dflag:-}" brava   0.01 --gpu
    done
    echo
    python3 summarize_reproduce.py bvk "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 3: effect of k on runtime  (report Section 6.4)
# ---------------------------------------------------------------------------
# Same instances and epsilon as stage 'tight', Julia only. k=0 is not re-run:
# stage 'tight' already produced it under the same parameters.
run_k() {
    echo "=== Stage 'k': effect of top-k on runtime, eps=1e-4, Julia ==="
    local specs=(
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
        "amazon           $INST/ABCDE/amazon.txt                   "
        "dblp             $INST/ABCDE/dblp.txt                     "
    )
    for spec in "${specs[@]}"; do
        read -r name path dflag <<<"$spec"
        # k=0 comes from stage 'tight'; link it in so the summary sees all three
        [[ -f "$OUTDIR/tight_julia_$name.json" && ! -f "$OUTDIR/k_julia_${name}_k0.json" ]] && \
            cp "$OUTDIR/tight_julia_$name.json" "$OUTDIR/k_julia_${name}_k0.json"
        for k in 10 100; do
            local out="$OUTDIR/k_julia_${name}_k${k}.json"
            [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; continue; }
            echo "[$(date +%H:%M:%S)] $name k=$k"
            # Pinned to the C++ reference's budget allocation on purpose: this stage
            # reproduces the single-seed sweep as first reported, before stage 'kx'
            # showed the paper's allocation to be cheaper and made it the default.
            julia --project=.. "$JL_RUNNER" ${dflag:-} -i "$path" -o "$out" \
                -a kadabra -v julia --epsilon 0.0001 --delta 0.1 -k "$k" -s 0 -t "$THREADS" \
                --topk-variant code >/dev/null
        done
    done
    echo
    python3 summarize_reproduce.py k "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 4: multi-seed top-k sweep + paper-vs-code budget allocation (Section 6.3)
# ---------------------------------------------------------------------------
# Two questions at once:
#   (a) is the "top-k sometimes costs more" effect of stage 'k' real, or one seed's
#       noise?  -> three seeds per configuration.
#   (b) does the disagreement between the KADABRA paper and its C++ reference over
#       which rank gap sizes which side of a vertex's confidence interval matter?
#       -> three variants:
#            code   the reference's budget allocation, paper's external-exclusion test
#                   (what every other measurement in this report uses)
#            paper  the paper's allocation throughout (self-consistent with the test)
#            cpp    the reference verbatim, g(v_k) in place of f(v_k)
# k=0 does not use the top-k branch at all, so it is run once per seed, not per variant.
run_kx() {
    echo "=== Stage 'kx': multi-seed top-k sweep x budget-allocation variant, eps=1e-4 ==="
    local specs=(
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    )
    for seed in 1 2 3; do
        for spec in "${specs[@]}"; do
            read -r name path dflag <<<"$spec"
            kx_run "$OUTDIR/kx_${name}_k0_code_s${seed}.json"  "$path" "${dflag:-}" 0   code  "$seed"
            for k in 10 100; do
                for variant in code paper cpp; do
                    kx_run "$OUTDIR/kx_${name}_k${k}_${variant}_s${seed}.json" \
                           "$path" "${dflag:-}" "$k" "$variant" "$seed"
                done
            done
        done
    done
    echo
    python3 summarize_topk.py "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 4b: the same multi-seed top-k sweep on the two large 'tight' graphs
# ---------------------------------------------------------------------------
# Gives Section 6.3's figure error bars on all six instances, and checks that the
# boundary-pair repair still removes the anomaly on the two large graphs -- `dblp` is
# where the original single-seed sweep put it at 1.06x. These two graphs take 2-3 h
# per run, so the stage is a multi-day one, written seed-outermost: a complete extra
# seed lands before the next one starts.
run_kxl() {
    echo "=== Stage 'kxl': the repaired allocation on amazon/dblp, eps=1e-4 ==="
    local specs=(
        "amazon           $INST/ABCDE/amazon.txt                   "
        "dblp             $INST/ABCDE/dblp.txt                     "
    )
    # Only the repaired allocation, and only k in {10,100}. These two graphs run 2-3 h
    # apiece, so the full k x variant grid of stages kx/kxs is out of reach; what this
    # buys is the one claim that needs the large instances -- that the repaired version
    # never rises above k=0 -- on six graphs instead of four. k=0 is variant-independent
    # and is the per-seed denominator, so it is run once per seed and kept under the
    # historical `_code_` filename that summarize_topk.py and make_plots.py expect.
    for seed in 1 2 3; do
        for spec in "${specs[@]}"; do
            read -r name path dflag <<<"$spec"
            kx_run "$OUTDIR/kx_${name}_k0_code_s${seed}.json" \
                   "$path" "${dflag:-}" 0 code "$seed"
            for k in 10 100; do
                kx_run "$OUTDIR/kx_${name}_k${k}_paper_bd_s${seed}.json" \
                       "$path" "${dflag:-}" "$k" paper_bd "$seed"
            done
        done
    done
    echo
    python3 summarize_topk.py "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 4c: small k, where the paper/reference disagreement actually bites
# ---------------------------------------------------------------------------
# The two allocations can only differ on ranks whose gap to their neighbour exceeds
# KADABRA's tie-collapse threshold sqrt(start_factor)*eps/4; below it both are reset
# to the uniform eps. On these graphs at eps=1e-4 that threshold is crossed around
# rank 5, so k=10 and k=100 are dominated by vertices the two variants treat
# identically. k=3 and k=5 are not (diagnose_topk_delta.jl shows why).
run_kxs() {
    echo "=== Stage 'kxs': small-k budget-allocation variants, eps=1e-4 ==="
    local specs=(
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    )
    for seed in 1 2 3; do
        for spec in "${specs[@]}"; do
            read -r name path dflag <<<"$spec"
            for k in 3 5; do
                for variant in code paper cpp; do
                    kx_run "$OUTDIR/kx_${name}_k${k}_${variant}_s${seed}.json" \
                           "$path" "${dflag:-}" "$k" "$variant" "$seed"
                done
            done
        done
    done
    echo
    python3 summarize_topk.py "$OUTDIR"
}

# ---------------------------------------------------------------------------
# Stage 4d: does closing the tie-collapse guard's boundary-pair hole remove the
#           "top-k costs more than k=0" anomaly?
# ---------------------------------------------------------------------------
# Two candidate repairs to the tie-collapse guard, which never covers the pair
# (v_k, v_{k+1}) that the exclusion test depends on:
#   paper_bd  collapse that one pair, as the guard already does for every other
#             adjacent pair inside the top k
#   paper_ex  re-anchor the guard's second arm from v_{k+1} onto v_k, the vertex
#             exclusion actually compares against (subsumes the pair above)
# Compared against the k=0 and `:paper` runs that stages kx and kxs already produced,
# so only the two new variants are run here.
run_kxb() {
    echo "=== Stage 'kxb': boundary-pair collapse, eps=1e-4, 3 seeds ==="
    local specs=(
        "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
        "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
        "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    )
    for seed in 1 2 3; do
        for spec in "${specs[@]}"; do
            read -r name path dflag <<<"$spec"
            for k in 3 5 10 100; do
                for variant in paper_bd paper_ex; do
                    kx_run "$OUTDIR/kx_${name}_k${k}_${variant}_s${seed}.json" \
                           "$path" "${dflag:-}" "$k" "$variant" "$seed"
                done
            done
        done
    done
    echo
    python3 summarize_topk.py "$OUTDIR"
}

case "$STAGE" in
    tight)  run_tight ;;
    tightx) run_tightx ;;
    bvk)    run_bvk ;;
    k)      run_k ;;
    kx)     run_kx ;;
    kxs)    run_kxs ;;
    kxb)    run_kxb ;;
    kxl)    run_kxl ;;
    all)    run_tight; echo; run_bvk; echo; run_tightx; echo; run_k; echo; run_kx; echo; run_kxs ;;
    *) echo "unknown stage: $STAGE (expected tight|tightx|bvk|k|kx|kxs|kxb|kxl|all)" >&2; exit 2 ;;
esac

echo
echo "Done. Raw per-run JSON is in $OUTDIR"
echo "Re-running is incremental: existing .json files are skipped, so delete a"
echo "file (or the whole directory) to force that measurement to be recomputed."
