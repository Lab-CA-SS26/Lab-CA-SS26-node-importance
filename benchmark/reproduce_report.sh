#!/usr/bin/env bash
#
# reproduce_report.sh --- regenerate every measurement reported in the lab report.
#
# Produces the two result tables:
#   Table "cpp_vs_julia_tight"  (Report section 6.1)  -- C++ vs Julia KADABRA at eps=1e-4
#   Table "brava_vs_kadabra"    (Report section 6.2)  -- BRAVA-GNN vs KADABRA at eps=1e-2
#
# and prints, for each, the LaTeX table body plus the summary statistics quoted
# in the report text.
#
# Usage:
#   ./reproduce_report.sh [--stage tight|bvk|all] [--threads N] [--outdir DIR]
#
#   --stage tight   only the eps=1e-4 C++ vs Julia comparison  (~6h, dominated by amazon/dblp)
#   --stage bvk     only the eps=1e-2 BRAVA vs KADABRA comparison (~20 min)
#   --stage k       only the top-k runtime sweep, Julia at eps=1e-4 (hours; reuses tight's k=0)
#   --stage all     all three (default)
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
            julia --project=.. "$JL_RUNNER" ${dflag:-} -i "$path" -o "$out" \
                -a kadabra -v julia --epsilon 0.0001 --delta 0.1 -k "$k" -s 0 -t "$THREADS" >/dev/null
        done
    done
    echo
    python3 summarize_reproduce.py k "$OUTDIR"
}

case "$STAGE" in
    tight) run_tight ;;
    bvk)   run_bvk ;;
    k)     run_k ;;
    all)   run_tight; echo; run_bvk; echo; run_k ;;
    *) echo "unknown stage: $STAGE (expected tight|bvk|k|all)" >&2; exit 2 ;;
esac

echo
echo "Done. Raw per-run JSON is in $OUTDIR"
echo "Re-running is incremental: existing .json files are skipped, so delete a"
echo "file (or the whole directory) to force that measurement to be recomputed."
