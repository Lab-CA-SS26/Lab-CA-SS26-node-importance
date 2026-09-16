#!/usr/bin/env bash
#
# run_seed_check.sh --- the seed experiment from TODO.md:
#   "why does Julia draw more samples than the C++ reference?"
#
# For each graph and each seed, run six arms at eps=1e-4, delta=0.1, k=0, 8 threads:
#   julia   arm 1: Julia as is (default check interval max(1000, tau/10))
#   cpp     arm 2: the C++ reference (checks every 11 samples per thread)
#   ci11    arm 3: Julia with --check-interval 11 (fine-grained checks)
#   seq     arm 4: Julia with --no-parallel (single sampling stream)
#   sib     arm 5: Julia with --stop-in-batch (workers test the stop flag before every sample)
#   fix     arm 6: arm 5 plus --consistent-pairs (checks count unfinished batches too)
#
# Same seeds for every arm. Raw per-run JSON lands in results/seed_check/.
#
# Usage:
#   ./run_seed_check.sh [--threads N] [--seeds "1 2 3 4 5"] [--outdir DIR] [--big]
#
#   --big   also run amazon and dblp (2-3 h per run; off by default)
#
# Run inside tmux on the server, and give the machine to nothing else while it runs:
#   tmux new-session -d -s seedcheck './run_seed_check.sh 2>&1 | tee ~/seedcheck.log'
set -euo pipefail

THREADS=8
SEEDS="1 2 3 4 5"
OUTDIR="$(pwd)/results/seed_check"
BIG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --big)     BIG=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"
JL_RUNNER=simexpal_runners/run_experiments.jl
CPP_RUNNER=simexpal_runners/run_experiments_cpp
INST=../Instances

for f in "$JL_RUNNER" "$CPP_RUNNER"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[[ -x "$CPP_RUNNER" ]] || { echo "ERROR: $CPP_RUNNER is not executable" >&2; exit 1; }

mkdir -p "$OUTDIR"
export JULIA_NUM_THREADS="$THREADS"

echo "run_seed_check.sh: threads=$THREADS seeds=[$SEEDS] outdir=$OUTDIR big=$BIG"
echo

# The C++ parser strips leading dashes and looks up key "d"; --directed is never read and
# the graph loads UNDIRECTED with no warning. Only -d works. Julia specs carry --directed,
# so translate here (see reproduce_report.sh).
cpp_dflag() { [[ -n "${1:-}" ]] && echo "-d" || echo ""; }

# julia_arm <out.json> <graph> <dflag> <seed> [extra runner args...]
julia_arm() {
    local out="$1" graph="$2" dflag="$3" seed="$4"; shift 4
    [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; return 0; }
    echo "[$(date +%H:%M:%S)] $(basename "$out")"
    julia --project=.. "$JL_RUNNER" ${dflag:-} -i "$graph" -o "$out" \
        -a kadabra -v julia --epsilon 0.0001 --delta 0.1 -k 0 -s "$seed" -t "$THREADS" "$@" >/dev/null
}

# cpp_arm <out.json> <graph> <dflag> <seed>
cpp_arm() {
    local out="$1" graph="$2" dflag="$3" seed="$4"
    [[ -f "$out" ]] && { echo "  skip (exists): $(basename "$out")"; return 0; }
    echo "[$(date +%H:%M:%S)] $(basename "$out")"
    "$CPP_RUNNER" $(cpp_dflag "$dflag") -i "$graph" -o "$out" \
        -threads "$THREADS" -k 0 -delta 0.1 -epsilon 0.0001 -a kadabra -v cpp -s "$seed" >/dev/null
}

specs=(
    "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
    "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
    "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
)
if [[ "$BIG" == "1" ]]; then
    specs+=(
        "amazon           $INST/ABCDE/amazon.txt                   "
        "dblp             $INST/ABCDE/dblp.txt                     "
    )
fi

for spec in "${specs[@]}"; do
    read -r name path dflag <<<"$spec"
    echo "=== $name ==="
    for seed in $SEEDS; do
        cpp_arm    "$OUTDIR/sc_cpp_${name}_s${seed}.json"   "$path" "${dflag:-}" "$seed"
        julia_arm  "$OUTDIR/sc_julia_${name}_s${seed}.json" "$path" "${dflag:-}" "$seed"
        julia_arm  "$OUTDIR/sc_ci11_${name}_s${seed}.json"  "$path" "${dflag:-}" "$seed" --check-interval 11
        julia_arm  "$OUTDIR/sc_sib_${name}_s${seed}.json"   "$path" "${dflag:-}" "$seed" --stop-in-batch
        julia_arm  "$OUTDIR/sc_fix_${name}_s${seed}.json"   "$path" "${dflag:-}" "$seed" --stop-in-batch --consistent-pairs
        julia_arm  "$OUTDIR/sc_seq_${name}_s${seed}.json"   "$path" "${dflag:-}" "$seed" --no-parallel
    done
    echo
done

echo "Done. Raw per-run JSON is in $OUTDIR"
echo "Summarize with: python3 summarize_seed_check.py $OUTDIR"
