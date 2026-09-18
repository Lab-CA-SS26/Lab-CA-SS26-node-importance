#!/usr/bin/env bash
#
# rerun_cpp_sameday.sh --- re-time the C++ reference in the same week as the Julia rerun.
#
# rerun_after_fix.sh reuses the August C++ runs, which is fine for sample counts (the C++
# source is unchanged since 2026-07-22) but makes Section 6.1's Julia/C++ *time* ratios compare
# runs from different weeks. The seed experiment showed day-to-day variance of ~10% on this
# machine, so this re-times both C++ sets Section 6.1 uses, after the rerun has finished:
#   tight   6 graphs, eps=1e-4, 8 threads, seed 0 (as reproduce_report.sh --stage tight)
#   ts      4 graphs x t in {1..48} x seeds {0,1,2}, eps=1e-2 (as ~/run_thread_scaling.sh)
#
#   tmux new-session -d -s cppsd '~/Lab-CA-SS26-node-importance/benchmark/rerun_cpp_sameday.sh 2>&1 | tee -a ~/rerun_cpp_sameday.log'
set -uo pipefail

B="$HOME/Lab-CA-SS26-node-importance/benchmark"
INST="$HOME/Lab-CA-SS26-node-importance/Instances"
OUT="${OUT:-$HOME/rerun_fix}"
CPP="simexpal_runners/run_experiments_cpp"
log() { echo "[$(date '+%F %T')] $*"; }

if tmux has-session -t rerun 2>/dev/null; then
    log "waiting for tmux session 'rerun' to finish"
    while tmux has-session -t rerun 2>/dev/null; do sleep 60; done
fi
cd "$B"
mkdir -p "$OUT/tight_cpp_sameday" "$OUT/ts_cpp_sameday"

# The C++ runner reads only -d; --directed is silently ignored (see CLAUDE.md).
tight_specs=(
    "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
    "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    -d"
    "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt -d"
    "email-EuAll      $INST/TestInstances/email-EuAll.txt      -d"
    "amazon           $INST/ABCDE/amazon.txt                   "
    "dblp             $INST/ABCDE/dblp.txt                     "
)
log "=== tight C++ ==="
for spec in "${tight_specs[@]}"; do
    read -r name path dflag <<<"$spec"
    f="$OUT/tight_cpp_sameday/tight_cpp_${name}.json"
    [[ -f "$f" ]] && continue
    log "tight cpp $name"
    "$CPP" ${dflag:-} -i "$path" -o "$f" -threads 8 -k 0 -delta 0.1 -epsilon 0.0001 \
        -a kadabra -v cpp -s 0 >/dev/null 2>&1 || log "FAILED tight cpp $name"
done

ts_specs=(
    "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt -d"
    "amazon           $INST/ABCDE/amazon.txt                   "
    "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
    "com-lj           $INST/ABCDE/com-lj.txt                   "
)
log "=== thread-scaling C++ ==="
for spec in "${ts_specs[@]}"; do
    read -r name path dflag <<<"$spec"
    for t in 1 2 4 8 16 24 32 48; do
        for s in 0 1 2; do
            f="$OUT/ts_cpp_sameday/cpp_${name}_t${t}_s${s}.json"
            [[ -f "$f" ]] && continue
            "$CPP" ${dflag:-} -i "$path" -o "$f" -threads $t -k 0 -delta 0.1 -epsilon 0.01 \
                -a kadabra -v cpp -s $s >/dev/null 2>&1 || log "FAILED ts cpp $name t=$t s=$s"
        done
    done
done

python3 - "$OUT" <<'EOF'
import glob, json, os, sys
directed = {"soc-Epinions1", "soc-Slashdot0902", "email-EuAll"}
bad = n = 0
for f in glob.glob(os.path.join(sys.argv[1], "*_cpp_sameday", "*.json")):
    n += 1; p = json.load(open(f))["parameters"]; b = os.path.basename(f)
    if any(g in b for g in directed) and not p.get("directed"):
        bad += 1; print("  !! " + b + ": directed=false")
print(f"  verified {n} C++ runs, {bad} with problems")
EOF
log "=== C++ same-day rerun finished ==="
