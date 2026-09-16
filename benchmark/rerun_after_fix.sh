#!/usr/bin/env bash
#
# rerun_after_fix.sh --- re-measure every KADABRA number the report quotes, with the
# stopping-coordination fix of 2026-09-16 (stop_in_batch, consistent_pairs) as the default.
#
# Runs on the server, in tmux, after the seed sweep (tmux session `seedcheck`) has finished:
#   tmux new-session -d -s rerun '~/Lab-CA-SS26-node-importance/benchmark/rerun_after_fix.sh 2>&1 | tee -a ~/rerun_fix.log'
#
# Order: cheapest and most central first, so each report section can be updated as soon as
# its stage lands, and the two multi-day stages come last.
#   A  tight   Section 6.1 table, A.7          Julia, 6 graphs, eps=1e-4          ~6 h
#   B  ts      Section 6.1 thread scaling      Julia, 4 graphs x 8 threads x 3    ~2 h
#   C  bvk     Section 6.4 table + Figure 3    KADABRA eps=1e-2, seed 0 and 1-3   ~1 h
#   D  topk    Section 6.3, A.5                stages kx, kxs, kxb (4 graphs)      ~8 h
#   E  tightx  Section 6.4 eps=1e-4 column     com-youtube, com-lj, cit-Patents   ~18 h
#   F  kxl     Section 6.3 amazon/dblp         k in {0,10,100} x 3 seeds          ~40 h
#
# Not re-measured, because the fix does not touch them: every C++ run (tight_cpp_*, the
# C++ thread-scaling runs) and every BRAVA-GNN run. They are copied from $SEED_DIR into the
# output directories so the stage scripts skip them. The single-seed `k` stage and the
# `bias_fix` runs are not used by the report any more and are not repeated.
#
# Every stage is incremental (existing JSON is skipped), so the script can be re-launched.
set -uo pipefail

ROOT="$HOME/Lab-CA-SS26-node-importance"
B="$ROOT/benchmark"
OUT="${OUT:-$HOME/rerun_fix}"
SEED_DIR="${SEED_DIR:-$HOME/rerun_fix_seed}"
INST="$ROOT/Instances"
JL="simexpal_runners/run_experiments.jl"

log() { echo "[$(date '+%F %T')] $*"; }

# verify_dir <dir> <glob> : every Julia KADABRA JSON must carry the fix and a sane config
verify_dir() {
    python3 - "$1" "$2" <<'EOF'
import glob, json, os, sys
d, pat = sys.argv[1], sys.argv[2]
directed = {"soc-Epinions1", "soc-Slashdot0902", "email-EuAll"}
bad = n = 0
for f in sorted(glob.glob(os.path.join(d, pat))):
    j = json.load(open(f)); p = j["parameters"]
    if p.get("version") != "julia" or p.get("algorithm") != "kadabra":
        continue
    n += 1
    b = os.path.basename(f)
    probs = []
    if p.get("stop_in_batch") is not True: probs.append(f"stop_in_batch={p.get('stop_in_batch')}")
    if p.get("consistent_pairs") is not True: probs.append(f"consistent_pairs={p.get('consistent_pairs')}")
    if any(g in b for g in directed) and not p.get("directed"): probs.append("directed=false")
    if "_t" not in b and p.get("threads") != 8: probs.append(f"threads={p.get('threads')}")
    if probs:
        bad += 1; print("  !! " + b + ": " + ", ".join(probs))
print(f"  verified {n} Julia KADABRA runs in {d} ({pat}), {bad} with problems")
EOF
}

stage() {  # stage <name> <command...>
    local name="$1"; shift
    log "=== stage $name: start ==="
    if "$@"; then log "=== stage $name: done ==="; else log "=== stage $name: FAILED (exit $?) ==="; fi
}

# ---------------------------------------------------------------------------
# 0. wait for the seed sweep, then deploy the new default
# ---------------------------------------------------------------------------
if tmux has-session -t seedcheck 2>/dev/null; then
    log "waiting for tmux session 'seedcheck' to finish"
    while tmux has-session -t seedcheck 2>/dev/null; do sleep 60; done
fi
last_exit="$(grep '^EXIT_' "$HOME/seedcheck.log" | tail -1)"
if [[ "$last_exit" != "EXIT_0" ]]; then
    log "seed sweep ended with '${last_exit:-no exit marker}'; not deploying. Fix it and re-launch."
    exit 1
fi
log "seed sweep finished cleanly"

for f in src/kadabra.jl benchmark/simexpal_runners/run_experiments.jl benchmark/run_seed_check.sh; do
    if [[ -f "$ROOT/$f.pending" ]]; then mv "$ROOT/$f.pending" "$ROOT/$f"; log "deployed $f"; fi
done
grep -q "stop_in_batch::Bool = true" "$ROOT/src/kadabra.jl" || { log "src/kadabra.jl does not carry the fix default; abort"; exit 1; }

cd "$B"
export JULIA_NUM_THREADS=8
smoke="$(mktemp -d)"
julia --project=.. "$JL" -i "$INST/TestInstances/p2p-Gnutella31.txt" -o "$smoke/smoke.json" \
    -a kadabra -v julia --epsilon 0.01 --delta 0.1 -k 0 -s 1 -t 8 >/dev/null 2>&1 \
    || { log "smoke run failed; abort"; exit 1; }
verify_dir "$smoke" "smoke.json" | tee /dev/stderr | grep -q ", 0 with problems" \
    || { log "smoke run does not carry the fix; abort"; exit 1; }
rm -rf "$smoke"
log "smoke run carries the fix"

mkdir -p "$OUT"/{tight,ts,bvk/kadabra_seeds,topk}

# ---------------------------------------------------------------------------
# A. Section 6.1: Julia vs C++ at eps = 1e-4 (C++ reused)
# ---------------------------------------------------------------------------
run_tight() {
    cp -n "$SEED_DIR"/tight_cpp_*.json "$OUT/tight/"
    ./reproduce_report.sh --stage tight --outdir "$OUT/tight"
    verify_dir "$OUT/tight" "tight_julia_*.json"
}
stage A-tight run_tight

# ---------------------------------------------------------------------------
# B. Section 6.1: thread scaling at eps = 1e-2 (C++ reused); ~/run_thread_scaling.sh
# ---------------------------------------------------------------------------
run_ts() {
    cp -n "$SEED_DIR"/cpp_*_t*_s*.json "$OUT/ts/"
    local specs=(
        "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
        "amazon           $INST/ABCDE/amazon.txt                   "
        "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
        "com-lj           $INST/ABCDE/com-lj.txt                   "
    )
    for spec in "${specs[@]}"; do
        read -r name path dflag <<<"$spec"
        for t in 1 2 4 8 16 24 32 48; do
            for s in 0 1 2; do
                local f="$OUT/ts/julia_${name}_t${t}_s${s}.json"
                [[ -f "$f" ]] && continue
                log "ts julia $name t=$t s=$s"
                JULIA_NUM_THREADS=$t julia --project=.. "$JL" ${dflag:-} -i "$path" -o "$f" \
                    -a kadabra -v julia --epsilon 0.01 --delta 0.1 -k 0 -s $s -t $t >/dev/null 2>&1 \
                    || log "FAILED ts julia $name t=$t s=$s"
            done
        done
    done
    verify_dir "$OUT/ts" "julia_*.json"
    python3 - "$OUT/ts" <<'EOF'
import glob, json, os, re, sys
bad = [os.path.basename(f) for f in glob.glob(os.path.join(sys.argv[1], "julia_*.json"))
       if json.load(open(f))["parameters"]["threads"] != int(re.search(r"_t(\d+)_", f).group(1))]
print(f"  thread-count mismatches: {bad if bad else 'none'}")
EOF
}
stage B-thread-scaling run_ts

# ---------------------------------------------------------------------------
# C. Section 6.4: KADABRA at eps = 1e-2, seed 0 (bvk stage) and seeds 1-3 (BRAVA-GNN reused)
# ---------------------------------------------------------------------------
run_bvk() {
    cp -n "$SEED_DIR"/bvk_brava_*.json "$OUT/bvk/"
    ./reproduce_report.sh --stage bvk --outdir "$OUT/bvk"
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
    for s in 1 2 3; do
        for spec in "${specs[@]}"; do
            read -r name path dflag <<<"$spec"
            local f="$OUT/bvk/kadabra_seeds/bvk_kadabra_${name}_s${s}.json"
            [[ -f "$f" ]] && continue
            log "bvk kadabra $name s=$s"
            julia --project=.. "$JL" ${dflag:-} -i "$path" -o "$f" -a kadabra -v julia \
                --epsilon 0.01 --delta 0.1 -k 0 -s $s -t 8 >/dev/null 2>&1 || log "FAILED bvk kadabra $name s=$s"
        done
    done
    verify_dir "$OUT/bvk" "bvk_kadabra_*.json"
    verify_dir "$OUT/bvk/kadabra_seeds" "bvk_kadabra_*.json"
}
stage C-bvk run_bvk

# ---------------------------------------------------------------------------
# D. Section 6.3 and A.5: top-k sweeps on the four small graphs
# ---------------------------------------------------------------------------
run_topk_small() {
    ./reproduce_report.sh --stage kx  --outdir "$OUT/topk"
    ./reproduce_report.sh --stage kxs --outdir "$OUT/topk"
    ./reproduce_report.sh --stage kxb --outdir "$OUT/topk"
    verify_dir "$OUT/topk" "kx_*.json"
}
stage D-topk run_topk_small

# ---------------------------------------------------------------------------
# E. Section 6.4: KADABRA at eps = 1e-4 on the three largest graphs
# ---------------------------------------------------------------------------
run_tightx() {
    ./reproduce_report.sh --stage tightx --outdir "$OUT/tight"
    verify_dir "$OUT/tight" "tight_julia_*.json"
}
stage E-tightx run_tightx

# ---------------------------------------------------------------------------
# F. Section 6.3: amazon and dblp top-k (multi-day)
# ---------------------------------------------------------------------------
run_kxl() {
    ./reproduce_report.sh --stage kxl --outdir "$OUT/topk"
    verify_dir "$OUT/topk" "kx_*.json"
}
stage F-kxl run_kxl

log "=== ALL STAGES FINISHED ==="
