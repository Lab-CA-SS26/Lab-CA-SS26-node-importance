#!/usr/bin/env bash
#
# reproduce_all.sh --- every measurement, table and figure of the lab report, in one run.
#
# This is the single entry point. It chains what was run piecemeal while the report was
# written (reproduce_report.sh's stages, rerun_after_fix.sh's thread-scaling and seed loops,
# the BRAVA-GNN seed evaluation, the C++ accuracy scoring of Appendix A.7) and then writes
# the tables, figures and every number quoted in the prose from the fresh output.
#
#   step         what                                              report          time (8 thr)
#   build        C++ runner, Julia environment                                     minutes
#   weights      BRAVA-GNN checkpoints (committed, or --retrain)   6.4             0 / ~50 min GPU
#   tight        C++ vs Julia KADABRA, eps=1e-4, 6 graphs    TIMED 6.1, Table 1    ~6 h
#   threads      thread scaling, C++ and Julia, 1..48 thr    TIMED 6.1, Figure 1   ~4 h
#   bvk          BRAVA-GNN vs KADABRA, eps=1e-2, 9 graphs    TIMED 6.4, Table 2    ~1 h
#   brava-seeds  the three checkpoints against the paper's Table 2    6.4, Table 2    ~30 min
#   topk         top-k sweeps (4 graphs x k x allocation x 3 seeds),  6.3, Fig 2, T 6 ~10 h
#                and the C++ binary in top-k mode
#   cpp-quality  C++ per-vertex output, scored against ground truth   A.7, Table 8   ~6 h
#   tightx       KADABRA eps=1e-4 on the three largest graphs         6.4, A.4 (T 5) ~18 h
#   kxl          top-k on amazon and dblp, 3 seeds                    6.3, Figure 2   ~40 h
#   report       tables, figures, and the numbers quoted in the prose
#
# Usage (on the benchmark server, from anywhere, inside tmux --- the whole run takes ~4 days):
#   tmux new-session -d -s repro './benchmark/reproduce_all.sh 2>&1 | tee -a ~/reproduce_all.log'
#
#   --out DIR        results tree (default ~/reproduce_all); laid out like benchmark/results/
#   --from STEP      start at STEP (earlier steps are assumed done)
#   --only STEP      run just STEP
#   --retrain        train the three BRAVA-GNN checkpoints instead of using the committed ones
#   --dry-run        print every command instead of running it
#   --from-archive   run only the report step, on the committed runs in benchmark/results/
#                    (seconds; checks that the tables and figures follow from the data)
#
# Every step is incremental: a run whose JSON already exists is skipped, so an interrupted
# run is resumed by launching the script again. Delete a file to force that measurement.
#
# Prerequisites: Instances/ populated (README.md, "Instances"), including
# Instances/ground_truth/ and Instances/Training/ for --retrain; the C++ reference under
# ../cpp_reference; a CUDA GPU for the BRAVA-GNN GPU runs and for --retrain.
#
# Output: $OUT/report/{tables,figures} and $OUT/claims/*.txt. If the Report repo is checked
# out next to benchmark/, its tables are copied in first, so $OUT/report/tables/*.tex are
# complete tables and `diff -r $OUT/report/tables ../Report/tables` compares them directly;
# otherwise each table is written as its bare body.
#
# Not reproduced here, and why:
#   * the exact ground truth itself (Appendix A.2): hours to days per graph with Brandes;
#     it is an input, fetched or recomputed as README.md describes.
#   * the single-seed k sweep and the pre-fix runs that the report cites as history only.
#   * bit-identical numbers: KADABRA's multithreaded sampling and BRAVA-GNN's GPU training
#     are not deterministic. Expect agreement within the seed spread the report states.
#
# Timed steps (tight, threads, bvk) need the machine to themselves.
set -uo pipefail

B="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$B")"
OUT="$HOME/reproduce_all"
FROM="" ONLY="" RETRAIN=0 DRY=0 ARCHIVE=0
STEPS=(build weights tight threads bvk brava-seeds topk cpp-quality tightx kxl report)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT="$2"; shift 2 ;;
        --from)    FROM="$2"; shift 2 ;;
        --only)    ONLY="$2"; shift 2 ;;
        --retrain) RETRAIN=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --from-archive) ARCHIVE=1; ONLY=report; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for s in "$FROM" "$ONLY"; do
    [[ -z "$s" || " ${STEPS[*]} " == *" $s "* ]] || { echo "unknown step: $s (${STEPS[*]})" >&2; exit 2; }
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

if (( ARCHIVE )); then
    # the archived runs, arranged in the layout the report step reads
    for pair in rerun_fix:rerun_fix kxc:topk_variant/measured cpp_quality:cpp_quality \
                brava_retrained:brava_retrained; do
        [[ -e "$OUT/${pair%%:*}" ]] || ln -s "$B/results/${pair#*:}" "$OUT/${pair%%:*}"
    done
fi
RF="$OUT/rerun_fix"          # the name update_report_from_rerun.py and summarize_cpp_quality.py expect
INST="$ROOT/Instances"
GT="$INST/ground_truth/test_instances"
JL="simexpal_runners/run_experiments.jl"
CPP="simexpal_runners/run_experiments_cpp"
CACHE="$B/cache"
export JULIA_NUM_THREADS=8

log() { echo "[$(date '+%F %T')] $*"; }
x()   { if (( DRY )); then echo "  + $*" >&2; else "$@"; fi; }   # run, or print under --dry-run

# graph specs: name, path, directed flag (Julia spelling; the C++ runner needs -d, see cpp_d)
SMALL=(
    "p2p-Gnutella31   $INST/TestInstances/p2p-Gnutella31.txt   "
    "soc-Epinions1    $INST/TestInstances/soc-Epinions1.txt    --directed"
    "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    "email-EuAll      $INST/TestInstances/email-EuAll.txt      --directed"
)
TIGHT=("${SMALL[@]}"
    "amazon           $INST/ABCDE/amazon.txt                   "
    "dblp             $INST/ABCDE/dblp.txt                     "
)
BVK=("${TIGHT[@]}"
    "com-youtube      $INST/ABCDE/com-youtube.txt              "
    "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
    "com-lj           $INST/ABCDE/com-lj.txt                   "
)
THREADS=(
    "soc-Slashdot0902 $INST/TestInstances/soc-Slashdot0902.txt --directed"
    "amazon           $INST/ABCDE/amazon.txt                   "
    "cit-Patents      $INST/ABCDE/cit-Patents.txt              "
    "com-lj           $INST/ABCDE/com-lj.txt                   "
)
# The C++ runner silently ignores --directed and loads the graph undirected; only -d works.
cpp_d() { [[ -n "${1:-}" ]] && echo "-d" || true; }

# verify_dir <dir> <glob>: every Julia KADABRA run must carry the stopping fix, the right
# direction, and (outside thread scaling) 8 threads. A run that does not is a wrong number
# with a valid JSON, so this is checked rather than assumed.
verify_dir() {
    (( DRY )) && return 0
    python3 - "$1" "$2" <<'EOF'
import glob, json, os, sys
d, pat = sys.argv[1], sys.argv[2]
directed = {"soc-Epinions1", "soc-Slashdot0902", "email-EuAll"}
bad = n = 0
for f in sorted(glob.glob(os.path.join(d, pat))):
    p = json.load(open(f))["parameters"]
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

# ---------------------------------------------------------------------------
step_build() {
    x make -C "$B/simexpal_runners" build
    x julia --project="$ROOT" -e 'using Pkg; Pkg.instantiate()'
}

# ---------------------------------------------------------------------------
# BRAVA-GNN checkpoints. run_experiments.jl and eval_brava_paper.jl load
# cache/bravagnn_weights.jld2 and fall back to an UNTRAINED model with only a warning if it
# is missing --- so this step fails hard instead.
step_weights() {
    mkdir -p "$CACHE"
    if (( RETRAIN )); then
        for s in 1 2 3; do
            log "train BRAVA-GNN seed $s"
            (cd "$ROOT" && x julia --project=. src/train_bravagnn.jl --seed=$s)
        done
    else
        for f in "$B"/results/brava_retrained/weights/bravagnn_weights*; do
            x cp "$f" "$CACHE/"
        done
    fi
    (( DRY )) && return 0
    for s in "" _S2 _S3; do
        [[ -f "$CACHE/bravagnn_weights$s.jld2" ]] || { log "missing cache/bravagnn_weights$s.jld2"; return 1; }
    done
}

# ---------------------------------------------------------------------------
step_tight() {
    x ./reproduce_report.sh --stage tight --outdir "$RF/tight"
    verify_dir "$RF/tight" "tight_julia_*.json"
}

# ---------------------------------------------------------------------------
step_threads() {
    mkdir -p "$RF/ts"
    local name path dflag t s f
    for spec in "${THREADS[@]}"; do
        read -r name path dflag <<<"$spec"
        for t in 1 2 4 8 16 24 32 48; do
            for s in 0 1 2; do
                f="$RF/ts/julia_${name}_t${t}_s${s}.json"
                if [[ ! -f "$f" ]]; then
                    log "threads julia $name t=$t s=$s"
                    JULIA_NUM_THREADS=$t x julia --project=.. "$JL" ${dflag:-} -i "$path" -o "$f" \
                        -a kadabra -v julia --epsilon 0.01 --delta 0.1 -k 0 -s $s -t $t >/dev/null \
                        || log "FAILED threads julia $name t=$t s=$s"
                fi
                f="$RF/ts/cpp_${name}_t${t}_s${s}.json"
                if [[ ! -f "$f" ]]; then
                    log "threads cpp $name t=$t s=$s"
                    x "$CPP" $(cpp_d "${dflag:-}") -i "$path" -o "$f" -threads $t -k 0 \
                        -delta 0.1 -epsilon 0.01 -a kadabra -v cpp -s $s >/dev/null \
                        || log "FAILED threads cpp $name t=$t s=$s"
                fi
            done
        done
    done
    verify_dir "$RF/ts" "julia_*.json"
}

# ---------------------------------------------------------------------------
# Seed 0 of both methods (timed) via reproduce_report.sh, then KADABRA seeds 1-3 for the
# +/- on its columns. BRAVA-GNN's spread comes from the training seed (step brava-seeds).
step_bvk() {
    x ./reproduce_report.sh --stage bvk --outdir "$RF/bvk"
    mkdir -p "$RF/bvk/kadabra_seeds"
    local name path dflag s f
    for s in 1 2 3; do
        for spec in "${BVK[@]}"; do
            read -r name path dflag <<<"$spec"
            f="$RF/bvk/kadabra_seeds/bvk_kadabra_${name}_s${s}.json"
            [[ -f "$f" ]] && continue
            log "bvk kadabra $name s=$s"
            x julia --project=.. "$JL" ${dflag:-} -i "$path" -o "$f" -a kadabra -v julia \
                --epsilon 0.01 --delta 0.1 -k 0 -s $s -t 8 >/dev/null || log "FAILED bvk kadabra $name s=$s"
        done
    done
    verify_dir "$RF/bvk" "bvk_kadabra_*.json"
    verify_dir "$RF/bvk/kadabra_seeds" "bvk_kadabra_*.json"
}

# ---------------------------------------------------------------------------
step_brava_seeds() {
    local log_file="$OUT/brava_retrained/logs/eval_seeds.log"
    mkdir -p "$(dirname "$log_file")"
    (( DRY )) || : > "$log_file"
    for s in "" _S2 _S3; do
        (( DRY )) || echo "############ benchmark/cache/bravagnn_weights$s.jld2" >> "$log_file"
        if (( DRY )); then x julia --project=.. eval_brava_paper.jl "$CACHE/bravagnn_weights$s.jld2"
        else julia --project=.. eval_brava_paper.jl "$CACHE/bravagnn_weights$s.jld2" >> "$log_file"; fi
    done
}

# ---------------------------------------------------------------------------
step_topk() {
    for st in kx kxs kxb; do
        x ./reproduce_report.sh --stage $st --outdir "$RF/topk"
    done
    x ./reproduce_report.sh --stage kxc --outdir "$OUT/kxc"
    verify_dir "$RF/topk" "kx_*.json"
}

# ---------------------------------------------------------------------------
# Appendix A.7. The C++ runner keeps its per-vertex scores only with -centralities 1, so
# these are separate, untimed runs at the tight configuration: seed 0 (the configuration of
# Table 1) and seed 1 (a second draw). Scored exactly as run_experiments.jl scores Julia.
# The burn-in size tau comes from the matching Julia run.
step_cpp_quality() {
    local q="$OUT/cpp_quality" name path dflag s f
    mkdir -p "$q/raw_report_set" "$q/raw_second_set"
    for spec in "${TIGHT[@]}"; do
        read -r name path dflag <<<"$spec"
        for s in 0 1; do
            [[ $s == 0 ]] && f="$q/raw_report_set/cpp_$name.stats.json" || f="$q/raw_second_set/cpp_$name.stats.json"
            [[ -f "$f" ]] && continue
            log "cpp-quality $name s=$s"
            x "$CPP" $(cpp_d "${dflag:-}") -i "$path" -o "$f" -threads 8 -k 0 -delta 0.1 \
                -epsilon 0.0001 -a kadabra -v cpp -s $s -centralities 1 >/dev/null \
                || log "FAILED cpp-quality $name s=$s"
        done
    done
    (( DRY )) && { echo "  + score_cpp_centralities.py / check_burnin_bias.py -> $q/*.jsonl"; return 0; }
    python3 - "$RF/tight" "$q/julia_tau.json" <<'EOF'
import glob, json, os, sys
taus = {os.path.basename(f)[len("tight_julia_"):-len(".json")]: json.load(open(f))["kadabra_tau"]
        for f in glob.glob(os.path.join(sys.argv[1], "tight_julia_*.json"))}
json.dump(taus, open(sys.argv[2], "w"))
EOF
    # summarize_cpp_quality.py reads the second draw from set_aug10.jsonl (its historical name)
    python3 score_cpp_centralities.py "$GT" "$q"/raw_report_set/*.stats.json > "$q/set_report.jsonl"
    python3 score_cpp_centralities.py "$GT" "$q"/raw_second_set/*.stats.json > "$q/set_aug10.jsonl"
    python3 check_burnin_bias.py "$GT" "$q/julia_tau.json" 1e-4 "$q"/raw_report_set/*.stats.json > "$q/debias.jsonl"
}

# ---------------------------------------------------------------------------
step_tightx() {
    x ./reproduce_report.sh --stage tightx --outdir "$RF/tight"
    verify_dir "$RF/tight" "tight_julia_*.json"
}

step_kxl() {
    x ./reproduce_report.sh --stage kxl --outdir "$RF/topk"
    verify_dir "$RF/topk" "kx_*.json"
}

# ---------------------------------------------------------------------------
# Tables and figures, then every script whose output the prose quotes. The claim checkers
# print "!! CLAIM DOES NOT HOLD" where a sentence of the report is no longer supported.
step_report() {
    local rep="$OUT/report" c="$OUT/claims"
    mkdir -p "$rep" "$c"
    if [[ -d "$ROOT/Report/tables" && ! -d "$rep/tables" ]]; then
        x cp -R "$ROOT/Report/tables" "$rep/tables"
    fi
    x python3 update_report_from_rerun.py --results "$OUT" --report "$rep"
    (( DRY )) && return 0
    local tx; tx="$(mktemp -d)"
    ln -s "$RF"/tight/tight_julia_*.json "$RF"/bvk/bvk_*.json "$tx/"
    python3 summarize_reproduce.py tight  "$RF/tight"          > "$c/6.1_cpp_vs_julia.txt"
    python3 summarize_threads.py          "$RF/ts"             > "$c/6.1_thread_scaling.txt"
    python3 summarize_topk.py             "$RF/topk"           > "$c/6.3_topk.txt"
    python3 summarize_topk_claims.py      "$RF/topk" "$OUT/kxc" > "$c/6.3_topk_claims.txt"
    python3 summarize_seeds.py "$RF/bvk/kadabra_seeds" "$OUT/brava_retrained/logs/eval_seeds.log" \
                               "$RF/bvk"                        > "$c/6.4_brava_vs_kadabra.txt"
    python3 summarize_reproduce.py tightx "$tx"                > "$c/6.4_tight_accuracy.txt"
    python3 summarize_cpp_quality.py "$OUT" --julia-from rerun_fix > "$c/A.7_cpp_quality.txt"
    rm -rf "$tx"
    log "tables and figures: $rep; quoted numbers: $c"
    # Expected: 6.4_tight_accuracy.txt flags email-EuAll, where BRAVA-GNN wins tau_b; the
    # report claims tau_b on eight of nine graphs and says so. Anything else is news.
    grep -H "CLAIM DOES NOT HOLD" "$c"/*.txt || true
}

# ---------------------------------------------------------------------------
cd "$B"
log "reproduce_all.sh: out=$OUT from=${FROM:-start} only=${ONLY:-all} retrain=$RETRAIN dry=$DRY"
started=0
[[ -z "$FROM" ]] && started=1
for s in "${STEPS[@]}"; do
    [[ "$s" == "$FROM" ]] && started=1
    (( started )) || continue
    [[ -n "$ONLY" && "$s" != "$ONLY" ]] && continue
    log "=== step $s: start ==="
    if "step_${s//-/_}"; then
        log "=== step $s: done ==="
    else
        log "=== step $s: FAILED (exit $?) --- fix it and re-launch with --from $s ==="
        exit 1
    fi
done
log "=== reproduce_all.sh finished ==="
