#!/usr/bin/env julia
# Why is our BRAVA-GNN below the numbers the BRAVA-GNN paper reports?
#
# `train_bravagnn.jl` transposes BEFORE applying the clique mask:
#
#     A_t = A'                        # then
#     A   = spdiagm(mask) * A         # A_masked = D A
#     A_t = spdiagm(mask) * A_t       # A_t      = D Aᵀ      <-- rows of Aᵀ zeroed
#
# `brava_centrality` transposes AFTER:
#
#     A   = spdiagm(mask) * A         # A_masked = D A
#     A_t = A'                        # A_t      = Aᵀ D      <-- COLUMNS of Aᵀ zeroed
#
# The upstream PyTorch does what training does (utils.graph_to_adj_bet: adj_temp_t is
# taken from the unmasked adjacency, then `.multiply(degree_arr)` scales its rows).
# So inference feeds the in-stream a matrix the model was never trained on, and pruned
# vertices — whose in-features should be identically zero, tying them at the bottom of
# the ranking exactly where their true BC of 0 belongs — instead receive live signal.
#
# This script scores both variants with the same weights and prints tau_b against the
# cached exact ground truth, using the same corkendall-over-all-nodes metric as
# run_experiments.jl. Nothing in src/ is modified.

using Pkg
const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using Graphs, SparseArrays, StatsBase, Flux, JLD2, Printf

include(joinpath(ROOT, "src", "BRAVAGNN.jl"))
using .BRAVAGNN
include(joinpath(ROOT, "benchmark", "scripts", "BenchmarkUtils.jl"))
using .BenchmarkUtils

const GRAPHS = [
    ("p2p-Gnutella31",   "Instances/TestInstances/p2p-Gnutella31.txt",   false),
    ("soc-Epinions1",    "Instances/TestInstances/soc-Epinions1.txt",    true),
    ("soc-Slashdot0902", "Instances/TestInstances/soc-Slashdot0902.txt", true),
    ("email-EuAll",      "Instances/TestInstances/email-EuAll.txt",      true),
    ("com-youtube",      "Instances/ABCDE/com-youtube.txt",              false),
    ("amazon",           "Instances/ABCDE/amazon.txt",                   false),
    ("dblp",             "Instances/ABCDE/dblp.txt",                     false),
    ("cit-Patents",      "Instances/ABCDE/cit-Patents.txt",              false),
    ("com-lj",           "Instances/ABCDE/com-lj.txt",                   false),
]

const M_HOPS = 6

"Adjacency built exactly as brava_centrality builds it, before any masking."
function raw_adjacency(g)
    N = nv(g)
    I_idx, J_idx = Int[], Int[]
    sizehint!(I_idx, 2 * ne(g)); sizehint!(J_idx, 2 * ne(g))
    for v in 1:N, u in outneighbors(g, v)
        push!(I_idx, v); push!(J_idx, u)
    end
    return sparse(I_idx, J_idx, ones(Float32, length(I_idx)), N, N)
end

"Score every vertex with `model`, building A_t the current way or the training way."
function score(model, g, A_raw, mask; fixed::Bool)
    D = spdiagm(mask)
    A = D * A_raw
    A_t = fixed ? SparseMatrixCSC{Float32,Int}(D * sparse(A_raw')) :
                  SparseMatrixCSC{Float32,Int}(A')
    pr = compute_pagerank_feature(A)
    X_out = compute_degree_masses(A,   pr, M_HOPS; use_pr = true)
    X_in  = compute_degree_masses(A_t, pr, M_HOPS; use_pr = true)
    return model(A, A_t, X_in, X_out), X_in
end

function ground_truth(name, n)
    path = joinpath(ROOT, "Instances", "ground_truth", "test_instances", "$(name)_bet.csv")
    isfile(path) || return nothing
    lines = readlines(path)
    length(lines) >= 2 || return nothing
    shift = parse(Int, split(lines[2], ",")[1]) == 0 ? 1 : 0
    maxnode = 0
    for l in lines[2:end]
        maxnode = max(maxnode, parse(Int, split(l, ",")[1]) + shift)
    end
    bt = zeros(Float64, maxnode)
    for l in lines[2:end]
        p = split(l, ",")
        bt[parse(Int, p[1]) + shift] = parse(Float64, p[2])
    end
    return bt
end

"corkendall over all ground-truth vertices, as run_experiments.jl computes tau_overall."
function tau_b(bt_exact, scores)
    approx = zeros(Float64, length(bt_exact))
    for (v, c) in enumerate(scores)
        v <= length(bt_exact) && (approx[v] = c)
    end
    return corkendall(bt_exact, approx)
end

function main()
    wpath = joinpath(ROOT, "benchmark", "cache", "bravagnn_weights.jld2")
    isfile(wpath) || error("no weights at $wpath")
    @load wpath model
    println("weights: $wpath")
    @printf("\n%-18s %7s %8s %8s %8s %8s   %s\n",
            "graph", "dir", "masked%", "tau NOW", "tau FIX", "delta", "in-feat of masked v")
    println(repeat("-", 92))

    for (name, rel, directed) in GRAPHS
        path = joinpath(ROOT, rel)
        if !isfile(path)
            @printf("%-18s  (missing: %s)\n", name, rel); continue
        end
        g = BenchmarkUtils.load_graph_from_edgelist(path, directed)
        A_raw = raw_adjacency(g)
        mask = brava_clique_mask(g)
        pruned = count(==(0.0f0), mask)
        frac = 100 * pruned / length(mask)

        s_now, Xin_now = score(model, g, A_raw, mask; fixed = false)
        s_fix, _       = score(model, g, A_raw, mask; fixed = true)

        # How much in-stream signal leaks into vertices that should be silenced?
        idx = findall(==(0.0f0), mask)
        leak = isempty(idx) ? 0.0 :
               100 * count(j -> any(!iszero, @view Xin_now[1:M_HOPS, j]), idx) / length(idx)

        bt = ground_truth(name, nv(g))
        if bt === nothing
            @printf("%-18s %7s %7.1f%%  (no ground truth)\n", name, directed ? "D" : "U", frac)
            continue
        end
        t_now, t_fix = tau_b(bt, s_now), tau_b(bt, s_fix)
        @printf("%-18s %7s %7.1f%% %8.3f %8.3f %+8.3f   %5.1f%% still nonzero\n",
                name, directed ? "D" : "U", frac, t_now, t_fix, t_fix - t_now, leak)
        flush(stdout)
    end
    println("\nDIAGNOSE DONE")
end

main()
