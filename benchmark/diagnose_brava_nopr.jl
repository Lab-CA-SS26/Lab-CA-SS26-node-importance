#!/usr/bin/env julia
# Second half of the BRAVA-GNN gap investigation (see diagnose_brava_mask.jl for the first).
#
# Our runs enable the PageRank input channel (`use_pr = true`, the default in both
# `brava_centrality` and `train_bravagnn.jl`). The BRAVA-GNN paper never mentions
# PageRank, and exactly 1 of the 846 configurations in the authors' own
# `all_results.csv` carries the `_pr` suffix — it is an abandoned side experiment,
# not the reported model. Our tau_b matches that one ablation row graph for graph.
#
# This scores the same graphs with `bravagnn_weights_no_pr.jld2` (use_pr = false) to
# confirm the channel is what separates us from the paper's headline numbers, and
# crosses it with the A_t masking fix from diagnose_brava_mask.jl.

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

function raw_adjacency(g)
    N = nv(g)
    I_idx, J_idx = Int[], Int[]
    sizehint!(I_idx, 2 * ne(g)); sizehint!(J_idx, 2 * ne(g))
    for v in 1:N, u in outneighbors(g, v)
        push!(I_idx, v); push!(J_idx, u)
    end
    return sparse(I_idx, J_idx, ones(Float32, length(I_idx)), N, N)
end

function score(model, A_raw, mask; fixed::Bool, use_pr::Bool)
    D = spdiagm(mask)
    A = D * A_raw
    A_t = fixed ? SparseMatrixCSC{Float32,Int}(D * sparse(A_raw')) :
                  SparseMatrixCSC{Float32,Int}(A')
    pr = compute_pagerank_feature(A)
    X_out = compute_degree_masses(A,   pr, M_HOPS; use_pr = use_pr)
    X_in  = compute_degree_masses(A_t, pr, M_HOPS; use_pr = use_pr)
    return model(A, A_t, X_in, X_out)
end

function ground_truth(name)
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

function tau_b(bt_exact, scores)
    approx = zeros(Float64, length(bt_exact))
    for (v, c) in enumerate(scores)
        v <= length(bt_exact) && (approx[v] = c)
    end
    return corkendall(bt_exact, approx)
end

"How many distinct scores does the model give the pruned vertices? Kendall tau_b on a
 graph that is mostly zero-betweenness hinges on those vertices staying tied."
function tie_block(scores, mask)
    idx = findall(==(0.0f0), mask)
    isempty(idx) && return (0, 0)
    return (length(idx), length(unique(round.(Float64.(scores[idx]), digits = 12))))
end

function main()
    wp = joinpath(ROOT, "benchmark", "cache", "bravagnn_weights_no_pr.jld2")
    isfile(wp) || error("no no-PR weights at $wp")
    @load wp model
    println("weights: $wp  (use_pr = false)")
    @printf("\n%-18s %8s %10s %10s   %s\n",
            "graph", "masked%", "tau noPR", "tau noPR+fix", "distinct scores among pruned")
    println(repeat("-", 92))

    for (name, rel, directed) in GRAPHS
        path = joinpath(ROOT, rel)
        isfile(path) || (@printf("%-18s (missing %s)\n", name, rel); continue)
        g = BenchmarkUtils.load_graph_from_edgelist(path, directed)
        A_raw = raw_adjacency(g)
        mask = brava_clique_mask(g)
        frac = 100 * count(==(0.0f0), mask) / length(mask)

        s_now = score(model, A_raw, mask; fixed = false, use_pr = false)
        s_fix = score(model, A_raw, mask; fixed = true,  use_pr = false)
        bt = ground_truth(name)
        bt === nothing && (@printf("%-18s %7.1f%%  (no ground truth)\n", name, frac); continue)

        npruned, ndistinct = tie_block(s_fix, mask)
        @printf("%-18s %7.1f%% %10.3f %10.3f      %d distinct / %d pruned\n",
                name, frac, tau_b(bt, s_now), tau_b(bt, s_fix), ndistinct, npruned)
        flush(stdout)
    end
    println("\nDIAGNOSE DONE")
end

main()
