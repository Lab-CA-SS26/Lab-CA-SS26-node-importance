#!/usr/bin/env julia
# End-to-end check of the retrained BRAVA-GNN against the paper's Table 2.
#
# Unlike the two diagnose_* scripts, this calls `brava_centrality` itself, so it
# exercises the same code path `run_experiments.jl` uses -- weights loaded from
# benchmark/cache/bravagnn_weights.jld2, masking and feature construction included.
# tau_b and top-100 overlap are computed exactly as run_experiments.jl computes
# `tau_overall` and `overlap_topk`.

using Pkg
const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using Graphs, SparseArrays, StatsBase, Flux, JLD2, Printf

include(joinpath(ROOT, "src", "BRAVAGNN.jl"))
using .BRAVAGNN
include(joinpath(ROOT, "benchmark", "scripts", "BenchmarkUtils.jl"))
using .BenchmarkUtils

# (name, path, directed, paper Table 2 tau_b x100)
const GRAPHS = [
    ("p2p-Gnutella31",   "Instances/TestInstances/p2p-Gnutella31.txt",   false, 89.7),
    ("com-youtube",      "Instances/ABCDE/com-youtube.txt",              false, 92.2),
    ("amazon",           "Instances/ABCDE/amazon.txt",                   false, 85.9),
    ("cit-Patents",      "Instances/ABCDE/cit-Patents.txt",              false, 74.0),
    ("com-lj",           "Instances/ABCDE/com-lj.txt",                   false, 80.5),
    ("dblp",             "Instances/ABCDE/dblp.txt",                     false, 85.0),
    ("soc-Epinions1",    "Instances/TestInstances/soc-Epinions1.txt",    true,  92.6),
    ("soc-Slashdot0902", "Instances/TestInstances/soc-Slashdot0902.txt", true,  90.3),
    ("email-EuAll",      "Instances/TestInstances/email-EuAll.txt",      true,  99.2),
]

# tau_b of the PageRank-variant model these graphs were reported with (Report Table 6.4).
const REPORTED = Dict("p2p-Gnutella31" => 82.6, "com-youtube" => 73.6, "amazon" => 78.1,
    "cit-Patents" => 73.8, "com-lj" => 74.9, "dblp" => 79.0, "soc-Epinions1" => 72.8,
    "soc-Slashdot0902" => 80.4, "email-EuAll" => 45.9)

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

"Checkpoint to score: first positional argument, else the canonical seed-1 model."
function weights_path()
    a = filter(s -> !startswith(s, "--"), ARGS)
    isempty(a) ? joinpath(ROOT, "benchmark", "cache", "bravagnn_weights.jld2") :
                 abspath(a[1])
end

function main()
    wp = weights_path()
    isfile(wp) || error("no weights at $wp")
    cfg = replace(wp, ".jld2" => ".config.txt")
    println("--- checkpoint: $(basename(wp)) ---")
    isfile(cfg) && print(read(cfg, String))

    @printf("\n%-18s %8s %8s %8s %8s   %s\n",
            "graph", "paper", "NEW", "vs paper", "reported", "top-100 ovlp")
    println(repeat("-", 84))
    dp = Float64[]; dr = Float64[]

    for (name, rel, directed, paper) in GRAPHS
        path = joinpath(ROOT, rel)
        isfile(path) || (@printf("%-18s (missing %s)\n", name, rel); continue)
        g = BenchmarkUtils.load_graph_from_edgelist(path, directed)
        scores, _ = brava_centrality(g, 0, 0.01, 0.1; weight_path = wp)

        bt = ground_truth(name)
        bt === nothing && (@printf("%-18s (no ground truth)\n", name); continue)
        approx = zeros(Float64, length(bt))
        for (v, c) in enumerate(scores)
            v <= length(bt) && (approx[v] = c)
        end
        t = 100 * corkendall(bt, approx)
        k = min(100, length(bt))
        ovl = length(intersect(sortperm(bt, rev = true)[1:k], sortperm(approx, rev = true)[1:k]))

        push!(dp, t - paper); push!(dr, t - REPORTED[name])
        @printf("%-18s %8.1f %8.1f %+8.1f %8.1f   %3d/100\n",
                name, paper, t, t - paper, REPORTED[name], ovl)
        flush(stdout)
    end

    @printf("\n%-18s %8s %8s %+8.1f  (mean delta vs paper)\n", "MEAN", "", "", sum(dp)/length(dp))
    @printf("%-18s %8s %8s %+8.1f  (mean delta vs currently reported)\n",
            "", "", "", sum(dr)/length(dr))
    println("\nEVAL DONE")
end

main()
