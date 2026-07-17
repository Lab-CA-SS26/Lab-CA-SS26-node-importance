using Pkg
Pkg.activate(mktempdir())
Pkg.add(["Graphs", "DataFrames", "CSV", "JLD2", "StatsBase"])

using Graphs
using DataFrames
using CSV
using JLD2
using StatsBase

include(joinpath("..", "src", "kadabra.jl"))

const INSTANCES_DIR = joinpath("..", "cpp_reference", "example_input")
const TEST_FILE = joinpath(INSTANCES_DIR, "facebook_combined.txt")
const IS_DIRECTED = false

const EPSILONS = [0.1, 0.05, 0.01, 0.005, 0.001]
const DELTAS = [0.1, 0.05, 0.01]

function load_graph(filepath::String, directed::Bool)
    edges_list = Tuple{Int, Int}[]
    max_node = 0
    for line in eachline(filepath)
        parts = split(strip(line))
        if length(parts) >= 2
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            push!(edges_list, (u, v))
            max_node = max(max_node, u, v)
        end
    end
    
    # 0-indexed adjustment
    min_node = isempty(edges_list) ? 0 : minimum(min(u, v) for (u, v) in edges_list)
    shift = min_node == 0 ? 1 : 0
    
    g = directed ? SimpleDiGraph(max_node + shift) : SimpleGraph(max_node + shift)
    for (u, v) in edges_list
        add_edge!(g, u + shift, v + shift)
    end
    return g
end

function run_error_bounds_benchmark()
    println("--- Kadabra Error Bounds Evaluation ---")
    println("Loading graph: $TEST_FILE")
    g = load_graph(TEST_FILE, IS_DIRECTED)
    n = nv(g)
    
    # 1. Ground Truth (Exact BC)
    dataset_name = basename(TEST_FILE)
    cache_file = joinpath(@__DIR__, "$(dataset_name)_exact_bc.jld2")
    
    if isfile(cache_file)
        println("Loading cached exact betweenness centrality from $cache_file...")
        @load cache_file exact_scores
    else
        println("Calculating exact betweenness (this may take a while)...")
        exact_time = @elapsed begin
            exact_scores = betweenness_centrality(g, normalize=true)
        end
        println("Exact calculation finished in $(round(exact_time, digits=2)) seconds. Saving to cache...")
        @save cache_file exact_scores
    end
    
    results = DataFrame(
        Epsilon = Float64[],
        Delta = Float64[],
        Runtime_s = Float64[],
        Kendall_Tau = Float64[]
    )
    
    # Warmup
    println("Warming up Kadabra...")
    _ = kadabra_centrality(g, 0, 0.1, 0.1)
    
    println("Starting grid search...")
    for eps in EPSILONS
        for delta in DELTAS
            println("Evaluating epsilon = $eps, delta = $delta ...")
            
            runtime = @elapsed begin
                # k=0 means we compute approximate BC for all nodes
                scores = kadabra_centrality(g, 0, eps, delta)
            end
            
            tau = corkendall(exact_scores, scores)
            
            println("  -> Time: $(round(runtime, digits=2))s, Tau: $(round(tau, digits=4))")
            
            push!(results, (eps, delta, runtime, tau))
        end
    end
    
    println("\n=== Final Error Bounds Results ===")
    display(results)
    
    output_file = joinpath(@__DIR__, "error_bounds_results.csv")
    CSV.write(output_file, results)
    println("\nResults saved to $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_error_bounds_benchmark()
end
