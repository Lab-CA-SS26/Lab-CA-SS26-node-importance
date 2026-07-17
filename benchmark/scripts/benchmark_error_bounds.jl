using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using Graphs
using DataFrames
using CSV
using JLD2
using StatsBase

include(joinpath("..", "..", "src", "kadabra.jl"))
include("BenchmarkUtils.jl")
using .BenchmarkUtils

const EPSILONS = [0.1, 0.05, 0.01, 0.005, 0.001]
const DELTAS = [0.1, 0.05, 0.01]

function run_error_bounds_benchmark()
    println("--- Kadabra Error Bounds Evaluation ---")
    
    instances_file = joinpath(dirname(dirname(@__DIR__)), "Instances", "instances.txt")
    instances = BenchmarkUtils.read_instances(instances_file)
    
    results = DataFrame(
        Dataset = String[],
        Directed = Bool[],
        Epsilon = Float64[],
        Delta = Float64[],
        Runtime_s = Float64[],
        Kendall_Tau = Float64[]
    )
    
    for (rel_path, is_directed) in instances
        filepath = joinpath(dirname(dirname(@__DIR__)), "Instances", rel_path)
        if !isfile(filepath)
            @warn "File not found: $filepath. Skipping."
            continue
        end
        
        dataset_name = basename(filepath)
        println("\n=== Evaluating Graph: $dataset_name (Directed: $is_directed) ===")
        
        g = BenchmarkUtils.load_graph_from_edgelist(filepath, is_directed)
        n = nv(g)
        
        # 1. Ground Truth (Exact BC)
        cache_file = joinpath(@__DIR__, "..", "cache", "$(dataset_name)_exact_bc.jld2")
        
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
        
        # Warmup
        println("Warming up Kadabra...")
        _ = kadabra_centrality(g, 0, 0.1, 0.1)
        
        println("Starting grid search...")
        for eps in EPSILONS
            for delta in DELTAS
                println("  Evaluating epsilon = $eps, delta = $delta ...")
                
                runtime = @elapsed begin
                    # k=0 means we compute approximate BC for all nodes
                    scores = kadabra_centrality(g, 0, eps, delta)
                end
                
                tau = corkendall(exact_scores, scores)
                
                println("    -> Time: $(round(runtime, digits=2))s, Tau: $(round(tau, digits=4))")
                
                push!(results, (dataset_name, is_directed, eps, delta, runtime, tau))
            end
        end
    end
    
    println("\n=== Final Error Bounds Results ===")
    display(results)
    
    output_file = joinpath(@__DIR__, "..", "results", "error_bounds_results.csv")
    CSV.write(output_file, results)
    println("\nResults saved to $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_error_bounds_benchmark()
end
