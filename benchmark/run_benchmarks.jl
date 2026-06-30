using Pkg
# Activate temporary environment for benchmarking dependencies
Pkg.activate(mktempdir())
Pkg.add(["Graphs", "SparseArrays", "DataFrames", "CSV", "StatsBase", "Flux", "JLD2"])

using Graphs
using SparseArrays
using DataFrames
using CSV
using StatsBase
using Flux
using JLD2

# Include the local source modules
include("../src/BRAVAGNN.jl")
include("../src/kadabra.jl")
include("BenchmarkUtils.jl")

using .BRAVAGNN
using .BenchmarkUtils

# Configuration
const INSTANCES_DIR = joinpath(dirname(@__DIR__), "Instances", "example_input") # Example input directory to test
const OUTPUT_FILE = joinpath(@__DIR__, "benchmark_results.csv")
const KADABRA_EPSILON = 0.01
const KADABRA_DELTA = 0.1

"""
    load_graph_from_edgelist(filepath::String, is_directed::Bool)

A robust loader for space-separated edge lists.
"""
function load_graph_from_edgelist(filepath::String, is_directed::Bool)
    edges_list = Tuple{Int, Int}[]
    max_node = 0
    
    for line in eachline(filepath)
        line = strip(line)
        if isempty(line) || startswith(line, "#") || startswith(line, "%")
            continue
        end
        parts = split(line)
        if length(parts) >= 2
            # Handle both 0-indexed and 1-indexed safely by shifting
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            push!(edges_list, (u, v))
            max_node = max(max_node, u, v)
        end
    end
    
    # Check if 0-indexed
    min_node = isempty(edges_list) ? 0 : minimum(min(u, v) for (u, v) in edges_list)
    shift = min_node == 0 ? 1 : 0
    
    g = is_directed ? SimpleDiGraph(max_node + shift) : SimpleGraph(max_node + shift)
    for (u, v) in edges_list
        add_edge!(g, u + shift, v + shift)
    end
    
    return g
end

"""
    get_exact_betweenness(g::AbstractGraph, cache_file::String)

Computes or loads the exact betweenness centrality for the graph.
"""
function get_exact_betweenness(g::AbstractGraph, cache_file::String)
    if isfile(cache_file)
        println("Loading cached exact betweenness centrality...")
        @load cache_file exact_bc
        return exact_bc
    else
        println("Computing exact betweenness centrality (This may take a while)...")
        t_exact = @elapsed exact_bc = betweenness_centrality(g)
        println("Exact BC computed in $(round(t_exact, digits=2))s. Saving to cache.")
        @save cache_file exact_bc
        return exact_bc
    end
end

function run_benchmarks()
    # Initialize the results DataFrame
    results = DataFrame(
        Dataset = String[],
        Directed = Bool[],
        Nodes = Int[],
        Edges = Int[],
        Method = String[],
        WallClock_s = Float64[],
        Speedup = Float64[],
        Kendall_Tau = Float64[]
    )
    
    # We will test on the facebook_combined graph inside cpp_reference/example_input as a sanity check
    # In a real run, this would loop over INSTANCES_DIR
    test_files = [joinpath(dirname(@__DIR__), "cpp_reference", "example_input", "facebook_combined.txt")]
    
    # Initialize a dummy trained BRAVA model
    model = BRAVAModel(m_hops=5, hidden_dim=12)
    
    for filepath in test_files
        if !isfile(filepath)
            @warn "File not found: $filepath. Skipping."
            continue
        end
        
        dataset_name = basename(filepath)
        is_directed = false # Infer or configure based on dataset
        println("\n=== Benchmarking Dataset: $dataset_name ===")
        
        # 1. Load Graph
        g = load_graph_from_edgelist(filepath, is_directed)
        N = nv(g)
        M = ne(g)
        println("Loaded graph: $N nodes, $M edges.")
        
        # Format graph for C++ if necessary
        cpp_input_file = joinpath(tempdir(), "kadabra_cpp_input.txt")
        export_to_edgelist(g, cpp_input_file)
        
        # 2. Ground Truth (with Caching)
        cache_file = joinpath(@__DIR__, "$(dataset_name)_exact_bc.jld2")
        t_exact = @elapsed exact_bc = get_exact_betweenness(g, cache_file)
        
        push!(results, (dataset_name, is_directed, N, M, "Exact (Brandes)", t_exact, 1.0, 100.0))
        
        # 3. Evaluate Julia KADABRA
        println("Running Julia KADABRA...")
        # Warmup
        _ = kadabra_centrality(g, 0, KADABRA_EPSILON, KADABRA_DELTA)
        t_jl = @elapsed scores_jl = kadabra_centrality(g, 0, KADABRA_EPSILON, KADABRA_DELTA)
        tau_jl = corkendall(scores_jl, exact_bc) * 100
        push!(results, (dataset_name, is_directed, N, M, "KADABRA (Julia)", t_jl, t_exact / t_jl, tau_jl))
        
        # 4. Evaluate C++ KADABRA
        println("Running C++ KADABRA...")
        t_cpp, scores_cpp = run_cpp_kadabra(cpp_input_file, is_directed, KADABRA_EPSILON, KADABRA_DELTA, N)
        tau_cpp = corkendall(scores_cpp, exact_bc) * 100
        push!(results, (dataset_name, is_directed, N, M, "KADABRA (C++)", t_cpp, t_exact / max(t_cpp, 1e-6), tau_cpp))
        
        # 5. Evaluate BRAVA-GNN
        println("Running BRAVA-GNN...")
        # Compile sparse adjacency
        A = sparse(g)
        A_t = A'
        
        # Strict timing encompasses feature preprocessing + inference
        t_gnn = @elapsed begin
            X_out = compute_degree_masses(A, 5)
            X_in = compute_degree_masses(A_t, 5)
            scores_gnn = model(A, A_t, X_in, X_out)
        end
        
        tau_gnn = corkendall(scores_gnn, exact_bc) * 100
        push!(results, (dataset_name, is_directed, N, M, "BRAVA-GNN", t_gnn, t_exact / t_gnn, tau_gnn))
        
        println("Done with $dataset_name.\n")
    end
    
    # Save and Print Results
    CSV.write(OUTPUT_FILE, results)
    println("=== Final Benchmark Results ===")
    println(results)
    println("Results saved to: $OUTPUT_FILE")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
