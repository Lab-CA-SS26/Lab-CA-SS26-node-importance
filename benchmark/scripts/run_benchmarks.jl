using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using Graphs
using SparseArrays
using DataFrames
using CSV
using StatsBase
using Flux
using JLD2

# Include the local source modules
include("../../src/BRAVAGNN.jl")
using .BRAVAGNN
include("../../src/kadabra.jl")
include("BenchmarkUtils.jl")

using .BenchmarkUtils

# Configuration
const INSTANCES_DIR = joinpath(dirname(dirname(@__DIR__)), "Instances", "example_input") # Example input directory to test
const OUTPUT_FILE = joinpath(@__DIR__, "..", "results", "benchmark_results.csv")
const KADABRA_EPSILON = 0.01
const KADABRA_DELTA = 0.1



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
        t_exact = @elapsed exact_bc = betweenness_centrality(g, normalize = true)
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
        Memory_MB = Float64[],
        Speedup = Float64[],
        Kendall_Tau = Float64[],
    )

    instances_file = joinpath(dirname(dirname(@__DIR__)), "Instances", "instances.txt")
    instances = BenchmarkUtils.read_instances(instances_file)

    datasets = []

    # 1. Real graphs
    for (rel_path, is_directed) in instances
        filepath = joinpath(dirname(dirname(@__DIR__)), "Instances", rel_path)
        if isfile(filepath)
            g = BenchmarkUtils.load_graph_from_edgelist(filepath, is_directed)
            push!(datasets, (basename(filepath), is_directed, g))
        else
            @warn "File not found: $filepath. Skipping."
        end
    end

    # 2. Synthetic Graphs for Scalability Testing
    # println("\nGenerating synthetic graphs for scalability...")
    # for n in [1000, 5000, 10000]
    #     println("Generating Barabási–Albert graph N=$n...")
    #     g_syn = barabasi_albert(n, 10)
    #     push!(datasets, ("BA_$(n)_10", false, g_syn))
    # end

    # Initialize and load trained BRAVA model
    model = BRAVAModel(m_hops = 5, hidden_dim = 12, num_layers = 2)
    weight_path = joinpath(@__DIR__, "..", "cache", "bravagnn_weights.jld2")
    if isfile(weight_path)
        println("Loading trained BRAVA-GNN weights from $weight_path...")
        @load weight_path model
    else
        println("No trained weights found. Using random initialized BRAVA-GNN weights.")
    end

    # Load exact Brandes timings
    timing_file = joinpath(@__DIR__, "..", "results", "exact_brandes_timings.csv")
    timing_df = isfile(timing_file) ? CSV.read(timing_file, DataFrame) : DataFrame()

    for (dataset_name, is_directed, g) in datasets
        println("\n=== Benchmarking Dataset: $dataset_name ===")
        N = nv(g)
        M = ne(g)
        println("Loaded graph: $N nodes, $M edges.")

        # Format graph for C++ if necessary
        cpp_input_file = joinpath(tempdir(), "kadabra_cpp_input.txt")
        export_to_edgelist(g, cpp_input_file)

        # 2. Ground Truth (with Caching)
        cache_file = joinpath(@__DIR__, "..", "cache", "$(dataset_name)_exact_bc.jld2")
        exact_bc = get_exact_betweenness(g, cache_file)

        t_exact = 0.0
        if !isempty(timing_df)
            t_row = filter(row -> row.Dataset == dataset_name, timing_df)
            if !isempty(t_row)
                t_exact = t_row.Runtime_s[1]
            end
        end
        if t_exact == 0.0
            println("Warning: Exact Brandes timing not found in CSV. Defaulting to 0.0s")
        end

        push!(
            results,
            (dataset_name, is_directed, N, M, "Exact (Brandes)", t_exact, 0.0, 1.0, 100.0),
        )

        # 3. Evaluate Julia KADABRA
        println("Running Julia KADABRA...")
        # Warmup
        _ = kadabra_centrality(g, 0, KADABRA_EPSILON, KADABRA_DELTA)

        # We manually use @allocated to track memory
        mem_jl_bytes = @allocated begin
            t_jl = @elapsed scores_jl =
                kadabra_centrality(g, 0, KADABRA_EPSILON, KADABRA_DELTA)
        end
        tau_jl = corkendall(scores_jl, exact_bc) * 100
        push!(
            results,
            (
                dataset_name,
                is_directed,
                N,
                M,
                "KADABRA (Julia)",
                t_jl,
                mem_jl_bytes / 1024^2,
                t_exact / t_jl,
                tau_jl,
            ),
        )

        # 4. Evaluate C++ KADABRA
        println("Running C++ KADABRA...")
        t_cpp, scores_cpp =
            run_cpp_kadabra(cpp_input_file, is_directed, KADABRA_EPSILON, KADABRA_DELTA, N)
        tau_cpp = corkendall(scores_cpp, exact_bc) * 100
        # C++ memory tracking from Julia is tricky, defaulting to 0 for now
        push!(
            results,
            (
                dataset_name,
                is_directed,
                N,
                M,
                "KADABRA (C++)",
                t_cpp,
                0.0,
                t_exact / max(t_cpp, 1e-6),
                tau_cpp,
            ),
        )

        # 5. Evaluate BRAVA-GNN
        println("Running BRAVA-GNN...")
        A = sparse(g)
        A_t = A'

        # Set to test mode to disable Dropout during inference!
        Flux.testmode!(model)

        # Warmup
        pr_feat = compute_pagerank_feature(A)
        _X_out = compute_degree_masses(A, pr_feat, 5)
        _X_in = compute_degree_masses(A_t, pr_feat, 5)
        _ = model(A, A_t, _X_in, _X_out)

        mem_gnn_bytes = @allocated begin
            t_gnn = @elapsed begin
                pr_feat_val = compute_pagerank_feature(A)
                X_out = compute_degree_masses(A, pr_feat_val, 5)
                X_in = compute_degree_masses(A_t, pr_feat_val, 5)
                scores_gnn = model(A, A_t, X_in, X_out)
            end
        end

        tau_gnn = corkendall(vec(scores_gnn), exact_bc) * 100
        push!(
            results,
            (
                dataset_name,
                is_directed,
                N,
                M,
                "BRAVA-GNN",
                t_gnn,
                mem_gnn_bytes / 1024^2,
                t_exact / t_gnn,
                tau_gnn,
            ),
        )

        # 6. Evaluate Heuristic Baselines
        println("Running Heuristics...")

        # Degree
        mem_deg = @allocated t_deg = @elapsed scores_deg = degree(g)
        tau_deg = corkendall(scores_deg, exact_bc) * 100
        push!(
            results,
            (
                dataset_name,
                is_directed,
                N,
                M,
                "Degree Centrality",
                t_deg,
                mem_deg / 1024^2,
                t_exact / max(t_deg, 1e-6),
                tau_deg,
            ),
        )

        # PageRank
        mem_pr = @allocated t_pr = @elapsed scores_pr = pagerank(g)
        tau_pr = corkendall(scores_pr, exact_bc) * 100
        push!(
            results,
            (
                dataset_name,
                is_directed,
                N,
                M,
                "PageRank",
                t_pr,
                mem_pr / 1024^2,
                t_exact / max(t_pr, 1e-6),
                tau_pr,
            ),
        )

        println("Done with $dataset_name.\n")
    end

    # Save and Print Results
    CSV.write(OUTPUT_FILE, results)
    println("=== Final Benchmark Results ===")
    display(results)
    println("Results saved to: $OUTPUT_FILE")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
