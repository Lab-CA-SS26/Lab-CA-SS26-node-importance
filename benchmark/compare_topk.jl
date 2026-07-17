using Pkg
Pkg.activate(mktempdir())
Pkg.add(["Graphs", "SparseArrays", "DataFrames", "CSV", "Flux", "JLD2", "StatsBase"])

using Graphs
using SparseArrays
using DataFrames
using CSV
using Flux
using CUDA
using JLD2
using StatsBase

include(joinpath("..", "src", "kadabra.jl"))
include(joinpath("..", "src", "BRAVAGNN.jl"))
using .BRAVAGNN

const INSTANCES_DIR = joinpath("..", "cpp_reference", "example_input")
const TEST_FILE = joinpath(INSTANCES_DIR, "facebook_combined.txt")
const IS_DIRECTED = false

const KADABRA_EPSILON = 0.01
const KADABRA_DELTA = 0.1
const K_VALUES = [10, 50, 100]

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

# Calculate the intersection fraction of the top-k items (Precision at K)
function top_k_overlap(true_scores, pred_scores, k::Int)
    true_topk = Set(sortperm(true_scores, rev=true)[1:k])
    pred_topk = Set(sortperm(pred_scores, rev=true)[1:k])
    overlap = length(intersect(true_topk, pred_topk))
    return overlap / k
end

# Calculate Kendall Tau strictly on the true Top-K nodes
function top_k_kendall_tau(true_scores, pred_scores, k::Int)
    true_topk_idx = sortperm(true_scores, rev=true)[1:k]
    return corkendall(true_scores[true_topk_idx], pred_scores[true_topk_idx])
end

# NDCG at K
function ndcg_at_k(true_scores, pred_scores, k::Int)
    pred_topk_idx = sortperm(pred_scores, rev=true)[1:k]
    true_topk_idx = sortperm(true_scores, rev=true)[1:k]
    
    function dcg(scores, idxs)
        val = 0.0
        for i in 1:k
            rel = scores[idxs[i]]
            val += rel / log2(i + 1)
        end
        return val
    end
    
    actual_dcg = dcg(true_scores, pred_topk_idx)
    ideal_dcg = dcg(true_scores, true_topk_idx)
    
    return ideal_dcg > 0 ? actual_dcg / ideal_dcg : 0.0
end

function run_topk_comparison()
    println("--- Top-k Evaluation ---")
    println("Loading graph: $TEST_FILE")
    g = load_graph(TEST_FILE, IS_DIRECTED)
    n = nv(g)
    
    # 1. Ground Truth (Exact BC)
    dataset_name = basename(TEST_FILE)
    cache_file = joinpath(@__DIR__, "$(dataset_name)_exact_bc.jld2")
    
    if isfile(cache_file)
        println("Loading cached exact betweenness centrality from $cache_file...")
        @load cache_file exact_scores
        exact_time = 0.0
    else
        println("Calculating exact betweenness (this may take a while)...")
        exact_time = @elapsed begin
            exact_scores = betweenness_centrality(g, normalize=true)
        end
        println("Exact calculation finished in $(round(exact_time, digits=2)) seconds. Saving to cache...")
        @save cache_file exact_scores
    end
    
    # 2. BRAVA-GNN setup
    model = BRAVAModel(m_hops=5, hidden_dim=12)
    # If a trained model exists, load it:
    weight_path = joinpath("..", "bravagnn_weights.jld2")
    if isfile(weight_path)
        println("Loading trained BRAVA-GNN weights from $weight_path...")
        @load weight_path model
    else
        println("No trained weights found. Using random initialized BRAVA-GNN weights.")
    end
    
    device = CUDA.functional() ? gpu : cpu
    if CUDA.functional()
        println("Using CUDA GPU for BRAVA-GNN inference...")
    else
        println("Using CPU for BRAVA-GNN inference...")
    end
    
    model = model |> device

    println("Running BRAVA-GNN warmup...")
    # Warmup
    A = sparse(g)
    A_t = A'
    X_out = compute_degree_masses(A, 5)
    X_in = compute_degree_masses(A_t, 5)
    
    A_gpu = device(A)
    A_t_gpu = device(A_t)
    X_in_gpu = device(X_in)
    X_out_gpu = device(X_out)
    
    _ = model(A_gpu, A_t_gpu, X_in_gpu, X_out_gpu)
    if CUDA.functional() CUDA.synchronize() end
    
    println("Running BRAVA-GNN evaluation...")
    brava_time = @elapsed begin
        brava_scores = model(A_gpu, A_t_gpu, X_in_gpu, X_out_gpu)
        if CUDA.functional() CUDA.synchronize() end
    end
    
    brava_scores = brava_scores |> cpu
    println("BRAVA-GNN finished in $(round(brava_time, digits=2)) seconds.")
    
    # 3. KADABRA and Baselines evaluation
    results = DataFrame(
        k = Int[],
        Kadabra_Time_s = Float64[],
        Kadabra_Overlap = Float64[],
        Kadabra_Tau = Float64[],
        Kadabra_NDCG = Float64[],
        BRAVA_Time_s = Float64[],
        BRAVA_Overlap = Float64[],
        BRAVA_Tau = Float64[],
        BRAVA_NDCG = Float64[],
        Degree_Overlap = Float64[],
        Degree_Tau = Float64[],
        Degree_NDCG = Float64[],
        PageRank_Overlap = Float64[],
        PageRank_Tau = Float64[],
        PageRank_NDCG = Float64[]
    )
    
    # Compute fast baselines
    println("Computing Heuristic Baselines...")
    degree_scores = degree(g)
    pagerank_scores = pagerank(g)
    
    # Warmup KADABRA
    _ = kadabra_centrality(g, 10, KADABRA_EPSILON, KADABRA_DELTA)
    
    for k in K_VALUES
        println("\nEvaluating k = $k ...")
        
        kadabra_time = @elapsed begin
            kadabra_scores = kadabra_centrality(g, k, KADABRA_EPSILON, KADABRA_DELTA)
        end
        
        overlap_kad = top_k_overlap(exact_scores, kadabra_scores, k)
        overlap_brava = top_k_overlap(exact_scores, vec(brava_scores), k)
        overlap_deg = top_k_overlap(exact_scores, degree_scores, k)
        overlap_pr = top_k_overlap(exact_scores, pagerank_scores, k)
        
        tau_kad = top_k_kendall_tau(exact_scores, kadabra_scores, k)
        tau_brava = top_k_kendall_tau(exact_scores, vec(brava_scores), k)
        tau_deg = top_k_kendall_tau(exact_scores, degree_scores, k)
        tau_pr = top_k_kendall_tau(exact_scores, pagerank_scores, k)

        ndcg_kad = ndcg_at_k(exact_scores, kadabra_scores, k)
        ndcg_brava = ndcg_at_k(exact_scores, vec(brava_scores), k)
        ndcg_deg = ndcg_at_k(exact_scores, degree_scores, k)
        ndcg_pr = ndcg_at_k(exact_scores, pagerank_scores, k)
        
        println("  KADABRA -> Time: $(round(kadabra_time, digits=2))s, Overlap: $(round(overlap_kad*100, digits=2))%, Tau: $(round(tau_kad, digits=4)), NDCG: $(round(ndcg_kad, digits=4))")
        println("  BRAVA   -> Time: $(round(brava_time, digits=2))s, Overlap: $(round(overlap_brava*100, digits=2))%, Tau: $(round(tau_brava, digits=4)), NDCG: $(round(ndcg_brava, digits=4))")
        println("  Degree  -> Overlap: $(round(overlap_deg*100, digits=2))%, Tau: $(round(tau_deg, digits=4)), NDCG: $(round(ndcg_deg, digits=4))")
        println("  PageRank-> Overlap: $(round(overlap_pr*100, digits=2))%, Tau: $(round(tau_pr, digits=4)), NDCG: $(round(ndcg_pr, digits=4))")
        
        push!(results, (
            k, kadabra_time, overlap_kad, tau_kad, ndcg_kad, 
            brava_time, overlap_brava, tau_brava, ndcg_brava,
            overlap_deg, tau_deg, ndcg_deg,
            overlap_pr, tau_pr, ndcg_pr
        ))
    end
    
    println("\n=== Final Top-k Results ===")
    display(results)
    
    CSV.write("topk_results.csv", results)
    println("\nResults saved to topk_results.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_topk_comparison()
end
