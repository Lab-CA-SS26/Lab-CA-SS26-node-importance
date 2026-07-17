using Pkg
Pkg.activate(".")
Pkg.add(["Graphs", "SparseArrays", "DataFrames", "CSV", "Flux", "JLD2", "Random", "Statistics", "Optimisers", "CUDA", "cuDNN"])

using Graphs
using SparseArrays
using CSV
using DataFrames
using Flux
using CUDA
using cuDNN
using JLD2
using Random
using Statistics
using Optimisers

include("BRAVAGNN.jl")
using .BRAVAGNN

const TRAINING_DIR = joinpath(dirname(@__DIR__), "Instances", "Training")
const BATCH_SIZE = 1024
const EPOCHS = 10
const LEARNING_RATE = 5e-3

# Parse the graph depending on whether it is BO/UDHY (Directed) or SBO (Undirected)
function load_training_graph(filepath::String, is_directed::Bool)
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
    
    g = is_directed ? SimpleDiGraph(max_node + shift) : SimpleGraph(max_node + shift)
    for (u, v) in edges_list
        add_edge!(g, u + shift, v + shift)
    end
    return g
end

function train()
    println("--- BRAVA-GNN Training Script ---")
    
    # 1. Load Data
    graph_files = filter(f -> endswith(f, ".txt"), readdir(TRAINING_DIR))
    println("Found $(length(graph_files)) training graphs.")
    
    training_data = []
    
    for gf in graph_files
        name = replace(gf, ".txt" => "")
        is_directed = occursin("Dir", name)
        
        # Load graph
        g = load_training_graph(joinpath(TRAINING_DIR, gf), is_directed)
        
        # Load scores
        scores_df = CSV.read(joinpath(TRAINING_DIR, "$(name)_scores.csv"), DataFrame)
        scores = scores_df.betweenness
        
        # Precompute PageRank feature once for the graph
        A = Float32.(sparse(g))
        A_t = A'
        
        pr_feat = compute_pagerank_feature(A)
        
        X_out = compute_degree_masses(A, pr_feat, 5)
        X_in = compute_degree_masses(A_t, pr_feat, 5)
        
        # GPU transfer if available
        device = CUDA.functional() ? gpu : cpu
        
        A_gpu = device(A)
        A_t_gpu = device(A_t)
        X_in_gpu = device(X_in)
        X_out_gpu = device(X_out)
        
        push!(training_data, (A=A_gpu, A_t=A_t_gpu, X_in=X_in_gpu, X_out=X_out_gpu, scores=scores))
    end
    
    device = CUDA.functional() ? gpu : cpu
    
    # 2. Initialize Model
    model = BRAVAModel(m_hops=5, hidden_dim=12, num_layers=2) |> device
    opt_state = Flux.setup(Flux.Adam(LEARNING_RATE), model)
    
    # 3. Training Loop
    for epoch in 1:EPOCHS
        epoch_loss = 0.0
        n_batches = 0
        
        for virtual_copy in 1:50
            Random.shuffle!(training_data)
            
            for data in training_data
                N = length(data.scores)
                k = N * 20
                
                U = rand(1:N, k)
                V = rand(1:N, k)
                diffs = data.scores[U] .- data.scores[V]
                Y = sign.(diffs)
                
                # To match PyTorch, we can filter out ties if we want, or just let gradient be 0.
                # PyTorch computes the margin loss directly. Our margin_ranking_loss handles Y=0 natively
                # by having a gradient of 0, but filtering is slightly faster:
                valid_idx = findall(x -> x != 0, diffs)
                U = U[valid_idx]
                V = V[valid_idx]
                Y = Y[valid_idx]
                
                U_dev = device(U)
                V_dev = device(V)
                Y_dev = device(Float32.(Y))
                
                loss_val, grads = Flux.withgradient(model) do m
                    preds = m(data.A, data.A_t, data.X_in, data.X_out)
                    margin_ranking_loss(preds, U_dev, V_dev, Y_dev)
                end
                
                Flux.update!(opt_state, model, grads[1])
                
                epoch_loss += loss_val
                n_batches += 1
            end
        end
        
        println("Epoch $epoch / $EPOCHS - Avg Loss: $(round(epoch_loss / max(n_batches, 1), digits=4))")
    end
    
    # 4. Save Model (Move back to CPU before saving)
    model = model |> cpu
    save_path = joinpath(dirname(@__DIR__), "benchmark", "cache", "bravagnn_weights.jld2")
    @save save_path model
    println("Training complete. Model saved to $save_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
