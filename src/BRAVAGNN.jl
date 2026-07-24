module BRAVAGNN

using LinearAlgebra
using SparseArrays
using Random
using Flux
using Graphs

export compute_degree_masses,
    compute_pagerank_feature,
    BRAVALayer,
    BRAVAModel,
    PairwiseDataLoader,
    margin_ranking_loss,
    brava_centrality,
    brava_clique_mask

# ==============================================================================
# 1. Preprocessing Heuristics
# ==============================================================================

"""
    brava_clique_mask(g::AbstractGraph)

Returns a `Vector{Float32}` of length `nv(g)` where the value is 0.0f0 for pruned nodes
and 1.0f0 for kept nodes, following the BRAVA clique-neighborhood preprocessing rule.
"""
function brava_clique_mask(g::AbstractGraph)
    N = nv(g)
    mask = ones(Float32, N)
    
    for v in 1:N
        if indegree(g, v) == 0 || outdegree(g, v) == 0
            mask[v] = 0.0f0
            continue
        end
        
        in_nodes = inneighbors(g, v)
        out_nodes = outneighbors(g, v)
        
        is_clique = true
        for u in in_nodes
            for w in out_nodes
                if w != u && !has_edge(g, u, w)
                    is_clique = false
                    break
                end
            end
            if !is_clique
                break
            end
        end
        
        if is_clique
            mask[v] = 0.0f0
        end
    end
    
    return mask
end

# ==============================================================================
# 2. Multi-hop Degree Mass Pipeline
# ==============================================================================

"""
    compute_pagerank_feature(A::AbstractSparseMatrix, alpha::Float32=0.85f0, iters::Int=80)

Computes the PageRank of the graph and applies the exact same transformation as the PyTorch reference:
log1p(PageRank * N) followed by Max-Normalization.
"""
function compute_pagerank_feature(
    A::AbstractSparseMatrix,
    alpha::Float32 = 0.85f0,
    iters::Int = 80,
)
    N = size(A, 1)
    # Compute out-degree (row sums)
    deg = vec(sum(A, dims = 2))
    deg_inv = Float32[d > 0 ? 1.0f0 / d : 0.0f0 for d in deg]

    # Transition matrix P
    P = spdiagm(deg_inv) * A
    P_t = P'

    p = fill(1.0f0 / N, N)
    for _ = 1:iters
        new_p = alpha .* (P_t * p) .+ (1.0f0 - alpha) / N
        if maximum(abs.(new_p .- p)) < 1e-9
            p = new_p
            break
        end
        p = new_p
    end

    pr_feat = log1p.(p .* N)
    return pr_feat ./ (maximum(pr_feat) + 1.0f-6)
end

"""
    compute_degree_masses(A, pr_feat::AbstractVector{Float32}, m::Int=5)

Computes the degree mass features up to order `m` for a sparse adjacency matrix `A`,
applies log1p, and appends the precomputed PageRank feature to perfectly match `degree_mix_mass_5_pr`.
"""
function compute_degree_masses(A, pr_feat::AbstractVector{Float32}, m::Int = 5)
    N = size(A, 1)
    F = zeros(Float32, m + 1, N)

    v = A * ones(Float32, N)
    F[1, :] .= v

    for k = 2:m
        v = A * v
        F[k, :] .= v
    end

    # Apply log1p exactly like PyTorch (but ONLY to degree features)
    F[1:m, :] .= log1p.(F[1:m, :])

    # Append PageRank as the final feature (already normalized and scaled)
    F[m+1, :] .= pr_feat
    return F
end

# ==============================================================================
# 2. Compact, Size-Invariant GNN Architecture
# ==============================================================================

"""
Row-wise L2 normalization mapped across columns (features).
"""
function norm2_features(X::AbstractMatrix)
    norms = sqrt.(sum(X .^ 2, dims = 1)) .+ eps(Float32)
    return X ./ norms
end

"""
    BRAVALayer(W, b)

A single message-passing layer for BRAVA-GNN.
Matches PyTorch's GNN_Layer exactly: Y = A * (X W) + b
In column-major layout: Y = (A * (W X)^T)^T + b
"""
struct BRAVALayer{T,B}
    W::T
    b::B
end

Flux.@layer BRAVALayer

import Flux.ChainRulesCore: rrule
using Flux.ChainRulesCore: unthunk, NoTangent

sparse_dense_mul(A, B) = A * B

function rrule(::typeof(sparse_dense_mul), A, B)
    Y = A * B
    function sparse_dense_mul_pullback(ΔY)
        ΔB = A' * unthunk(ΔY)
        return (NoTangent(), NoTangent(), ΔB)
    end
    return Y, sparse_dense_mul_pullback
end

function (l::BRAVALayer)(X::AbstractMatrix, A_transposed)
    Z = l.W * X
    out = copy(sparse_dense_mul(A_transposed, Z')')
    return out .+ l.b
end

"""
    BRAVAModel(embedding, layers, dropout, mlp)

The dual-stream BRAVA-GNN model.
"""
struct BRAVAModel{E,L,D,M}
    embedding::E
    layers::L
    dropout::D
    mlp::M
end

Flux.@layer BRAVAModel

function BRAVAModel(;
    m_hops::Int = 5,
    hidden_dim::Int = 12,
    num_layers::Int = 4,
    p_drop::Float32 = 0.3f0,
)
    # 1. DegreeMassEmbedding + PageRank
    embedding = Dense(m_hops + 1 => hidden_dim, bias = true)

    # 2. PyTorch uses independent GNN_Layers (not shared)
    layers = Tuple([
        BRAVALayer(
            Dense(hidden_dim => hidden_dim, bias = true).weight,
            Dense(hidden_dim => hidden_dim, bias = true).bias,
        ) for _ = 1:num_layers
    ])

    # 3. Dropout
    dropout = Dropout(p_drop)

    # 4. MLP for prediction
    mlp = Chain(
        Dense(hidden_dim => 24, relu),
        Dropout(p_drop),
        Dense(24 => 24, relu),
        Dropout(p_drop),
        Dense(24 => 1),
    )

    return BRAVAModel(embedding, layers, dropout, mlp)
end

function (m::BRAVAModel)(A, A_t, X_in::AbstractMatrix, X_out::AbstractMatrix)
    # --- Initial Embeddings ---
    # PyTorch: x = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
    # In PyTorch, gc1 is GNN_Layer_Init which applies AW + b. But wait!
    # Our embedding just does W X + b (no adjacency multiplication on the first step for degree_mass).
    # This matches PyTorch's logic for degree_mix_mass, which skips A*W for the first layer.
    H_out = norm2_features(relu.(m.embedding(X_out)))
    H_in = norm2_features(relu.(m.embedding(X_in)))

    y_out = vec(m.mlp(H_out))
    y_in = vec(m.mlp(H_in))

    num_layers = length(m.layers)
    for (i, layer) in enumerate(m.layers)
        # Message passing + Bias
        # Python implementation uses regular adj (A) for H_out (forward stream)
        # and adj_t (A_t) for H_in (backward stream)
        H_out_new = layer(H_out, A)
        H_in_new = layer(H_in, A_t)

        # Non-linearity
        H_out = m.dropout(relu.(H_out_new))
        H_in = m.dropout(relu.(H_in_new))

        # Normalize all except last layer
        if i < num_layers
            H_out = norm2_features(H_out)
            H_in = norm2_features(H_in)
        end

        y_out = y_out .+ vec(m.mlp(H_out))
        y_in = y_in .+ vec(m.mlp(H_in))
    end

    # Multiplicative Score Fusion
    return y_in .* y_out
end

# ==============================================================================
# 3. Pairwise Mini-batch Data Loader
# ==============================================================================

"""
    PairwiseDataLoader

Iterator that yields mini-batches `(U, V, Y)` for Margin Ranking Loss without storing \$O(N^2)\$ pairs.
It dynamically samples uniformly from `1:N` and rejects pairs with tied centrality.
"""
struct PairwiseDataLoader
    scores::Vector{Float32}
    batchsize::Int
    batches_per_epoch::Int
end

Base.length(dl::PairwiseDataLoader) = dl.batches_per_epoch

function Base.iterate(dl::PairwiseDataLoader, state = 1)
    if state > dl.batches_per_epoch
        return nothing
    end

    N = length(dl.scores)
    U = Vector{Int}(undef, dl.batchsize)
    V = Vector{Int}(undef, dl.batchsize)
    Y = Vector{Float32}(undef, dl.batchsize)

    count = 1
    while count <= dl.batchsize
        u = rand(1:N)
        v = rand(1:N)

        # Omit pairs with equal ground-truth betweenness
        diff = dl.scores[u] - dl.scores[v]
        if diff != 0
            U[count] = u
            V[count] = v
            Y[count] = sign(diff) # 1 if u > v, -1 if u < v
            count += 1
        end
    end

    return ((U, V, Y), state + 1)
end

# ==============================================================================
# 4. Margin Ranking Loss
# ==============================================================================

"""
    margin_ranking_loss(s, U, V, Y)

Computes the pairwise margin ranking loss for the predicted scores `s` over indices `U` and `V` with labels `Y`.
`Y` is +1 if s_U should be > s_V, and -1 if s_U should be < s_V.
"""
function margin_ranking_loss(s, U, V, Y)
    # L(u, v) = max(0, 1 - Y * (s_u - s_v))
    diffs = s[U] .- s[V]
    margins = 1.0f0 .- Y .* diffs
    return sum(relu.(margins)) / length(U)
end

# ==============================================================================
# 5. Brava Centrality Inference Wrapper
# ==============================================================================

using Graphs
using JLD2

"""
    brava_centrality(g::AbstractGraph, k::Int, err::Float64, delta::Float64; m_hops::Int=5)

Inference wrapper to compute betweenness centrality scores using the BRAVA GNN model.
Provides a similar API interface to `kadabra_centrality` for benchmarking.
Returns a tuple of `(centrality_scores, num_samples)`, where `num_samples` is 1 for GNN.
"""
function brava_centrality(
    g::AbstractGraph,
    k::Int,
    err::Float64,
    delta::Float64;
    m_hops::Int = 6,
    weight_path::Union{String,Nothing} = nothing,
    model = nothing,
    use_gpu::Bool = false,
)
    N = nv(g)
    
    # Extract sparse adjacency matrix safely (bypassing StaticGraphs.jl UInt mismatch bugs)
    # We build the I, J vectors directly from the edges.
    n_edges = is_directed(g) ? ne(g) : 2 * ne(g)
    I_idx = Vector{Int}(undef, n_edges)
    J_idx = Vector{Int}(undef, n_edges)
    V_val = ones(Float32, n_edges)
    
    count = 1
    for e in edges(g)
        I_idx[count] = src(e)
        J_idx[count] = dst(e)
        count += 1
        if !is_directed(g)
            I_idx[count] = dst(e)
            J_idx[count] = src(e)
            count += 1
        end
    end
    
    A = sparse(I_idx, J_idx, V_val, N, N)
    A_t = SparseMatrixCSC{Float32,Int}(A')

    # 1. Feature Extraction
    pr_feat = compute_pagerank_feature(A)
    X_out = compute_degree_masses(A, pr_feat, m_hops)
    X_in = compute_degree_masses(A_t, pr_feat, m_hops)

    # 2. Initialize Model 
    if model === nothing
        if weight_path === nothing
            weight_path = joinpath(@__DIR__, "..", "benchmark", "cache", "bravagnn_weights.jld2")
        end
        
        if isfile(weight_path)
            # Load the pre-trained model from the server cache
            @load weight_path model
        else
            # Fallback to untrained model for local runtime tests if missing
            model = BRAVAModel(; m_hops = m_hops, hidden_dim = 12, num_layers = 2)
        end
    end

    # 3. Handle GPU transfer if requested
    if use_gpu
        try
            @eval Main import CUDA
            # Use Core.eval to evaluate in the latest world age and avoid world age errors in Julia 1.12
            cuda_functional = Core.eval(Main, :(CUDA.functional()))
            if cuda_functional
                device = Flux.gpu
            else
                println("Warning: GPU requested but CUDA is not functional. Falling back to CPU.")
                device = Flux.cpu
            end
        catch e
            println("Warning: Could not load CUDA.jl (is it installed?). Falling back to CPU. Error: ", e)
            device = Flux.cpu
        end
    else
        device = Flux.cpu
    end

    A_dev = device(A)
    A_t_dev = device(A_t)
    X_in_dev = device(X_in)
    X_out_dev = device(X_out)
    model = model |> device

    # 4. Inference
    scores_dev = model(A_dev, A_t_dev, X_in_dev, X_out_dev)
    
    # 5. Bring scores back to CPU
    scores = scores_dev |> Flux.cpu

    return scores, 1
end

end # module BRAVAGNN
