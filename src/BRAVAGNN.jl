module BRAVAGNN

using LinearAlgebra
using SparseArrays
using Random
using Flux

export compute_degree_masses, BRAVALayer, BRAVAModel, PairwiseDataLoader, margin_ranking_loss

# ==============================================================================
# 1. Multi-hop Degree Mass Pipeline
# ==============================================================================

"""
    compute_degree_masses(A, m::Int=5)

Computes the degree mass features up to order `m` for a sparse adjacency matrix `A`.
For an outgoing stream over A (which captures destinations reachable from sources), 
this computes out-degrees. For an incoming stream over A^T, pass `A'` to compute in-degrees.

Returns a dense `Float32` matrix of size `(m+1) × N`, where N is the number of nodes.
This layout is perfectly optimal for Flux.jl (Feature × Batch).
"""
function compute_degree_masses(A, m::Int=5)
    N = size(A, 1)
    # Pre-allocate output feature matrix F of size (m+1) x N in column-major
    F = zeros(Float32, m + 1, N)
    
    # d^(0) = row-degree vector (using Float32 for NN compatibility)
    # Since A corresponds to destinations, row sum is the out-degree.
    # Note: A * ones(N) computes the row sum.
    v = A * ones(Float32, N)
    F[1, :] .= v
    
    # Iteratively compute A^k * d
    curr_mass = copy(v)
    for k in 1:m
        v = A * v
        curr_mass .= curr_mass .+ v
        F[k+1, :] .= curr_mass
    end
    return F
end

# ==============================================================================
# 2. Compact, Size-Invariant GNN Architecture
# ==============================================================================

"""
Row-wise L2 normalization mapped across columns (features).
Given F x N matrix X, normalizes each node's F-dimensional feature vector.
"""
function norm2_features(X::AbstractMatrix)
    # Avoid division by zero
    norms = sqrt.(sum(X.^2, dims=1)) .+ eps(Float32)
    return X ./ norms
end

"""
    BRAVALayer(W)

A single message-passing layer for BRAVA-GNN.
Since Flux expects features as columns (F x N), the equation 
H_out = Norm2(ReLU(A^T H_out W)) translates perfectly to:
X_out = Norm2(ReLU(W * X_out * A)) in our column-major layout.
"""
struct BRAVALayer{T}
    W::T
end

Flux.@layer BRAVALayer

# W * X applies the feature transformation
# (W * X) * A applies the sparse aggregation over neighbors
# Notice that A here corresponds to outgoing edge propagation.
function (l::BRAVALayer)(X::AbstractMatrix, A)
    Z = l.W * X
    # Convert Dense * Sparse into (Sparse^T * Dense^T)^T to leverage fast cuSPARSE Sparse * Dense routines
    out = copy((A' * Z')')
    return norm2_features(relu.(out))
end

"""
    BRAVAModel(embedding, layers, mlp)

The dual-stream BRAVA-GNN model.
"""
struct BRAVAModel{E, L, M}
    embedding::E
    layers::L
    mlp::M
end

Flux.@layer BRAVAModel

function BRAVAModel(; m_hops::Int=5, hidden_dim::Int=12)
    # 1. DegreeMassEmbedding
    embedding = Dense(m_hops + 1 => hidden_dim, relu)
    
    # 2. Two message passing layers with shared weights
    # We use Dense without bias to represent the W matrix
    layer1 = BRAVALayer(Dense(hidden_dim => hidden_dim, bias=false).weight)
    layer2 = BRAVALayer(Dense(hidden_dim => hidden_dim, bias=false).weight)
    layers = (layer1, layer2)
    
    # 3. Shared MLP for prediction (dimensions 12 -> 24 -> 24 -> 1)
    mlp = Chain(
        Dense(hidden_dim => 24, relu),
        Dropout(0.3),
        Dense(24 => 24, relu),
        Dropout(0.3),
        Dense(24 => 1) # Outputs a scalar score
    )
    
    return BRAVAModel(embedding, layers, mlp)
end

"""
    (m::BRAVAModel)(A, A_t, X_in::AbstractMatrix, X_out::AbstractMatrix)

Forward pass of the BRAVA model.
A_t is the transpose of A (for incoming streams).
X_in and X_out are the degree mass features for incoming and outgoing streams respectively.
"""
function (m::BRAVAModel)(A, A_t, X_in::AbstractMatrix, X_out::AbstractMatrix)
    # --- Initial Embeddings ---
    H_out = norm2_features(m.embedding(X_out))
    H_in  = norm2_features(m.embedding(X_in))
    
    # Accumulated intermediate scores
    y_out = vec(m.mlp(H_out))
    y_in  = vec(m.mlp(H_in))
    
    # --- Dual Message Passing ---
    for layer in m.layers
        # Outgoing stream uses A_t for aggregation in the paper, which corresponds to A in our column-major math
        H_out = layer(H_out, A)
        # Incoming stream uses A for aggregation in the paper, which corresponds to A_t in our column-major math
        H_in  = layer(H_in, A_t)
        
        y_out = y_out .+ vec(m.mlp(H_out))
        y_in  = y_in .+ vec(m.mlp(H_in))
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

function Base.iterate(dl::PairwiseDataLoader, state=1)
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
function margin_ranking_loss(s::AbstractVector, U::AbstractVector{Int}, V::AbstractVector{Int}, Y::AbstractVector{Float32})
    # L(u, v) = max(0, 1 - Y * (s_u - s_v))
    diffs = s[U] .- s[V]
    margins = 1.0f0 .- Y .* diffs
    return sum(relu.(margins)) / length(U)
end

end # module BRAVAGNN
