import Pkg
Pkg.activate(mktempdir())
Pkg.add(["Flux"])

include("src/BRAVAGNN.jl")
using .BRAVAGNN
using SparseArrays
using Flux

println("Testing Multi-hop Degree Mass Pipeline...")
# Create a dummy directed graph (A_ij = 1 means i -> j)
N = 100
A = sprand(N, N, 0.05)

# For outgoing features, we pass A
X_out = compute_degree_masses(A, 5)
@assert size(X_out) == (6, N)

# For incoming features, we pass A'
X_in = compute_degree_masses(A', 5)
@assert size(X_in) == (6, N)

println("Degree mass features tested successfully.")

println("Testing Compact GNN Architecture...")
model = BRAVAModel(m_hops=5, hidden_dim=12)

# Forward pass
scores = model(A, A', X_in, X_out)
@assert length(scores) == N

println("Forward pass tested successfully. Computed scores for $N nodes.")

println("Testing Pairwise Mini-batch Data Loader...")
dl = PairwiseDataLoader(scores, 32, 10) # batch size 32, 10 batches per epoch

batches_yielded = 0
for (U, V, Y) in dl
    global batches_yielded += 1
    @assert length(U) == 32
    @assert length(V) == 32
    @assert length(Y) == 32
    @assert all(abs.(Y) .== 1.0)
end
@assert batches_yielded == 10

println("Pairwise mini-batch loader tested successfully.")

println("Testing Margin Ranking Loss...")
(U, V, Y), _ = iterate(dl, 1)
loss = margin_ranking_loss(scores, U, V, Y)
@assert loss >= 0.0

println("Margin ranking loss computed successfully: ", loss)
println("All tests passed!")
