using Pkg
Pkg.activate(".")

using Graphs
using SparseArrays
include("src/BRAVAGNN.jl")
using .BRAVAGNN

# Create a simple directed graph
g = SimpleDiGraph(5)
# 1 -> 2, 1 -> 3, 2 -> 3  (Node 2's neighbors form a clique)
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 2, 3)

# 4 is a leaf (only in-degree)
add_edge!(g, 3, 4)
# 5 is isolated
# Node 1 has in-degree 0 (pruned)
# Node 4 has out-degree 0 (pruned)
# Node 5 has in-degree 0 and out-degree 0 (pruned)
# Node 2 has in-degree 1 (from 1) and out-degree 1 (to 3). 
# Edge 1->3 exists, so Node 2's neighbors form a clique (pruned)
# Node 3 has in-degree 2, out-degree 1 (to 4).
# Node 3's in-neighbors are 1, 2. Out-neighbor is 4.
# Edge 1->4 does NOT exist, so Node 3 is NOT pruned. Wait, Node 3 should be kept!

println("Graph edges: ", collect(edges(g)))
mask = brava_clique_mask(g)
println("Mask: ", mask)

A = Float32.(sparse(g))
A_scaled = spdiagm(mask) * A

println("Original A:")
display(Matrix(A))
println("Scaled A:")
display(Matrix(A_scaled))
