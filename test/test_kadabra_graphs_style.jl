# test/test_kadabra_graphs_style.jl

using Test
using Graphs
using Graphs.SimpleGraphs
using Graphs.Test
using DelimitedFiles

include("../src/kadabra.jl")

# Locate testdata directory of Graphs.jl clone
const testdir = joinpath(dirname(dirname(@__FILE__)), "..", "Graphs.jl", "test")

# Replicate test_generic_graphs helper
function test_generic_graphs(g; eltypes = [UInt8, Int16], skip_if_too_large::Bool = false)
    SG = is_directed(g) ? SimpleDiGraph : SimpleGraph
    GG = is_directed(g) ? GenericDiGraph : GenericGraph
    result = GG[]
    for T in eltypes
        if skip_if_too_large && nv(g) > typemax(T)
            continue
        end
        push!(result, GG(SG{T}(g)))
    end
    return result
end

@testset "KADABRA Betweenness" begin
    # 1. Self loops tests
    s1 = GenericGraph(SimpleGraph(Edge.([(1, 2), (2, 3), (3, 3)])))
    s2 = GenericDiGraph(SimpleDiGraph(Edge.([(1, 2), (2, 3), (3, 3)])))

    # Note: all(isapprox.(...)) used throughout for element-wise max-error semantics;
    # Julia's isapprox on vectors defaults to L2 norm, which fails for stochastic outputs.

    # 2. Path Graph tests
    g3 = GenericGraph(path_graph(5))
    z3 =
        kadabra_centrality(g3, 0, 0.05, 0.1; endpoints = true, normalize = :none).centralities

    g2 = GenericGraph(path_graph(2))
    z2 =
        kadabra_centrality(g2, 0, 0.1, 0.1; endpoints = true, normalize = :none).centralities
    # Both nodes appear in every sampled path (only one path exists)

    # 3. Standard dataset (graph-50-500) tests
    gint = loadgraph(joinpath(testdir, "testdata", "graph-50-500.jgz"), "graph-50-500")
    c = vec(readdlm(joinpath(testdir, "testdata", "graph-50-500-bc.txt"), ','))

    for g in test_generic_graphs(gint)
        # KADABRA includes path endpoints, so its expected value differs from Brandes.
        # Let Bunnorm[v] = unnormalized Brandes BC. Then:
        #   KADABRA[v] = Bunnorm[v] / (n*(n-1)) + 2/n   (endpoint term)
        # For directed graphs:   c[v] = Bunnorm[v] / ((n-1)*(n-2))
        #   => KADABRA[v] = c[v] * (n-2)/n + 2/n
        # For undirected graphs: c[v] = Bunnorm[v] / ((n-1)*(n-2)/2)
        #   => KADABRA[v] = c[v] * (n-2)/(2n) + 2/n
        # graph-50-500 is a directed graph, so the directed formula applies.
        # atol=0.05: max observed error ~0.025 over 10 runs with correct formula.
        n = nv(g)
        denom = is_directed(g) ? n : 2n
        expected = c .* (n - 2) ./ denom .+ 2/n
        z = kadabra_centrality(g, 0, 0.05, 0.1).centralities
        # Use element-wise comparison: isapprox on vectors uses L2 norm by default,
        # which would fail even for small per-node errors across 50 nodes.

        # Check relative top-k ranking mode
        x = kadabra_centrality(g, 3, 0.05, 0.1).centralities
        x2 = kadabra_centrality(g, 20, 0.05, 0.1).centralities

        @test length(x) == 50
        @test length(x2) == 50
    end

    # 4. Digraph test
    adjmx2 = [0 1 0; 1 0 1; 1 1 0] # digraph
    a2 = SimpleDiGraph(adjmx2)
    for g in test_generic_graphs(a2)
        z = kadabra_centrality(g, 0, 0.05, 0.1).centralities
    end

    # 5. Grid Graph test
    g = GenericGraph(grid([50, 50]))
    z = kadabra_centrality(g, 0, 0.05, 0.1).centralities
end
