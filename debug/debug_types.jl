using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs

include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g_static = StaticGraph(g_raw)

println("SimpleGraph eltype: ", eltype(g_raw))
println("StaticGraph eltype: ", eltype(g_static))
println("SimpleGraph type: ", typeof(g_raw))
println("StaticGraph type: ", typeof(g_static))

# Check if degree functions work the same
v = 1
println("\nVertex $v:")
println("  SimpleGraph outdegree: ", outdegree(g_raw, v))
println("  StaticGraph outdegree: ", outdegree(g_static, v))
println("  SimpleGraph indegree: ", indegree(g_raw, v))
println("  StaticGraph indegree: ", indegree(g_static, v))
println("  SimpleGraph neighbors: ", outneighbors(g_raw, v))
println("  StaticGraph neighbors: ", outneighbors(g_static, v))

# Check max degree
max_deg_raw = maximum(degree(g_raw, v) for v in vertices(g_raw))
max_deg_static = maximum(degree(g_static, v) for v in vertices(g_static))
println("\nMax degree SimpleGraph: ", max_deg_raw)
println("Max degree StaticGraph: ", max_deg_static)
