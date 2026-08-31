using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs

include("kadabra_yesterday.jl")

println("Loading graph...")
g_raw = Graph("com-youtube.txt") # Assuming edges list or similar, wait actually let's use the same graph loading as debug_mt6.jl
