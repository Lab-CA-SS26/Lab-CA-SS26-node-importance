using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random, Base.Threads

include("src/kadabra.jl")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

println("Loading graph...")
g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g = StaticGraph(g_raw)
println("Graph loaded")

# Timer to print stacktraces after 10 seconds
t = Timer(10.0) do _
    println("TIMER TRIGGERED! Printing stacktraces:")
    for tid in 1:nthreads()
        println("--- Thread $tid ---")
        # In Julia, there isn't a direct way to get another thread's stacktrace easily,
        # but wait, I can just throw an error in all threads by setting a global flag!
    end
end

global_hang_flag = Threads.Atomic{Bool}(false)
# Let's modify kadabra to check global_hang_flag
