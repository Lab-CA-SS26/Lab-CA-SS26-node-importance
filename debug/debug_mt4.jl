using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g_static = StaticGraph(g_raw)

n = nv(g_static)
T = eltype(g_static)

println("Testing BFS speed: SimpleGraph vs StaticGraph")
println("eltype(g_static) = $T")

# Time 100 BFS samples on SimpleGraph
ws_raw = KadabraWorkspace(g_raw)
counts_raw = zeros(Int, nv(g_raw))
rng1 = Random.Xoshiro(42)
t1 = time()
for _ = 1:100
    s = rand(rng1, 1:nv(g_raw))
    t = rand(rng1, 1:nv(g_raw))
    while s == t; t = rand(rng1, 1:nv(g_raw)); end
    sample_shortest_path!(counts_raw, ws_raw, g_raw, rng1, Int64(s), Int64(t))
end
elapsed_raw = time() - t1
println("SimpleGraph{Int64}: 100 samples in $(round(elapsed_raw, digits=3))s ($(round(elapsed_raw*10, digits=1))ms/sample)")

# Time 100 BFS samples on StaticGraph
ws_static = KadabraWorkspace(g_static)
counts_static = zeros(Int, n)
rng2 = Random.Xoshiro(42)
t2 = time()
for _ = 1:100
    s = rand(rng2, 1:n)
    t = rand(rng2, 1:n)
    while s == t; t = rand(rng2, 1:n); end
    sample_shortest_path!(counts_static, ws_static, g_static, rng2, T(s), T(t))
end
elapsed_static = time() - t2
println("StaticGraph{UInt32}: 100 samples in $(round(elapsed_static, digits=3))s ($(round(elapsed_static*10, digits=1))ms/sample)")
println("Ratio: $(round(elapsed_static/elapsed_raw, digits=1))x")

# Now test: does the 16-thread version actually run the inner loop or is it stuck elsewhere?
println("\nTesting if Phase 2 inner loop executes at all with StaticGraph + 16 threads...")
flush(stdout)

nthreads_val = Threads.nthreads()
println("nthreads = $nthreads_val")

approx_local = [zeros(Int, n) for _ = 1:nthreads_val]
workspaces = [KadabraWorkspace(g_static) for _ = 1:nthreads_val]
thread_seeds = [rand(Random.default_rng(), UInt64) for _ = 1:nthreads_val]

progress = [Threads.Atomic{Int}(0) for _ = 1:nthreads_val]

println("Launching 16 threads, each doing 10 batches of 100 samples...")
flush(stdout)

t3 = time()
Threads.@threads for tid = 1:nthreads_val
    ws = workspaces[tid]
    cnts = approx_local[tid]
    t_rng = Random.Xoshiro(thread_seeds[tid])
    
    for batch = 1:10
        for _ = 1:100
            GC.safepoint()
            s = rand(t_rng, 1:n)
            t = rand(t_rng, 1:n)
            while s == t; t = rand(t_rng, 1:n); end
            sample_shortest_path!(cnts, ws, g_static, t_rng, T(s), T(t); endpoints=false)
        end
        Threads.atomic_add!(progress[tid], 1)
    end
end
elapsed_mt = time() - t3
println("All threads done in $(round(elapsed_mt, digits=2))s")
for tid = 1:nthreads_val
    println("  Thread $tid completed $(progress[tid][]) batches")
end
flush(stdout)
