using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")

println("Threads: $(Threads.nthreads())")

# Exact same warmup as run_experiments.jl
dummy_g_raw = SimpleGraph{Int64}(3)
add_edge!(dummy_g_raw, 1, 2)
add_edge!(dummy_g_raw, 2, 3)
dummy_g = StaticGraph(dummy_g_raw)

println("dummy_g: nv=$(nv(dummy_g)), ne=$(ne(dummy_g)), eltype=$(eltype(dummy_g))")

# Check params
n = nv(dummy_g)
diam_est = max(estimate_diameter(dummy_g), 2.0)
omega = 0.5 / (0.01^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / 0.1))
tau = max(round(Int, omega / 10), 1)
println("diam=$diam_est, omega=$omega, tau=$tau")
println("nthreads=$(Threads.nthreads()), tau_per_thread=$(cld(tau, Threads.nthreads()))")
check_interval = min(10000, max(100, tau ÷ 100))
println("check_interval=$check_interval")
flush(stdout)

println("\nRunning kadabra warmup on dummy graph...")
flush(stdout)
t1 = time()
res = kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false)
println("Warmup done in $(round(time()-t1, digits=2))s, n_samples=$(res.n_samples)")
flush(stdout)
