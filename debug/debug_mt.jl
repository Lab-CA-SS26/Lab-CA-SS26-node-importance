using Pkg
Pkg.activate(".")

using Graphs
using Random

include("src/kadabra.jl")

# Load graph
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils
g = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)

n = nv(g)
println("Graph: nv=$n, ne=$(ne(g))")

# Reproduce the parameter computation
err = 0.01
delta = 0.1
k_input = 0
start_factor = 100

absolute = (k_input == 0)
k = Int(k_input == 0 ? n : min(k_input, n))

diam_est = max(estimate_diameter(g), 2.0)
omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
tau = max(round(Int, omega / start_factor), 1)

nthreads = Threads.nthreads()
tau_per_thread = cld(tau, nthreads)

# Parallel check_interval
check_interval_par = min(10000, max(100, tau ÷ 100))
# Sequential check_interval
check_interval_seq = max(1000, tau ÷ 10)

println("diam_est = $diam_est")
println("omega = $omega")
println("tau = $tau")
println("nthreads = $nthreads")
println("tau_per_thread = $tau_per_thread")
println("check_interval (parallel) = $check_interval_par")
println("check_interval (sequential) = $check_interval_seq")
println("union_sample = $(Int(absolute ? k : min(n, k + 20)))")
println("k = $k (absolute=$absolute)")
println()

# Key insight: with nthreads=16, the threshold for checking is:
threshold = check_interval_par * nthreads
println("Threshold for check (check_interval * nthreads) = $threshold")
println("Phase 1 total samples = $(tau_per_thread * nthreads)")
println()

# Now let's test: does check_finished ever return true?
# Run sequentially first to see the sample count needed
println("Running sequential kadabra to find sample count needed...")
flush(stdout)
res_seq = kadabra_centrality(g, k_input, err, delta; parallel=false, start_factor=100)
println("Sequential finished with n_samples = $(res_seq.n_samples)")
flush(stdout)

# Now run parallel
if nthreads > 1
    println("\nRunning parallel kadabra with $nthreads threads...")
    flush(stdout)
    res_par = kadabra_centrality(g, k_input, err, delta; parallel=true, start_factor=100)
    println("Parallel finished with n_samples = $(res_par.n_samples)")
else
    println("\nSkipping parallel test (only 1 thread available)")
    println("Re-run with: julia --project=@. -t 16 debug_mt.jl")
end
