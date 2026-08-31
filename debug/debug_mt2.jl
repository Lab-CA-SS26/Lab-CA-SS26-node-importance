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

err = 0.01
delta = 0.1
k_input = 0
start_factor = 100

absolute = (k_input == 0)
k = Int(k_input == 0 ? n : min(k_input, n))

diam_est = max(estimate_diameter(g), 2.0)
omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
tau = max(round(Int, omega / start_factor), 1)

nthreads_val = Threads.nthreads()
tau_per_thread = cld(tau, nthreads_val)
check_interval = min(10000, max(100, tau ÷ 100))

println("omega = $omega, tau = $tau, nthreads = $nthreads_val")
println("check_interval = $check_interval")
println("k = $k (absolute=$absolute)")
println()

if nthreads_val <= 1
    println("ERROR: Need multiple threads. Run with: julia --project=@. -t 16 debug_mt2.jl")
    exit(1)
end

# Reproduce the parallel algorithm with debug prints
approx_local = [zeros(Int, n) for _ = 1:nthreads_val]
workspaces = [KadabraWorkspace(g) for _ = 1:nthreads_val]
n_pairs = Threads.Atomic{Int}(0)

base_rng = Random.default_rng()
thread_seeds = [rand(base_rng, UInt64) for _ = 1:nthreads_val]

# PHASE 1
println("Phase 1: burn-in with tau_per_thread=$tau_per_thread ...")
flush(stdout)
Threads.@threads for tid = 1:nthreads_val
    ws = workspaces[tid]
    counts = approx_local[tid]
    t_rng = Random.Xoshiro(thread_seeds[tid])
    for _ = 1:tau_per_thread
        GC.safepoint()
        s = rand(t_rng, 1:n)
        t = rand(t_rng, 1:n)
        while s == t; t = rand(t_rng, 1:n); end
        sample_shortest_path!(counts, ws, g, t_rng, eltype(g)(s), eltype(g)(t); endpoints=false)
    end
end
n_pairs[] = tau_per_thread * nthreads_val
println("Phase 1 done. n_pairs = $(n_pairs[])")
flush(stdout)

# PHASE 2 with debug
global_approx = zeros(Int, n)
union_sample = Int(absolute ? k : min(n, k + 20))
delta_l_guess = fill(delta / (4 * n), n)
delta_u_guess = fill(delta / (4 * n), n)
bet_buf = zeros(Float64, union_sample)
err_l_buf = zeros(Float64, union_sample)
err_u_buf = zeros(Float64, union_sample)
top_k_nodes = collect(1:n)

stop_flag = Threads.Atomic{Bool}(false)
check_lock = Threads.SpinLock()
last_check = Threads.Atomic{Int}(tau_per_thread * nthreads_val)
check_count = Threads.Atomic{Int}(0)

println("Phase 2: main loop, omega=$omega ...")
flush(stdout)

start_time = time()

Threads.@threads for tid = 1:nthreads_val
    ws = workspaces[tid]
    counts = approx_local[tid]
    t_rng = Random.Xoshiro(thread_seeds[tid])
    local_pairs = 0

    while !stop_flag[] && n_pairs[] < omega
        for _ = 1:check_interval
            GC.safepoint()
            s = rand(t_rng, 1:n)
            t = rand(t_rng, 1:n)
            while s == t; t = rand(t_rng, 1:n); end
            sample_shortest_path!(counts, ws, g, t_rng, eltype(g)(s), eltype(g)(t); endpoints=false)
            local_pairs += 1
        end

        Threads.atomic_add!(n_pairs, local_pairs)
        local_pairs = 0

        if trylock(check_lock)
            try
                current_n_pairs_global = n_pairs[]
                
                if current_n_pairs_global - last_check[] >= check_interval * nthreads_val
                    last_check[] = current_n_pairs_global
                    Threads.atomic_add!(check_count, 1)

                    fill!(global_approx, 0)
                    for i = 1:nthreads_val
                        global_approx .+= approx_local[i]
                    end

                    # Debug: Check sum vs n_pairs
                    total_sum = sum(global_approx)

                    if absolute
                        for i = 1:union_sample
                            top_k_nodes[i] = i
                        end
                    else
                        copyto!(top_k_nodes, 1:n)
                        partialsort!(top_k_nodes, 1:union_sample, by=x->global_approx[x], rev=true)
                    end

                    result = check_finished(
                        global_approx,
                        view(top_k_nodes, 1:union_sample),
                        current_n_pairs_global,
                        k,
                        err,
                        delta_l_guess,
                        delta_u_guess,
                        omega,
                        absolute,
                        bet_buf,
                        err_l_buf,
                        err_u_buf,
                    )

                    elapsed = time() - start_time
                    
                    if check_count[] <= 5 || check_count[] % 10 == 0
                        # Show debug stats
                        max_bet = maximum(bet_buf[1:min(10, union_sample)])
                        max_err_l = maximum(err_l_buf[1:min(10, union_sample)])
                        max_err_u = maximum(err_u_buf[1:min(10, union_sample)])
                        println("[tid=$tid, $(round(elapsed,digits=1))s] check #$(check_count[]): n_pairs=$current_n_pairs_global / $(round(Int,omega)), sum(approx)=$total_sum, check_finished=$result, max_bet=$(round(max_bet,digits=6)), max_err_l=$(round(max_err_l,digits=6)), max_err_u=$(round(max_err_u,digits=6))")
                        flush(stdout)
                    end

                    if result
                        Threads.atomic_xchg!(stop_flag, true)
                    end
                end
            finally
                unlock(check_lock)
            end
        end
    end
end

elapsed = time() - start_time
println("\nDone! n_pairs=$(n_pairs[]), checks=$(check_count[]), time=$(round(elapsed,digits=1))s, stop_flag=$(stop_flag[])")
println("Terminated because: $(stop_flag[] ? "check_finished returned true" : "n_pairs >= omega")")
flush(stdout)
