using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g = StaticGraph(g_raw)

n = nv(g)
T_g = eltype(g)
println("Graph (StaticGraph{$T_g}): nv=$n, ne=$(ne(g))")
println("Threads: $(Threads.nthreads())")

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

println("omega=$omega, tau=$tau, tau_per_thread=$tau_per_thread")
println("check_interval=$check_interval, k=$k, absolute=$absolute")
println("union_sample=$(Int(absolute ? k : min(n, k + 20)))")
flush(stdout)

# Setup
approx_local = [zeros(Int, n) for _ = 1:nthreads_val]
workspaces = [KadabraWorkspace(g) for _ = 1:nthreads_val]
n_pairs = Threads.Atomic{Int}(0)
base_rng = Random.default_rng()
thread_seeds = [rand(base_rng, UInt64) for _ = 1:nthreads_val]

# Phase 1
println("\nPhase 1...")
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
        sample_shortest_path!(counts, ws, g, t_rng, T_g(s), T_g(t); endpoints=false)
    end
end
n_pairs[] = tau_per_thread * nthreads_val
println("Phase 1 done. n_pairs=$(n_pairs[])")
flush(stdout)

# Phase 2
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

# Per-thread progress counters
thread_batches = [Threads.Atomic{Int}(0) for _ = 1:nthreads_val]
thread_trylock_attempts = [Threads.Atomic{Int}(0) for _ = 1:nthreads_val]
thread_trylock_wins = [Threads.Atomic{Int}(0) for _ = 1:nthreads_val]
thread_check_runs = [Threads.Atomic{Int}(0) for _ = 1:nthreads_val]
thread_alive = [Threads.Atomic{Bool}(true) for _ = 1:nthreads_val]

println("\nPhase 2 starting...")
flush(stdout)

start_time = time()

# Monitor thread in a separate task
monitor = Threads.@spawn begin
    while !stop_flag[] && n_pairs[] < omega
        sleep(2.0)
        elapsed = time() - start_time
        alive_count = count(t -> t[], thread_alive)
        total_batches = sum(t -> t[], thread_batches)
        total_trylock = sum(t -> t[], thread_trylock_attempts)
        total_wins = sum(t -> t[], thread_trylock_wins)
        total_checks = sum(t -> t[], thread_check_runs)
        println("[$(round(elapsed, digits=1))s] n_pairs=$(n_pairs[]) / $(round(Int,omega)), alive=$alive_count/$(nthreads_val), batches=$total_batches, trylock_attempts=$total_trylock, trylock_wins=$total_wins, checks_run=$total_checks, stop=$(stop_flag[])")
        flush(stdout)
    end
end

Threads.@threads for tid = 1:nthreads_val
    try
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
                sample_shortest_path!(counts, ws, g, t_rng, T_g(s), T_g(t); endpoints=false)
                local_pairs += 1
            end
            
            Threads.atomic_add!(thread_batches[tid], 1)
            Threads.atomic_add!(n_pairs, local_pairs)
            local_pairs = 0

            Threads.atomic_add!(thread_trylock_attempts[tid], 1)
            if trylock(check_lock)
                Threads.atomic_add!(thread_trylock_wins[tid], 1)
                try
                    current_n_pairs_global = n_pairs[]
                    
                    if current_n_pairs_global - last_check[] >= check_interval * nthreads_val
                        last_check[] = current_n_pairs_global
                        Threads.atomic_add!(thread_check_runs[tid], 1)

                        fill!(global_approx, 0)
                        for i = 1:nthreads_val
                            global_approx .+= approx_local[i]
                        end

                        if absolute
                            for i = 1:union_sample
                                top_k_nodes[i] = i
                            end
                        else
                            copyto!(top_k_nodes, 1:n)
                            partialsort!(top_k_nodes, 1:union_sample, by=x->global_approx[x], rev=true)
                        end

                        if check_finished(
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
                            Threads.atomic_xchg!(stop_flag, true)
                        end
                    end
                finally
                    unlock(check_lock)
                end
            end
        end
    catch e
        println("THREAD $tid CRASHED: $e")
        flush(stdout)
        Threads.atomic_xchg!(thread_alive[tid], false)
    end
end

elapsed = time() - start_time
println("\nDone! n_pairs=$(n_pairs[]), time=$(round(elapsed, digits=1))s, stop=$(stop_flag[])")
println("Terminated because: $(stop_flag[] ? "check_finished" : "n_pairs >= omega")")
for tid = 1:nthreads_val
    println("  Thread $tid: batches=$(thread_batches[tid][]), trylock=$(thread_trylock_attempts[tid][]), wins=$(thread_trylock_wins[tid][]), checks=$(thread_check_runs[tid][]), alive=$(thread_alive[tid][])")
end
flush(stdout)
