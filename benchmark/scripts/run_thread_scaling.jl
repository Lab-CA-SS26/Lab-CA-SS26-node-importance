# Master script to run thread scaling benchmarks
# Spawns Julia subprocesses with different thread counts.

include("BenchmarkUtils.jl")
instances_file = joinpath(dirname(dirname(@__DIR__)), "Instances", "instances.txt")
instances = BenchmarkUtils.read_instances(instances_file)

# You can adjust these thread counts depending on your server's hardware (e.g. up to 128)
thread_counts = [1, 2, 4, 8, 16, 24, 32, 48]

out_csv = joinpath(@__DIR__, "..", "results", "thread_scaling_results.csv")

# Initialize CSV
open(out_csv, "w") do io
    println(io, "Dataset,Threads,TimeSeconds")
end

worker_script = joinpath(@__DIR__, "worker_thread_scaling.jl")

println("Starting Thread Scaling Benchmark...")
println("Results will be saved to: $out_csv")
println("-"^40)

for (rel_path, is_directed) in instances
    println("=== Benchmarking Graph: $rel_path ===")
    for t in thread_counts
        println("  -> Spawning worker with $t threads...")
        # Spawn a new Julia process with exactly `t` threads
        # Pass the graph path and the directed flag as arguments
        flag = is_directed ? "D" : "U"
        cmd = `julia -t $t $worker_script $rel_path $flag`
        try
            run(cmd)
        catch e
            println("  [ERROR] Failed to run with $t threads: $e")
        end
    end
end

println("-"^40)
println("All scaling benchmarks completed successfully!")
println("Check $out_csv for the results.")
