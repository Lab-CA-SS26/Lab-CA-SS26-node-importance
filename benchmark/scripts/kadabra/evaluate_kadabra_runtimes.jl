using JSON
using Statistics
using Printf

# Base directories assuming script is run from `benchmark/` folder
output_dir = "output"
results_dir = "results/kadabra"

if !isdir(results_dir)
    mkpath(results_dir)
end

println("Scanning Kadabra runtime results...")

# Dict to group runtimes:
# (instance, implementation, graph_type, k, threads) => Vector{Float64}
grouped_runtimes = Dict{Tuple{String, String, String, String, String}, Vector{Float64}}()

if !isdir(output_dir)
    println("Output directory $output_dir does not exist. Exiting.")
    exit(1)
end

for exp_dir in readdir(output_dir)
    if !startswith(exp_dir, "kadabra-")
        continue
    end
    
    parts = split(exp_dir, "~")
    impl = parts[1] # e.g. kadabra-julia or kadabra-cpp
    variants_str = length(parts) > 1 ? parts[2] : ""
    variants = split(variants_str, ",")
    
    # Extract known dimensions
    graph_type = "undirected"
    k_val = "0"
    threads = "1"
    
    for v in variants
        if startswith(v, "k") && tryparse(Int, v[2:end]) !== nothing
            k_val = v[2:end]
        elseif startswith(v, "t") && tryparse(Int, v[2:end]) !== nothing
            threads = v[2:end]
        elseif v == "directed" || v == "undirected"
            graph_type = v
        end
    end
    
    # Read the instance stats
    for stats_file_name in readdir(joinpath(output_dir, exp_dir))
        if !endswith(stats_file_name, ".stats.json")
            continue
        end
        instance_name = replace(stats_file_name, ".stats.json" => "")
        stats_file = joinpath(output_dir, exp_dir, stats_file_name)
        
        try
            stats_dict = JSON.parsefile(stats_file)
            if !haskey(stats_dict, "execution_time_seconds")
                continue
            end
            runtime = Float64(stats_dict["execution_time_seconds"])
            
            key = (instance_name, impl, graph_type, k_val, threads)
            if !haskey(grouped_runtimes, key)
                grouped_runtimes[key] = Float64[]
            end
            push!(grouped_runtimes[key], runtime)
            
        catch e
            println("  Error processing $stats_file: $e")
        end
    end
end

out_csv = joinpath(results_dir, "kadabra_runtimes_summary.csv")
open(out_csv, "w") do f
    write(f, "Instance,Implementation,GraphType,k,Threads,NumSeeds,Runtime_Mean,Runtime_Median,Runtime_Std\n")
    
    for (key, runtimes) in sort(collect(grouped_runtimes))
        inst, impl, gtype, k_val, threads = key
        n = length(runtimes)
        mu = mean(runtimes)
        med = median(runtimes)
        sd = n > 1 ? std(runtimes) : 0.0
        
        @printf(f, "%s,%s,%s,%s,%s,%d,%.6f,%.6f,%.6f\n",
            inst, impl, gtype, k_val, threads, n, mu, med, sd
        )
    end
end

println("Runtime evaluation complete. Results saved to $out_csv")
