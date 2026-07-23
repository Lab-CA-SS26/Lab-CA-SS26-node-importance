using JSON
using StatsBase
using Printf
using Statistics

# Base directories assuming script is run from `benchmark/` folder
instances_dir = "../Instances"
output_dir = "output"
results_dir = "results/kadabra"

if !isdir(results_dir)
    mkpath(results_dir)
end

function get_max_rank(true_top_k, approx_perm)
    m = 0
    for v in true_top_k
        r = findfirst(==(v), approx_perm)
        if r !== nothing
            m = max(m, r)
        else
            m = length(approx_perm)
        end
    end
    return m
end

println("Finding exact score files...")
score_files = String[]
for (root, dirs, files) in walkdir(instances_dir)
    for file in files
        if endswith(file, "-score.txt")
            push!(score_files, joinpath(root, file))
        end
    end
end

# Dictionary to hold raw metrics across seeds: 
# (Instance, Impl, GraphType, k, Threads) => List of Dicts
grouped_metrics = Dict{Tuple{String, String, String, String, String}, Vector{Dict{String, Float64}}}()

for score_file_path in score_files
    score_file = basename(score_file_path)
    instance_name = replace(score_file, "-score.txt" => "")
    println("Processing exact scores for instance: $instance_name")
    
    # Read exact scores
    bt_exact = Float64[]
    for line in eachline(score_file_path)
        push!(bt_exact, parse(Float64, strip(line)))
    end
    n_nodes = length(bt_exact)
    
    p_exact = sortperm(bt_exact, rev=true)
    top10_exact = p_exact[1:min(10, n_nodes)]
    top100_exact = p_exact[1:min(100, n_nodes)]
    
    if !isdir(output_dir)
        continue
    end
    
    for exp_dir in readdir(output_dir)
        if !startswith(exp_dir, "kadabra-")
            continue
        end
        
        parts = split(exp_dir, "~")
        impl = parts[1]
        variants_str = length(parts) > 1 ? parts[2] : ""
        variants = split(variants_str, ",")
        
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
        
        stats_file = joinpath(output_dir, exp_dir, "$instance_name.stats.json")
        if !isfile(stats_file)
            continue
        end
        
        try
            stats_dict = JSON.parsefile(stats_file)
            if !haskey(stats_dict, "centralities")
                continue
            end
            
            cents = stats_dict["centralities"]
            bt_approx = zeros(Float64, n_nodes)
            for (k_str, v) in cents
                node_id = parse(Int, k_str)
                if node_id >= 1 && node_id <= n_nodes
                    bt_approx[node_id] = v
                end
            end
            
            p_approx = sortperm(bt_approx, rev=true)
            
            top10_approx = p_approx[1:min(10, n_nodes)]
            overlap10 = length(intersect(top10_exact, top10_approx))
            tau10 = corkendall(bt_exact[top10_exact], bt_approx[top10_exact])
            max_rank10 = get_max_rank(top10_exact, p_approx)
            
            top100_approx = p_approx[1:min(100, n_nodes)]
            overlap100 = length(intersect(top100_exact, top100_approx))
            tau100 = corkendall(bt_exact[top100_exact], bt_approx[top100_exact])
            max_rank100 = get_max_rank(top100_exact, p_approx)
            
            tau0 = corkendall(bt_exact, bt_approx)
            
            # Calculate true candidate set size for top-k bounding
            candidate_size = Float64(n_nodes)
            if k_val != "0" && haskey(stats_dict, "lower_bounds") && haskey(stats_dict, "upper_bounds")
                k_int = min(parse(Int, k_val), n_nodes)
                
                lower_bounds_arr = zeros(Float64, n_nodes)
                upper_bounds_arr = zeros(Float64, n_nodes)
                for (v_str, b_val) in stats_dict["lower_bounds"]
                    node_id = parse(Int, v_str)
                    if node_id >= 1 && node_id <= n_nodes
                        lower_bounds_arr[node_id] = b_val
                    end
                end
                for (v_str, b_val) in stats_dict["upper_bounds"]
                    node_id = parse(Int, v_str)
                    if node_id >= 1 && node_id <= n_nodes
                        upper_bounds_arr[node_id] = b_val
                    end
                end
                
                k_th_lower_bound = partialsort(lower_bounds_arr, k_int, rev = true)
                candidate_size = Float64(count(u -> u >= k_th_lower_bound, upper_bounds_arr))
            end
            
            key = (instance_name, impl, graph_type, k_val, threads)
            if !haskey(grouped_metrics, key)
                grouped_metrics[key] = []
            end
            push!(grouped_metrics[key], Dict(
                "tau0" => tau0,
                "overlap10" => overlap10,
                "tau10" => tau10,
                "max_rank10" => max_rank10,
                "overlap100" => overlap100,
                "tau100" => tau100,
                "max_rank100" => max_rank100,
                "candidate_size" => Float64(candidate_size)
            ))
            
        catch e
            println("  Error processing $stats_file: $e")
        end
    end
end

out_csv = joinpath(results_dir, "kadabra_accuracy_summary.csv")
open(out_csv, "w") do f
    write(f, "Instance,Implementation,GraphType,k,Threads,NumSeeds," *
             "Tau0_Mean,Tau0_Std," *
             "Overlap10_Mean,Overlap10_Std,Tau10_Mean,Tau10_Std,MaxRank10_Mean,MaxRank10_Std," *
             "Overlap100_Mean,Overlap100_Std,Tau100_Mean,Tau100_Std,MaxRank100_Mean,MaxRank100_Std," *
             "CandidateSize_Mean,CandidateSize_Std\n")
    
    for (key, metrics_list) in sort(collect(grouped_metrics))
        inst, impl, gtype, k_val, threads = key
        n = length(metrics_list)
        
        # Helper to extract a vector for a specific metric
        get_vec = metric -> [m[metric] for m in metrics_list]
        
        tau0 = get_vec("tau0")
        overlap10 = get_vec("overlap10")
        tau10 = get_vec("tau10")
        max_rank10 = get_vec("max_rank10")
        overlap100 = get_vec("overlap100")
        tau100 = get_vec("tau100")
        max_rank100 = get_vec("max_rank100")
        candidate_size = get_vec("candidate_size")
        
        @printf(f, "%s,%s,%s,%s,%s,%d,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.1f,%.1f,%.2f,%.2f,%.4f,%.4f,%.1f,%.1f,%.1f,%.1f\n",
            inst, impl, gtype, k_val, threads, n,
            mean(tau0), n > 1 ? std(tau0) : 0.0,
            mean(overlap10), n > 1 ? std(overlap10) : 0.0,
            mean(tau10), n > 1 ? std(tau10) : 0.0,
            mean(max_rank10), n > 1 ? std(max_rank10) : 0.0,
            mean(overlap100), n > 1 ? std(overlap100) : 0.0,
            mean(tau100), n > 1 ? std(tau100) : 0.0,
            mean(max_rank100), n > 1 ? std(max_rank100) : 0.0,
            mean(candidate_size), n > 1 ? std(candidate_size) : 0.0
        )
    end
end

println("Accuracy evaluation complete. Results saved to $out_csv")
