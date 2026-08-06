using Pkg
Pkg.activate(".")
using Graphs
using StatsBase
using JSON
using CSV
using DataFrames

include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

out_dir = "results/kadabra_accuracy"
mkpath(out_dir)

results_file = joinpath(out_dir, "evaluation_results.csv")
out_df = DataFrame(
    Run_Type=String[],
    Graph=String[],
    K_Param=Int[],
    Threads=Int[],
    Epsilon=Float64[],
    Execution_Time=Float64[],
    Tau_Overall=Float64[],
    Tau_TopK=Float64[],
    Overlap_TopK=Int[],
    Max_AE=Float64[],
    MAE=Float64[],
    NDCG_TopK=Float64[]
)

function evaluate_file(json_path, k_param)
    local res_dict
    try
        res_dict = JSON.parsefile(json_path)
    catch e
        println("Skipping $json_path: Error parsing JSON")
        return nothing
    end
    
    tau_overall = get(res_dict, "tau_overall", NaN)
    tau_topk = get(res_dict, "tau_topk", NaN)
    overlap = get(res_dict, "overlap_topk", 0)
    max_ae = get(res_dict, "max_ae", NaN)
    mae = get(res_dict, "mae", NaN)
    ndcg_topk = get(res_dict, "ndcg_topk", NaN)
    
    execution_time = get(res_dict, "execution_time_seconds", NaN)
    params = get(res_dict, "parameters", Dict())
    threads = get(params, "threads", 1)
    epsilon = get(params, "epsilon", NaN)
    
    if tau_overall === nothing
        tau_overall = NaN
    end
    if tau_topk === nothing
        tau_topk = NaN
    end
    if overlap === nothing
        overlap = 0
    end
    if max_ae === nothing
        max_ae = NaN
    end
    if mae === nothing
        mae = NaN
    end
    if ndcg_topk === nothing
        ndcg_topk = NaN
    end
    
    return (threads, epsilon, execution_time, tau_overall, tau_topk, overlap, max_ae, mae, ndcg_topk)
end

output_dirs = filter(isdir, readdir("benchmark/output", join=true))

for dir in output_dirs
    dir_name = basename(dir)
    # Parse K param from directory name, e.g., kadabra-julia~err1,k10,t4,undirected
    m = match(r"k(\d+)", dir_name)
    k_param = m !== nothing ? parse(Int, m.captures[1]) : 0
    
    run_type = split(dir_name, "~")[1]
    
    json_files = filter(f -> endswith(f, ".stats.json"), readdir(dir, join=true))
    for json_file in json_files
        println("Evaluating: $json_file (k=$k_param)")
        res = evaluate_file(json_file, k_param)
        if res !== nothing
            graph_name = replace(basename(json_file), ".stats.json" => "")
            push!(out_df, (run_type, graph_name, k_param, res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9]))
        end
    end
end

CSV.write(results_file, out_df)
println("Evaluation complete. Results saved to $results_file")
