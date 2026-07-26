using Pkg
Pkg.activate(".")
using Graphs
using StatsBase
using JSON
using CSV
using DataFrames

include("../../../benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

out_dir = "benchmark/results/kadabra_accuracy"
mkpath(out_dir)

gt_dir = "Instances/ground_truth/test_instances"
gt_cache = Dict{String, Tuple{Vector{Float64}, Int}}() # name -> (bt_exact, max_node)

function load_gt(inst)
    if haskey(gt_cache, inst)
        return gt_cache[inst]
    end
    gt_path = joinpath(gt_dir, "$(inst)_bet.csv")
    if !isfile(gt_path)
        return nothing
    end
    lines = readlines(gt_path)
    if length(lines) < 2
        return nothing
    end
    shift = parse(Int, split(lines[2], ",")[1]) == 0 ? 1 : 0
    max_node = 0
    for line in lines[2:end]
        parts = split(line, ",")
        u = parse(Int, parts[1])
        max_node = max(max_node, u + shift)
    end
    
    bt_exact = zeros(Float64, max_node)
    for line in lines[2:end]
        parts = split(line, ",")
        u = parse(Int, parts[1])
        bt_exact[u + shift] = parse(Float64, parts[2])
    end
    
    gt_cache[inst] = (bt_exact, max_node)
    return gt_cache[inst]
end

function evaluate_run(cents, bt_exact, max_node, k_param)
    bt_approx = zeros(Float64, max_node)
    for (k_str, v) in cents
        idx = parse(Int, k_str)
        if idx <= max_node
            bt_approx[idx] = v
        end
    end
    
    eval_k = k_param == 0 ? max_node : k_param
    eval_k = min(eval_k, max_node)
    
    p_exact = sortperm(bt_exact, rev=true)
    p_approx = sortperm(bt_approx, rev=true)
    
    topk_exact = p_exact[1:eval_k]
    topk_approx = p_approx[1:eval_k]
    
    tau_topk = NaN
    overlap = NaN
    try
        tau_topk = corkendall(bt_exact[topk_exact], bt_approx[topk_exact])
        overlap = length(intersect(topk_exact, topk_approx))
    catch
    end
    
    return overlap, tau_topk
end

df = DataFrame(
    algo=String[],
    instance_name=String[],
    k=Int[],
    epsilon=Float64[],
    delta=Float64[],
    overlap_top_k=Union{Float64, Int}[],
    kendall_tau_top_k=Float64[]
)

for out_d in readdir("benchmark/output", join=true)
    if !isdir(out_d) continue end
    algo = split(basename(out_d), "~")[1] # e.g. kadabra-cpp, kadabra-julia, brava-julia
    
    for json_file in readdir(out_d, join=true)
        if !endswith(json_file, ".stats.json") continue end
        
        inst = replace(basename(json_file), ".stats.json" => "")
        
        local res_dict
        try
            res_dict = JSON.parsefile(json_file)
        catch
            continue
        end
        
        cents = get(res_dict, "centralities", nothing)
        params = get(res_dict, "parameters", nothing)
        if cents === nothing || params === nothing
            continue
        end
        
        k_val = get(params, "k", 0)
        epsilon = get(params, "epsilon", NaN)
        delta = get(params, "delta", NaN)
        
        gt = load_gt(inst)
        if gt === nothing
            # Push NaN if no ground truth
            push!(df, (algo, inst, k_val, epsilon, delta, NaN, NaN))
        else
            overlap, tau = evaluate_run(cents, gt[1], gt[2], k_val)
            push!(df, (algo, inst, k_val, epsilon, delta, overlap, tau))
        end
    end
end

# The user wants CSVs created per algorithm, with epsilon and delta in filename
# And rows should exist for k=0,10,100 for each instance, even if empty.
# We will do a full outer join with a template DataFrame to ensure k=0,10,100 exist.

instances_list = [replace(f, "_bet.csv" => "") for f in readdir(gt_dir) if endswith(f, "_bet.csv")]
k_req = [0, 10, 100]

algorithms = ["kadabra-cpp", "kadabra-julia", "brava-julia"]

# To get unique (epsilon, delta) pairs per algorithm
groups = groupby(df, [:algo, :epsilon, :delta])

# If there are no groups at all, create empty ones for the default algorithms
if isempty(groups)
    for a in algorithms
        out_csv = joinpath(out_dir, "$(a)_epsNaN_deltaNaN_results.csv")
        empty_df = DataFrame(instance_name=String[], k=Int[], epsilon=Float64[], overlap_top_k=Float64[], kendall_tau_top_k=Float64[])
        CSV.write(out_csv, empty_df)
    end
else
    for g in groups
        algo = first(g.algo)
        eps = first(g.epsilon)
        delt = first(g.delta)
        
        # Build template
        template = DataFrame(
            instance_name = repeat(instances_list, inner=length(k_req)),
            k = repeat(k_req, outer=length(instances_list))
        )
        
        # Join with actual data
        sub_df = DataFrame(g)
        joined = leftjoin(template, sub_df, on=[:instance_name, :k])
        
        joined.epsilon .= eps
        joined.delta .= delt
        
        final_df = joined[:, [:instance_name, :k, :epsilon, :overlap_top_k, :kendall_tau_top_k]]
        
        out_csv = joinpath(out_dir, "$(algo)_eps$(eps)_delta$(delt)_results.csv")
        CSV.write(out_csv, final_df)
        println("Created $out_csv")
        
        filter!(a -> a != algo, algorithms)
    end
end

for a in algorithms
    out_csv = joinpath(out_dir, "$(a)_epsNaN_deltaNaN_results.csv")
    empty_df = DataFrame(instance_name=String[], k=Int[], epsilon=Float64[], overlap_top_k=Float64[], kendall_tau_top_k=Float64[])
    CSV.write(out_csv, empty_df)
    println("Created $out_csv")
end
