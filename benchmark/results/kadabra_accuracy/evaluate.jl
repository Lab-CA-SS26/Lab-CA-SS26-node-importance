using Pkg
Pkg.activate(".")
using Graphs
using StatsBase
using JSON
using CSV
using DataFrames
using YAML
using Statistics

include("../../../benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

out_dir = "benchmark/results/kadabra_accuracy"
mkpath(out_dir)

gt_dir = "Instances/ground_truth/test_instances"
gt_cache = Dict{String, Tuple{Vector{Float64}, Int, Int}}() # name -> (bt_exact, max_node, shift)

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
    
    gt_cache[inst] = (bt_exact, max_node, shift)
    return gt_cache[inst]
end

function evaluate_run(cents, bt_exact, max_node, k_param, shift)
    bt_approx = zeros(Float64, max_node)
    for (k_str, v) in cents
        idx = parse(Int, k_str) + shift
        if 1 <= idx <= max_node
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
    depth_to_cover = NaN
    try
        tau_topk = corkendall(bt_exact[topk_exact], bt_approx[topk_exact])
        overlap = length(intersect(topk_exact, topk_approx))
        
        if eval_k == 0 || eval_k == max_node
            depth_to_cover = max_node
        else
            approx_ranks = Dict{Int, Int}()
            for (rank, node) in enumerate(p_approx)
                approx_ranks[node] = rank
            end
            depth_to_cover = maximum(approx_ranks[node] for node in topk_exact)
        end
    catch
    end
    
    return overlap, tau_topk, depth_to_cover
end

# 1. Parse experiments.yml to get valid configurations
yml = YAML.load_file("benchmark/experiments.yml")

instsets = Dict{String, Vector{String}}()
for inst_group in yml["instances"]
    for set_name in inst_group["set"]
        instsets[set_name] = [replace(item, ".txt" => "") for item in inst_group["items"]]
    end
end

axis_defaults = Dict{String, String}()
variant_to_axis = Dict{String, String}()
for axis_def in yml["variants"]
    axis = axis_def["axis"]
    axis_defaults[axis] = axis_def["items"][1]["name"]
    for item in axis_def["items"]
        variant_to_axis[item["name"]] = axis
    end
end

valid_runs = Set{Tuple{String, String, String}}()

for block in yml["matrix"]["include"]
    algos = block["experiments"]
    sets = get(block, "instsets", [])
    
    insts = String[]
    for s in sets
        append!(insts, instsets[s])
    end
    
    block_variants = get(block, "variants", [])
    
    axis_to_block_variants = Dict{String, Vector{String}}()
    for v in block_variants
        if haskey(variant_to_axis, v)
            ax = variant_to_axis[v]
            if !haskey(axis_to_block_variants, ax)
                axis_to_block_variants[ax] = String[]
            end
            push!(axis_to_block_variants[ax], v)
        end
    end
    
    for (ax, def) in axis_defaults
        if !haskey(axis_to_block_variants, ax)
            axis_to_block_variants[ax] = [def]
        end
    end
    
    axes_keys = collect(keys(axis_defaults))
    
    function cartesian(idx)
        if idx > length(axes_keys)
            return [String[]]
        end
        ax = axes_keys[idx]
        tails = cartesian(idx + 1)
        res = Vector{String}[]
        for v in axis_to_block_variants[ax]
            for t in tails
                push!(res, [v; t])
            end
        end
        return res
    end
    
    all_combos = cartesian(1)
    
    for algo in algos
        for combo in all_combos
            sorted_combo = sort(combo)
            param_str = join(sorted_combo, ",")
            for inst in insts
                push!(valid_runs, (algo, inst, param_str))
            end
        end
    end
end
println("Parsed experiments.yml. Evaluating only $(length(valid_runs)) valid configurations.")

df = DataFrame(
    algo=String[],
    instance_name=String[],
    k=Int[],
    epsilon=Float64[],
    delta=Float64[],
    seed=String[],
    overlap_top_k=Float64[],
    kendall_tau_top_k=Float64[],
    depth_to_cover_top_k=Float64[]
)

for out_d in readdir("benchmark/output", join=true)
    if !isdir(out_d) continue end
    basename_dir = basename(out_d)
    parts = split(basename_dir, "~")
    if length(parts) != 2 continue end
    algo = parts[1]
    param_str = parts[2]
    
    for json_file in readdir(out_d, join=true)
        if !endswith(json_file, ".stats.json") continue end
        
        inst = replace(basename(json_file), ".stats.json" => "")
        
        if !((algo, inst, param_str) in valid_runs)
            continue
        end
        
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
        
        seed = "unknown"
        for v in split(param_str, ",")
            if haskey(variant_to_axis, v) && variant_to_axis[v] == "seed"
                seed = v
                break
            end
        end
        
        gt = load_gt(inst)
        if gt === nothing
            push!(df, (algo, inst, k_val, epsilon, delta, seed, NaN, NaN, NaN))
        else
            overlap, tau, depth = evaluate_run(cents, gt[1], gt[2], k_val, gt[3])
            push!(df, (algo, inst, k_val, epsilon, delta, seed, Float64(overlap), tau, Float64(depth)))
        end
    end
end

println("Evaluation complete. Aggregating results across seeds...")

# Helper to ignore NaNs when aggregating
mean_nan(x) = isempty(filter(!isnan, x)) ? NaN : mean(filter(!isnan, x))
std_nan(x) = length(filter(!isnan, x)) <= 1 ? NaN : std(filter(!isnan, x))

grouped = groupby(df, [:algo, :instance_name, :k, :epsilon, :delta])
agg_df = combine(grouped,
    :overlap_top_k => mean_nan => :overlap_top_k_mean,
    :overlap_top_k => std_nan => :overlap_top_k_std,
    :kendall_tau_top_k => mean_nan => :kendall_tau_top_k_mean,
    :kendall_tau_top_k => std_nan => :kendall_tau_top_k_std,
    :depth_to_cover_top_k => mean_nan => :depth_to_cover_top_k_mean,
    :depth_to_cover_top_k => std_nan => :depth_to_cover_top_k_std
)

instances_list = [replace(f, "_bet.csv" => "") for f in readdir(gt_dir) if endswith(f, "_bet.csv")]
k_req = [0, 10, 100]
algorithms = ["kadabra-cpp", "kadabra-julia", "brava-julia"]

final_grouped = groupby(agg_df, [:algo, :epsilon, :delta])

if isempty(final_grouped)
    for a in algorithms
        out_csv = joinpath(out_dir, "$(a)_epsNaN_deltaNaN_results.csv")
        empty_df = DataFrame(instance_name=String[], k=Int[], epsilon=Float64[], overlap_top_k_mean=Float64[], overlap_top_k_std=Float64[], kendall_tau_top_k_mean=Float64[], kendall_tau_top_k_std=Float64[], depth_to_cover_top_k_mean=Float64[], depth_to_cover_top_k_std=Float64[])
        CSV.write(out_csv, empty_df)
    end
else
    for g in final_grouped
        algo = first(g.algo)
        eps = first(g.epsilon)
        delt = first(g.delta)
        
        template = DataFrame(
            instance_name = repeat(instances_list, inner=length(k_req)),
            k = repeat(k_req, outer=length(instances_list))
        )
        
        sub_df = DataFrame(g)
        joined = leftjoin(template, sub_df, on=[:instance_name, :k])
        
        joined.epsilon .= eps
        joined.delta .= delt
        
        final_df = joined[:, [:instance_name, :k, :epsilon, :overlap_top_k_mean, :overlap_top_k_std, :kendall_tau_top_k_mean, :kendall_tau_top_k_std, :depth_to_cover_top_k_mean, :depth_to_cover_top_k_std]]
        
        out_csv = joinpath(out_dir, "$(algo)_eps$(eps)_delta$(delt)_results.csv")
        CSV.write(out_csv, final_df)
        println("Created $out_csv")
        
        filter!(a -> a != algo, algorithms)
    end
end

for a in algorithms
    out_csv = joinpath(out_dir, "$(a)_epsNaN_deltaNaN_results.csv")
    empty_df = DataFrame(instance_name=String[], k=Int[], epsilon=Float64[], overlap_top_k_mean=Float64[], overlap_top_k_std=Float64[], kendall_tau_top_k_mean=Float64[], kendall_tau_top_k_std=Float64[], depth_to_cover_top_k_mean=Float64[], depth_to_cover_top_k_std=Float64[])
    CSV.write(out_csv, empty_df)
    println("Created $out_csv")
end
