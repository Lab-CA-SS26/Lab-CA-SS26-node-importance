using JSON
using Statistics
using DataFrames
using CSV
using ArgParse

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--brandes_file"
        help = "Path to the Brandes JSON output"
        required = true
        "--kadabra_file"
        help = "Path to the KADABRA JSON output"
        required = true
        "-k"
        help = "Top-K value to evaluate"
        arg_type = Int
        default = 100
        "--output_csv"
        help = "Path to output the results CSV"
        default = "kendall_tau_results.csv"
    end

    return parse_args(s)
end

function kendall_tau(rank_a::Vector{Int}, rank_b::Vector{Int})
    n = length(rank_a)
    if n < 2
        return 1.0
    end
    
    concordant = 0
    discordant = 0
    
    for i in 1:n
        for j in (i+1):n
            a_diff = rank_a[i] - rank_a[j]
            b_diff = rank_b[i] - rank_b[j]
            
            if a_diff * b_diff > 0
                concordant += 1
            elseif a_diff * b_diff < 0
                discordant += 1
            end
        end
    end
    
    total = concordant + discordant
    return total == 0 ? 0.0 : (concordant - discordant) / total
end

function main()
    parsed_args = parse_cmdline()
    
    brandes_path = parsed_args["brandes_file"]
    kadabra_path = parsed_args["kadabra_file"]
    k = parsed_args["k"]
    out_csv = parsed_args["output_csv"]
    
    # 1. Read JSONs
    brandes_data = JSON.parsefile(brandes_path)
    kadabra_data = JSON.parsefile(kadabra_path)
    
    # 2. Extract centralities
    brandes_cent = brandes_data["centralities"]
    kadabra_cent = kadabra_data["centralities"]
    
    kadabra_lower = get(kadabra_data, "lower_bounds", Dict{String, Any}())
    kadabra_upper = get(kadabra_data, "upper_bounds", Dict{String, Any}())
    
    # Parse to array of (node, centrality)
    b_nodes = [(parse(Int, node), Float64(val)) for (node, val) in brandes_cent]
    k_nodes = [(parse(Int, node), Float64(val)) for (node, val) in kadabra_cent]
    
    # Sort descending
    sort!(b_nodes, by = x -> x[2], rev = true)
    sort!(k_nodes, by = x -> x[2], rev = true)
    
    # 3. Identify Exact Top-K
    actual_k = min(k, length(b_nodes))
    top_k_exact = b_nodes[1:actual_k]
    top_k_exact_nodes = [n for (n, v) in top_k_exact]
    
    # 4. Find ranks in Brandes and Kadabra for these specific nodes
    b_rank_dict = Dict(node => rank for (rank, (node, v)) in enumerate(b_nodes))
    k_rank_dict = Dict(node => rank for (rank, (node, v)) in enumerate(k_nodes))
    
    rank_in_b = Int[]
    rank_in_k = Int[]
    
    missing_in_k = 0
    for node in top_k_exact_nodes
        push!(rank_in_b, b_rank_dict[node])
        if haskey(k_rank_dict, node)
            push!(rank_in_k, k_rank_dict[node])
        else
            push!(rank_in_k, length(k_nodes) + 1)
            missing_in_k += 1
        end
    end
    
    tau = kendall_tau(rank_in_b, rank_in_k)
    
    # 5. Bound Analysis
    k_k_nodes = k_nodes[1:actual_k]
    
    boundary_ambiguity = 0
    if !isempty(kadabra_lower) && !isempty(kadabra_upper)
        k_th_node_id = string(k_k_nodes[end][1])
        k_th_lower_bound = get(kadabra_lower, k_th_node_id, 0.0)
        
        for i in (actual_k + 1):length(k_nodes)
            node_id_str = string(k_nodes[i][1])
            u_bound = get(kadabra_upper, node_id_str, 0.0)
            if u_bound > k_th_lower_bound
                boundary_ambiguity += 1
            end
        end
    else
        println("Warning: KADABRA bounds not found in JSON. Did you run the updated run_experiments.jl?")
    end
    
    println("--- Kendall Tau Analysis (Top-\$actual_k) ---")
    println("Graph: ", brandes_data["parameters"]["input_file"])
    println("Kendall Tau Score: ", round(tau, digits=4))
    println("Nodes in Exact Top-K missing from Kadabra output: ", missing_in_k)
    println("Boundary Ambiguity (Nodes outside Kadabra's Top-K with overlapping bounds): ", boundary_ambiguity)
    
    df = DataFrame(
        Graph = [basename(brandes_data["parameters"]["input_file"])],
        K = [actual_k],
        Kendall_Tau = [tau],
        Missing_In_Kadabra = [missing_in_k],
        Boundary_Ambiguity = [boundary_ambiguity]
    )
    
    if isfile(out_csv)
        CSV.write(out_csv, df, append=true)
    else
        CSV.write(out_csv, df)
    end
    println("Results appended to \$out_csv")
end

main()
