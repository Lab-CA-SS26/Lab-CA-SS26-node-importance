using ArgParse
using JSON
using Graphs
using StatsBase
import JLD2

const USE_STATIC_GRAPHS = true
if USE_STATIC_GRAPHS
    using StaticGraphs
end

# Wait, let's load what we need dynamically based on the algorithm argument or include them directly.


# In the benchmark script, they used `include` directly:
include(joinpath(@__DIR__, "..", "scripts", "BenchmarkUtils.jl"))
using .BenchmarkUtils

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_file", "-i"
        help = "Path to the input graph file"
        required = true
        "--output_file", "-o"
        help = "Path to the output JSON file"
        required = true
        "--threads", "-t"
        help = "Number of threads to use (Julia uses JULIA_NUM_THREADS env by default, but we accept this parameter for logging)"
        arg_type = Int
        default = 1
        "-k"
        help = "Top-K value (0 means absolute error bound for all)"
        arg_type = Int
        default = 0
        "--delta", "-d"
        help = "Confidence parameter"
        arg_type = Float64
        default = 0.1
        "--epsilon", "-e"
        help = "Error bound parameter"
        arg_type = Float64
        default = 0.01
        "-a", "--algorithm"
        help = "Algorithm to run: kadabra or brava"
        arg_type = String
        default = "kadabra"
        "-v", "--version"
        help = "Implementation version"
        arg_type = String
        default = "julia"
        "--directed"
        help = "Treat graph as directed"
        action = :store_true
        "--seed", "-s"
        help = "Random seed (0 means no seed / random)"
        arg_type = Int
        default = 0
        "--gpu"
        help = "Use GPU for BRAVA inference if available"
        action = :store_true
    end

    return parse_args(s)
end

include(joinpath(@__DIR__, "..", "..", "src", "kadabra.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "BRAVAGNN.jl"))

function main()
    println("ARGS: ", ARGS)
    try
        parsed_args = parse_cmdline()

        input_file = parsed_args["input_file"]
        output_file = parsed_args["output_file"]
        threads = Threads.nthreads()
        println("Aktive Threads: ", Threads.nthreads())
        k = parsed_args["k"]
        delta = parsed_args["delta"]
        epsilon = parsed_args["epsilon"]
        algo = parsed_args["algorithm"]
        version = parsed_args["version"]
        is_directed = parsed_args["directed"]

        # ---------------------------------------------------------
        # Pre-load BRAVA model if needed
        # ---------------------------------------------------------
        brava_model = nothing
        if algo == "brava"
            weight_path = joinpath(@__DIR__, "..", "cache", "bravagnn_weights.jld2")
            if isfile(weight_path)
                Main.JLD2.@load weight_path model
                brava_model = model
            end
        end

        # ---------------------------------------------------------
        # JIT WARMUP
        # ---------------------------------------------------------
        println("Performing JIT Warmup...")
        dummy_g_raw = Main.Graphs.SimpleGraph(100, 500)
            dummy_g =
                USE_STATIC_GRAPHS ?
                (is_directed ? StaticDiGraph(dummy_g_raw) : StaticGraph(dummy_g_raw)) :
                dummy_g_raw

            if algo == "kadabra"
                Main.kadabra_centrality(
                    dummy_g,
                    k,
                    epsilon,
                    delta;
                    start_factor = 10,
                    endpoints = false,
                    rng = parsed_args["seed"] == 0 ? nothing : Main.Random.Xoshiro(parsed_args["seed"])
                )
            elseif algo == "brava"
                Main.BRAVAGNN.brava_centrality(
                    dummy_g, 
                    k, 
                    epsilon, 
                    delta;
                    model = brava_model,
                    use_gpu = parsed_args["gpu"]
                )
            elseif algo == "brandes"
                Main.Graphs.betweenness_centrality(
                    dummy_g;
                    normalize = false,
                    endpoints = false,
                )
            end
            println("Warmup done")

        # Load graph
        io_start_time = time_ns()
        g_raw = BenchmarkUtils.load_graph_from_edgelist(input_file, is_directed)
        g =
            USE_STATIC_GRAPHS ? (is_directed ? StaticDiGraph(g_raw) : StaticGraph(g_raw)) :
            g_raw
        io_end_time = time_ns()
        io_time = (io_end_time - io_start_time) / 1e9

        # JIT WARMUP: Run the algorithm on a tiny dummy graph to compile all functions
        dummy_g_raw = typeof(g_raw)(3) # Create empty graph of exact same type (e.g. SimpleGraph{Int32} vs Int64)
        Main.Graphs.add_edge!(dummy_g_raw, 1, 2)
        Main.Graphs.add_edge!(dummy_g_raw, 2, 3)
        dummy_g =
            USE_STATIC_GRAPHS ?
            (is_directed ? StaticDiGraph(dummy_g_raw) : StaticGraph(dummy_g_raw)) :
            dummy_g_raw

        if algo == "kadabra"
            
            Main.kadabra_centrality(
                dummy_g,
                k,
                epsilon,
                delta;
                start_factor = 10,
                endpoints = false,
                rng = parsed_args["seed"] == 0 ? nothing : Main.Random.Xoshiro(parsed_args["seed"])
            )
        elseif algo == "brava"
            Main.BRAVAGNN.brava_centrality(
                dummy_g, 
                k, 
                epsilon, 
                delta;
                model = brava_model
            )
        elseif algo == "brandes"
            Main.Graphs.betweenness_centrality(
                dummy_g;
                normalize = false,
                endpoints = false,
            )
        end
        # END WARMUP
        println("Warmup done")

        local lower_bounds = nothing
        local upper_bounds = nothing

        if algo == "kadabra"
            # Start timing
            start_time = time_ns()

            # Run kadabra
            res = Main.kadabra_centrality(
                g,
                k,
                epsilon,
                delta;
                start_factor = 100,
                endpoints = false,
                rng = parsed_args["seed"] == 0 ? nothing : Main.Random.Xoshiro(parsed_args["seed"])
            )
            centralities = res.centralities
            lower_bounds = res.lower_bounds
            upper_bounds = res.upper_bounds
            n_samples = res.n_samples

            end_time = time_ns()
            execution_time = (end_time - start_time) / 1e9

        elseif algo == "brava"
            # For brava
            start_time = time_ns()

            centralities, n_samples = Main.BRAVAGNN.brava_centrality(
                g, 
                k, 
                epsilon, 
                delta;
                model = brava_model,
                use_gpu = parsed_args["gpu"]
            )

            end_time = time_ns()
            execution_time = (end_time - start_time) / 1e9

        elseif algo == "brandes"
            # For brandes (exact)
            start_time = time_ns()

            centralities =
                Main.Graphs.betweenness_centrality(g; normalize = false, endpoints = false)
            n_samples = 0

            end_time = time_ns()
            execution_time = (end_time - start_time) / 1e9

        else
            error("Unknown algorithm: $algo")
        end

        # Read Ground truth
        graph_name = replace(basename(input_file), ".txt" => "")
        gt_path = joinpath(@__DIR__, "..", "..", "Instances", "ground_truth", "test_instances", "$(graph_name)_bet.csv")
        
        tau_overall = NaN
        tau_topk = NaN
        overlap = 0
        max_ae = NaN
        mae = NaN
        ndcg_topk = NaN
        
        if isfile(gt_path)
            lines = readlines(gt_path)
            if length(lines) >= 2
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
                
                bt_approx = zeros(Float64, max_node)
                for (v, cent) in enumerate(centralities)
                    if v <= max_node
                        bt_approx[v] = cent
                    end
                end
                
                try
                    tau_overall = corkendall(bt_exact, bt_approx)
                catch
                    tau_overall = NaN
                end
                
                max_ae = 0.0
                sum_ae = 0.0
                for v in 1:max_node
                    ae = abs(bt_exact[v] - bt_approx[v])
                    if ae > max_ae
                        max_ae = ae
                    end
                    sum_ae += ae
                end
                mae = max_node > 0 ? sum_ae / max_node : 0.0
                
                eval_k = k == 0 ? 100 : k
                eval_k = min(eval_k, max_node)
                
                p_exact = sortperm(bt_exact, rev=true)
                p_approx = sortperm(bt_approx, rev=true)
                
                topk_exact = p_exact[1:eval_k]
                topk_approx = p_approx[1:eval_k]
                
                try
                    tau_topk = corkendall(bt_exact[topk_exact], bt_approx[topk_exact])
                    overlap = length(intersect(topk_exact, topk_approx))
                catch
                    tau_topk = NaN
                end
                
                dcg = 0.0
                idcg = 0.0
                for i in 1:eval_k
                    u_approx = topk_approx[i]
                    u_exact = topk_exact[i]
                    
                    dcg += bt_exact[u_approx] / log2(i + 1)
                    idcg += bt_exact[u_exact] / log2(i + 1)
                end
                ndcg_topk = idcg > 0.0 ? dcg / idcg : 1.0
            end
        else
            println("Ground truth not found at $gt_path")
        end

        # Format JSON
        results = Dict(
            "parameters" => Dict(
                "input_file" => input_file,
                "algorithm" => algo,
                "version" => version,
                "threads" => threads,
                "k" => k,
                "delta" => delta,
                "epsilon" => epsilon,
                "directed" => is_directed,
            ),
            "execution_time_seconds" => execution_time,
            "io_time_seconds" => io_time,
            "num_samples" => n_samples,
            "tau_overall" => isnan(tau_overall) ? nothing : tau_overall,
            "tau_topk" => isnan(tau_topk) ? nothing : tau_topk,
            "overlap_topk" => overlap,
            "max_ae" => isnan(max_ae) ? nothing : max_ae,
            "mae" => isnan(mae) ? nothing : mae,
            "ndcg_topk" => isnan(ndcg_topk) ? nothing : ndcg_topk,
        )

        # Write out
        open(output_file, "w") do f
            JSON.print(f, results, 4)
        end
    catch e
        println("ERROR: ", e)
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        rethrow(e)
    end
end

main()
