using ArgParse
using JSON
using Graphs

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
    threads = parsed_args["threads"]
    # println("Aktive Threads: ", Threads.nthreads())
    k = parsed_args["k"]
    delta = parsed_args["delta"]
    epsilon = parsed_args["epsilon"]
    algo = parsed_args["algorithm"]
    version = parsed_args["version"]
    is_directed = parsed_args["directed"]

    # Load graph
    io_start_time = time_ns()
    g_raw = BenchmarkUtils.load_graph_from_edgelist(input_file, is_directed)
    g = USE_STATIC_GRAPHS ? (is_directed ? StaticDiGraph(g_raw) : StaticGraph(g_raw)) : g_raw
    io_end_time = time_ns()
    io_time = (io_end_time - io_start_time) / 1e9

    # JIT WARMUP: Run the algorithm on a tiny dummy graph to compile all functions
    dummy_g_raw = typeof(g_raw)(3) # Create empty graph of exact same type (e.g. SimpleGraph{Int32} vs Int64)
    Main.Graphs.add_edge!(dummy_g_raw, 1, 2)
    Main.Graphs.add_edge!(dummy_g_raw, 2, 3)
    dummy_g = USE_STATIC_GRAPHS ? (is_directed ? StaticDiGraph(dummy_g_raw) : StaticGraph(dummy_g_raw)) : dummy_g_raw
    
    if algo == "kadabra"
        Main.kadabra_centrality(dummy_g, k, epsilon, delta; start_factor=10, endpoints=false)
    elseif algo == "brava"
        Main.BRAVAGNN.brava_centrality(dummy_g, k, epsilon, delta)
    elseif algo == "brandes"
        Main.Graphs.betweenness_centrality(dummy_g; normalize=false, endpoints=false)
    end
    # END WARMUP

    if algo == "kadabra"
        # Start timing
        start_time = time_ns()
        
        # Run kadabra
        res = Main.kadabra_centrality(g, k, epsilon, delta; start_factor=100, endpoints=false)
        centralities = res.centralities
        n_samples = res.n_samples
        
        end_time = time_ns()
        execution_time = (end_time - start_time) / 1e9
        
    elseif algo == "brava"
        # For brava
        start_time = time_ns()
        
        centralities, n_samples = Main.BRAVAGNN.brava_centrality(g, k, epsilon, delta)
        
        end_time = time_ns()
        execution_time = (end_time - start_time) / 1e9
        
    elseif algo == "brandes"
        # For brandes (exact)
        start_time = time_ns()
        
        centralities = Main.Graphs.betweenness_centrality(g; normalize=false, endpoints=false)
        n_samples = 0
        
        end_time = time_ns()
        execution_time = (end_time - start_time) / 1e9
        
    else
        error("Unknown algorithm: $algo")
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
            "directed" => is_directed
        ),
        "execution_time_seconds" => execution_time,
        "io_time_seconds" => io_time,
        "num_samples" => n_samples,
        "centralities" => Dict{String, Float64}()
    )

    # Convert centralities array/dict to the expected schema
    for (v, cent) in enumerate(centralities)
        if cent > 0.0
            # Assuming 1-based indexing for vertices in Julia, converting to 0-based for JSON?
            # Or just use the 1-based vertex ID as string.
            results["centralities"][string(v)] = cent
        end
    end

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
