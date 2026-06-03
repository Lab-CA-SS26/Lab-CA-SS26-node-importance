# compare_kadabra.jl
using Graphs
using Printf

include("../src/kadabra.jl")

function load_snap_graph(filepath::String, directed::Bool)
    edges = Tuple{Int, Int}[]
    max_id = 0
    open(filepath, "r") do io
        for line in eachline(io)
            if startswith(line, "#") || isempty(strip(line))
                continue
            end
            parts = split(line)
            u = parse(Int, parts[1]) + 1
            v = parse(Int, parts[2]) + 1
            max_id = max(max_id, u, v)
            push!(edges, (u, v))
        end
    end
    
    g = directed ? DiGraph(max_id) : Graph(max_id)
    for (u, v) in edges
        add_edge!(g, u, v)
    end
    return g
end

function print_usage()
    println("Usage: julia compare_kadabra.jl [-k <k>] [-d] [-s <start_factor>] <err> <delta> <filepath>")
    println("Options:")
    println("  -k <Int>          Number of top vertices to approximate (default: 0 for absolute mode)")
    println("  -d                Graph is directed (default: undirected)")
    println("  -s <start_factor> Start factor parameter (default: 100)")
end

function main()
    args = ARGS
    if length(args) < 3
        print_usage()
        return
    end
    
    # Parse options
    k = 0
    directed = false
    start_factor = 100
    
    idx = 1
    while idx <= length(args)
        if args[idx] == "-k"
            k = parse(Int, args[idx+1])
            deleteat!(args, idx:(idx+1))
        elseif args[idx] == "-d"
            directed = true
            deleteat!(args, idx)
        elseif args[idx] == "-s"
            start_factor = parse(Int, args[idx+1])
            deleteat!(args, idx:(idx+1))
        else
            idx += 1
        end
    end
    
    if length(args) != 3
        print_usage()
        return
    end
    
    err = parse(Float64, args[1])
    delta = parse(Float64, args[2])
    filepath = args[3]
    
    if !isfile(filepath)
        println("Error: File not found at ", filepath)
        return
    end
    
    println(directed ? "Directed graph" : "Undirected graph")
    
    # Load graph
    t_load = @elapsed g = load_snap_graph(filepath, directed)
    n = nv(g)
    m = ne(g)
    
    println("Number of nodes: ", n)
    println("Number of edges: ", m)
    println("Graph loading took: ", @sprintf("%.4f", t_load), " seconds.")
    println("Running KADABRA with err=", err, ", delta=", delta, ", k=", k, " on ", Threads.nthreads(), " thread(s)...")
    
    # Warmup compiler with a tiny dummy graph of the exact same type
    print("Warming up Julia compiler... ")
    dummy_g = typeof(g)(3)
    add_edge!(dummy_g, 1, 2)
    add_edge!(dummy_g, 2, 3)
    # Run a microscopic iteration to compile everything
    kadabra_centrality(dummy_g, 0, 0.5, 0.5; start_factor=10)
    println("Done.\n")

    # Run KADABRA and measure execution time
    t_start = time()
    approx_bet = kadabra_centrality(g, k, err, delta; start_factor=start_factor)
    t_total = time() - t_start
    
    println("\nFinished KADABRA run.")
    println("Total time: ", @sprintf("%.6f", t_total), " seconds")
    
    if k == 0
        # Absolute mode
        max_val = maximum(approx_bet)
        max_node = argmax(approx_bet) - 1
        println("Max approximated centrality: ", @sprintf("%.6f", max_val), " (Node ", max_node, ")")
    else
        # Top-k mode
        # Sort and get indices (convert to 0-based node IDs for C++ comparison!)
        sorted_nodes = sortperm(approx_bet, rev=true)
        println("\nTop ", k, " centralities:")
        for idx in 1:k
            node_1based = sorted_nodes[idx]
            node_0based = node_1based - 1
            val = approx_bet[node_1based]
            println(string(lpad(idx, 8), ")"), string(lpad(node_0based, 8)), " ", @sprintf("%.6f", val))
        end
    end
end

main()
