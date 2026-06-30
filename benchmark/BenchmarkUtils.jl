module BenchmarkUtils

using Graphs
using SparseArrays

export run_cpp_kadabra, export_to_edgelist

"""
    export_to_edgelist(g::AbstractGraph, filepath::String)

Exports a `Graphs.AbstractGraph` to a space-separated edge list format compatible 
with the C++ KADABRA implementation (0-indexed nodes).
"""
function export_to_edgelist(g::AbstractGraph, filepath::String)
    open(filepath, "w") do f
        for e in edges(g)
            # C++ KADABRA assumes 0-indexed contiguous integer IDs
            # Graphs.jl uses 1-indexed. So we subtract 1.
            u = src(e) - 1
            v = dst(e) - 1
            println(f, u, " ", v)
        end
    end
end

"""
    run_cpp_kadabra(graph_file::String, is_directed::Bool, epsilon::Float64, delta::Float64, num_nodes::Int)

Executes the C++ KADABRA binary on `graph_file`.
It dynamically compiles the binary if missing.
Returns `(execution_time_seconds, scores_array)`.
"""
function run_cpp_kadabra(graph_file::String, is_directed::Bool, epsilon::Float64, delta::Float64, num_nodes::Int)
    # Check if compiled binary exists
    cpp_dir = joinpath(dirname(@__DIR__), "cpp_reference")
    binary_path = joinpath(cpp_dir, "kadabra")
    
    if !isfile(binary_path)
        println("Compiling C++ KADABRA reference...")
        run(`make -C $cpp_dir`)
    end
    
    # KADABRA Command
    # -k N dumps all node scores
    args = ["-k", string(num_nodes), string(epsilon), string(delta), graph_file]
    if is_directed
        pushfirst!(args, "-d")
    end
    cmd = Cmd([binary_path; args])
    
    # Execute and capture output
    output = read(cmd, String)
    
    execution_time = -1.0
    scores = zeros(Float64, num_nodes)
    
    # Parse output using Regex
    for line in split(output, '\n')
        # Total time parser
        if startswith(line, "Total time:")
            time_match = match(r"Total time: ([0-9.]+)", line)
            if time_match !== nothing
                execution_time = parse(Float64, time_match[1])
            end
        end
        
        # Parse individual node scores (Lines like: "       1)      107 0.450715 0.47515 0.500898")
        # Optional '?' character for nodes tied or with low confidence
        score_match = match(r"^\s*\??\s*\d+\)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
        if score_match !== nothing
            node_idx = parse(Int, score_match[1]) + 1 # Convert back to 1-indexed for Julia
            betweenness_val = parse(Float64, score_match[3]) # Middle value is the estimated score
            
            if 1 <= node_idx <= num_nodes
                scores[node_idx] = betweenness_val
            end
        end
    end
    
    if execution_time < 0.0
        @warn "Failed to parse KADABRA C++ execution time."
    end
    
    return execution_time, scores
end

end # module
