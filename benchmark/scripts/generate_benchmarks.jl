using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using Graphs
using DelimitedFiles
using DataFrames
using CSV
using JLD2

include("BenchmarkUtils.jl")

"""
    with_timeout(f, timeout_sec)

Runs function `f` asynchronously. If it doesn't finish within `timeout_sec`, 
it throws an InterruptException to the task and returns `nothing`.
"""
function with_timeout(f, timeout_sec)
    c = Channel(1)
    task = @async begin
        try
            put!(c, f())
        catch e
            put!(c, e)
        end
    end
    
    timer = Timer(timeout_sec) do _
        if !istaskdone(task)
            Base.throwto(task, InterruptException())
        end
    end
    
    try
        res = take!(c)
        close(timer)
        if isa(res, Exception)
            if isa(res, InterruptException)
                return nothing
            end
            throw(res)
        end
        return res
    catch e
        close(timer)
        if isa(e, InterruptException) || (isa(e, TaskFailedException) && isa(e.task.exception, InterruptException))
            return nothing
        end
        rethrow(e)
    end
end

function generate_benchmarks()
    # Path to Instances directory relative to this script
    instances_dir = joinpath(dirname(dirname(@__DIR__)), "Instances")
    instances_file = joinpath(instances_dir, "instances.txt")
    
    # We output timings and JLD2 caches into the benchmark dir itself for easy loading by other scripts
    benchmark_dir = dirname(dirname(@__DIR__)) * "/benchmark"
    out_csv = joinpath(benchmark_dir, "exact_brandes_timings.csv")
    
    if !isfile(instances_file)
        println("Error: Instances list not found at: $instances_file")
        return
    end
    
    println("Reading instances from $instances_file")
    
    # Initialize DataFrame for timings
    if isfile(out_csv)
        results = CSV.read(out_csv, DataFrame)
        println("Loaded existing timings from $out_csv")
        processed_datasets = unique(results.Dataset)
    else
        results = DataFrame(
            Dataset = String[],
            Nodes = Int[],
            Edges = Int[],
            Runtime_s = Float64[],
            Status = String[]
        )
        processed_datasets = String[]
    end
    
    instances = BenchmarkUtils.read_instances(instances_file)
    
    for (rel_path, is_directed) in instances
        full_path = joinpath(instances_dir, rel_path)
        if !isfile(full_path)
            println("Graph file not found: $full_path. Skipping.")
            continue
        end
        
        dataset_name = basename(full_path)
        if dataset_name in processed_datasets
            println("\nSkipping: $rel_path (Already processed)")
            continue
        end
        
        println("\nProcessing: $rel_path (Directed: $is_directed)")
        
        # 1. Load graph
        g = BenchmarkUtils.load_graph_from_edgelist(full_path, is_directed)
        N = nv(g)
        M = ne(g)
        println("  Nodes: $N, Edges: $M")
        
        # 2. Compute Exact Betweenness Centrality with a 1-hour timeout (3600 seconds)
        println("  Computing exact betweenness centrality (Timeout: 1h)...")
        start_time = time()
        
        exact_bc = with_timeout(3600*6) do
            betweenness_centrality(g, normalize=true)
        end
        
        elapsed = time() - start_time
        
        if exact_bc === nothing
            println("  [TIMEOUT] Computation exceeded 6 hour. Skipping to next.")
            push!(results, (dataset_name, N, M, elapsed, "Timeout"))
        else
            println("  Computed in $(round(elapsed, digits=2)) seconds.")
            push!(results, (dataset_name, N, M, elapsed, "Success"))
            
            # 3. Save scores as txt (for easy inspection)
            base_path, _ = splitext(full_path)
            out_file_txt = "$(base_path)-score.txt"
            
            println("  Saving txt scores to: $out_file_txt")
            open(out_file_txt, "w") do io
                for score in exact_bc
                    println(io, score)
                end
            end
            
            # 4. Save as JLD2 directly into the benchmark folder to make it easy for comparison later
            # (Using exact_bc and exact_scores variables since run_benchmarks.jl and compare_topk.jl use different names)
            exact_scores = exact_bc
            jld2_file = joinpath(benchmark_dir, "$(dataset_name)_exact_bc.jld2")
            println("  Saving JLD2 cache to: $jld2_file")
            @save jld2_file exact_bc exact_scores
        end
        
        # Intermediate save in case of crash later
        CSV.write(out_csv, results)
    end
    
    println("\nAll benchmarks processed!")
    println("Timings saved to: $out_csv")
    display(results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_benchmarks()
end
