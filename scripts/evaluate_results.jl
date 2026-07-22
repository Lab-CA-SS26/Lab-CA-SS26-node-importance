using CSV
using DataFrames
using Statistics

function evaluate_results()
    results = DataFrame()
    for lang in ["cpp", "julia"]
        file = joinpath(@__DIR__, "..", "benchmark", "results", "kadabra", lang, "results.csv")
        if isfile(file)
            df = CSV.read(file, DataFrame)
            if nrow(df) > 0
                results = vcat(results, df)
            end
        else
            println("Warning: $file not found.")
        end
    end

    if nrow(results) == 0
        println("No results found.")
        return
    end

    # Calculate mean in case of multiple runs with same parameters
    summary = combine(groupby(results, [:graph, :k, :threads, :version]), 
                      :execution_time => mean => :execution_time)

    # Pivot to compare cpp vs julia side by side
    comp = unstack(summary, [:graph, :k, :threads], :version, :execution_time)
    
    # Fill missing columns if one language didn't run
    if !("cpp" in names(comp))
        comp[!, :cpp] .= missing
    end
    if !("julia" in names(comp))
        comp[!, :julia] .= missing
    end

    # Calculate speedup (CPP time / Julia time, > 1 means Julia is faster)
    comp.speedup_julia_vs_cpp = comp.cpp ./ comp.julia

    # Sort for better readability
    sort!(comp, [:graph, :k, :threads])

    println("=== Kadabra Performance Comparison (Execution Time in seconds) ===")
    show(comp, allrows=true, allcols=true)
    println()
    
    out_file = joinpath(@__DIR__, "..", "benchmark", "results", "kadabra", "comparison.csv")
    CSV.write(out_file, comp)
    println("Saved comparison to $out_file")
end

evaluate_results()
