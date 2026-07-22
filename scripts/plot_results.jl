using CSV
using DataFrames
using Plots
using Statistics

function plot_results()
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

    # Aggregate in case there are multiple runs for the same parameters
    results = combine(groupby(results, [:graph, :k, :threads, :version]), 
                      :execution_time => mean => :execution_time)

    out_dir = joinpath(@__DIR__, "..", "benchmark", "results", "kadabra", "plots")
    mkpath(out_dir)

    # Group by graph and k to create separate plots
    for (key, subdf) in pairs(groupby(results, [:graph, :k]))
        graph = key.graph
        k = key.k
        
        # Sort by threads to make line plots connect properly
        sort!(subdf, :threads)

        p = plot(title="Kadabra Performance\nGraph: $graph, k=$k",
                 xlabel="Number of Threads",
                 ylabel="Execution Time (s)",
                 legend=:topright,
                 marker=:circle,
                 linewidth=2,
                 grid=true)
        
        for lang in unique(subdf.version)
            lang_data = filter(row -> row.version == lang, subdf)
            if nrow(lang_data) > 0
                plot!(p, lang_data.threads, lang_data.execution_time, label="Kadabra $lang")
            end
        end

        # Make x-axis ticks align with actual thread counts if possible
        if length(unique(subdf.threads)) > 0
            xticks!(p, unique(subdf.threads))
        end

        plot_file = joinpath(out_dir, "plot_$(graph)_k$(k).png")
        savefig(p, plot_file)
        println("Saved plot to $plot_file")
    end
end

plot_results()
