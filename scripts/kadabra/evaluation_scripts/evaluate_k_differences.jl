using CSV
using DataFrames
using Statistics
using Plots

function evaluate_k_differences()
    input_file = joinpath(@__DIR__, "..", "..", "..", "benchmark", "results", "kadabra", "comparison.csv")
    if !isfile(input_file)
        println("Error: $input_file not found.")
        return
    end

    df = CSV.read(input_file, DataFrame)

    # We want to see how runtime differs across k=0, 10, 100 for each graph and version.
    # Let's fix threads to 8 to see the baseline algorithmic difference without high thread scaling noise.
    df_t8 = filter(row -> row.threads == 8, df)

    if nrow(df_t8) == 0
        println("No data for t=8 found.")
        return
    end

    # Pivot to have k as columns
    long_df = stack(df_t8, [:cpp, :julia], variable_name=:version, value_name=:runtime)
    
    wide_df = unstack(long_df, [:graph, :version], :k, :runtime, renamecols=x -> Symbol("k_", x))
    sort!(wide_df, [:graph, :version])

    println("=== Runtime (s) across different values of k (at t=8) ===")
    show(wide_df, allrows=true)
    println()

    # Create a plot for each graph to visualize this
    out_dir = joinpath(@__DIR__, "..", "..", "..", "benchmark", "results", "kadabra", "plots")
    mkpath(out_dir)

    for graph_name in unique(df.graph)
        graph_df = filter(row -> row.graph == graph_name && row.threads == 8, long_df)
        if nrow(graph_df) > 0
            sort!(graph_df, :k)
            
            p = plot(title="Effect of k on Runtime\n($graph_name, t=8)",
                     xlabel="k (Top-k Nodes, 0 = All Nodes)",
                     ylabel="Execution Time (s)",
                     legend=:topright,
                     marker=:circle,
                     xticks=[0, 10, 100],
                     linewidth=2)
                     
            for ver in unique(graph_df.version)
                ver_df = filter(row -> row.version == ver, graph_df)
                plot!(p, ver_df.k, ver_df.runtime, label=uppercase(String(ver)))
            end
            
            plot_file = joinpath(out_dir, "plot_k_diff_$(graph_name).png")
            savefig(p, plot_file)
            println("Saved plot to $plot_file")
        end
    end
end

evaluate_k_differences()
