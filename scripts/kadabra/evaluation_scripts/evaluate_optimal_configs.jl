using CSV
using DataFrames
using Statistics

function evaluate_optimal_configs()
    input_file = joinpath(@__DIR__, "..", "..", "..", "benchmark", "results", "kadabra", "comparison.csv")
    if !isfile(input_file)
        println("Error: $input_file not found.")
        return
    end

    df = CSV.read(input_file, DataFrame)

    # We want to compare C++ at t=8 against Julia at t=16
    df_cpp8 = filter(row -> row.threads == 8, df)
    df_julia16 = filter(row -> row.threads == 16, df)

    # Join on graph and k
    joined = innerjoin(df_cpp8[!, [:graph, :k, :cpp]], df_julia16[!, [:graph, :k, :julia]], on = [:graph, :k])
    
    # Calculate ratio: Julia_t16 / C++_t8
    # ratio > 1 means Julia is slower. ratio < 1 means Julia is faster.
    joined.ratio = joined.julia ./ joined.cpp

    println("=== Comparison of Optimal Configurations (Julia t=16 vs C++ t=8) ===")
    
    mean_val = mean(joined.ratio)
    median_val = median(joined.ratio)
    min_val = minimum(joined.ratio)
    max_val = maximum(joined.ratio)
    
    println("Mean Ratio (Julia/C++): $(round(mean_val, digits=3))")
    println("Median Ratio (Julia/C++): $(round(median_val, digits=3))")
    println("Min Ratio (Julia/C++): $(round(min_val, digits=3))")
    println("Max Ratio (Julia/C++): $(round(max_val, digits=3))")
    
    println("\nDetailed view:")
    show(joined, allrows=true)
    println()
    
    # Optionally, save to CSV
    output_file = joinpath(@__DIR__, "..", "..", "..", "benchmark", "results", "kadabra", "optimal_configs_comparison.csv")
    CSV.write(output_file, joined)
    println("Saved detailed comparison to $output_file")
end

evaluate_optimal_configs()
