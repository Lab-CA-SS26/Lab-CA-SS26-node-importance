using CSV
using DataFrames
using Printf

results_dir = "results/kadabra"
runtimes_file = joinpath(results_dir, "kadabra_runtimes_summary.csv")

if !isfile(runtimes_file)
    println("Runtimes summary file not found. Please run evaluate_kadabra_runtimes.jl first.")
    exit(1)
end

println("Calculating optimal configs from runtimes summary...")
df = CSV.read(runtimes_file, DataFrame)

# We want the optimal (minimum) Runtime_Mean for each (Instance, k, Implementation)
# across all available thread counts.

# Group by Instance, k, Implementation and find the row with the minimum Runtime_Mean
gdf = groupby(df, [:Instance, :k, :Implementation])

optimal_rows = []
for g in gdf
    min_idx = argmin(g.Runtime_Mean)
    push!(optimal_rows, g[min_idx, :])
end
df_optimal = DataFrame(optimal_rows)

# Now pivot to compare Julia and CPP
df_julia = filter(row -> row.Implementation == "kadabra-julia", df_optimal)
df_cpp = filter(row -> row.Implementation == "kadabra-cpp", df_optimal)

rename!(df_julia, :Runtime_Mean => :Julia_Min_Runtime, :Threads => :Julia_Optimal_Threads)
rename!(df_cpp, :Runtime_Mean => :CPP_Min_Runtime, :Threads => :CPP_Optimal_Threads)

select!(df_julia, [:Instance, :k, :Julia_Min_Runtime, :Julia_Optimal_Threads])
select!(df_cpp, [:Instance, :k, :CPP_Min_Runtime, :CPP_Optimal_Threads])

df_merged = innerjoin(df_julia, df_cpp, on=[:Instance, :k])

# Ratio as before: T_julia / T_cpp (wait, old optimal configs comparison: cpp=3.66, julia=6.05, ratio=1.65 => this is julia / cpp!)
df_merged.Ratio_Julia_vs_CPP = df_merged.Julia_Min_Runtime ./ df_merged.CPP_Min_Runtime

out_csv = joinpath(results_dir, "kadabra_optimal_configs_comparison.csv")
CSV.write(out_csv, df_merged)

println("Optimal config evaluation complete. Results saved to $out_csv")
