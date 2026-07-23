using CSV
using DataFrames
using Printf

results_dir = "results/kadabra"
runtimes_file = joinpath(results_dir, "kadabra_runtimes_summary.csv")

if !isfile(runtimes_file)
    println("Runtimes summary file not found. Please run evaluate_kadabra_runtimes.jl first.")
    exit(1)
end

println("Calculating speedups from runtimes summary...")
df = CSV.read(runtimes_file, DataFrame)

# Separate into Julia and CPP
df_julia = filter(row -> row.Implementation == "kadabra-julia", df)
df_cpp = filter(row -> row.Implementation == "kadabra-cpp", df)

# Rename runtime columns for merging
rename!(df_julia, :Runtime_Mean => :Julia_Runtime_Mean, :Runtime_Median => :Julia_Runtime_Median, :Runtime_Std => :Julia_Runtime_Std)
rename!(df_cpp, :Runtime_Mean => :CPP_Runtime_Mean, :Runtime_Median => :CPP_Runtime_Median, :Runtime_Std => :CPP_Runtime_Std)

# Select only necessary columns to avoid clutter
select!(df_julia, [:Instance, :GraphType, :k, :Threads, :Julia_Runtime_Mean, :Julia_Runtime_Median, :Julia_Runtime_Std])
select!(df_cpp, [:Instance, :GraphType, :k, :Threads, :CPP_Runtime_Mean, :CPP_Runtime_Median, :CPP_Runtime_Std])

# Merge on exact config matching
df_merged = innerjoin(df_julia, df_cpp, on=[:Instance, :GraphType, :k, :Threads])

# Compute Speedup (CPP time / Julia time) => speedup > 1 means Julia is faster
# Previous comparison.csv computed speedup as Julia time / CPP time ? 
# Usually Speedup = T_baseline / T_new. Let's provide both or standard speedup.
# The user's old file had speedup_julia_vs_cpp, where if Julia=20, CPP=10, speedup=0.5
# We'll match that definition for consistency: T_cpp / T_julia. Wait, 10 / 20 = 0.5. 
# Oh, in the user's old file: cpp=10.95, julia=20.30, speedup=0.539. This is cpp / julia!
df_merged.Speedup_Mean = df_merged.CPP_Runtime_Mean ./ df_merged.Julia_Runtime_Mean
df_merged.Speedup_Median = df_merged.CPP_Runtime_Median ./ df_merged.Julia_Runtime_Median

out_csv = joinpath(results_dir, "kadabra_speedup_comparison.csv")
CSV.write(out_csv, df_merged)

println("Speedup evaluation complete. Results saved to $out_csv")
