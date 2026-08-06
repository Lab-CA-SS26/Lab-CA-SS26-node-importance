import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

results_path = "results/kadabra_accuracy/evaluation_results.csv"
if not os.path.exists(results_path):
    print("Results file not found.")
    exit(1)

df = pd.read_csv(results_path)

if df.empty:
    print("No results to plot.")
    exit(0)

# Filter out rows with NaN in key columns
df = df.dropna(subset=['Tau_Overall', 'Execution_Time'])

out_dir = "results/kadabra_julia_specific"
os.makedirs(out_dir, exist_ok=True)

# 1. Strong Scaling Curve (Execution Time vs Threads) for kadabra-julia
df_julia = df[(df['Run_Type'] == 'kadabra-julia') & (df['K_Param'] == 0)]
if not df_julia.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_julia, x='Threads', y='Execution_Time', hue='Graph', marker='o')
    plt.title("Strong Scaling: Execution Time vs Threads (Kadabra-Julia, k=0)")
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Number of Threads")
    
    # Try to set sensible log scales if there's enough range
    if len(df_julia['Threads'].unique()) > 1:
        plt.xscale('log', base=2)
        plt.xticks(sorted(df_julia['Threads'].unique()), sorted(df_julia['Threads'].unique()))
    
    if df_julia['Execution_Time'].max() / df_julia['Execution_Time'].min() > 10:
        plt.yscale('log')
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/strong_scaling.png")
    plt.close()

# 2. Language Overhead Comparison (Julia vs C++ at t=1)
df_single = df[(df['Threads'] == 1) & (df['K_Param'] == 0) & (df['Run_Type'].isin(['kadabra-julia', 'kadabra-cpp']))]
if not df_single.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_single, x='Graph', y='Execution_Time', hue='Run_Type')
    plt.title("Language Overhead: Julia vs C++ (Single Thread, k=0)")
    plt.ylabel("Execution Time (s)")
    plt.xticks(rotation=45)
    
    if df_single['Execution_Time'].max() / df_single['Execution_Time'].min() > 10:
        plt.yscale('log')
        
    plt.tight_layout()
    plt.savefig(f"{out_dir}/language_comparison_t1.png")
    plt.close()

# 3. Accuracy vs Runtime (Pareto Front)
# For this we need to show Max_AE vs Execution Time
# Usually we look at different epsilons (0.01 vs 0.0001) for the same graph
df_pareto = df[(df['Run_Type'] == 'kadabra-julia') & (df['Threads'] == 1)]
if not df_pareto.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pareto, x='Execution_Time', y='Max_AE', hue='Graph', style='Epsilon', s=100)
    plt.title("Accuracy vs Runtime (Pareto Front for Julia Kadabra)")
    plt.xlabel("Execution Time (s)")
    plt.ylabel("Maximum Absolute Error (Max_AE)")
    
    if df_pareto['Execution_Time'].max() / df_pareto['Execution_Time'].min() > 10:
        plt.xscale('log')
    if df_pareto['Max_AE'].max() / df_pareto['Max_AE'].min() > 10:
        plt.yscale('log')
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/accuracy_vs_runtime_pareto.png")
    plt.close()

print(f"Plots generated in {out_dir}/")
