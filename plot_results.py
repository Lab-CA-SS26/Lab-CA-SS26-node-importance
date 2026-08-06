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

# Filter out rows with NaN
df = df.dropna(subset=['Tau_Overall', 'Overlap_TopK'])

# 1. Bar Plot for Kendall Tau Overall
plt.figure(figsize=(10, 6))
sns.barplot(x="Graph", y="Tau_Overall", hue="Run_Type", data=df)
plt.title("Kendall Tau (Overall) per Graph")
plt.ylabel("Kendall Tau")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/kadabra_accuracy/kendall_tau_overall.png")
plt.close()

# 2. Bar Plot for Top K Kendall Tau
plt.figure(figsize=(10, 6))
sns.barplot(x="Graph", y="Tau_TopK", hue="Run_Type", data=df)
plt.title("Kendall Tau (Top K) per Graph")
plt.ylabel("Kendall Tau (Top K)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/kadabra_accuracy/kendall_tau_topk.png")
plt.close()

# 3. Bar Plot for Overlap Top K
plt.figure(figsize=(10, 6))
sns.barplot(x="Graph", y="Overlap_TopK", hue="Run_Type", data=df)
plt.title("Overlap Top K per Graph")
plt.ylabel("Overlap")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/kadabra_accuracy/overlap_topk.png")
plt.close()

print("Plots generated in results/kadabra_accuracy/")
