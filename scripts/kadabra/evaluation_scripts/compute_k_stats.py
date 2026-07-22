import csv
from collections import defaultdict
import statistics
import os

input_file = "benchmark/results/kadabra/comparison.csv"
output_file = "benchmark/results/kadabra/k_differences_stats.csv"

# read data
data = defaultdict(lambda: defaultdict(dict))
with open(input_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # We focus on t=8
        if int(row["threads"]) == 8:
            graph = row["graph"]
            k = int(row["k"])
            data[graph]["cpp"][k] = float(row["cpp"])
            data[graph]["julia"][k] = float(row["julia"])

stats_data = []

for version in ["cpp", "julia"]:
    ratios_10 = []
    ratios_100 = []
    
    for graph, vers_data in data.items():
        v_data = vers_data[version]
        if 0 in v_data and 10 in v_data and 100 in v_data:
            t0 = v_data[0]
            t10 = v_data[10]
            t100 = v_data[100]
            
            ratios_10.append(t10 / t0)
            ratios_100.append(t100 / t0)
            
    # Compute stats
    stats_data.append({
        "version": version,
        "k_comparison": "10_vs_0",
        "mean_ratio": statistics.mean(ratios_10),
        "median_ratio": statistics.median(ratios_10),
        "min_ratio_speedup": min(ratios_10),
        "max_ratio_slowdown": max(ratios_10)
    })
    
    stats_data.append({
        "version": version,
        "k_comparison": "100_vs_0",
        "mean_ratio": statistics.mean(ratios_100),
        "median_ratio": statistics.median(ratios_100),
        "min_ratio_speedup": min(ratios_100),
        "max_ratio_slowdown": max(ratios_100)
    })

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["version", "k_comparison", "mean_ratio", "median_ratio", "min_ratio_speedup", "max_ratio_slowdown"])
    writer.writeheader()
    for row in stats_data:
        writer.writerow(row)

print("Saved stats to", output_file)
for row in stats_data:
    print(f"{row['version']} {row['k_comparison']}: Mean={row['mean_ratio']:.3f}, Median={row['median_ratio']:.3f}, Min={row['min_ratio_speedup']:.3f}, Max={row['max_ratio_slowdown']:.3f}")
