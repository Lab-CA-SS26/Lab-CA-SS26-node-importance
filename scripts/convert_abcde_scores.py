import os
import csv
import glob

score_files = glob.glob("Instances/ABCDE/*-score.txt")
out_dir = "Instances/ground_truth/test_instances"
os.makedirs(out_dir, exist_ok=True)

for score_file in score_files:
    graph_name = os.path.basename(score_file).replace("-score.txt", "")
    out_file = os.path.join(out_dir, f"{graph_name}_bet.csv")
    
    with open(score_file, 'r') as f:
        lines = f.readlines()
        
    with open(out_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['node', 'betweenness'])
        for i, line in enumerate(lines):
            score = line.strip()
            if score:
                writer.writerow([i, score])
    print(f"Converted {score_file} -> {out_file}")

print("Done converting ABCDE score files.")
