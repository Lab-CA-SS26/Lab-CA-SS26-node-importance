import pickle
import csv
import glob
import sys

def convert(in_file):
    out_file = in_file.replace('.pickle', '.csv')
    print(f"Converting {in_file} to {out_file}...")
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    # Extract the dictionary of scores
    # The structure is usually [[graph_object, {node_id: score}]]
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list) and len(data[0]) == 2:
        scores = data[0][1]
    elif isinstance(data, tuple) and len(data) == 2:
        scores = data[1]
    elif isinstance(data, dict):
        scores = data
    else:
        print(f"Unknown structure for {in_file}")
        return

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node', 'betweenness'])
        
        # We sort the keys so that nodes are in numerical order
        for k in sorted(scores.keys()):
            writer.writerow([k, scores[k]])

if __name__ == "__main__":
    folder = "Instances/ground_truth/test_instances/*.pickle"
    files = glob.glob(folder)
    for f in files:
        convert(f)
