import pickle
import os
import networkx as nx
import numpy as np
import time

DATASETS = ["amazon", "cit-Patents", "com-lj", "com-youtube", "dblp"]

SOURCE_DIR = "./datasets/abcde"
DEST_DIR = "./datasets/data_splits"

def process_abcde_dataset(name):
    edge_path = os.path.join(SOURCE_DIR, f"{name}.txt")
    score_path = os.path.join(SOURCE_DIR, f"{name}-score.txt")

    if not os.path.exists(edge_path) or not os.path.exists(score_path):
        print(f"[SKIP] {name}: Files not found.")
        return

    print(f"[PROCESS] Importing {name}...")
    t0 = time.time()

    try:
        print(f"   -> Loading scores from {score_path}...")
        scores = np.loadtxt(score_path, dtype=np.float32)
        num_nodes_score = len(scores)

        print(f"   -> Loading edges from {edge_path}...")
        
        G = nx.read_edgelist(edge_path, nodetype=int, create_using=nx.Graph())
        
        sorted_nodes = sorted(list(G.nodes()))
        num_nodes_graph = len(sorted_nodes)
        
        print(f"   -> Graph Nodes: {num_nodes_graph} | Score Entries: {num_nodes_score}")

        if num_nodes_graph != num_nodes_score:
            print(f"[WARNING] Node count mismatch for {name}! GNN might fail.")
        
        mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
        G = nx.relabel_nodes(G, mapping)
        
        if num_nodes_score > num_nodes_graph:
            G.add_nodes_from(range(num_nodes_score))

        cent_mat = np.zeros((len(scores), 1), dtype=np.float32)
        cent_mat[:, 0] = scores

        node_seq = list(range(len(scores)))
        
        output_data = [
            [G],
            [node_seq],
            [len(scores)],
            cent_mat
        ]

        dest_file = os.path.join(DEST_DIR, f"{name}_bet.pickle")
        with open(dest_file, "wb") as f:
            pickle.dump(output_data, f)

        t1 = time.time()
        print(f"[DONE] Saved {name} to {dest_file} ({t1-t0:.2f}s)")

    except Exception as e:
        print(f"[ERROR] Failed processing {name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    for name in DATASETS:
        process_abcde_dataset(name)