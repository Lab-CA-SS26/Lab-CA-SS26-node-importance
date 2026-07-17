import networkx as nx
import numpy as np

def compute_stats(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for n, d in G.degree()]
    var_deg = np.var(degrees)
    max_deg = np.max(degrees)
    
    # core decomposition
    G.remove_edges_from(nx.selfloop_edges(G))
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    
    print(f"n: {n}, m: {m}, m/n: {m/n:.2f}")
    print(f"var: {var_deg:.2f}, max_deg: {max_deg}, max_core: {max_core}")

compute_stats("Instances/TestInstances/email-Enron.txt")
