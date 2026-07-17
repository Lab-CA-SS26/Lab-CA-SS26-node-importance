import networkx as nx
import numpy as np
import pickle
import os

TEST_GRAPHS = [
    # "ego-Twitter",
    # "soc-Slashdot0902",
    # "gemsec-Facebook",
    # "musae-github",
    # "twitch-gamers",
    # "com-Orkut",
    # "com-DBLP",
    # "web-BerkStan",
    # "web-NotreDame",
    # "email-Enron",
    # "p2p-Gnutella30",
    # "SF"
    "soc-Slashdot0811",
]

def load_graph(name):
    paths = [
        f"./datasets/data_splits/{name}/betweenness/training.pickle",
        f"./datasets/graphs/{name}_data_bet.pickle",
        f"./datasets/data_splits/{name}_bet.pickle",
        f"./datasets/data_splits/{name}/betweenness/test.pickle",
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"Found: {p}")
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, list):
                        if len(data) == 4 and isinstance(data[0], list):
                            return data[0]
                        elif len(data) > 0 and isinstance(data[0], list) and isinstance(data[0][0], (nx.Graph, nx.DiGraph)):
                            return [x[0] for x in data]
                        elif len(data) > 0 and isinstance(data[0], (nx.Graph, nx.DiGraph)):
                             return data

                    if isinstance(data, (nx.Graph, nx.DiGraph)):
                        return [data]
                    
                    print(f"Warning: Data loaded from {p} but structure unrecognized. Type: {type(data)}")

            except Exception as e:
                print(f"Error loading pickle {p}: {e}")
                continue

    raw_path = f"./datasets/raw/{name}.txt"

    if os.path.exists(raw_path):
        return [load_from_txt(raw_path, directed=True)]

    return []

def load_from_txt(path, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('%'): continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading TXT {path}: {e}")
    return G

def estimate_gamma(degrees):
    d = np.array([x for x in degrees if x > 0])
    if len(d) < 10: return 0.0
    
    best_ks = float('inf')
    best_gamma = 0.0
    
    x_mins = sorted(list(set(d)))[:50]
    
    for xm in x_mins:
        data = d[d >= xm]
        n = len(data)
        if n < 10: continue
        
        denom = np.sum(np.log(data / (xm - 0.5)))
        if denom == 0: continue
        alpha = 1 + n / denom
        
        data_sorted = np.sort(data)
        y_empirical = np.arange(n, 0, -1) / n
        y_theoretical = (data_sorted / (xm - 0.5)) ** (-alpha + 1)
        ks = np.max(np.abs(y_empirical - y_theoretical))
        
        if ks < best_ks:
            best_ks = ks
            best_gamma = alpha
            
    return best_gamma

def main():
    print(f"{'Dataset':<25} {'Nodes':<10} {'Edges':<10} {'Avg Deg (k)':<15} {'Gamma':<15}")
    print("-" * 75)
    
    ks = []
    gammas = []
    
    for name in TEST_GRAPHS:
        graphs = load_graph(name)
        if not graphs:
            print(f"{name:<25} {'Not Found':<10} {'-':<10} {'-':<15} {'-'}")
            continue
            
        for i, G in enumerate(graphs):
            G_u = nx.Graph(G)
            G_u.remove_edges_from(nx.selfloop_edges(G_u))
            
            n = G_u.number_of_nodes()
            e = G_u.number_of_edges()
            if n == 0: 
                print(f"{name}_{i:<23} {'Empty':<10} {'-':<10} {'-':<15} {'-'}")
                continue
            
            k = 2 * e / n
            degrees = [d for n, d in G_u.degree()]
            gamma = estimate_gamma(degrees)
            
            ks.append(k)
            gammas.append(gamma)
            
            row_label = f"{name}" if len(graphs) == 1 else f"{name}_{i}"
            print(f"{row_label:<25} {n:<10} {e:<10} {k:<15.4f} {gamma:<15.4f}")
        
    print("-" * 75)
    if ks:
        print(f"{'AVERAGE':<25} {'-':<10} {'-':<10} {np.mean(ks):<15.4f} {np.mean(gammas):<15.4f}")

if __name__ == "__main__":
    main()