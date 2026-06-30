import os
import networkx as nx
import networkit as nk
import random
import csv

N_NODES = 100000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Instances", "Training")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_graph(G, name, is_directed):
    # Calculate exact betweenness using networkit
    # Convert networkx to networkit
    nkG = nk.Graph(G.number_of_nodes(), directed=is_directed)
    for u, v in G.edges():
        if not nkG.hasEdge(u, v):
            nkG.addEdge(u, v)
    
    print(f"Calculating betweenness for {name}...")
    bc = nk.centrality.Betweenness(nkG)
    bc.run()
    scores = bc.scores()
    
    # Save edges
    with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
            
    # Save scores
    with open(os.path.join(OUTPUT_DIR, f"{name}_scores.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "betweenness"])
        for i, score in enumerate(scores):
            writer.writerow([i, score])

print("Generating 10 Directed Scale-Free Graphs (BO)...")
for i in range(10):
    G = nx.scale_free_graph(N_NODES)
    G = nx.DiGraph(G) # remove parallel edges and self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    save_graph(G, f"BO_{i}", True)

print("Generating 10 Undirected Scale-Free Graphs (SBO)...")
for i in range(10):
    G = nx.scale_free_graph(N_NODES)
    G = nx.Graph(G) # undirected, no parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    save_graph(G, f"SBO_{i}", False)

print("Generating 10 Uniformly Directed Hyperbolic Random Graphs (UDHY)...")
for i in range(10):
    # empirical params from paper: avg degree ~ 15, gamma ~ 2.5
    T = random.uniform(0.01, 0.5)
    gen = nk.generators.HyperbolicGenerator(N_NODES, k=15, gamma=2.5, T=T)
    nkG_undirected = gen.generate()
    
    # orient uniformly at random
    G = nx.DiGraph()
    G.add_nodes_from(range(N_NODES))
    for u, v in nkG_undirected.iterEdges():
        if random.random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
            
    save_graph(G, f"UDHY_{i}", True)

print("Data generation complete!")
