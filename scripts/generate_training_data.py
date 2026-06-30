import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
import os
import argparse
import sys
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

np.random.seed(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Instances", "Training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test graphs stored as DiGraph but semantically undirected (0% reciprocity,
# bidirectional underlying relation). Symmetrize at load so ground-truth BC
# is computed on a graph where A = A^T. See docs/graph_regimes.md.
UNDIRECTED_STORED_AS_DIGRAPH = {
    "p2p-Gnutella31",
}


def load_real_data(file_path, is_undirected=False):
    original_to_new = {}
    new_id = 0
    edges = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 2: continue
            from_node, to_node = int(parts[0]), int(parts[1])
            
            if from_node not in original_to_new:
                original_to_new[from_node] = new_id
                new_id += 1
            
            if to_node not in original_to_new:
                original_to_new[to_node] = new_id
                new_id += 1
            
            edges.append((original_to_new[from_node], original_to_new[to_node]))
            if is_undirected:
                edges.append((original_to_new[to_node], original_to_new[from_node]))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    return G

def create_graph(graph_type, idx=0, num_nodes=100000):
    # Required regime suffix on SF/HY: _Dir, _Sym, _SymBA (SF only), or _mNN (HY only).
    # _SymBA must be matched before _Sym since _Sym is its prefix.
    regime = None
    base = graph_type
    for suf in ("_mNN", "_Dir", "_SymBA", "_Sym"):
        if graph_type.endswith(suf):
            regime, base = suf[1:], graph_type[:-len(suf)]
            break

    if base == "ER":
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = True)
        return g_nx

    if base == "SF" or base.startswith("SF_"):
        if regime is None:
            raise ValueError(f"{graph_type}: SF requires an explicit regime suffix (_Dir, _Sym, or _SymBA)")
        if regime == "SymBA":
            m = int(np.random.randint(2, 6))
            print(f" [DEBUG] SF SymBA Params: N={num_nodes}, m={m}")
            return nx.barabasi_albert_graph(num_nodes, m).to_directed()
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        if regime == "Sym":
            g_nx = g_nx.to_undirected().to_directed()
        return g_nx


    if base == "GRP":
        s = np.random.randint(200,1000)
        v = np.random.randint(200,1000)
        p_in = np.random.randint(2,25)*0.0001
        p_out = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = True)
        assert nx.is_directed(g_nx)==True,"Not directed"
        return g_nx

    if base.startswith("HY"):
        if regime is None:
            raise ValueError(f"{graph_type}: HY requires an explicit regime suffix (_Dir, _Sym, or _mNN)")

        if regime == "mNN":
            # Geometric directed hyperbolic graph (Kasyanov et al. arXiv:2303.01002).
            # Out-degree is exactly m to the m hyperbolic-nearest neighbours.
            # Periphery-dominated regime (m/ν ≥ ~10²) yields the truncated
            # power-law in-degree with exponent -3 (paper §III.E, Fig 7).
            # m ≥ 12 keeps the LCC ≈ N at this scale; smaller m shatters
            # because the periphery's local m-NN clusters no longer bridge
            # angular sectors. The paper never claims connectivity (Fig 7
            # only reports in-degree distributions), so we pin empirically.
            m = int(np.random.choice([12, 16, 20]))
            m_over_nu = float(np.random.uniform(100.0, 2000.0))
            nu = m / m_over_nu
            R = float(np.arccosh(1.0 + num_nodes / (2.0 * np.pi * nu)))
            print(f" [DEBUG] HY mNN Params: N={num_nodes}, m={m}, R={R:.3f}, nu={nu:.5f}, m/nu={m_over_nu:.1f}")
            r_arr, theta_arr = _sample_hy_disk_uniform(num_nodes, R)
            return _build_hy_mNN_edges(r_arr, theta_arr, m)

        # k = average degree, gamma = power law exponent, T = temperature.
        # Sampled from real-graph (k, gamma) stats with ±5% jitter.
        REAL_DATA_STATS = [
            (21.4106, 2.1604),  # soc-Slashdot0811  (directed: total degree)
            (32.4296, 2.4494),  # gemsec-Facebook
            (15.3317, 2.5023),  # musae-github
            (80.8684, 2.1334),  # twitch-gamers
            (76.2814, 2.2849),  # com-Orkut
            (6.6221, 3.5380),   # com-DBLP
            (22.1841, 2.6245),  # web-BerkStan      (directed: total degree)
            (9.0239, 2.0757),   # web-NotreDame     (directed: total degree)
            (10.0202, 2.3982),  # email-Enron
            (4.8159, 7.0192),   # p2p-Gnutella30
        ]
        target_k, target_gamma = REAL_DATA_STATS[np.random.randint(len(REAL_DATA_STATS))]

        k = target_k * np.random.uniform(0.95, 1.05)
        # Clamp gamma > 2 (HyperbolicGenerator requirement).
        gamma = max(2.1, target_gamma * np.random.uniform(0.95, 1.05))
        T = np.random.uniform(0, 0.5)

        print(f" [DEBUG] HY Params: k={k:.4f}, gamma={gamma:.4f}, T={T:.4f}")
        hg = generators.HyperbolicGenerator(num_nodes, k, gamma, T)
        g_nx = nkit2nx(hg.generate())
        if regime == "Dir":
            # Random per-edge orientation → DiGraph with ~0% reciprocity.
            dg = nx.DiGraph()
            dg.add_nodes_from(g_nx.nodes())
            for u, v in g_nx.edges():
                if np.random.rand() < 0.5:
                    dg.add_edge(u, v)
                else:
                    dg.add_edge(v, u)
            return dg
        # _Sym: symmetric DiGraph → downstream BC = undirected BC.
        return g_nx.to_directed()


def _sample_hy_disk_uniform(n, R):
    """Sample n points uniformly on a hyperbolic disk of radius R (α=1).

    Hyperbolic polar area element dA = sinh(ρ)dρdθ (paper eq 11), so
    uniform-on-disk has radial CDF F(r) = (cosh r − 1)/(cosh R − 1) (eq 12).
    Inverse-CDF: r = arccosh(1 + U·(cosh R − 1)).
    """
    r = np.arccosh(1.0 + np.random.uniform(0.0, np.cosh(R) - 1.0, size=n))
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    return r.astype(np.float32), theta.astype(np.float32)


def _build_hy_mNN_edges(r, theta, m, chunk_size=2000):
    """Directed m-NN graph from hyperbolic coordinates.

    Pairwise distance via the hyperbolic cosine theorem (paper eq 14):
        cosh(d_ij) = cosh(r_i)cosh(r_j) − sinh(r_i)sinh(r_j) cos(θ_i − θ_j).
    Computed in row-chunks (CUDA if available, else NumPy) to bound memory.
    Each row's m smallest entries (excluding self) become out-edges.
    """
    n = int(r.shape[0])
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        r_t = torch.from_numpy(r).to(device)
        theta_t = torch.from_numpy(theta).to(device)
        cosh_r = torch.cosh(r_t)
        sinh_r = torch.sinh(r_t)
        srcs, tgts = [], []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ci = cosh_r[start:end].unsqueeze(1)
            si = sinh_r[start:end].unsqueeze(1)
            ti = theta_t[start:end].unsqueeze(1)
            arg = ci * cosh_r.unsqueeze(0) - si * sinh_r.unsqueeze(0) * torch.cos(ti - theta_t.unsqueeze(0))
            d = torch.acosh(torch.clamp(arg, min=1.0))
            d[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = float("inf")
            _, nbrs = torch.topk(d, k=m, largest=False, dim=1)
            src = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, m).reshape(-1)
            srcs.append(src.cpu().numpy())
            tgts.append(nbrs.reshape(-1).cpu().numpy())
        src_all = np.concatenate(srcs)
        tgt_all = np.concatenate(tgts)
    except ImportError:
        cosh_r = np.cosh(r)
        sinh_r = np.sinh(r)
        srcs, tgts = [], []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ci = cosh_r[start:end][:, None]
            si = sinh_r[start:end][:, None]
            ti = theta[start:end][:, None]
            arg = ci * cosh_r[None, :] - si * sinh_r[None, :] * np.cos(ti - theta[None, :])
            np.clip(arg, 1.0, None, out=arg)
            d = np.arccosh(arg)
            d[np.arange(end - start), np.arange(start, end)] = np.inf
            nbrs = np.argpartition(d, m, axis=1)[:, :m]
            srcs.append(np.repeat(np.arange(start, end), m))
            tgts.append(nbrs.reshape(-1))
        src_all = np.concatenate(srcs)
        tgt_all = np.concatenate(tgts)

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(src_all.tolist(), tgt_all.tolist()))
    return g


def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

def nkit2nx(g_nkit):
    return nxadapter.nk2nx(g_nkit)


def cal_exact_bet(g_nkit):
    exact_bet = centrality.Betweenness(g_nkit, normalized=True).run().ranking()
    exact_bet_dict = {k: v for k, v in exact_bet}
    return exact_bet_dict

def save_instance(G, bet_dict, name):
    edges_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    scores_path = os.path.join(OUTPUT_DIR, f"{name}_scores.csv")
    
    # Save edges
    with open(edges_path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
            
    # Save scores
    with open(scores_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "betweenness"])
        # bet_dict may not contain isolated nodes, handle properly up to max node id
        max_id = G.number_of_nodes()
        for i in range(max_id):
            writer.writerow([i, bet_dict.get(i, 0.0)])

def main():
    print("Load and process data graphs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Wiki-Vote", "soc-Epinions1", "web-Google", "soc-Slashdot0811","p2p-Gnutella31"])
    parser.add_argument("--num_nodes", type=int, default=100000)
    # Manually extract --datasets values before argparse to handle names starting with '-'
    # (e.g. negative lat/lon coordinates like -0.3560_117.2740_r195239_n102157)
    _datasets = None
    if '--datasets' in sys.argv:
        _start = sys.argv.index('--datasets')
        _vals, _i = [], _start + 1
        while _i < len(sys.argv) and not sys.argv[_i].startswith('--'):
            _vals.append(sys.argv[_i])
            _i += 1
        _datasets = _vals
        sys.argv = sys.argv[:_start] + sys.argv[_i:]
    args = parser.parse_args()
    if _datasets is not None:
        args.datasets = _datasets

    output_dir = "./datasets/graphs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data in args.datasets:
        is_synthetic = data in ["SF", "ER", "GRP"] or data.startswith("HY") or data.startswith("SF_")

        if is_synthetic:
            # Check if this type already exists.
            if os.path.exists(os.path.join(OUTPUT_DIR, f"{data}_0.txt")):
                print(f"Skipping {data} (exists)")
                continue

            # HY_N / SF_N: N from name. Bare "HY"/"SF"/"ER"/"GRP": 5.
            if data.startswith("HY_") or data.startswith("SF_"):
                try:
                    num_of_graphs = int(data.split("_")[1])
                except ValueError:
                    num_of_graphs = 5
            else:
                num_of_graphs = 5

            print("###################")
            print(f"Generating synthetic graph type : {data}")
            print(f"Number of graphs to be generated:{num_of_graphs}")
            list_bet_data = list()
            for i in range(num_of_graphs):
                print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
                g_nx = create_graph(data, i, args.num_nodes)

                num_before = g_nx.number_of_nodes()
                if nx.number_of_isolates(g_nx)>0:
                    g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
                    g_nx = nx.convert_node_labels_to_integers(g_nx)

                num_after = g_nx.number_of_nodes()
                print(f" -> Nodes: {num_before} -> {num_after}")

                g_nkit = nx2nkit(g_nx)
                bet_dict = cal_exact_bet(g_nkit)
                
                # Write natively to .txt and .csv instead of picking
                save_instance(g_nx, bet_dict, f"{data}_{i}")

            print("")
            print(f"{data} Graphs saved to Instances/Training")

        else:
            if os.path.exists(os.path.join(OUTPUT_DIR, f"{data}.txt")):
                print(f"Skipping {data} (exists)")
                continue

            # In the original python pipeline, real graphs were also stored here. 
            # We already placed the real graphs in Instances/TestInstances, so if
            # they are requested, parse from there
            input_path_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Instances", "TestInstances", f"{data}.txt")

            G = None
            if os.path.exists(input_path_raw):
                print(f"Processing {data} (raw)...")
                G = load_real_data(input_path_raw)
            else:
                print(f"Warning: raw data for {data} not found in datasets/raw")
                continue

            if nx.number_of_isolates(G)>0:
                G.remove_nodes_from(list(nx.isolates(G)))
                G = nx.convert_node_labels_to_integers(G)

            if data in UNDIRECTED_STORED_AS_DIGRAPH:
                print(f"Symmetrizing {data} (declared_regime=undirected_stored_as_digraph)")
                G = G.to_undirected().to_directed()

            g_nkit = nx2nkit(G)

            print(f"Calculating Betweenness for {data}...")
            bet_dict = cal_exact_bet(g_nkit)
            save_instance(G, bet_dict, data)
            print(f"{data} bet saved to Instances/Training")

    print("End.")


if __name__ == "__main__":
    main()