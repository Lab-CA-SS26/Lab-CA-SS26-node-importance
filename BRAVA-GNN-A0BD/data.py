import os
import pickle
import numpy as np
import fcntl
from config import train_path, test_path
from graph_regimes import assert_regime, normalize_for_model
from utils import graph_to_adj_bet

def load_train_data(train_graphs):
    list_graph, list_n_seq, list_num_node, bc_mat = [], [], [], []
    latest_mtime = 0

    for g in train_graphs:
        path = train_path(g)
        if not os.path.exists(path): continue

        print(f"Loading {path}")
        latest_mtime = max(latest_mtime, os.path.getmtime(path))
        with open(path, "rb") as f:
            data = pickle.load(f)

        list_graph.extend(data[0])
        list_n_seq.extend(data[1])
        list_num_node.extend(data[2])

        cent_mat, nodes = data[3], data[2]
        for k in range(cent_mat.shape[1]):
            bc_mat.append(cent_mat[:nodes[k], k])

    seen, unique = set(), []
    for i, g in enumerate(list_graph):
        if id(g) not in seen:
            seen.add(id(g))
            unique.append(i)
    if len(unique) < len(list_graph):
        print(f"Deduped: {len(list_graph)} -> {len(unique)}")
        list_graph = [list_graph[i] for i in unique]
        list_n_seq = [list_n_seq[i] for i in unique]
        list_num_node = [list_num_node[i] for i in unique]
        bc_mat = [bc_mat[i] for i in unique]

    return list_graph, list_n_seq, list_num_node, bc_mat, latest_mtime

def load_test_data(test_graphs):
    test_data = {}
    latest_mtime = 0

    for g in test_graphs:
        path = test_path(g)
        if not os.path.exists(path):
            print(f"Dataset {g} not found, skipping")
            continue

        print(f"  Loading {path}")
        latest_mtime = max(latest_mtime, os.path.getmtime(path))
        with open(path, "rb") as f:
            d = pickle.load(f)

        for G in d[0]:
            assert_regime(g, G)
        graphs = [normalize_for_model(G) for G in d[0]]

        c_mat, l_nodes = d[3], d[2]
        c_list = [c_mat[:l_nodes[k], k] for k in range(c_mat.shape[1])]
        test_data[g] = (graphs, d[1], d[2], c_list)

    return test_data, latest_mtime

def build_adj_cache(list_graph_train, list_n_seq_train, list_num_node_train,
                    list_graph_val, list_n_seq_val, list_num_node_val,
                    model_size, gtype, val_key, latest_mtime):
    os.makedirs("pickles", exist_ok=True)
    cache_path = f"pickles/adj_data_scipy_{gtype}_{model_size}_{val_key}.pickle"
    lock_path = cache_path + ".lock"

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        if os.path.exists(cache_path) and os.path.getmtime(cache_path) > latest_mtime:
            print(f"Loading cached adjacency from {cache_path}")
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
        else:
            print("Building adjacency matrices.")
            adj_train, adj_t_train = graph_to_adj_bet(
                list_graph_train, list_n_seq_train, list_num_node_train)
            adj_val, adj_t_val = graph_to_adj_bet(
                list_graph_val, list_n_seq_val, list_num_node_val)
            result = [adj_train, adj_t_train, adj_val, adj_t_val]
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

        fcntl.flock(lf, fcntl.LOCK_UN)

    return result


def load_test_adj_or_build(name, preprocessing=True):
    """Per-test adj+bc cache. Returns (adj_list, adj_t_list, num_nodes_list, bc_list),
    or None if source pickle missing. Cache invalidated on source mtime change."""
    source_path = test_path(name)
    if not os.path.exists(source_path):
        return None

    os.makedirs("pickles", exist_ok=True)
    prep_tag = "prep" if preprocessing else "noprep"
    cache_path = f"pickles/test_adj_{name}_{prep_tag}.pickle"
    lock_path = cache_path + ".lock"

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(source_path):
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
        else:
            print(f"  Building test adj cache for {name} (preprocessing={preprocessing})")
            with open(source_path, "rb") as f:
                d = pickle.load(f)
            for G in d[0]:
                assert_regime(name, G)
            graphs = [normalize_for_model(G) for G in d[0]]
            n_seq, num_nodes, c_mat = d[1], d[2], d[3]
            bc_list = [c_mat[:num_nodes[k], k] for k in range(c_mat.shape[1])]
            adj, adj_t = graph_to_adj_bet(graphs, n_seq, num_nodes, preprocessing=preprocessing)
            result = (adj, adj_t, num_nodes, bc_list)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

        fcntl.flock(lf, fcntl.LOCK_UN)

    return result


