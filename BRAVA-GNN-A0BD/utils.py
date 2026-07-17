from networkit import *
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import scipy.sparse as sp
import random
import numpy as np
import torch


def get_out_edges(g_nkit, node_sequence):
    global all_out_dict
    all_out_dict = {n: set() for n in node_sequence}
    for n in node_sequence:
        g_nkit.forEdgesOf(n, nkit_outedges)
    return all_out_dict


def get_in_edges(g_nkit, node_sequence):
    global all_in_dict
    all_in_dict = {n: set() for n in node_sequence}
    for n in node_sequence:
        g_nkit.forInEdgesOf(n, nkit_inedges)
    return all_in_dict


def nkit_inedges(u, v, weight, edgeid):
    all_in_dict[u].add(v)


def nkit_outedges(u, v, weight, edgeid):
    all_out_dict[u].add(v)


def nx2nkit(g_nx):
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    for i in range(node_num):
        g_nkit.addNode()
    for e1, e2 in g_nx.edges():
        g_nkit.addEdge(e1, e2)
    assert g_nx.number_of_nodes() == g_nkit.numberOfNodes(), "Number of nodes not matching"
    assert g_nx.number_of_edges() == g_nkit.numberOfEdges(), "Number of edges not matching"
    return g_nkit


def clique_check(index, node_sequence, all_out_dict, all_in_dict):
    node = node_sequence[index]
    in_nodes = all_in_dict[node]
    out_nodes = all_out_dict[node]
    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if not tmp_out_nodes.issubset(all_out_dict[in_n]):
            return False
    return True


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def _prepare_graph(graph, node_sequence):
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    adj_temp = nx.adjacency_matrix(graph, nodelist=node_sequence)
    return graph, adj_temp


def _compute_clique_mask(graph, adj_temp, node_sequence, node_num):
    adj_temp_t = adj_temp.transpose()
    arr_temp1 = np.sum(adj_temp, axis=1)
    arr_temp2 = np.sum(adj_temp_t, axis=1)
    arr_multi = np.where(np.multiply(arr_temp1, arr_temp2) > 0, 1.0, 0.0)
    degree_arr = arr_multi.reshape(-1, 1)
    non_zero_ind = np.nonzero(degree_arr.flatten())[0]

    g_nkit = nx2nkit(graph)
    in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
    all_out_dict = get_out_edges(g_nkit, node_sequence)
    all_in_dict = get_in_edges(g_nkit, in_n_seq)

    for index in non_zero_ind:
        if clique_check(index, node_sequence, all_out_dict, all_in_dict):
            degree_arr[index, 0] = 0.0
    return degree_arr


def graph_to_adj_bet(list_graph, list_n_sequence, list_node_num, init_type="AW", nhid=0, preprocessing=True):
    list_adjacency = []
    list_adjacency_t = []

    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}", end='\r')
        graph, adj_temp = _prepare_graph(list_graph[i], list_n_sequence[i])
        adj_temp_t = adj_temp.transpose()
        node_num = list_node_num[i]

        if preprocessing:
            degree_arr = _compute_clique_mask(graph, adj_temp, list_n_sequence[i], node_num)
        else:
            degree_arr = np.ones((node_num, 1))

        adj_temp = adj_temp.multiply(csr_matrix(degree_arr))
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))

        list_adjacency.append(adj_temp.tocsr())
        list_adjacency_t.append(adj_temp_t.tocsr())

    print("")
    return list_adjacency, list_adjacency_t


def compute_pagerank_feature(adj_csr, alpha=0.85, iters=80):
    """Returns log(1 + PageRank * N) as (N, 1) numpy array — matches diagnostic."""
    N = adj_csr.shape[0]
    deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    P = sp.diags(deg_inv) @ adj_csr
    p = np.ones(N) / N
    for _ in range(iters):
        new_p = alpha * (P.T @ p) + (1 - alpha) / N
        if np.abs(new_p - p).max() < 1e-9:
            p = new_p; break
        p = new_p
    return np.log1p(p * N).reshape(-1, 1).astype(np.float32)


def compute_landmark_feature(adj_csr, count, mode="ff", seed=42):
    """Returns log(1+dist) to `count` landmarks as (N, count) float32 array.
    mode: 'ff' (farthest-first), 'random', 'deg' (highest-degree),
    'stats' (random; GNN layer compresses to 3 dims at forward time).
    """
    from scipy.sparse.csgraph import shortest_path
    A = ((adj_csr + adj_csr.T) > 0).astype(np.float64).tocsr()
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    count = min(count, N)

    def _bfs_one(idx):
        d = shortest_path(A, directed=False, indices=np.asarray([idx]),
                          unweighted=True).ravel()
        return d  # may contain inf for unreachable

    if mode == "ff":
        # Farthest-first: reuse per-landmark BFS vectors to avoid a final batch BFS.
        first = int(rng.integers(N))
        d0 = _bfs_one(first)
        finite0 = d0[np.isfinite(d0)]
        cap = float(finite0.max()) * 2.0 if finite0.size else 1.0
        d0 = np.where(np.isinf(d0), cap, d0)
        per_lm = [d0]
        min_dist = d0.copy()
        while len(per_lm) < count:
            next_lm = int(np.argmax(min_dist))
            new_d = _bfs_one(next_lm)
            new_d = np.where(np.isinf(new_d), cap, new_d)
            per_lm.append(new_d)
            min_dist = np.minimum(min_dist, new_d)
        dists = np.stack(per_lm, axis=1)  # (N, count)
    else:
        if mode == "deg":
            deg = np.asarray(A.sum(axis=1)).ravel()
            landmarks = np.argsort(-deg)[:count]
        else:  # random or stats
            landmarks = rng.choice(N, size=count, replace=False)
        dists = shortest_path(A, directed=False, indices=landmarks,
                              unweighted=True).T  # (N, count)
        finite = dists[np.isfinite(dists)]
        cap = float(finite.max()) * 2.0 if finite.size else 1.0
        dists = np.where(np.isinf(dists), cap, dists)
    return np.log1p(dists).astype(np.float32)


def _topk_dict(predict_arr, true_arr, node_num):
    pred_indices = np.argsort(-predict_arr)
    true_indices = np.argsort(-true_arr)
    accs = {}
    for p in [0.01, 0.05, 0.1]:
        k = max(1, int(node_num * p))
        intersect = np.intersect1d(pred_indices[:k], true_indices[:k], assume_unique=True)
        accs[p] = len(intersect) / k
    return accs


def ranking_correlation(y_out, true_val, node_num, compute_filtered=False):
    """Returns Kendall-Tau against full arrays. With compute_filtered=True, also returns
    KT restricted to the bc>0 subset (drops both arrays' bc=0 entries)."""
    predict_arr = y_out.reshape(-1).cpu().detach().numpy()[:node_num]
    true_arr = true_val.reshape(-1).cpu().detach().numpy()[:node_num]
    kt, _ = kendalltau(predict_arr, true_arr)
    if not compute_filtered:
        return kt
    keep = true_arr > 0
    if keep.sum() < 2:
        return kt, float("nan")
    kt_f, _ = kendalltau(predict_arr[keep], true_arr[keep])
    return kt, kt_f


def ranking_correlation_topk(y_out, true_val, node_num, compute_filtered=False):
    """Returns (KT, Top-K dict). With compute_filtered=True, also returns the same
    pair restricted to bc>0 nodes (Top-K threshold scales with bc>0 count)."""
    predict_arr = y_out.reshape(-1)[:node_num].cpu().detach().numpy()
    true_arr = true_val.reshape(-1)[:node_num].cpu().detach().numpy()
    kt, _ = kendalltau(predict_arr, true_arr)
    topk_accs = _topk_dict(predict_arr, true_arr, node_num)
    if not compute_filtered:
        return kt, topk_accs
    keep = true_arr > 0
    n_keep = int(keep.sum())
    if n_keep < 2:
        return kt, topk_accs, float("nan"), {0.01: float("nan"), 0.05: float("nan"), 0.1: float("nan")}
    kt_f, _ = kendalltau(predict_arr[keep], true_arr[keep])
    topk_accs_f = _topk_dict(predict_arr[keep], true_arr[keep], n_keep)
    return kt, topk_accs, kt_f, topk_accs_f


def loss_cal(y_out, true_val, num_nodes, device):
    y_out = y_out.reshape(-1)
    true_val = true_val.reshape(-1)

    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes * 20
    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

    return torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)


def convert_to_line_graph_sample(graph_nx, edge_seq, cent_arr):
    LG = nx.line_graph(graph_nx)
    edge_to_idx = {}
    for k, (u, v) in enumerate(edge_seq):
        edge_to_idx[(u, v)] = k
        edge_to_idx[(v, u)] = k
    next_idx = len(edge_seq)
    relabel = {}
    for node in LG.nodes():
        if node in edge_to_idx:
            relabel[node] = edge_to_idx[node]
        else:
            relabel[node] = next_idx
            next_idx += 1
    LG_int = nx.relabel_nodes(LG, relabel)
    E = len(edge_seq)
    lg_n_seq = list(range(E))
    return LG_int, lg_n_seq, E, cent_arr


def _lg_to_adj(lg, n_seq, num_node):
    node_to_pos = {n: p for p, n in enumerate(n_seq)}
    rows, cols = [], []
    directed = lg.is_directed()
    for u, v in lg.edges():
        if u not in node_to_pos or v not in node_to_pos:
            continue
        pu, pv = node_to_pos[u], node_to_pos[v]
        rows.append(pu); cols.append(pv)
        if not directed:
            rows.append(pv); cols.append(pu)
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(num_node, num_node))
    return adj, adj.T.tocsr()


def get_binary_mask(scores, k):
    k = min(max(1, int(k)), scores.shape[-1])
    _, indices = torch.topk(scores, k)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, indices, 1.0)
    return mask