import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
import argparse
import os
import sys
import collections
import fcntl
import time

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_DATA_ROOT = os.path.join(_PROJECT_ROOT, "datasets")
sys.path.insert(0, _PROJECT_ROOT)
from config import TEST_GRAPHS_ALL
from results import save_results, append_csv
from layer import GNN_Layer, GNN_Layer_Init


def _count_gflops(model, adj, adj_t):
    """Count inference GFLOPs for GNN-Bet via forward hooks."""
    counter = [0]
    hooks = []

    def _gnn_init_hook(module, inputs, output):
        adj_in = inputs[0]
        N = adj_in.shape[0]
        counter[0] += 2 * adj_in._nnz() * module.out_features  # spmm(adj, W)
        if module.bias is not None:
            counter[0] += N * module.out_features

    def _gnn_layer_hook(module, inputs, output):
        x, adj_in = inputs
        N, in_f = x.shape
        out_f = module.out_features
        counter[0] += 2 * N * in_f * out_f          # mm(x, W)
        counter[0] += 2 * adj_in._nnz() * out_f     # spmm(adj, XW)
        if module.bias is not None:
            counter[0] += N * out_f

    def _linear_hook(module, inputs, output):
        x = inputs[0]
        batch = x.numel() // x.shape[-1]
        counter[0] += 2 * batch * module.in_features * module.out_features
        if module.bias is not None:
            counter[0] += batch * module.out_features

    for m in model.modules():
        if isinstance(m, GNN_Layer_Init):
            hooks.append(m.register_forward_hook(_gnn_init_hook))
        elif isinstance(m, GNN_Layer):
            hooks.append(m.register_forward_hook(_gnn_layer_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    try:
        with torch.no_grad():
            model(adj, adj_t)
    finally:
        for h in hooks:
            h.remove()

    return counter[0] / 1e9

#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="HY")
parser.add_argument("--top_k", action="store_true", help="Calculate and log Top-K accuracy metrics")
parser.add_argument("--seed", type=int, default=20, help="Random seed")
parser.add_argument("--test_graphs", nargs="*", default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

gtype = args.g
print(gtype)
if gtype == "SF":
    data_path = _DATA_ROOT + "/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = _DATA_ROOT + "/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = _DATA_ROOT + "/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")

elif gtype == "HY":
    data_path = _DATA_ROOT + "/data_splits/HY/betweenness/"
    print("Hyperbolic graphs selected.")



#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

TEST_GRAPHS = args.test_graphs if args.test_graphs else TEST_GRAPHS_ALL


real_data_dict = dict()
max_test_nodes = 0

for data in TEST_GRAPHS:
    path = f"{_DATA_ROOT}/data_splits/{data}_bet.pickle"
    if os.path.exists(path):
        print(f"Loading {data} graph...")
        with open(path,"rb") as fopen:
            list_graph,list_n_seq,list_num_node,bc_mat = pickle.load(fopen)
            real_data_dict[data] = [list_graph,list_n_seq,list_num_node,bc_mat]
            # Track max nodes for model sizing
            if list_num_node:
                max_test_nodes = max(max_test_nodes, max(list_num_node))
    else:
        print(f"Skipping {data} (not found)")

# Determine model size based on largest graph in train or test
max_train_nodes = max(list_num_node_train) if list_num_node_train else 0
model_size = max(10000, max_train_nodes, max_test_nodes)
print(f"Model initialized with size: {model_size}")

#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

val_idx = 3
val_name = TEST_GRAPHS[val_idx] if len(TEST_GRAPHS) > val_idx else ""
list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val = [], [], [], []
if val_name in real_data_dict:
    print(f"Preparing validation set: {val_name}")
    g_val, seq_val, num_val, bc_val = real_data_dict[val_name]
    list_adj_val, list_adj_t_val = graph_to_adj_bet(g_val, seq_val, num_val, model_size)
    list_num_node_val, bc_mat_val = num_val, bc_val

def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test, compute_filtered=False):
    model.eval()
    loss_val = 0
    list_kt = list()
    list_kt_filtered = list()
    topk_lists = collections.defaultdict(list)
    topk_lists_filtered = collections.defaultdict(list)
    total_inference_time = 0

    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        y_out = model(adj,adj_t)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.time()
        total_inference_time += (end - start)

        # Handle format difference: bc_mat can be matrix or list of arrays depending on dataset source
        if isinstance(bc_mat_test, np.ndarray) and bc_mat_test.ndim > 1:
             true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        else:
             true_arr = torch.from_numpy(bc_mat_test[j] if isinstance(bc_mat_test, list) else bc_mat_test).float()

        true_val = true_arr.to(device)

        if args.top_k:
            out = ranking_correlation_topk(y_out, true_val, num_nodes, model_size, compute_filtered=compute_filtered)
            if not compute_filtered:
                kt, topk_dict = out
            else:
                kt, topk_dict, kt_f, topk_dict_f = out
                list_kt_filtered.append(kt_f)
                for p, acc in topk_dict_f.items():
                    topk_lists_filtered[p].append(acc)
            list_kt.append(kt)
            for p, acc in topk_dict.items():
                topk_lists[p].append(acc)
        else:
            out = ranking_correlation(y_out, true_val, num_nodes, model_size, compute_filtered=compute_filtered)
            if not compute_filtered:
                list_kt.append(out)
            else:
                kt, kt_f = out
                list_kt.append(kt)
                list_kt_filtered.append(kt_f)

    mean_kt = np.mean(np.array(list_kt))
    avg_inference_time = total_inference_time / num_samples_test
    print(f"   Average KT score on test graphs is: {mean_kt:.4f} and std: {np.std(np.array(list_kt)):.4f}")

    topk_means = {}
    if args.top_k:
        topk_means = {p: np.mean(vals) for p, vals in topk_lists.items()}
        print(f"   Top 1%: {topk_means.get(0.01, 0):.4f} | Top 5%: {topk_means.get(0.05, 0):.4f} | Top 10%: {topk_means.get(0.1, 0):.4f}")

    if not compute_filtered:
        return mean_kt, topk_means, avg_inference_time

    mean_kt_filtered = float(np.nanmean(np.array(list_kt_filtered))) if list_kt_filtered else float("nan")
    topk_means_filtered = ({p: np.nanmean(vals) for p, vals in topk_lists_filtered.items()}
                           if args.top_k else {})
    print(f"   Average KT (filtered, bc>0): {mean_kt_filtered:.4f}")
    if args.top_k:
        print(f"   Top 1% (filtered): {topk_means_filtered.get(0.01, 0):.4f} | "
              f"Top 5% (filtered): {topk_means_filtered.get(0.05, 0):.4f} | "
              f"Top 10% (filtered): {topk_means_filtered.get(0.1, 0):.4f}")
    return mean_kt, topk_means, avg_inference_time, mean_kt_filtered, topk_means_filtered


#Model parameters
hidden = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
num_epoch = 10

print("Training")
print(f"Total Number of epoches: {num_epoch}")
training_start_time = time.time()
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
training_end_time = time.time()
total_training_time = training_end_time - training_start_time

print("Testing on real datasets")
scores_per_dataset = collections.defaultdict(list)
scores_filtered_per_dataset = collections.defaultdict(list)
topk_scores_per_dataset = collections.defaultdict(dict)
topk_scores_filtered_per_dataset = collections.defaultdict(dict)
times_per_dataset = collections.defaultdict(list)
flops_per_dataset = collections.defaultdict(list)

for data in TEST_GRAPHS:
    if data not in real_data_dict:
        continue
    print(f"Testing on {data} dataset")
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = real_data_dict[data]
    list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)
    with torch.no_grad():
        kt, topk_means, avg_time, kt_f, topk_means_f = test(
            list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test,
            compute_filtered=True)
        scores_per_dataset[data].append(kt)
        scores_filtered_per_dataset[data].append(kt_f)
        times_per_dataset[data].append(avg_time)
        flops_per_dataset[data].append(
            _count_gflops(model, list_adj_test[0].to(device), list_adj_t_test[0].to(device)))
        if args.top_k:
            topk_scores_per_dataset[data] = topk_means
            topk_scores_filtered_per_dataset[data] = topk_means_f

alg_name = f"Original_Murata_{gtype}_S{args.seed}"

print("\n" + "="*30)
print("Writing results to CSVs")
print("="*30)

save_results(alg_name, TEST_GRAPHS, scores_per_dataset, times_per_dataset,
             topk_scores_per_dataset, total_training_time, top_k=args.top_k,
             results_dir=os.path.join(_PROJECT_ROOT, "results/betweenness"),
             scores_filtered=scores_filtered_per_dataset,
             topk_scores_filtered=(topk_scores_filtered_per_dataset if args.top_k else None),
             flops=flops_per_dataset)

print("Done.")