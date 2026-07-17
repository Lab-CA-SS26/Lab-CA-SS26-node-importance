 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_close import GNN_Close
import argparse
import os
import sys
import collections
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import TEST_GRAPHS_ALL
from results import save_results, append_csv

parser = argparse.ArgumentParser()
parser.add_argument("--g", default="SF")
parser.add_argument("--top_k", action="store_true")
parser.add_argument("--seed", type=int, default=20)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

gtype = args.g
print(gtype)
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/closeness/"
    print("Scale-free graphs selected.")
elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/closeness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/closeness/"
    print("Gaussian Random Partition graphs selected.")

print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,cc_mat_train = pickle.load(fopen)

with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,cc_mat_test = pickle.load(fopen)

TEST_GRAPHS = TEST_GRAPHS_ALL

real_data_dict = dict()
max_test_nodes = 0

for data in TEST_GRAPHS:
    path = f"./datasets/data_splits/{data}_close.pickle"
    if os.path.exists(path):
        print(f"Loading {data} graph...")
        with open(path, "rb") as fopen:
            list_graph, list_n_seq, list_num_node, cc_mat = pickle.load(fopen)
            real_data_dict[data] = [list_graph, list_n_seq, list_num_node, cc_mat]
            if list_num_node:
                max_test_nodes = max(max_test_nodes, max(list_num_node))
    else:
        print(f"Skipping {data} (not found)")

max_train_nodes = max(list_num_node_train) if list_num_node_train else 0
model_size = max(10000, max_train_nodes, max_test_nodes)
print(f"Model initialized with size: {model_size}")

print(f"Graphs to adjacency conversion.")
list_adj_train,list_adj_mod_train = graph_to_adj_close(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_mod_test = graph_to_adj_close(list_graph_test,list_n_seq_test,list_num_node_test,model_size)


def train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train):
    model.train()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i].to(device)
        adj_mod = list_adj_mod_train[i].to(device)
        num_nodes = list_num_node_train[i]

        optimizer.zero_grad()
        y_out = model(adj,adj_mod)
        true_arr = torch.from_numpy(cc_mat_train[:,i]).float()
        true_val = true_arr.to(device)

        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test):
    model.eval()
    list_kt = list()
    topk_lists = collections.defaultdict(list)
    total_inference_time = 0
    num_samples_test = len(list_adj_test)

    for j in range(num_samples_test):
        adj = list_adj_test[j].to(device)
        adj_mod = list_adj_mod_test[j].to(device)
        num_nodes = list_num_node_test[j]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        y_out = model(adj,adj_mod)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_inference_time += time.time() - start

        if isinstance(cc_mat_test, np.ndarray) and cc_mat_test.ndim > 1:
            true_arr = torch.from_numpy(cc_mat_test[:,j]).float()
        else:
            true_arr = torch.from_numpy(cc_mat_test[j] if isinstance(cc_mat_test, list) else cc_mat_test).float()
        true_val = true_arr.to(device)

        if args.top_k:
            kt, topk_dict = ranking_correlation_topk(y_out,true_val,num_nodes,model_size)
            list_kt.append(kt)
            for p, acc in topk_dict.items():
                topk_lists[p].append(acc)
        else:
            kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
            list_kt.append(kt)

    mean_kt = np.mean(np.array(list_kt))
    avg_inference_time = total_inference_time / num_samples_test
    print(f"   Average KT score on test graphs is: {mean_kt:.4f} and std: {np.std(np.array(list_kt)):.4f}")

    topk_means = {}
    if args.top_k:
        topk_means = {p: np.mean(vals) for p, vals in topk_lists.items()}
        print(f"   Top 1%: {topk_means.get(0.01, 0):.4f} | Top 5%: {topk_means.get(0.05, 0):.4f} | Top 10%: {topk_means.get(0.1, 0):.4f}")

    return mean_kt, topk_means, avg_inference_time


hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Close(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 15

print("Training")
print(f"Total Number of epoches: {num_epoch}")
training_start_time = time.time()
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train)
    with torch.no_grad():
        test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test)
total_training_time = time.time() - training_start_time

print("Testing on real datasets")
scores_per_dataset = collections.defaultdict(list)
topk_scores_per_dataset = collections.defaultdict(dict)
times_per_dataset = collections.defaultdict(list)

for data in TEST_GRAPHS:
    if data not in real_data_dict:
        continue
    print(f"Testing on {data} dataset")
    list_graph_test,list_n_seq_test,list_num_node_test,cc_mat_test = real_data_dict[data]
    list_adj_test,list_adj_mod_test = graph_to_adj_close(list_graph_test,list_n_seq_test,list_num_node_test,model_size)
    with torch.no_grad():
        kt, topk_means, avg_time = test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test)
        scores_per_dataset[data].append(kt)
        times_per_dataset[data].append(avg_time)
        if args.top_k:
            topk_scores_per_dataset[data] = topk_means

alg_name = f"Original_Murata_Close_{gtype}_S{args.seed}"

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
save_results(alg_name, TEST_GRAPHS, scores_per_dataset, times_per_dataset,
             topk_scores_per_dataset, total_training_time, top_k=args.top_k,
             results_dir=os.path.join(_PROJECT_ROOT, "results/closeness"))

print("Done.")