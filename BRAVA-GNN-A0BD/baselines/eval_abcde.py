import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import random
import torch
import numpy as np
import networkx as nx
import time
import argparse
from scipy.stats import kendalltau
from torch_geometric.data import Data
from abcde.models import ABCDE, DrBC
from abcde.metrics import kendall_tau as abcde_kendall_tau
from results import append_csv
from config import TEST_GRAPHS_ALL
from graph_regimes import DIRECTED_SKIP, assert_regime

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="abcde", choices=["abcde", "drbc"])
parser.add_argument("--model_path", default=None)
parser.add_argument("--algorithm_name", default=None)
parser.add_argument("--test_graphs", nargs="*", default=None)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if args.model_path is None:
    args.model_path = ("baselines/abcde/drbc.ckpt" if args.model == "drbc"
                       else "baselines/abcde/best.ckpt")
if args.algorithm_name is None:
    base = "DrBC" if args.model == "drbc" else "ABCDE_Train"
    args.algorithm_name = f"{base}_S{args.seed}" if args.seed is not None else base

MODEL_PATH = args.model_path
TEST_GRAPHS = args.test_graphs if args.test_graphs else TEST_GRAPHS_ALL

def get_graph_data(name):
    if name in ["SF", "HY", "ER", "GRP"] or name.startswith("SF_") or name.startswith("HY_"):
        path = f"./datasets/data_splits/{name}/betweenness/test.pickle"
    else:
        path = f"./datasets/data_splits/{name}_bet.pickle"

    if not os.path.exists(path):
        return None, None, None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            graphs, sequences, bc_mats = data[0], data[1], data[3]
            G, node_seq = graphs[0], sequences[0]
            bc_scores = bc_mats[:, 0] if isinstance(bc_mats, np.ndarray) else bc_mats[0]
            num_nodes = len(node_seq)
            if len(bc_scores) > num_nodes:
                bc_scores = bc_scores[:num_nodes]
            return G, node_seq, bc_scores
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None, None


def convert_to_pyg(G, node_seq, bc_scores):
    mapping = {original_id: i for i, original_id in enumerate(node_seq)}
    G = nx.relabel_nodes(G, mapping)
    G_di = G if isinstance(G, nx.DiGraph) else G.to_directed()
    edge_index = np.array(G_di.edges).T
    degrees = nx.degree_centrality(G)
    degrees_arr = np.array([degrees[i] for i in range(len(G))], dtype='float32')
    label = np.array(bc_scores, dtype='float32')
    data = Data(
        x=torch.from_numpy(np.expand_dims(degrees_arr, -1)),
        y=torch.from_numpy(np.expand_dims(label, -1)),
        edge_index=torch.from_numpy(edge_index))
    data.src_ids = torch.zeros(1, dtype=torch.long)
    data.tgt_ids = torch.zeros(1, dtype=torch.long)
    data.num_nodes = G.number_of_nodes()
    return data


def load_drbc_model(path):
    print(f"Loading DrBC model from {path}...")
    model = DrBC()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    patched = {}
    for key, value in state_dict.items():
        if key == "conv.weight":
            patched["conv.lin.weight"] = value.T
        else:
            patched[key] = value
    model.load_state_dict(patched, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DrBC parameters: {n_params:,} ({n_params/1e3:.1f}K)")
    return model


def load_patched_model(path):
    print(f"Loading model from {path}...")
    model = ABCDE(
        nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
        conv_sizes=(48, 48, 32, 32, 24, 24),
        drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
        lr_reduce_patience=2, dropout=0.1)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    target_state_dict = model.state_dict()
    for key, value in state_dict.items():
        new_key = key
        new_val = value
        if "conv_blocks" in key and ".weight" in key and "lin.weight" not in key:
            parts = key.split('.')
            if len(parts) > 3 and parts[3] == '0':
                new_key = key.replace(".weight", ".lin.weight")
                new_val = value.T
        if new_key in target_state_dict:
            target_shape = target_state_dict[new_key].shape
            if new_val.shape != target_shape and new_val.shape == torch.Size([1]) and len(target_shape) == 1:
                new_val = new_val.repeat(target_shape[0])
        new_state_dict[new_key] = new_val
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def _count_gflops(model, pyg_data):
    """Inference GFLOPs for ABCDE/DrBC via forward hooks.

    GCNConv hooks count linear transform (N*in*out MACs) + sparse propagation
    (E*out MACs). Remaining nn.Linear modules (e.g. output MLP) are counted
    separately, skipping any that live inside a GCNConv.
    """
    import torch.nn as nn
    counter = [0]
    hooks = []
    linear_in_gcn_ids = set()

    try:
        from torch_geometric.nn import GCNConv

        for m in model.modules():
            if isinstance(m, GCNConv) and hasattr(m, "lin"):
                linear_in_gcn_ids.add(id(m.lin))

        def _gcn_hook(module, inputs, kwargs, output):
            x = inputs[0] if inputs else kwargs.get('x')
            edge_index = inputs[1] if len(inputs) > 1 else kwargs.get('edge_index')
            if x is None or edge_index is None:
                return
            N = x.shape[0]
            E = edge_index.shape[1]
            counter[0] += 2 * N * module.in_channels * module.out_channels  # linear
            counter[0] += 2 * E * module.out_channels                        # propagation

        for m in model.modules():
            if isinstance(m, GCNConv):
                hooks.append(m.register_forward_hook(_gcn_hook, with_kwargs=True))
    except ImportError:
        pass  # fall through to nn.Linear-only counting

    def _linear_hook(module, inputs, output):
        if id(module) in linear_in_gcn_ids:
            return
        x = inputs[0]
        batch = x.numel() // x.shape[-1]
        counter[0] += 2 * batch * module.in_features * module.out_features
        if module.bias is not None:
            counter[0] += batch * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    try:
        with torch.no_grad():
            model(pyg_data)
    finally:
        for h in hooks:
            h.remove()

    return counter[0] / 1e9


def calculate_topk(pred, label, p):
    k = max(1, int(len(label) * p))
    top_pred = np.argpartition(pred, -k)[-k:]
    top_true = np.argpartition(label, -k)[-k:]
    return len(np.intersect1d(top_pred, top_true)) / k


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = (load_drbc_model(MODEL_PATH) if args.model == "drbc"
             else load_patched_model(MODEL_PATH))
    kt_raw_ours, kt_raw_paper, kt_mask_ours, kt_mask_paper = [], [], [], []
    kt_filt_ours = []
    top1_raw, top5_raw, top10_raw = [], [], []
    top1_mask, top5_mask, top10_mask = [], [], []
    top1_filt, top5_filt, top10_filt = [], [], []
    times_list = []
    flops_list = []
    all_lists = (kt_raw_ours, kt_raw_paper, kt_mask_ours, kt_mask_paper,
                 kt_filt_ours,
                 top1_raw, top5_raw, top10_raw,
                 top1_mask, top5_mask, top10_mask,
                 top1_filt, top5_filt, top10_filt, times_list, flops_list)

    for name in TEST_GRAPHS:
        print(f"Processing {name}...", end=" ", flush=True)
        if args.model == "abcde" and name in DIRECTED_SKIP:
            for lst in all_lists:
                lst.append("")
            print("Skipped (directed — out of domain for ABCDE)")
            continue
        G, node_seq, bc_scores = get_graph_data(name)
        if G is None:
            for lst in all_lists:
                lst.append("")
            print("Skipped")
            continue

        assert_regime(name, G)
        pyg_data = convert_to_pyg(G, node_seq, bc_scores)
        try:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                pred_scores = model(pyg_data).view(-1).cpu().numpy()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.time() - start

            degrees_arr = pyg_data.x.numpy().flatten()
            leaf_sel = degrees_arr * len(degrees_arr) < 1.1
            pred_masked = pred_scores.copy()
            if leaf_sel.any():
                pred_masked[leaf_sel] = pred_masked.min() - np.finfo(np.float32).eps

            kt_ro, _ = kendalltau(pred_scores, bc_scores)
            kt_mo, _ = kendalltau(pred_masked, bc_scores)
            kt_rp = abcde_kendall_tau(bc_scores, pred_scores)
            kt_mp = abcde_kendall_tau(bc_scores, pred_masked)
            t1_r = calculate_topk(pred_scores, bc_scores, 0.01)
            t5_r = calculate_topk(pred_scores, bc_scores, 0.05)
            t10_r = calculate_topk(pred_scores, bc_scores, 0.10)
            t1_m = calculate_topk(pred_masked, bc_scores, 0.01)
            t5_m = calculate_topk(pred_masked, bc_scores, 0.05)
            t10_m = calculate_topk(pred_masked, bc_scores, 0.10)

            keep = np.asarray(bc_scores) > 0
            if int(keep.sum()) >= 2:
                kt_fo, _ = kendalltau(pred_scores[keep], np.asarray(bc_scores)[keep])
                t1_f = calculate_topk(pred_scores[keep], np.asarray(bc_scores)[keep], 0.01)
                t5_f = calculate_topk(pred_scores[keep], np.asarray(bc_scores)[keep], 0.05)
                t10_f = calculate_topk(pred_scores[keep], np.asarray(bc_scores)[keep], 0.10)
            else:
                kt_fo, t1_f, t5_f, t10_f = float("nan"), float("nan"), float("nan"), float("nan")

            print(f"KT r/p/m/mp/f: {kt_ro:.4f}/{kt_rp:.4f}/{kt_mo:.4f}/{kt_mp:.4f}/{kt_fo:.4f} | "
                  f"T1 r/m/f: {t1_r:.4f}/{t1_m:.4f}/{t1_f:.4f} | Time: {elapsed:.4f}s")
            kt_raw_ours.append(f"{kt_ro:.4f}")
            kt_raw_paper.append(f"{kt_rp:.4f}")
            kt_mask_ours.append(f"{kt_mo:.4f}")
            kt_mask_paper.append(f"{kt_mp:.4f}")
            kt_filt_ours.append(f"{kt_fo:.4f}" if not np.isnan(kt_fo) else "")
            top1_raw.append(f"{t1_r:.4f}")
            top5_raw.append(f"{t5_r:.4f}")
            top10_raw.append(f"{t10_r:.4f}")
            top1_mask.append(f"{t1_m:.4f}")
            top5_mask.append(f"{t5_m:.4f}")
            top10_mask.append(f"{t10_m:.4f}")
            top1_filt.append(f"{t1_f:.4f}" if not np.isnan(t1_f) else "")
            top5_filt.append(f"{t5_f:.4f}" if not np.isnan(t5_f) else "")
            top10_filt.append(f"{t10_f:.4f}" if not np.isnan(t10_f) else "")
            times_list.append(f"{elapsed:.6f}")
            flops_list.append(f"{_count_gflops(model, pyg_data):.4f}")
        except Exception as e:
            print(f"Error: {e}")
            for lst in all_lists:
                lst.append("")

    alg = args.algorithm_name
    header = "Algorithm," + ",".join(TEST_GRAPHS_ALL)

    def _pad(values):
        by_name = dict(zip(TEST_GRAPHS, values))
        return [by_name.get(n, "-") or "-" for n in TEST_GRAPHS_ALL]

    append_csv("all_results.csv", header, f"{alg}," + ",".join(_pad(kt_raw_ours)), results_dir="results/betweenness")
    append_csv("all_results.csv", header, f"{alg}_filtered," + ",".join(_pad(kt_filt_ours)), results_dir="results/betweenness")
    append_csv("all_results.csv", header, f"{alg}_paperKT," + ",".join(_pad(kt_raw_paper)), results_dir="results/betweenness")
    append_csv("all_results.csv", header, f"{alg}_masked," + ",".join(_pad(kt_mask_ours)), results_dir="results/betweenness")
    append_csv("all_results.csv", header, f"{alg}_masked_paperKT," + ",".join(_pad(kt_mask_paper)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top1%," + ",".join(_pad(top1_raw)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top5%," + ",".join(_pad(top5_raw)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_Top10%," + ",".join(_pad(top10_raw)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top1%," + ",".join(_pad(top1_filt)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top5%," + ",".join(_pad(top5_filt)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_filtered_Top10%," + ",".join(_pad(top10_filt)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_masked_Top1%," + ",".join(_pad(top1_mask)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_masked_Top5%," + ",".join(_pad(top5_mask)), results_dir="results/betweenness")
    append_csv("all_results_topk.csv", header, f"{alg}_masked_Top10%," + ",".join(_pad(top10_mask)), results_dir="results/betweenness")
    append_csv("all_results_wallclock.csv", header, f"{alg}," + ",".join(_pad(times_list)), results_dir="results/betweenness")
    append_csv("all_results_flops.csv", header, f"{alg}," + ",".join(_pad(flops_list)), results_dir="results/betweenness")


if __name__ == "__main__":
    main()