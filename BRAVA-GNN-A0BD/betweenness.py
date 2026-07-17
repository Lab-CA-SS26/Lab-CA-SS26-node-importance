import collections
import time
import os
import numpy as np
import torch

from config import parse_args, set_seeds, resolve_train_graphs, resolve_test_graphs, get_algo_name
from data import load_train_data, build_adj_cache, load_test_adj_or_build
from training import train, evaluate
from results import save_results
from model_bet import GNN_Bet
from utils import sparse_mx_to_torch_sparse_tensor, compute_landmark_feature, compute_pagerank_feature
from layer import parse_init_type as _parse_init_type
from flops import count_inference_gflops


def _dump_predictions(model, adj_csr, adj_t_csr, num_nodes, bc_true, lm_np, pr_np,
                      device, out_path):
    """One forward pass; write per-node arrays to .npz for offline error analysis."""
    adj = sparse_mx_to_torch_sparse_tensor(adj_csr).to(device)
    adj_t = sparse_mx_to_torch_sparse_tensor(adj_t_csr).to(device)
    lm = torch.from_numpy(lm_np).float().to(device) if lm_np is not None else None
    pr = torch.from_numpy(pr_np).float().to(device) if pr_np is not None else None
    with torch.no_grad():
        y = model(adj, adj_t, landmarks=lm, pagerank=pr).reshape(-1)[:num_nodes]
    pred = y.detach().cpu().numpy()
    true = np.asarray(bc_true).reshape(-1)[:num_nodes]
    out_deg = np.asarray(adj_csr.sum(axis=1)).reshape(-1)[:num_nodes]
    in_deg = np.asarray(adj_csr.sum(axis=0)).reshape(-1)[:num_nodes]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, true_bc=true.astype(np.float32),
                        pred_bc=pred.astype(np.float32),
                        in_deg=in_deg.astype(np.int32),
                        out_deg=out_deg.astype(np.int32))


def _compute_addons(list_adj, list_num_node, has_pr, sp_count, sp_mode):
    """Compute (per-graph) addon features for an adjacency list. Returns
    (list_landmarks, list_pagerank) as lists of (num_node, K) numpy arrays
    (truncated to actual node count)."""
    if not has_pr and not sp_count:
        return None, None
    lm_list = [] if sp_count else None
    pr_list = [] if has_pr else None
    for adj_csr, num in zip(list_adj, list_num_node):
        if sp_count:
            lm = compute_landmark_feature(adj_csr, sp_count, mode=sp_mode)
            lm_list.append(lm[:num])
        if has_pr:
            pr = compute_pagerank_feature(adj_csr)
            pr_list.append(pr[:num])
    return lm_list, pr_list

args = parse_args()
set_seeds(args.seed)

run_name = get_algo_name(args)

TRAIN_GRAPHS = resolve_train_graphs(args.train_type)
TEST_GRAPHS = args.test_graphs if args.test_graphs else resolve_test_graphs(args.run_all_tests)

print(f"Train: {args.train_type} | Init: {args.init_type} | Layers: {args.num_layers} | "
      f"Nhid: {args.nhid} | Dropout: {args.dropout} | Seed: {args.seed} | Epochs: {args.epochs}")

timings = {}

t0 = time.perf_counter()
list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train, train_mtime = \
    load_train_data(TRAIN_GRAPHS)
timings["load_train"] = time.perf_counter() - t0

model_size = max(list_num_node_train) if list_num_node_train else 0
print(f"Model size: {model_size} (Train-set max; test graphs are processed at native size.)")

val_key = TEST_GRAPHS[0] if TEST_GRAPHS else None
val_data = load_test_adj_or_build(val_key, preprocessing=True) if val_key else None
if val_data is None:
    val_key = None
    list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val = [], [], [], []
else:
    list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val = val_data
    print(f"Validation: {val_key}")

adj_cache_path = f"pickles/adj_data_scipy_{args.train_type}_{model_size}_None.pickle"
cache_hit = os.path.exists(adj_cache_path) and os.path.getmtime(adj_cache_path) > train_mtime
t0 = time.perf_counter()
list_adj_train, list_adj_t_train, _, _ = build_adj_cache(
    list_graph_train, list_n_seq_train, list_num_node_train,
    [], [], [],
    model_size, args.train_type, None, train_mtime)
timings["build_adj_cache"] = time.perf_counter() - t0
timings["adj_cache_hit"] = cache_hit

# Parse init_type for optional _pr / _sp{N}_{mode} addon channels.
_, _has_pr, _sp_count, _sp_mode = _parse_init_type(args.init_type)
if _has_pr or _sp_count:
    print(f"Init addons: pr={_has_pr}  sp={_sp_count}{('_'+_sp_mode) if _sp_count else ''}")
    t0 = time.perf_counter()
    lm_train, pr_train = _compute_addons(
        list_adj_train, list_num_node_train, _has_pr, _sp_count, _sp_mode)
    if list_adj_val:
        lm_val, pr_val = _compute_addons(
            list_adj_val, list_num_node_val, _has_pr, _sp_count, _sp_mode)
    else:
        lm_val, pr_val = None, None
    timings["compute_train_val_addons"] = time.perf_counter() - t0
    print(f"  train+val addon compute: {timings['compute_train_val_addons']:.1f}s")
else:
    lm_train, pr_train, lm_val, pr_val = None, None, None, None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

t0 = time.perf_counter()
model = GNN_Bet(
    ninput=model_size, nhid=args.nhid, dropout=args.dropout, mode=args.mode,
    repeats=args.repeats, init_type=args.init_type, leverage=args.leverage,
    normalize=args.normalize, num_layers=args.num_layers,
    fusion=args.fusion, shared_encoders=not args.unshared_encoders,
).to(device)
if torch.cuda.is_available(): torch.cuda.synchronize()
timings["model_init"] = time.perf_counter() - t0

print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

per_epoch_train, per_epoch_val = [], []
training_start = time.time()
for e in range(args.epochs):
    print(f"Epoch {e+1}/{args.epochs}")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    avg_loss = train(model, optimizer, list_adj_train, list_adj_t_train, list_num_node_train,
          bc_mat_train, model_size, device, args.accumulate,
          list_landmarks_train=lm_train, list_pagerank_train=pr_train)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    per_epoch_train.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        val_kt, val_std, val_topk, _ = evaluate(
            model, list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val,
            model_size, device, args.top_k,
            list_landmarks_test=lm_val, list_pagerank_test=pr_val,
        )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    per_epoch_val.append(time.perf_counter() - t0)

    log_dict = {
        "epoch": e + 1,
        "train_loss": avg_loss,
        "val_kt": val_kt,
        "timing/epoch_train": per_epoch_train[-1],
        "timing/epoch_val": per_epoch_val[-1],
    }

    if args.top_k:
        log_dict.update({f"val_top{int(k*100)}": v for k, v in val_topk.items()})

total_training_time = time.time() - training_start
timings["train_total"] = total_training_time

scores = collections.defaultdict(list)
scores_filtered = collections.defaultdict(list)
times = collections.defaultdict(list)
flops = collections.defaultdict(list)
topk_scores = collections.defaultdict(dict)
topk_scores_filtered = collections.defaultdict(dict)

print("Testing on real datasets")
final_test_metrics = {}
per_test_eval_total = {}

for name in TEST_GRAPHS:
    if name == val_key:
        adj, adj_t, num, bc = list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val
        test_lm, test_pr = lm_val, pr_val
    else:
        loaded = load_test_adj_or_build(name, preprocessing=not args.no_preprocessing)
        if loaded is None:
            print(f"Dataset {name} not found, skipping")
            continue
        adj, adj_t, num, bc = loaded
        if _has_pr or _sp_count:
            test_lm, test_pr = _compute_addons(adj, num, _has_pr, _sp_count, _sp_mode)
        else:
            test_lm, test_pr = None, None
    print(f"Testing: {name}")

    t0 = time.perf_counter()
    with torch.no_grad():
        mean, std, topk, avg_time, mean_f, topk_f = evaluate(
            model, adj, adj_t, num, bc, model_size, device, args.top_k,
            list_landmarks_test=test_lm, list_pagerank_test=test_pr,
            compute_filtered=True,
        )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    per_test_eval_total[name] = time.perf_counter() - t0

    scores[name].append(mean)
    scores_filtered[name].append(mean_f)
    times[name].append(avg_time)
    _adj_fwd = sparse_mx_to_torch_sparse_tensor(adj[0]).to(device)
    _adj_rev = sparse_mx_to_torch_sparse_tensor(adj_t[0]).to(device)
    _lm = torch.from_numpy(test_lm[0]).float().to(device) if test_lm else None
    _pr = torch.from_numpy(test_pr[0]).float().to(device) if test_pr else None
    flops[name].append(count_inference_gflops(model, _adj_fwd, _adj_rev, _lm, _pr))
    if args.top_k:
        topk_scores[name] = topk
        topk_scores_filtered[name] = topk_f

    if args.dump_predictions:
        _dump_predictions(
            model, adj[0], adj_t[0], num[0], bc[0],
            test_lm[0] if test_lm else None,
            test_pr[0] if test_pr else None,
            device,
            os.path.join(args.results_dir, "betweenness", "predictions",
                         f"{run_name}__{name}.npz"),
        )

    final_test_metrics[f"test/{name}_KT"] = mean
    final_test_metrics[f"test/{name}_KT_filtered"] = mean_f
    final_test_metrics[f"test/{name}_Time"] = avg_time
    final_test_metrics[f"timing/test_{name}_eval_total"] = per_test_eval_total[name]
    if args.top_k:
        for p, acc in topk.items():
             final_test_metrics[f"test/{name}_Top{int(p*100)}"] = acc
        for p, acc in topk_f.items():
             final_test_metrics[f"test/{name}_Top{int(p*100)}_filtered"] = acc

print("\n=== Timing Summary ===")
print(f"  load_train        : {timings['load_train']:.2f}s")
print(f"  build_adj_cache   : {timings['build_adj_cache']:.2f}s ({'hit' if timings['adj_cache_hit'] else 'miss'})")
print(f"  model_init        : {timings['model_init']:.2f}s")
print(f"  training_total    : {timings['train_total']:.2f}s ({args.epochs} epochs)")
if per_epoch_train:
    print(f"    epoch train avg : {sum(per_epoch_train)/len(per_epoch_train):.2f}s "
          f"(min {min(per_epoch_train):.2f}s, max {max(per_epoch_train):.2f}s)")
    print(f"    epoch val avg   : {sum(per_epoch_val)/len(per_epoch_val):.2f}s")
print("  per-test eval total:")
for name in TEST_GRAPHS:
    if name in per_test_eval_total:
        print(f"    {name:<28s}  {per_test_eval_total[name]:6.2f}s")

save_results(run_name, TEST_GRAPHS, scores, times, topk_scores, total_training_time,
             args.top_k, os.path.join(args.results_dir, "betweenness"),
             scores_filtered=scores_filtered,
             topk_scores_filtered=topk_scores_filtered if args.top_k else None,
             flops=flops)

print("Done.")