import torch
import numpy as np
import collections
import time
from utils import sparse_mx_to_torch_sparse_tensor, loss_cal, ranking_correlation, ranking_correlation_topk

def train(model, optimizer, list_adj_train, list_adj_t_train, list_num_node_train,
          bc_mat_train, model_size, device, accum_steps=1,
          list_landmarks_train=None, list_pagerank_train=None):
    model.train()

    gpu_graphs = []
    for i in range(len(list_adj_train)):
        adj = sparse_mx_to_torch_sparse_tensor(list_adj_train[i]).to(device).coalesce()
        adj_t = sparse_mx_to_torch_sparse_tensor(list_adj_t_train[i]).to(device).coalesce()
        true_val = torch.from_numpy(bc_mat_train[i]).float().to(device)
        lm = (torch.from_numpy(list_landmarks_train[i]).float().to(device)
              if list_landmarks_train else None)
        pr = (torch.from_numpy(list_pagerank_train[i]).float().to(device)
              if list_pagerank_train else None)
        gpu_graphs.append((adj, adj_t, true_val, list_num_node_train[i], lm, pr))

    num_virtual_copies = 50
    optimizer.zero_grad()

    total_epoch_loss = 0.0
    total_steps = 0

    for _ in range(num_virtual_copies):
        indices = torch.randperm(len(gpu_graphs))
        for i, idx in enumerate(indices):
            adj, adj_t, true_val, node_num, lm, pr = gpu_graphs[idx]
            perm = torch.randperm(node_num, device=device)
            batch_true_val = true_val[perm]
            inv_perm = torch.argsort(perm)

            def permute_adj(src_adj, map_idx, n):
                new_idx = map_idx[src_adj.indices()]
                return torch.sparse_coo_tensor(new_idx, src_adj.values(), (n, n), device=device).coalesce()

            batch_adj = permute_adj(adj, inv_perm, node_num)
            batch_adj_t = permute_adj(adj_t, inv_perm, node_num)
            batch_lm = lm[perm] if lm is not None else None
            batch_pr = pr[perm] if pr is not None else None
            y_out = model(batch_adj, batch_adj_t, landmarks=batch_lm, pagerank=batch_pr)

            raw_loss = loss_cal(y_out, batch_true_val, node_num, device)

            total_epoch_loss += raw_loss.item()
            total_steps += 1

            loss = raw_loss
            if accum_steps > 1:
                loss = loss / accum_steps
            
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(indices):
                optimizer.step()
                optimizer.zero_grad()
    
    return total_epoch_loss / max(1, total_steps)


def evaluate(model, list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test,
             model_size, device, top_k=False,
             list_landmarks_test=None, list_pagerank_test=None,
             compute_filtered=False):
    """If compute_filtered=True, appends (mean_kt_filtered, topk_means_filtered) to the return tuple."""
    model.eval()
    list_kt = []
    list_kt_filtered = []
    topk_lists = collections.defaultdict(list)
    topk_lists_filtered = collections.defaultdict(list)
    total_time = 0
    n_samples = len(list_adj_test)

    for j in range(n_samples):
        adj = sparse_mx_to_torch_sparse_tensor(list_adj_test[j]).to(device)
        adj_t = sparse_mx_to_torch_sparse_tensor(list_adj_t_test[j]).to(device)
        num_nodes = list_num_node_test[j]
        lm = (torch.from_numpy(list_landmarks_test[j]).float().to(device)
              if list_landmarks_test else None)
        pr = (torch.from_numpy(list_pagerank_test[j]).float().to(device)
              if list_pagerank_test else None)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        y_out = model(adj, adj_t, landmarks=lm, pagerank=pr)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_time += time.time() - start

        true_val = torch.from_numpy(bc_mat_test[j]).float().to(device)

        if top_k:
            out = ranking_correlation_topk(y_out, true_val, num_nodes, compute_filtered=compute_filtered)
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
            out = ranking_correlation(y_out, true_val, num_nodes, compute_filtered=compute_filtered)
            if not compute_filtered:
                list_kt.append(out)
            else:
                kt, kt_f = out
                list_kt.append(kt)
                list_kt_filtered.append(kt_f)

    mean_kt = np.mean(list_kt)
    std_kt = np.std(list_kt)
    avg_time = total_time / n_samples if n_samples else 0
    topk_means = {p: np.mean(v) for p, v in topk_lists.items()} if top_k else {}

    print(f"   Average KT: {mean_kt:.4f} | Std: {std_kt:.4f}")
    if top_k:
        print(f"   Top 1%: {topk_means.get(0.01, 0):.4f} | Top 5%: {topk_means.get(0.05, 0):.4f} | Top 10%: {topk_means.get(0.1, 0):.4f}")

    if not compute_filtered:
        return mean_kt, std_kt, topk_means, avg_time

    mean_kt_filtered = np.nanmean(list_kt_filtered) if list_kt_filtered else float("nan")
    topk_means_filtered = ({p: np.nanmean(v) for p, v in topk_lists_filtered.items()}
                           if top_k else {})
    print(f"   Average KT (filtered, bc>0): {mean_kt_filtered:.4f}")
    if top_k:
        print(f"   Top 1% (filtered): {topk_means_filtered.get(0.01, 0):.4f} | "
              f"Top 5% (filtered): {topk_means_filtered.get(0.05, 0):.4f} | "
              f"Top 10% (filtered): {topk_means_filtered.get(0.1, 0):.4f}")
    return mean_kt, std_kt, topk_means, avg_time, mean_kt_filtered, topk_means_filtered