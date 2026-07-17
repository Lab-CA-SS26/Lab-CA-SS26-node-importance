import torch
import torch.nn as nn
from layer import GNN_Layer, GNN_Layer_Init


def count_inference_gflops(model, adj, adj_t, lm=None, pr=None):
    """Count inference GFLOPs for one forward pass via forward hooks (MACs × 2).

    Dual-tower model (adj, adj_t) fires each shared layer twice, which is correct.
    Returns GFLOPs (float).
    """
    counter = [0]
    hooks = []

    def _gnn_init_hook(module, inputs, output):
        adj_in = inputs[0]
        N = adj_in.shape[0]
        nnz = adj_in._nnz()
        out_f = module.out_features
        bit = module.base_init_type
        if bit.startswith("random_"):
            # randn cost is negligible; count only mm(features, W)
            counter[0] += 2 * N * module.input_dim * out_f
        elif bit.startswith("degree") or bit.startswith("ev_mix_") or "positional_encoding" in bit:
            if (bit.startswith("degree_mix_mass_") or bit.startswith("degree_mix_independent_")
                    or bit.startswith("ev_mix_mass_") or bit.startswith("ev_mix_independent_")):
                k = int(bit.split("_")[-1])
            elif bit in ("degree_mix_sum", "degree_mix_mass_3", "degree3", "degree_mix_d0d2"):
                k = 3
            elif bit in ("degree_mix_sum_2",):
                k = 2
            else:
                k = 1
            counter[0] += 2 * k * nnz                       # k spmm(adj, 1-col) hops
            counter[0] += 2 * N * module.input_dim * out_f  # mm(features, W)
        else:
            counter[0] += 2 * nnz * out_f  # AW: spmm(adj, W)
        if module.bias is not None:
            counter[0] += N * out_f

    def _gnn_layer_hook(module, inputs, output):
        x, adj_in = inputs
        N, in_f = x.shape
        out_f = module.out_features
        counter[0] += 2 * N * in_f * out_f       # mm(x, W)
        counter[0] += 2 * adj_in._nnz() * out_f  # spmm(adj, XW)
        if module.bias is not None:
            counter[0] += N * out_f

    def _linear_hook(module, inputs, output):
        x = inputs[0]
        batch = x.numel() // x.shape[-1]
        counter[0] += 2 * batch * module.in_features * module.out_features
        if module.bias is not None:
            counter[0] += batch * module.out_features

    # Track custom-layer ids to avoid double-counting any child nn.Linear modules.
    custom_ids = set()
    for m in model.modules():
        if isinstance(m, GNN_Layer_Init):
            hooks.append(m.register_forward_hook(_gnn_init_hook))
            custom_ids.add(id(m))
        elif isinstance(m, GNN_Layer):
            hooks.append(m.register_forward_hook(_gnn_layer_hook))
            custom_ids.add(id(m))

    for m in model.modules():
        if isinstance(m, nn.Linear) and id(m) not in custom_ids:
            hooks.append(m.register_forward_hook(_linear_hook))

    try:
        with torch.no_grad():
            model(adj, adj_t, landmarks=lm, pagerank=pr)
    finally:
        for h in hooks:
            h.remove()

    return counter[0] / 1e9
