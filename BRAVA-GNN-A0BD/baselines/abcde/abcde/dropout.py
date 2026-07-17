import torch
from torch_geometric.utils import degree, coalesce
from torch_geometric.utils.num_nodes import maybe_num_nodes


def _filter_adj(row, col, edge_attr, mask):
    row, col = row[mask], col[mask]
    return row, col, edge_attr[mask] if edge_attr is not None else None


def dropout_adj(edge_index, edge_attr=None, p=0.5, force_undirected=False,
                num_nodes=None, training=True):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training or p == 0:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = _filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)
    row_deg, col_deg = degree(row), degree(col)

    mask |= row_deg[row] < 5
    mask |= col_deg[col] < 5

    row, col, edge_attr = _filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr
