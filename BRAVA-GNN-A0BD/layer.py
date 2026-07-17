import math
import re
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from torch_sparse import spspmm
# from torch_scatter import scatter_max
import sys
import os

# from positional_encodings.torch_encodings import PositionalEncoding1D


def parse_init_type(init_type):
    """Split 'degree_mix_mass_6_pr_sp256_ff' into (base, has_pr, sp_count, sp_mode).

    Recognized trailing addons (in order, both optional):
      _pr            — adds 1 PageRank channel
      _sp{N}{_mode}  — adds N landmark-distance channels (or 3 for stats)
                       mode in {ff, deg, random, stats}; default 'random' if mode omitted.
    """
    rest = init_type
    sp_count, sp_mode = None, None
    sp_match = re.search(r"_sp(\d+)(?:_(ff|deg|random|stats))?$", rest)
    if sp_match:
        sp_count = int(sp_match.group(1))
        sp_mode = sp_match.group(2) or "random"
        rest = rest[:sp_match.start()]
    has_pr = False
    if rest.endswith("_pr"):
        has_pr = True
        rest = rest[:-3]
    return rest, has_pr, sp_count, sp_mode

class GNN_Layer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MasterUpdate(Module):
    def __init__(self, nhid):
        super(MasterUpdate, self).__init__()
        self.linear_update = torch.nn.Linear(2*nhid, nhid)
        self.linear_broadcast = torch.nn.Linear(nhid, nhid)
    
    def forward(self, master_state, node_embeddings):
        node_avg = torch.mean(node_embeddings, dim=0, keepdim=True)
        inp = torch.cat([master_state, node_avg], dim=1)
        new_master = torch.tanh(self.linear_update(inp))

        broadcast = self.linear_broadcast(new_master)

        return new_master, broadcast



class GNN_Layer_Init(Module):
    """First GNN layer: computes node features from the adjacency (degree hops, AW, or random)."""

    def __init__(self, in_features, out_features, bias=True, init_type="AW", leverage=False, normalize=False):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init_type  # full string including _pr / _sp{N}{_mode} suffixes
        self.base_init_type, self._has_pr, self._sp_count, self._sp_mode = parse_init_type(init_type)
        self.leverage = leverage
        self.normalize = normalize

        self.input_dim = 1
        if self.base_init_type == "degree_embedding":
            self.input_dim = out_features
        elif self.base_init_type == "positional_encoding":
            # 1D PE (uses degree1)
            self.input_dim = out_features
        elif self.base_init_type == "positional_encoding_3d":
            # 3D PE (uses d1, d2, d3 concatenated)
            self.input_dim = out_features * 3
        elif self.base_init_type in ["degree_mix_sum", "degree_mix_max"]:
            self.input_dim = 3
        elif self.base_init_type in ["degree_mix_sum_2", "degree_mix_d0d2"]:
            self.input_dim = 2
        elif self.base_init_type == "degree_mix_max_sum":
            self.input_dim = 6

        if (self.base_init_type.startswith("degree_mix_mass_")
                or self.base_init_type.startswith("degree_mix_independent_")
                or self.base_init_type.startswith("ev_mix_mass_")
                or self.base_init_type.startswith("ev_mix_independent_")
                or self.base_init_type.startswith("random_")):
            try:
                self.input_dim = int(self.base_init_type.split("_")[-1])
            except ValueError:
                self.input_dim = 1

        if self.leverage: self.input_dim += 1

        if self._has_pr:
            self.input_dim += 1
        if self._sp_count:
            self.input_dim += 3 if self._sp_mode == "stats" else self._sp_count

        if self.base_init_type == "degree_embedding":
            self.emb = torch.nn.Embedding(in_features, out_features)
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))

        elif self.base_init_type == "positional_encoding":
            from positional_encodings.torch_encodings import PositionalEncoding1D
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
            self.p_enc = PositionalEncoding1D(out_features)

        elif self.base_init_type == "positional_encoding_3d":
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))

        elif (self.base_init_type.startswith("degree")
              or self.base_init_type.startswith("ev_mix_")
              or self.base_init_type.startswith("random_")):
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
        else:
            # AW path: weight is (in_features, out_features)
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_pe_for_mass(self, mass_vector):
        """Sinusoidal PE computed analytically (not via lookup table) to avoid OOM when
        d3 masses exceed 1e8. Formula: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)."""
        if mass_vector.dim() == 1:
            mass_vector = mass_vector.unsqueeze(1)
            
        positions = mass_vector.float()
        D = self.out_features
        device = positions.device
        i = torch.arange(0, D, 2, device=device).float()
        div_term = torch.exp(i * (-math.log(10000.0) / D))
        phase = positions * div_term.unsqueeze(0)  # (N,1) * (1,D/2) = (N,D/2)
        
        pe = torch.zeros(positions.shape[0], D, device=device)
        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        
        return pe

    def forward(self, adj, landmarks=None, pagerank=None):
        bit = self.base_init_type
        if bit.startswith("random_"):
            k = int(bit.split("_")[-1])
            n = adj.shape[0]
            seed_val = (n * 2654435761 + adj._nnz() * 40503) & 0x7FFFFFFF
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed_val)
            x = torch.randn(n, k, generator=gen).to(adj.device)
            output = torch.mm(x, self.weight)
        elif bit.startswith("degree") or bit.startswith("ev_mix_") or "positional_encoding" in bit:

            if self.leverage or bit in ["degree_mix_max", "degree_mix_max_sum"]:
                adj = adj.coalesce()

            ones = torch.ones((adj.shape[0], 1), device=adj.device)
            d1 = torch.spmm(adj, ones)
            x = d1

            if bit == "degree0":
                x = ones
            elif bit == "degree_embedding":
                x = self.emb(d1.long().view(-1).clamp(0, self.in_features - 1))

            # --- 1D PE ---
            elif bit == "positional_encoding":
                x = self.get_pe_for_mass(d1)

            # --- 3D PE (Using d1, d2, d3) ---
            elif bit == "positional_encoding_3d":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                pe_d1 = self.get_pe_for_mass(d1)
                pe_d2 = self.get_pe_for_mass(d2)
                pe_d3 = self.get_pe_for_mass(d3)
                x = torch.cat([pe_d1, pe_d2, pe_d3], dim=1)

            elif bit == "degree1" or bit == "degree_pure":
                x = d1
            elif bit == "degree2" or bit == "degree":
                d2 = torch.spmm(adj, d1)
                x = d2
            elif bit == "degree3":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = d3
            elif bit == "degree_mix_sum":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d2, d3], dim=1)
            elif bit == "degree_mix_sum_2":
                d2 = torch.spmm(adj, d1)
                x = torch.cat([d1, d2], dim=1)
            elif bit == "degree_mix_mass_3":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d1+d2, d1+d2+d3], dim=1)
            elif bit == "degree_mix_max":
                from torch_scatter import scatter_max
                row, col = adj.indices()
                val = adj.values()
                d2 = F.relu(scatter_max(d1[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                d3 = F.relu(scatter_max(d2[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                x = torch.cat([d1, d2, d3], dim=1)
            elif bit == "degree_mix_max_sum":
                from torch_scatter import scatter_max
                d2_sum = torch.spmm(adj, d1)
                d3_sum = torch.spmm(adj, d2_sum)
                row, col = adj.indices()
                val = adj.values()
                d2_max = F.relu(scatter_max(d1[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                d3_max = F.relu(scatter_max(d2_max[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                x = torch.cat([d1, d2_sum, d3_sum, d1, d2_max, d3_max], dim=1)

            elif bit == "degree_mix_d0d2":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d1+d2+d3], dim=1)

            elif bit.startswith("degree_mix_mass_") or bit.startswith("degree_mix_independent_"):
                k = int(bit.split("_")[-1])
                current_d = d1
                deg_list = [d1]
                for _ in range(k - 1):
                    current_d = torch.spmm(adj, current_d)
                    deg_list.append(current_d)
                if bit.startswith("degree_mix_mass_"):
                    mass_list = []
                    current_mass = torch.zeros_like(d1)
                    for d in deg_list:
                        current_mass = current_mass + d
                        mass_list.append(current_mass)
                    x = torch.cat(mass_list, dim=1)
                else:
                    x = torch.cat(deg_list, dim=1)

            elif bit.startswith("ev_mix_mass_") or bit.startswith("ev_mix_independent_"):
                # Same shape as degree_mix_mass_k but seeds from degree-corrected
                # eigenvector centrality (5 power iterations) instead of all-ones.
                k = int(bit.split("_")[-1])
                ev = ones / adj.shape[0]
                for _ in range(5):
                    ev = torch.spmm(adj, ev)
                    ev = ev / (ev.norm() + 1e-8)
                ev_seed = ev * adj.shape[0] / (d1 + 1.0)
                current_e = torch.spmm(adj, ev_seed)
                ev_list = [current_e]
                for _ in range(k - 1):
                    current_e = torch.spmm(adj, current_e)
                    ev_list.append(current_e)
                if bit.startswith("ev_mix_mass_"):
                    mass_list = []
                    current_mass = torch.zeros_like(ev_seed)
                    for e in ev_list:
                        current_mass = current_mass + e
                        mass_list.append(current_mass)
                    x = torch.cat(mass_list, dim=1)
                else:
                    x = torch.cat(ev_list, dim=1)

            if "positional_encoding" not in bit:
                if bit != "degree0" and bit != "degree_embedding" and os.environ.get("BRAVA_NO_LOG1P") != "1":
                    x = torch.log(x + 1)
                    if self.normalize:
                        x = x / (x.max(dim=0)[0] + 1e-6)

            features = [x]
            if self.leverage:
                deg = d1.squeeze()
                row, col = adj.indices()
                vals = adj.values()
                deg_sum = deg[row] + deg[col]
                deg_sum = torch.where(deg_sum > 0, deg_sum, torch.ones_like(deg_sum))
                lev_vals = vals * (1.0 / deg_sum)
                lev_adj = torch.sparse_coo_tensor(adj.indices(), lev_vals, adj.size())
                sum_lev = torch.spmm(lev_adj, ones)
                lev_feat = 2 * sum_lev - 1
                features.append(lev_feat)

            # Per-graph max-normalize addon channels: training graphs (SF/HY diam ~5)
            # vs. test graphs (road diam ~40) have very different value ranges, so
            # without normalization the projection overfits to the training scale.
            if self._has_pr:
                if pagerank is None:
                    raise ValueError(f"init_type {self.init_type!r} requires pagerank but None passed")
                pr_norm = pagerank / (pagerank.max(dim=0, keepdim=True)[0] + 1e-6)
                features.append(pr_norm)
            if self._sp_count:
                if landmarks is None:
                    raise ValueError(f"init_type {self.init_type!r} requires landmarks but None passed")
                lm_norm = landmarks / (landmarks.max(dim=0, keepdim=True)[0] + 1e-6)
                if self._sp_mode == "stats":
                    lm_min = lm_norm.min(dim=1, keepdim=True)[0]
                    lm_max = lm_norm.max(dim=1, keepdim=True)[0]
                    lm_mean = lm_norm.mean(dim=1, keepdim=True)
                    features.append(torch.cat([lm_min, lm_max, lm_mean], dim=1))
                else:
                    features.append(lm_norm)

            x_in = torch.cat(features, dim=1)
            output = torch.mm(x_in, self.weight)
        else:
            # [H=A*W]
            support = self.weight
            if adj.shape[0] < support.shape[0]:
                support = support[:adj.shape[0], :]
            output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, nhid,dropout):
        super(MLP,self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid,2*nhid)
        self.linear2 = torch.nn.Linear(2*nhid,2*nhid)
        self.linear3 = torch.nn.Linear(2*nhid,1)


    def forward(self,input_vec,dropout):

        score_temp = F.relu(self.linear1(input_vec))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = F.relu(self.linear2(score_temp))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = self.linear3(score_temp)

        return score_temp