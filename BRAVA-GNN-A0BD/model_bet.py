import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout, mode="baseline", repeats=1, init_type="AW", leverage=False, normalize=False, num_layers=4, fusion="mul", shared_encoders=True):
        super(GNN_Bet, self).__init__()

        self.mode = mode
        self.repeats = repeats
        self.dropout = dropout
        self.num_layers = num_layers
        self.fusion = fusion
        self.shared_encoders = shared_encoders

        if shared_encoders:
            self.score_layer = MLP(nhid, self.dropout)
        else:
            self.score_layer_in  = MLP(nhid, self.dropout)
            self.score_layer_out = MLP(nhid, self.dropout)

        if fusion == "cat_mlp":
            self.fusion_mlp = nn.Sequential(nn.Linear(2, nhid), nn.ReLU(), nn.Linear(nhid, 1))

        self.gc1 = GNN_Layer_Init(ninput, nhid, init_type=init_type, leverage=leverage, normalize=normalize)

        self.layers = nn.ModuleList([GNN_Layer(nhid, nhid) for _ in range(self.num_layers)])

    def _score(self, x, is_in):
        if self.shared_encoders:
            return self.score_layer(x, self.dropout)
        return (self.score_layer_in if is_in else self.score_layer_out)(x, self.dropout)

    def forward(self, adj1, adj2, landmarks=None, pagerank=None):

        x = F.normalize(F.relu(self.gc1(adj1, landmarks=landmarks, pagerank=pagerank)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2, landmarks=landmarks, pagerank=pagerank)), p=2, dim=1)

        score1 = self._score(x,  is_in=True)
        score2 = self._score(x2, is_in=False)

        for i, layer in enumerate(self.layers):
            x = F.dropout(F.relu(layer(x, adj1)), self.dropout, training=self.training)
            x2 = F.dropout(F.relu(layer(x2, adj2)), self.dropout, training=self.training)

            if i < len(self.layers) - 1:  # normalize all except last layer
                x = F.normalize(x, p=2, dim=1)
                x2 = F.normalize(x2, p=2, dim=1)

            score1 = score1 + self._score(x,  is_in=True)
            score2 = score2 + self._score(x2, is_in=False)

        if self.fusion == "add":
            return score1 + score2
        elif self.fusion == "cat_mlp":
            return self.fusion_mlp(torch.cat([score1, score2], dim=1))
        else:  # mul
            return torch.mul(score1, score2)