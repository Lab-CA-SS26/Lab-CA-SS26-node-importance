from dataclasses import dataclass, asdict
import argparse
import json
import random
import numpy as np
import torch

from graph_regimes import UNDIRECTED, UNDIRECTED_STORED_AS_DIGRAPH, DIRECTED_SKIP

TEST_GRAPHS_ALL = (
    sorted(UNDIRECTED | UNDIRECTED_STORED_AS_DIGRAPH)
    + sorted(DIRECTED_SKIP)
)

TEST_GRAPHS_DEFAULT = [
    "web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31",
    "email-EuAll", "wiki-Talk"
]

TEST_GRAPHS_SMOKE = [
    "p2p-Gnutella31",                                       # undirected_stored_as_digraph
    "soc-Epinions1", "soc-Slashdot0902", "email-EuAll",     # directed
    "com-youtube", "amazon",                                # undirected
]

SOCIAL_NETWORKS = TEST_GRAPHS_ALL

DATASETS = SOCIAL_NETWORKS


def is_synthetic(g):
    return g in ["SF", "ER", "GRP"] or g.startswith("HY") or g.startswith("SF_")


def train_path(g):
    if is_synthetic(g):
        return f"./datasets/data_splits/{g}/betweenness/training.pickle"
    return f"./datasets/data_splits/{g}_bet.pickle"


def test_path(g):
    if is_synthetic(g):
        return f"./datasets/data_splits/{g}/betweenness/test.pickle"
    return f"./datasets/data_splits/{g}_bet.pickle"


def resolve_train_graphs(train_type):
    if "+" in train_type:
        return train_type.split("+")
    if train_type.startswith("HY"):
        return [train_type]
    if train_type.startswith("SF_"):
        return [train_type]
    return [train_type]


def resolve_test_graphs(run_all_tests):
    return TEST_GRAPHS_ALL if run_all_tests else TEST_GRAPHS_DEFAULT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_type", default="AW")
    parser.add_argument("--train_type", default="SF")
    parser.add_argument("--leverage", action="store_true")
    parser.add_argument("--run_all_tests", action="store_true")
    parser.add_argument("--nhid", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--top_k", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no_preprocessing", action="store_true")
    parser.add_argument("--dump_predictions", action="store_true",
                        help="Save per-node (true_bc, pred_bc, in_deg, out_deg) per test graph "
                             "to <results_dir>/betweenness/predictions/<run>__<graph>.npz")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--test_graphs", nargs="*", default=None)
    parser.add_argument("--fusion", default="mul", choices=["mul", "add", "cat_mlp"])
    parser.add_argument("--unshared_encoders", action="store_true")
    return parser.parse_args()


def get_algo_name(args_or_none=None, *, mode="baseline", init="AW", train="SF", nhid=12,
                  normalize=False, accumulate=1, layers=4, dropout=0.6,
                  epochs=10, seed=None, fusion="mul", unshared_encoders=False,
                  no_preprocessing=False):
    if args_or_none is not None:
        a = args_or_none
        mode = getattr(a, "mode", mode)
        init = getattr(a, "init_type", init)
        train = getattr(a, "train_type", train)
        nhid = getattr(a, "nhid", nhid)
        normalize = getattr(a, "normalize", normalize)
        accumulate = getattr(a, "accumulate", accumulate)
        layers = getattr(a, "num_layers", layers)
        dropout = getattr(a, "dropout", dropout)
        epochs = getattr(a, "epochs", epochs)
        seed = getattr(a, "seed", seed)
        fusion = getattr(a, "fusion", fusion)
        unshared_encoders = getattr(a, "unshared_encoders", unshared_encoders)
        no_preprocessing = getattr(a, "no_preprocessing", no_preprocessing)

    name = f"{mode}_{init}_{train}_{nhid}"
    if normalize: name += "_norm"
    if accumulate > 1: name += f"_accum{accumulate}"
    if layers != 4: name += f"_L{layers}"
    if abs(dropout - 0.6) > 1e-6: name += f"_drop{dropout}"
    if epochs != 10: name += f"_E{epochs}"
    if seed is not None:
        name += f"_S{seed}"
    if fusion != "mul": name += f"_fusion{fusion}"
    if unshared_encoders: name += "_unshared"
    if no_preprocessing: name += "_noprep"
    return name


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class Config:
    init_type: str = "AW"
    train_type: str = "SF"
    leverage: bool = False
    run_all_tests: bool = False
    nhid: int = 12
    num_layers: int = 4
    top_k: bool = False
    normalize: bool = False
    accumulate: int = 1
    seed: int = 20
    dropout: float = 0.1
    epochs: int = 10
    mode: str = "baseline"
    repeats: int = 1

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        for f in cls.__dataclass_fields__.values():
            if f.name in ("mode", "repeats"):
                continue
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action="store_true", default=f.default)
            else:
                parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
        ns = parser.parse_args()
        return cls(**{k: v for k, v in vars(ns).items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def algo_name(self):
        return get_algo_name(
            mode=self.mode, init=self.init_type, train=self.train_type,
            nhid=self.nhid, normalize=self.normalize, accumulate=self.accumulate,
            layers=self.num_layers, dropout=self.dropout, epochs=self.epochs,
            seed=self.seed,
        )

    @property
    def train_graphs(self):
        return resolve_train_graphs(self.train_type)

    @property
    def test_graphs(self):
        return resolve_test_graphs(self.run_all_tests)