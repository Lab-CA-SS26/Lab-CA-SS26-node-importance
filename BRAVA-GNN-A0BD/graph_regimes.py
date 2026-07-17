"""Per-graph regime classification and load-time normalization.

See docs/graph_regimes.md for the per-graph table and rationale.
"""
import networkx as nx

DIRECTED_SKIP = {
    "wiki-Talk", "wiki-topcats", "email-EuAll", "web-Google",
    "soc-Epinions1", "soc-Pokec", "soc-LiveJournal1", "soc-Slashdot0902",
}

UNDIRECTED_STORED_AS_DIGRAPH = {
    "p2p-Gnutella31",
}

UNDIRECTED = {"cit-Patents", "amazon", "com-youtube", "dblp", "com-lj"}


def assert_regime(name, G):
    """Fail fast if the pickle's class/symmetry disagrees with docs/graph_regimes.md."""
    if name in UNDIRECTED_STORED_AS_DIGRAPH:
        assert isinstance(G, nx.DiGraph), (
            f"{name}: declared undirected_stored_as_digraph but pickle is {type(G).__name__}")
        missing = next(((u, v) for u, v in G.edges() if not G.has_edge(v, u)), None)
        assert missing is None, (
            f"{name}: DiGraph not symmetric (missing reverse for {missing}). "
            f"Regenerate the pickle.")
    elif name in UNDIRECTED:
        assert isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph), (
            f"{name}: declared undirected but pickle is {type(G).__name__}")
    elif name in DIRECTED_SKIP:
        assert isinstance(G, nx.DiGraph), (
            f"{name}: declared directed but pickle is {type(G).__name__}")
    else:
        raise AssertionError(f"{name}: not in any regime bucket — update docs/graph_regimes.md")


def normalize_for_model(G):
    """Materialize nx.Graph as symmetric DiGraph so the model sees A = A^T.

    DiGraphs pass through unchanged (already either genuinely directed or
    pre-symmetrized in datasets/generate_graph.py).
    """
    if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        return G.to_directed()
    return G
