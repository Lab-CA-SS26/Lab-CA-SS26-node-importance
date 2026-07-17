"""Generate the test-set LaTeX table, split by directed/undirected regime.

Within each regime, rows are sorted by #Nodes ascending. Edge counts come
straight from networkx — `nx.Graph.number_of_edges()` for undirected, directed
count for nx.DiGraph (matches SNAP's reported edge counts for both genuinely-
directed graphs and the symmetrized p2p-Gnutella31).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calc_params
from graph_regimes import UNDIRECTED, UNDIRECTED_STORED_AS_DIGRAPH, DIRECTED_SKIP


def collect(names):
    rows = []
    for name in names:
        graphs = calc_params.load_graph(name)
        if not graphs:
            print(f"# skipping {name}: no pickle found", file=sys.stderr)
            continue
        G = graphs[0]
        m = G.number_of_edges()
        # p2p-Gnutella31 is stored as a pre-symmetrized DiGraph; halve to get
        # the undirected edge count matching SNAP's reported figure.
        if name in UNDIRECTED_STORED_AS_DIGRAPH:
            m //= 2
        rows.append((name, G.number_of_nodes(), m))
    rows.sort(key=lambda r: r[1])
    return rows


def emit_section(rows):
    for name, n, m in rows:
        display = name.replace("_", r"\_")
        print(f"    \\hspace{{0.5em}}{display} & {n:,} & {m:,} \\\\")


def generate_table_latex():
    undirected = collect(sorted(UNDIRECTED | UNDIRECTED_STORED_AS_DIGRAPH))
    directed = collect(sorted(DIRECTED_SKIP))

    print(r"\begin{table*}[t]")
    print(r"  \centering")
    print(r"  \renewcommand{\arraystretch}{1.2}")
    print(r"  \setlength{\tabcolsep}{2pt}")
    print(r"  \caption{Test set.}")
    print(r"  \label{tab:datasets}")
    print(r"  \begin{tabular}{@{}lrr@{}}")
    print(r"    \toprule")
    print(r"    \textbf{Graph} & \textbf{\#Nodes} & \textbf{\#Edges} \\")
    print(r"    \midrule")
    print(r"    \multicolumn{3}{@{}l}{\textit{Undirected}} \\")
    emit_section(undirected)
    print(r"    \midrule")
    print(r"    \multicolumn{3}{@{}l}{\textit{Directed}} \\")
    emit_section(directed)
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    generate_table_latex()
