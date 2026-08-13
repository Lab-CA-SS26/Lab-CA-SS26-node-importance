#!/usr/bin/env python3
"""Render the report's figures from raw benchmark JSON.

Usage:
    make_plots.py threads <ts-result-dir>  <outdir>   # Figure: thread scaling
    make_plots.py bvk     <bvk-result-dir> <outdir>   # Figure: BRAVA-GNN vs KADABRA

Writes PDF (vector, for LaTeX) and PNG (for quick inspection).
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# Categorical slots 1-3 of the validated palette (CVD-checked; see dataviz skill).
# Each series also carries a distinct marker, so identity is never colour-alone.
C_JULIA, C_CPP, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

THREADS = [1, 2, 4, 8, 16, 24, 32, 48]
SEEDS = [0, 1, 2]

# (|V|, |E|) from Report Table 1, used only to order graphs by scale on the x-axis.
SIZES = {
    "p2p-Gnutella31": (62586, 147892), "soc-Epinions1": (75879, 508837),
    "soc-Slashdot0902": (82168, 948464), "email-EuAll": (265214, 420045),
    "com-youtube": (1134890, 2987624), "amazon": (2146057, 5743146),
    "dblp": (4000148, 8649011), "cit-Patents": (3764117, 16511741),
    "com-lj": (3997962, 34681189),
}
BVK_GRAPHS = list(SIZES)


def style(ax):
    """Recessive grid and axes; the data should be the darkest thing on the page."""
    ax.grid(True, which="major", color="#e6e6e3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c9c4")
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def plot_threads(d, outdir):
    graphs = ["soc-Slashdot0902", "amazon", "cit-Patents", "com-lj"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.7))
    any_data = False

    for ax, g in zip(axes.flat, graphs):
        for impl, colour, marker, label in (
            ("julia", C_JULIA, "o", "Julia"),
            ("cpp", C_CPP, "s", "C++"),
        ):
            xs, med, lo, hi = [], [], [], []
            for t in THREADS:
                runs = [load(f"{d}/{impl}_{g}_t{t}_s{s}.json") for s in SEEDS]
                vals = [r["execution_time_seconds"] for r in runs if r]
                if not vals:
                    continue
                xs.append(t)
                med.append(statistics.median(vals))
                lo.append(statistics.median(vals) - min(vals))
                hi.append(max(vals) - statistics.median(vals))
            if not xs:
                continue
            any_data = True
            ax.errorbar(xs, med, yerr=[lo, hi], color=colour, marker=marker,
                        markersize=4.5, linewidth=1.6, capsize=2.5,
                        elinewidth=0.9, label=label, zorder=3)
            best = xs[med.index(min(med))]
            ax.axvline(best, color=colour, linewidth=0.8, linestyle=":", alpha=0.55, zorder=1)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(THREADS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_title(g, fontsize=9, color=INK)
        style(ax)

    if not any_data:
        print("no thread-scaling data found", file=sys.stderr)
        return
    for ax in axes[-1]:
        ax.set_xlabel("threads", fontsize=8.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("runtime (s)", fontsize=8.5, color=MUTED)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, outdir, "thread_scaling")


def plot_bvk(d, outdir):
    rows = []
    for g in BVK_GRAPHS:
        k = load(f"{d}/bvk_kadabra_{g}.json") or load(f"{d}/kadabra_{g}.stats.json")
        bc = load(f"{d}/bvk_brava_cpu_{g}.json") or load(f"{d}/brava_cpu_{g}.stats.json")
        bg = load(f"{d}/bvk_brava_gpu_{g}.json") or load(f"{d}/brava_gpu_{g}.stats.json")
        if not (k and bc and bg):
            continue
        rows.append((g, sum(SIZES[g]), bc["execution_time_seconds"],
                     bg["execution_time_seconds"], k["execution_time_seconds"],
                     bc.get("tau_overall"), k.get("tau_overall"),
                     bc.get("overlap_topk"), k.get("overlap_topk")))
    if not rows:
        print("no BRAVA/KADABRA data found", file=sys.stderr)
        return
    rows.sort(key=lambda r: r[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # (a) runtime vs graph scale -- shows where the crossover happens
    size = [r[1] for r in rows]
    for idx, colour, marker, label in ((2, C_JULIA, "o", "BRAVA-GNN (CPU)"),
                                       (3, C_THIRD, "^", "BRAVA-GNN (GPU)"),
                                       (4, C_CPP, "s", "KADABRA")):
        ax1.plot(size, [r[idx] for r in rows], color=colour, marker=marker,
                 markersize=5, linewidth=0, label=label, zorder=3)
    eu = next((r for r in rows if r[0] == "email-EuAll"), None)
    if eu:
        ax1.annotate("email-EuAll", xy=(eu[1], eu[4]), xytext=(5, -12),
                     textcoords="offset points", fontsize=7, color=MUTED)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("graph size $|V|+|E|$", fontsize=8.5, color=MUTED)
    ax1.set_ylabel("runtime (s)", fontsize=8.5, color=MUTED)
    ax1.set_title("(a) runtime vs. graph scale", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax1)

    # (b) the accuracy trade-off: global ranking vs the top of the ranking
    ax2.scatter([r[5] for r in rows], [r[7] for r in rows], color=C_JULIA,
                marker="o", s=34, label="BRAVA-GNN", zorder=3)
    ax2.scatter([r[6] for r in rows], [r[8] for r in rows], color=C_CPP,
                marker="s", s=34, label="KADABRA", zorder=3)
    for r in rows:  # pair each graph's two points so the shift is legible
        ax2.plot([r[5], r[6]], [r[7], r[8]], color=MUTED, linewidth=0.6,
                 alpha=0.35, zorder=2)
    ax2.set_xlabel(r"Kendall $\tau_b$ (all nodes)", fontsize=8.5, color=MUTED)
    ax2.set_ylabel("top-100 overlap", fontsize=8.5, color=MUTED)
    ax2.set_title("(b) accuracy trade-off", fontsize=9, color=INK)
    ax2.legend(frameon=False, fontsize=7.5, loc="best")
    style(ax2)

    fig.tight_layout()
    save(fig, outdir, "brava_vs_kadabra")


def save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("threads", "bvk"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    (plot_threads if sys.argv[1] == "threads" else plot_bvk)(
        sys.argv[2].rstrip("/"), sys.argv[3])
