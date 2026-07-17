import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import io
import re
from config import (DIRECTED_SKIP,
                    get_algo_name as _base_algo_name)

UNDIRECTED_GRAPHS = [
    "p2p-Gnutella31",
    "com-youtube",
    "amazon",
    "cit-Patents",
    "com-lj",
    "dblp",
]
DIRECTED_GRAPHS = [
    "soc-Epinions1",
    "soc-Slashdot0902",
    "email-EuAll",
    "web-Google",
    "soc-Pokec",
    "wiki-topcats",
    "wiki-Talk",
    "soc-LiveJournal1",
]
DATASETS = UNDIRECTED_GRAPHS + DIRECTED_GRAPHS
SOCIAL_NETWORKS = DATASETS

MANUAL_DEGREE_CODE = "degree_mix_mass_6"
MANUAL_TRAIN_TYPE = "SF_10_Dir+SF_10_Sym+HY_10_Dir"
MANUAL_LAYERS = 2
MANUAL_NHID = 12
MANUAL_DROPOUT = 0.3       # headline BRAVA model; also the baseline for t1/t2/t3/t4 ablations
MANUAL_EPOCHS = 10

RESULTS_KT       = "results/betweenness/all_results.csv"
RESULTS_KT_TOPK  = "results/betweenness/all_results_topk.csv"
RESULTS_TIME     = "results/betweenness/all_results_wallclock.csv"
RESULTS_TRAINING = "results/betweenness/all_results_training_time.csv"
RESULTS_FLOPS    = "results/betweenness/all_results_flops.csv"

KADABRA_ERR   = 0.01
KADABRA_DELTA = 0.1

SILVAN_ERR   = 0.01
# SILVAN_ERR   = 0.0005
SILVAN_DELTA = 0.05

# Bavarian sampling baseline (eval_bavarian.py defaults) and exact Brandes.
# Both feed the split table; the Brandes column is time-only.
BAVARIAN_ERR    = 0.01
BAVARIAN_DELTA  = 0.1
BAVARIAN_METHOD = "rk"
BRANDES_KEY     = "Brandes"
RESULTS_BRANDES = "results/betweenness/ground_truth_time.csv"


def get_algo_name(**kwargs):
    kwargs.setdefault('train', MANUAL_TRAIN_TYPE)
    return _base_algo_name(**kwargs)


# Headline config. Ablation captions flag only the *other* pinned params that
# differ from this, so a reader knows where the ablation grid sits. The swept
# dimension is never flagged (it varies by design).
HEADLINE_PINS = dict(
    train=MANUAL_TRAIN_TYPE,
    init=MANUAL_DEGREE_CODE,
    layers=MANUAL_LAYERS,
    nhid=MANUAL_NHID,
    dropout=MANUAL_DROPOUT,
)

# Friendly names for the multi-regime training mixes used across the ablations.
_TRAIN_NICKNAMES = {
    "SF_10_Dir+SF_10_Sym+HY_10_Dir+HY_10_Sym": "All-4",
    "SF_10_Dir+SF_10_Sym+HY_10_Dir":           "3-regime",
}

def _tex(s):
    return str(s).replace("_", r"\_")

def _train_descr(t):
    nick = _TRAIN_NICKNAMES.get(t)
    return fr"the \textit{{{nick}}} mix ({_tex(t)})" if nick else f"the {_tex(t)} mix"

def _caption_pin_note(pins, varied=()):
    """Caption fragment flagging pinned params that differ from the headline. Returns '' if all match."""
    parts = []
    for key in ("train", "init", "layers", "nhid", "dropout"):
        if key in varied or key not in pins or pins[key] == HEADLINE_PINS[key]:
            continue
        if key == "train":
            parts.append(f"trained on {_train_descr(pins[key])} "
                         f"rather than {_train_descr(HEADLINE_PINS[key])}")
        else:
            parts.append(f"{key} {_tex(pins[key])} "
                         f"rather than the headline {_tex(HEADLINE_PINS[key])}")
    if not parts:
        return ""
    note = "; ".join(parts)
    return " " + note[0].upper() + note[1:] + "."


ONE_CHOSEN = {
    "1": 7,  # -HY Sym (= MANUAL_TRAIN_TYPE: SF_10_Dir+SF_10_Sym+HY_10_Dir)
    "5": 3,  # dropout = MANUAL_DROPOUT (0.3), the selected cell in the dropout sweep
}

def load_and_aggregate(filepath, topk_pct=None, scale=1.0, update_seeds=True):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None, None

    df = pd.read_csv(filepath)

    if "Time" in df.columns and not any(ds in df.columns for ds in DATASETS):
        for ds in DATASETS:
            df[ds] = df["Time"]

    if "Algorithm" in df.columns:
        if topk_pct is not None:
            suffix = f"_Top{topk_pct}%"
            df = df[df["Algorithm"].str.endswith(suffix, na=False)].copy()
            df["Algorithm"] = df["Algorithm"].str[:-len(suffix)]
        # Pull the seed slot (_S\d+) off before stripping it, so duplicate runs
        # of the same seed can be collapsed to the latest row. Seeds 1-3 are
        # always run; a slot occasionally lands in the CSV twice.
        seed = df["Algorithm"].astype(str).str.extract(r"_S(\d+)(?=_|$)", expand=False)
        df["Algorithm"] = df["Algorithm"].apply(lambda x: re.sub(r"_S\d+(?=_|$)", "", str(x)))
        df["_Seed"] = seed
        has_seed = df["_Seed"].notna()
        seeded = df[has_seed].drop_duplicates(subset=["Algorithm", "_Seed"], keep="last")
        df = pd.concat([df[~has_seed], seeded]).drop(columns="_Seed")

    for c in df.columns:
        if c != "Algorithm":
            df[c] = pd.to_numeric(df[c], errors='coerce')

    grouped = df.groupby("Algorithm")
    mean_df = grouped.mean()
    std_df = grouped.std()
    if update_seeds:
        SEED_COUNTS.update(grouped.size().to_dict())

    if "DrBC" in mean_df.index:
        for g in DIRECTED_SKIP:
            if g in mean_df.columns:
                mean_df.loc["DrBC", g] = np.nan
            if g in std_df.columns:
                std_df.loc["DrBC", g] = np.nan

    if scale != 1.0:
        mean_df *= scale
        std_df  *= scale

    return mean_df, std_df

def load_brandes_time():
    """Exact Brandes betweenness wallclock from ground_truth_time.csv, reshaped
    into a one-row 'Brandes' frame (columns = graph names), ready to concat onto
    the aggregated time frame as one more 'algorithm'."""
    if not os.path.exists(RESULTS_BRANDES):
        print(f"Warning: {RESULTS_BRANDES} not found; Brandes column will be empty.")
        return None
    # Tolerate NUL-byte padding occasionally left by concurrent appends.
    with open(RESULTS_BRANDES, "rb") as fh:
        text = fh.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
    gt = pd.read_csv(io.StringIO(text))
    if not {"graph", "brandes_s"}.issubset(gt.columns):
        return None
    gt["graph"] = gt["graph"].astype(str).str.strip()
    gt = gt[gt["graph"].isin(DATASETS)].drop_duplicates(subset="graph", keep="last")
    if gt.empty:
        return None
    row = pd.to_numeric(gt.set_index("graph")["brandes_s"], errors="coerce")
    return pd.DataFrame({BRANDES_KEY: row}).T

NO_STDEV = False
NO_IMPROVEMENT = False
SHOW_SEEDS = False
RESIZEBOX = False
DECIMALS = 3
SEED_COUNTS: dict = {}

def _begin_tabular(col_spec):
    if RESIZEBOX:
        print(r"  \resizebox{\textwidth}{!}{%")
    print(r"  \begin{tabular}{" + col_spec + "}")

def _end_tabular():
    print(r"  \end{tabular}")
    if RESIZEBOX:
        print(r"  }")

def _label_with_seeds(key, label):
    if not SHOW_SEEDS:
        return label
    n = SEED_COUNTS.get(key)
    return f"{label} ({n})" if n is not None else label

def format_val_std(val, std, rank):
    if np.isnan(val): return "-"
    std_nan = NO_STDEV or np.isnan(std)
    s = f"{val:.{DECIMALS}f}" if std_nan else f"{val:.{DECIMALS}f} \\pm {std:.{DECIMALS}f}"
    if rank == 0: return f"$\\mathbf{{{s}}}$"
    elif rank == 1: return f"$\\underline{{{s}}}$"
    return f"${s}$"

def _fmt_md(val, std, rank):
    if np.isnan(val): return "-"
    std_nan = NO_STDEV or not isinstance(std, float) or np.isnan(std)
    s = f"{val:.{DECIMALS}f}" if std_nan else f"{val:.{DECIMALS}f} ±{std:.{DECIMALS}f}"
    if rank == 0: return f"**{s}**"
    if rank == 1: return f"*{s}*"
    return s

def _print_md_table(headers, rows):
    print("| " + " | ".join(str(h) for h in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")

def _summary_imp(mean_df, ds_list, keys, lower_is_better):
    """AVG-row 'imp' cell. For time, uses median of per-graph speedups (not ratio-of-means,
    which was dominated by large graphs)."""
    if mean_df is None or not ds_list or len(keys) < 2:
        return "-"
    our_key = keys[-1]
    if not lower_is_better:
        cols = [d for d in ds_list if d in mean_df.columns]
        def _mean(k):
            if k not in mean_df.index or not cols:
                return np.nan
            vals = mean_df.loc[k, cols]
            return vals.mean() if not vals.isna().any() else np.nan
        return calculate_improvement(_mean(our_key), [_mean(k) for k in keys[:-1]], False)
    per_graph = []
    for ds in ds_list:
        if ds not in mean_df.columns or our_key not in mean_df.index:
            continue
        ours = mean_df.loc[our_key, ds]
        if np.isnan(ours) or ours < 1e-9:
            continue
        bvals = [mean_df.loc[k, ds] for k in keys[:-1]
                 if k in mean_df.index and not np.isnan(mean_df.loc[k, ds])]
        if bvals:
            per_graph.append(min(bvals) / ours)
    return f"{np.median(per_graph):.1f}x" if per_graph else "-"

def calculate_improvement(our_val, baseline_vals, lower_is_better=False):
    if np.isnan(our_val): return "-"
    valid_baselines = [v for v in baseline_vals if not np.isnan(v)]
    if not valid_baselines: return "-"

    if lower_is_better:
        best_baseline = min(valid_baselines)
        if our_val < 1e-9: return "Inf"
        ratio = best_baseline / our_val
        return f"{ratio:.1f}x"
    else:
        best_baseline = max(valid_baselines)
        pct = ((our_val - best_baseline) / abs(best_baseline)) * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}\\%"

# Per-regime KT improvement tracking for the \extractmax{...} macros consumed downstream.
_REGIME_SUFFIX     = {"Undirected": "Undir", "Directed": "Dir"}
_REGIME_AVG_SUFFIX = {"Undirected AVG": "UndirAvg", "Directed AVG": "DirAvg", "AVG": "Avg"}
_REGIME_ORDER      = ("Undir", "Dir", "UndirAvg", "DirAvg", "Avg")

def _new_regime_imps():
    return {s: [] for s in _REGIME_ORDER}

def _record_regime_imp(imps, sfx, vals):
    """Append positive %-improvement of vals[-1] over best baseline into imps[sfx]."""
    if not sfx or np.isnan(vals[-1]):
        return
    valid_b = [v for v in vals[:-1] if not np.isnan(v)]
    if not valid_b:
        return
    pct = ((vals[-1] - max(valid_b)) / abs(max(valid_b))) * 100
    if pct > 0:
        imps[sfx].append(f"+{pct:.1f}")

def _emit_regime_macros(imps):
    for sfx in _REGIME_ORDER:
        nums = imps[sfx]
        if nums:
            print(f"\\begin{{extractmax}}{{{sfx}}}\\iffalse {' '.join(nums)} \\fi\\end{{extractmax}}")

def generate_combined_main_table(mean_kt, std_kt, mean_time, std_time, dotted=False, split_directed=False):
    kadabra_key = f"KADABRA_NK_err{KADABRA_ERR}_delta{KADABRA_DELTA}"
    silvan_key  = f"SILVAN_eps{SILVAN_ERR}_delta{SILVAN_DELTA}"
    our_key = get_algo_name(init=MANUAL_DEGREE_CODE,
                            nhid=MANUAL_NHID, layers=MANUAL_LAYERS, dropout=MANUAL_DROPOUT, epochs=MANUAL_EPOCHS)

    keys   = ["Original_Murata_SF", "ABCDE_Train", "DrBC", kadabra_key, silvan_key, our_key]
    labels = ["GNN-Bet", "ABCDE", "DrBC", "KAD.", "SIL.", "BRAVA"]
    n = len(keys)
    last = n - 1  # BRAVA index for improvement / speedup
    ncols = 1 + n + 1 + n + 1  # graph + KT methods + Imp + Time methods + Speedup

    hdr_methods = " & ".join(fr"\textbf{{{l}}}" for l in labels)

    print(r"\begin{table*}[t]")
    print(r"  \setlength{\tabcolsep}{3pt}")
    print(r"  \caption{Ranking accuracy (Kendall Tau, $\uparrow$) and inference time (seconds, $\downarrow$) on 14 real-world networks. Mean $\pm$ std.\ over 3 seeds. Best result per row is bolded, second-best underlined. \% Imp.\ is the percentage gain of BRAVA-GNN over the strongest baseline. Speedup (AVG rows) is the median of per-graph best-baseline-to-BRAVA time ratios.}")
    print(r"  \label{tab:combined_results}")
    print(r"  \centering")
    print(r"  \small")

    _begin_tabular("@{}" + "l" * ncols + "@{}")
    print(r"    \toprule")
    print(fr"    & \multicolumn{{{n + 1}}}{{c}}{{\textbf{{Accuracy (Kendall Tau $\uparrow$)}}}} & \multicolumn{{{n + 1}}}{{c}}{{\textbf{{Time (Seconds $\downarrow$)}}}} \\")
    print(fr"    \cmidrule(r){{2-{n + 2}}} \cmidrule(r){{{n + 3}-{2 * n + 3}}}")
    print(fr"    & {hdr_methods} & \textbf{{\% Imp.}} & {hdr_methods} & \textbf{{Speedup $\uparrow$}} \\")

    groups = ([("Undirected", UNDIRECTED_GRAPHS), ("Directed", DIRECTED_GRAPHS)]
              if split_directed else [("Real-World Networks", SOCIAL_NETWORKS)])

    def _avg_over(df, k, ds_list):
        # NaN-on-incomplete: AVG is only emitted if the method covers every dataset.
        # Skipping NaNs would silently compare ABCDE/DrBC over 6 graphs vs. BRAVA over 14.
        cols = [d for d in ds_list if d in df.columns]
        if k not in df.index or not cols:
            return np.nan
        vals = df.loc[k, cols]
        return vals.mean() if not vals.isna().any() else np.nan

    def _ranks(vals, lower_is_better):
        valid = sorted([(v, i) for i, v in enumerate(vals) if not np.isnan(v)],
                       key=lambda x: x[0], reverse=not lower_is_better)
        r = [-1] * n
        if len(valid) > 0: r[valid[0][1]] = 0
        if len(valid) > 1: r[valid[1][1]] = 1
        return r

    imps = _new_regime_imps()

    def _print_avg_row(label, ds_list):
        kt_avgs = [_avg_over(mean_kt, k, ds_list) for k in keys]
        t_avgs  = [_avg_over(mean_time, k, ds_list) for k in keys]
        kt_strs = [format_val_std(kt_avgs[i], float("nan"), r) for i, r in enumerate(_ranks(kt_avgs, False))]
        t_strs  = [format_val_std(t_avgs[i],  float("nan"), r) for i, r in enumerate(_ranks(t_avgs,  True))]
        kt_imp = _summary_imp(mean_kt,   ds_list, keys, False)
        t_imp  = _summary_imp(mean_time, ds_list, keys, True)
        print(f"    \\textbf{{{label}}} & " + " & ".join(kt_strs) + f" & {kt_imp} & " + " & ".join(t_strs) + f" & {t_imp} \\\\")
        _record_regime_imp(imps, _REGIME_AVG_SUFFIX.get(label), kt_avgs)

    for group_name, dataset_list in groups:
        suffix = _REGIME_SUFFIX.get(group_name, "")
        print(r"    \midrule")
        if split_directed:
            print(fr"    \multicolumn{{{ncols}}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

        for i, ds in enumerate(dataset_list):
            display_ds = ds.replace("_", r"\_")

            kt_vals = [mean_kt.loc[k, ds] if (k in mean_kt.index and ds in mean_kt.columns) else np.nan for k in keys]
            kt_stds = [std_kt.loc[k, ds]  if (k in std_kt.index  and ds in std_kt.columns)  else np.nan for k in keys]
            t_vals  = [mean_time.loc[k, ds] if (k in mean_time.index and ds in mean_time.columns) else np.nan for k in keys]
            t_stds  = [std_time.loc[k, ds]  if (k in std_time.index  and ds in std_time.columns)  else np.nan for k in keys]

            ranks_kt = _ranks(kt_vals, False)
            ranks_t  = _ranks(t_vals,  True)

            kt_strs = [format_val_std(kt_vals[j], kt_stds[j], ranks_kt[j]) for j in range(n)]
            t_strs  = [format_val_std(t_vals[j],  t_stds[j],  ranks_t[j])  for j in range(n)]
            kt_imp  = calculate_improvement(kt_vals[last], kt_vals[:last], False)
            t_imp   = calculate_improvement(t_vals[last],  t_vals[:last],  True)

            print(f"    \\hspace{{0.5em}}{display_ds} & " + " & ".join(kt_strs) + f" & {kt_imp} & " + " & ".join(t_strs) + f" & {t_imp} \\\\")
            _record_regime_imp(imps, suffix, kt_vals)

            if dotted and i < len(dataset_list) - 1:
                print(r"    \hdashline")

        if split_directed:
            _print_avg_row(f"{group_name} AVG", dataset_list)

    if split_directed:
        print(r"    \midrule")
        _print_avg_row("AVG", DATASETS)

    print(r"    \bottomrule")
    _end_tabular()
    print(r"\end{table*}")

    _emit_regime_macros(imps)

def generate_split_main_table(mean_kt, std_kt, mean_time, std_time, dotted=False, split_directed=False):
    """Split form of the combined main table: one table for ranking accuracy,
    one for inference time. Both gain a Bavarian baseline column; the time table
    also gains an exact-Brandes reference column."""
    kadabra_key  = f"KADABRA_NK_err{KADABRA_ERR}_delta{KADABRA_DELTA}"
    silvan_key   = f"SILVAN_eps{SILVAN_ERR}_delta{SILVAN_DELTA}"
    bavarian_key = f"Bavarian_{BAVARIAN_METHOD}_eps{BAVARIAN_ERR}_delta{BAVARIAN_DELTA}"
    our_key = get_algo_name(init=MANUAL_DEGREE_CODE,
                            nhid=MANUAL_NHID, layers=MANUAL_LAYERS, dropout=MANUAL_DROPOUT, epochs=MANUAL_EPOCHS)

    base_keys   = ["Original_Murata_SF", "ABCDE_Train", "DrBC", kadabra_key, silvan_key, bavarian_key]
    base_labels = ["GNN-Bet", "ABCDE", "DrBC", "KAD.", "SIL.", "BAV."]
    kt_keys,   kt_labels   = base_keys + [our_key],               base_labels + ["BRAVA"]
    # Exact Brandes sits just before BRAVA so the speedup column still compares
    # BRAVA against the sampling/learned baselines (min() skips slow Brandes).
    time_keys, time_labels = base_keys + [BRANDES_KEY, our_key],  base_labels + ["Brandes", "BRAVA"]

    for k, where in [(bavarian_key, mean_kt), (BRANDES_KEY, mean_time)]:
        if k not in where.index:
            print(f"% Note: no '{k}' rows found - that column renders as '-'.")

    groups = ([("Undirected", UNDIRECTED_GRAPHS), ("Directed", DIRECTED_GRAPHS)]
              if split_directed else [("Real-World Networks", SOCIAL_NETWORKS)])

    def _emit(mean_df, std_df, keys, labels, imp_hdr, lower_is_better, caption, tab_label, track_regime):
        n = len(keys)
        ncols = 1 + n + 1  # graph + methods + imp
        hdr_methods = " & ".join(fr"\textbf{{{l}}}" for l in labels)

        print(r"\begin{table*}[t]")
        print(r"  \setlength{\tabcolsep}{3pt}")
        print(f"  \\caption{{{caption}}}")
        print(f"  \\label{{{tab_label}}}")
        print(r"  \centering")
        print(r"  \small")
        _begin_tabular("@{}" + "l" * ncols + "@{}")
        print(r"    \toprule")
        print(fr"    & {hdr_methods} & \textbf{{{imp_hdr}}} \\")

        def _avg_over(k, ds_list):
            # NaN-on-incomplete: AVG only if the method covers every dataset.
            cols = [d for d in ds_list if d in mean_df.columns]
            if k not in mean_df.index or not cols:
                return np.nan
            vals = mean_df.loc[k, cols]
            return vals.mean() if not vals.isna().any() else np.nan

        def _ranks(vals):
            valid = sorted([(v, i) for i, v in enumerate(vals) if not np.isnan(v)],
                           key=lambda x: x[0], reverse=not lower_is_better)
            r = [-1] * n
            if len(valid) > 0: r[valid[0][1]] = 0
            if len(valid) > 1: r[valid[1][1]] = 1
            return r

        imps = _new_regime_imps()

        def _print_avg_row(label, ds_list):
            avgs = [_avg_over(k, ds_list) for k in keys]
            strs = [format_val_std(avgs[i], float("nan"), r) for i, r in enumerate(_ranks(avgs))]
            imp  = _summary_imp(mean_df, ds_list, keys, lower_is_better)
            print(f"    \\textbf{{{label}}} & " + " & ".join(strs) + f" & {imp} \\\\")
            if track_regime:
                _record_regime_imp(imps, _REGIME_AVG_SUFFIX.get(label), avgs)

        for group_name, dataset_list in groups:
            suffix = _REGIME_SUFFIX.get(group_name, "") if track_regime else ""
            print(r"    \midrule")
            if split_directed:
                print(fr"    \multicolumn{{{ncols}}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

            for i, ds in enumerate(dataset_list):
                display_ds = ds.replace("_", r"\_")
                vals = [mean_df.loc[k, ds] if (k in mean_df.index and ds in mean_df.columns) else np.nan for k in keys]
                stds = [std_df.loc[k, ds]  if (k in std_df.index  and ds in std_df.columns)  else np.nan for k in keys]
                ranks = _ranks(vals)
                strs  = [format_val_std(vals[j], stds[j], ranks[j]) for j in range(n)]
                imp   = calculate_improvement(vals[-1], vals[:-1], lower_is_better)
                print(f"    \\hspace{{0.5em}}{display_ds} & " + " & ".join(strs) + f" & {imp} \\\\")
                if track_regime:
                    _record_regime_imp(imps, suffix, vals)
                if dotted and i < len(dataset_list) - 1:
                    print(r"    \hdashline")

            if split_directed:
                _print_avg_row(f"{group_name} AVG", dataset_list)

        if split_directed:
            print(r"    \midrule")
            _print_avg_row("AVG", DATASETS)

        print(r"    \bottomrule")
        _end_tabular()
        print(r"\end{table*}")
        if track_regime:
            _emit_regime_macros(imps)

    _emit(mean_kt, std_kt, kt_keys, kt_labels, r"\% Imp.", False,
          r"Ranking accuracy (Kendall Tau, $\uparrow$) on 14 real-world networks. "
          r"Mean $\pm$ std.\ over 3 seeds. Best result per row is bolded, second-best "
          r"underlined. \% Imp.\ is the percentage gain of BRAVA-GNN over the strongest baseline.",
          "tab:main_accuracy", track_regime=True)
    print()
    _emit(mean_time, std_time, time_keys, time_labels, r"Speedup $\uparrow$", True,
          r"Inference time (seconds, $\downarrow$) on 14 real-world networks. "
          r"Mean $\pm$ std.\ over 3 seeds. Brandes is the exact algorithm (single run), shown "
          r"for reference. Best result per row is bolded, second-best underlined. Speedup "
          r"(AVG rows) is the median of per-graph best-baseline-to-BRAVA time ratios.",
          "tab:main_time", track_regime=False)

def generate_large_main_table(mean_kt, std_kt, mean_time, std_time, mean_flops=None, std_flops=None, dotted=False, fmt="latex", split_directed=False):
    our_key     = get_algo_name(init=MANUAL_DEGREE_CODE,
                                nhid=MANUAL_NHID, layers=MANUAL_LAYERS, dropout=MANUAL_DROPOUT, epochs=MANUAL_EPOCHS)
    kadabra_key = f"KADABRA_NK_err{KADABRA_ERR}_delta{KADABRA_DELTA}"
    silvan_key  = f"SILVAN_eps{SILVAN_ERR}_delta{SILVAN_DELTA}"

    keys   = ["Original_Murata_SF", "ABCDE_Train", "DrBC", kadabra_key, silvan_key, our_key]
    labels = ["GNN-Bet", "ABCDE", "DrBC",
              f"KAD.(ε={KADABRA_ERR:.0e}, δ={KADABRA_DELTA})",
              f"SILVAN(ε={SILVAN_ERR:.0e}, δ={SILVAN_DELTA})", "BRAVA"]
    labels_latex = ["GNN-Bet", "ABCDE", "DrBC",
                    f"KAD.", "SIL.", "BRAVA"]
    n = len(keys)
    groups = ([("Undirected", UNDIRECTED_GRAPHS), ("Directed", DIRECTED_GRAPHS)]
              if split_directed else [("Real-World Networks", SOCIAL_NETWORKS)])
    ncols = n + 1 + (0 if NO_IMPROVEMENT else 1)  # graph + methods [+ imp]
    col_spec = "@{}" + "l" * ncols + "@{}"

    def _print_subtable_latex(caption, label, mean_df, std_df, imp_hdr, lower_is_better,
                              _keys=None, _labels_latex=None, _no_stdev=False, track_regime=False):
        imps = _new_regime_imps()
        _k = _keys or keys
        _ll = _labels_latex or labels_latex
        _n = len(_k)
        _ncols = _n + 1 + (0 if NO_IMPROVEMENT else 1)
        _col_spec = "@{}" + "l" * _ncols + "@{}"
        hdr_parts = [f"\\textbf{{{l}}}" for l in _ll]
        if not NO_IMPROVEMENT:
            hdr_parts.append(f"\\textbf{{{imp_hdr}}}")
        hdrs = " & ".join(hdr_parts)
        print(r"\begin{table*}[t]")
        print(r"  \setlength{\tabcolsep}{3pt}")
        print(f"  \\caption{{{caption}}}")
        print(f"  \\label{{{label}}}")
        print(r"  \centering")
        print(r"  \small")
        _begin_tabular(_col_spec)
        print(r"    \toprule")
        print(f"    & {hdrs} \\\\")

        def _fvs(val, std, rank):
            return format_val_std(val, std if not _no_stdev else float("nan"), rank)

        def _avg_row_latex(label, ds_list):
            avgs = [mean_df.loc[k, [d for d in ds_list if d in mean_df.columns]].mean()
                    if (k in mean_df.index and any(d in mean_df.columns for d in ds_list)) else np.nan
                    for k in _k]
            a_valid = sorted([(v, j) for j, v in enumerate(avgs) if not np.isnan(v)],
                             key=lambda x: x[0], reverse=not lower_is_better)
            a_ranks = [-1] * _n
            if len(a_valid) > 0: a_ranks[a_valid[0][1]] = 0
            if len(a_valid) > 1: a_ranks[a_valid[1][1]] = 1
            a_strs = [_fvs(avgs[j], float("nan"), a_ranks[j]) for j in range(_n)]
            a_row = f"    \\textbf{{{label}}} & " + " & ".join(a_strs)
            if not NO_IMPROVEMENT:
                a_row += f" & {_summary_imp(mean_df, ds_list, _k, lower_is_better)}"
            print(a_row + r" \\")
            if track_regime and not lower_is_better:
                _record_regime_imp(imps, _REGIME_AVG_SUFFIX.get(label), avgs)

        for group_name, dataset_list in groups:
            suffix = _REGIME_SUFFIX.get(group_name, "") if track_regime else ""
            print(r"    \midrule")
            if split_directed:
                print(fr"    \multicolumn{{{_ncols}}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

            for i, ds in enumerate(dataset_list):
                display_ds = ds.replace("_", r"\_")
                vals = [mean_df.loc[k, ds] if (k in mean_df.index and ds in mean_df.columns) else np.nan for k in _k]
                stds = [std_df.loc[k, ds]  if (k in std_df.index  and ds in std_df.columns)  else np.nan for k in _k]
                valid = sorted([(v, j) for j, v in enumerate(vals) if not np.isnan(v)],
                               key=lambda x: x[0], reverse=not lower_is_better)
                ranks = [-1] * _n
                if len(valid) > 0: ranks[valid[0][1]] = 0
                if len(valid) > 1: ranks[valid[1][1]] = 1
                strs = [_fvs(vals[j], stds[j], ranks[j]) for j in range(_n)]
                row = f"    \\hspace{{0.5em}}{display_ds} & " + " & ".join(strs)
                if not NO_IMPROVEMENT:
                    row += f" & {calculate_improvement(vals[-1], vals[:-1], lower_is_better)}"
                print(row + r" \\")
                _record_regime_imp(imps, suffix, vals)
                if dotted and i < len(dataset_list) - 1:
                    print(r"    \hdashline")

            if split_directed:
                _avg_row_latex(f"{group_name} AVG", dataset_list)

        if split_directed:
            print(r"    \midrule")
            _avg_row_latex("AVG", DATASETS)

        print(r"    \bottomrule")
        _end_tabular()
        print(r"\end{table*}")
        if track_regime:
            _emit_regime_macros(imps)

    def _print_subtable_md(title, mean_df, std_df, imp_hdr, lower_is_better,
                           _keys=None, _labels=None, _no_stdev=False):
        _k = _keys or keys
        _l = _labels or labels
        print(f"\n### {title}\n")
        headers = ["Graph"] + _l + ([] if NO_IMPROVEMENT else [imp_hdr])
        rows = []
        _n = len(_k)

        def _fmd(val, std, rank):
            return _fmt_md(val, std if not _no_stdev else float("nan"), rank)

        def _avg_row_md(label, ds_list):
            avgs = [mean_df.loc[k, [d for d in ds_list if d in mean_df.columns]].mean()
                    if (k in mean_df.index and any(d in mean_df.columns for d in ds_list)) else np.nan
                    for k in _k]
            a_valid = sorted([(v, j) for j, v in enumerate(avgs) if not np.isnan(v)],
                             key=lambda x: x[0], reverse=not lower_is_better)
            a_ranks = [-1] * _n
            if len(a_valid) > 0: a_ranks[a_valid[0][1]] = 0
            if len(a_valid) > 1: a_ranks[a_valid[1][1]] = 1
            a_strs = [_fmd(avgs[j], float("nan"), a_ranks[j]) for j in range(_n)]
            a_row = [f"**{label}**"] + a_strs
            if not NO_IMPROVEMENT:
                a_row.append(_summary_imp(mean_df, ds_list, _k, lower_is_better))
            return a_row

        for group_name, dataset_list in groups:
            if split_directed:
                rows.append([f"**{group_name}**"] + [""] * (len(_l) + (0 if NO_IMPROVEMENT else 1)))
            for ds in dataset_list:
                vals = [mean_df.loc[k, ds] if (k in mean_df.index and ds in mean_df.columns) else np.nan for k in _k]
                stds = [std_df.loc[k, ds]  if (k in std_df.index  and ds in std_df.columns)  else np.nan for k in _k]
                valid = sorted([(v, j) for j, v in enumerate(vals) if not np.isnan(v)],
                               key=lambda x: x[0], reverse=not lower_is_better)
                ranks = [-1] * _n
                if len(valid) > 0: ranks[valid[0][1]] = 0
                if len(valid) > 1: ranks[valid[1][1]] = 1
                strs = [_fmd(vals[j], stds[j], ranks[j]) for j in range(_n)]
                row = [ds] + strs
                if not NO_IMPROVEMENT:
                    row.append(calculate_improvement(vals[-1], vals[:-1], lower_is_better))
                rows.append(row)

            if split_directed:
                rows.append(_avg_row_md(f"{group_name} AVG", dataset_list))

        if split_directed:
            rows.append(_avg_row_md("AVG", DATASETS))
        _print_md_table(headers, rows)

    if fmt == "markdown":
        _print_subtable_md(
            "Accuracy (Kendall Tau ↑) — GNN-Bet, ABCDE, DrBC, KADABRA, BRAVA",
            mean_kt, std_kt, "% Imp.", lower_is_better=False,
        )
        _print_subtable_md(
            "Inference Time (Seconds ↓) — GNN-Bet, ABCDE, DrBC, KADABRA, BRAVA",
            mean_time, std_time, "Speedup ↑", lower_is_better=True,
        )
        if mean_flops is not None and std_flops is not None:
            flops_keys   = [k for k in keys if k not in (kadabra_key, silvan_key)]
            flops_labels = [l for k, l in zip(keys, labels) if k not in (kadabra_key, silvan_key)]
            _print_subtable_md(
                "FLOPs (GFLOPs ↓) — GNN-Bet, ABCDE, DrBC, BRAVA",
                mean_flops, std_flops, "Reduction", lower_is_better=True,
                _keys=flops_keys, _labels=flops_labels, _no_stdev=True,
            )
    else:
        _print_subtable_latex(
            "Accuracy (Kendall Tau $\\uparrow$)"
            "Best results are \\textbf{bolded}, second best \\underline{underlined}.",
            "tab:large_main_kt", mean_kt, std_kt, r"\% Imp.", lower_is_better=False,
            track_regime=True,
        )
        _print_subtable_latex(
            "Inference Time (Seconds $\\downarrow$) "
            "Best results are \\textbf{bolded}, second best \\underline{underlined}.",
            "tab:large_main_time", mean_time, std_time, r"Speedup $\uparrow$", lower_is_better=True,
        )
        if mean_flops is not None and std_flops is not None:
            flops_keys         = [k for k in keys if k not in (kadabra_key, silvan_key)]
            flops_labels_latex = [l for k, l in zip(keys, labels_latex) if k not in (kadabra_key, silvan_key)]
            _print_subtable_latex(
                "FLOPs (GFLOPs $\\downarrow$) excluding KADABRA (sampling algorithm). "
                "Best results are \\textbf{bolded}, second best \\underline{underlined}.",
                "tab:large_main_flops", mean_flops, std_flops, "Reduction", lower_is_better=True,
                _keys=flops_keys, _labels_latex=flops_labels_latex, _no_stdev=True,
            )


def format_cell(mean, std, is_best=False, is_second=False, is_chosen=False):
    if np.isnan(mean): return "-"
    val_str = f"{mean:.{DECIMALS}f}" if (NO_STDEV or np.isnan(std)) else f"{mean:.{DECIMALS}f} \\pm {std:.{DECIMALS}f}"
    if is_chosen:
        val_str += "^{\\dagger}"
    if is_best:
        return f"$\\mathbf{{{val_str}}}$"
    if is_second:
        return f"$\\underline{{{val_str}}}$"
    return f"${val_str}$"

def _resolve_chosen(table_id, column_keys):
    chosen = ONE_CHOSEN.get(table_id)
    if isinstance(chosen, int):
        keys = list(column_keys.keys())
        return keys[chosen] if chosen < len(keys) else None
    return chosen

def generate_transposed_latex_table(title, label, mean_df, std_df, column_keys, table_id, split_threshold=7, higher_is_better=True, dotted=False, show_group_avgs=False, fmt="latex", split_directed=False, bold_headers=True):
    row_bests = {}
    row_seconds = {}

    for ds in DATASETS:
        vals = [(k, mean_df.loc[k, ds]) for k in column_keys.keys() if k in mean_df.index and ds in mean_df.columns]
        if vals:
            sorted_vals = sorted(vals, key=lambda x: x[1], reverse=higher_is_better)
            row_bests[ds] = sorted_vals[0][1]
            row_seconds[ds] = sorted_vals[1][1] if len(sorted_vals) > 1 else None
        else:
            row_bests[ds] = None
            row_seconds[ds] = None

    def _col_avgs(dataset_list):
        avgs = []
        for k in column_keys.keys():
            if k in mean_df.index:
                valid_ds = [d for d in dataset_list if d in mean_df.columns]
                avgs.append(mean_df.loc[k, valid_ds].mean() if valid_ds else None)
            else:
                avgs.append(None)
        return avgs

    all_col_avgs = _col_avgs(DATASETS)
    undirected_col_avgs = _col_avgs(UNDIRECTED_GRAPHS)
    directed_col_avgs   = _col_avgs(DIRECTED_GRAPHS)
    social_col_avgs     = _col_avgs(SOCIAL_NETWORKS)

    opt_func = max if higher_is_better else min

    def _best_second(col_avgs):
        valid = [(i, x) for i, x in enumerate(col_avgs) if x is not None]
        best = opt_func([x[1] for x in valid]) if valid else None
        srt = sorted(valid, key=lambda x: x[1], reverse=higher_is_better)
        second = srt[1][1] if len(srt) > 1 else None
        return best, second

    global_best_avg, global_second_avg = _best_second(all_col_avgs)
    undirected_best_avg, undirected_second_avg = _best_second(undirected_col_avgs)
    directed_best_avg,   directed_second_avg   = _best_second(directed_col_avgs)
    social_best_avg,     social_second_avg     = _best_second(social_col_avgs)

    all_items = list(column_keys.items())

    if len(all_items) <= split_threshold:
        chunks = [all_items]
    else:
        chunks = [all_items[i:i + split_threshold] for i in range(0, len(all_items), split_threshold)]

    groups = ([("Undirected", UNDIRECTED_GRAPHS), ("Directed", DIRECTED_GRAPHS)]
              if split_directed else [("Real-World Networks", SOCIAL_NETWORKS)])

    if fmt == "markdown":
        chunk_keys_flat = dict(all_items)
        col_labels = [_label_with_seeds(k, v) for k, v in chunk_keys_flat.items()]
        col_keys   = list(chunk_keys_flat.keys())

        def _avg_row_md(row_label, col_avgs, best_avg, second_avg):
            cells = [f"**{row_label}**"]
            for i, k in enumerate(col_keys):
                val = col_avgs[i]
                if val is not None:
                    s = std_df.loc[k, DATASETS].mean() if k in std_df.index else float("nan")
                    is_best   = best_avg   is not None and abs(val - best_avg)   < 1e-6
                    is_second = (not is_best) and second_avg is not None and abs(val - second_avg) < 1e-6
                    rank = 0 if is_best else (1 if is_second else -1)
                    cells.append(_fmt_md(val, s, rank))
                else:
                    cells.append("-")
            return cells

        print(f"\n### {title}\n")
        headers = ["Graph"] + col_labels
        rows = []
        group_avgs = ({
            id(UNDIRECTED_GRAPHS): (undirected_col_avgs, undirected_best_avg, undirected_second_avg, "Undirected AVG"),
            id(DIRECTED_GRAPHS):   (directed_col_avgs,   directed_best_avg,   directed_second_avg,   "Directed AVG"),
        } if split_directed else {
            id(SOCIAL_NETWORKS): (social_col_avgs, social_best_avg, social_second_avg, "Real-World AVG"),
        })
        for group_name, dataset_list in groups:
            if split_directed:
                rows.append([f"**{group_name}**"] + [""] * len(col_keys))
            for ds in dataset_list:
                row = [ds]
                for k in col_keys:
                    if k in mean_df.index and ds in mean_df.columns:
                        m = mean_df.loc[k, ds]
                        s = std_df.loc[k, ds]
                        is_best   = row_bests[ds]   is not None and abs(m - row_bests[ds])   < 1e-6
                        is_second = (not is_best) and row_seconds[ds] is not None and abs(m - row_seconds[ds]) < 1e-6
                        rank = 0 if is_best else (1 if is_second else -1)
                        row.append(_fmt_md(m, s, rank))
                    else:
                        row.append("-")
                rows.append(row)
            if show_group_avgs and id(dataset_list) in group_avgs:
                col_avgs, best_avg, second_avg, avg_label = group_avgs[id(dataset_list)]
                rows.append(_avg_row_md(avg_label, col_avgs, best_avg, second_avg))
        rows.append(_avg_row_md("AVG", all_col_avgs, global_best_avg, global_second_avg))
        _print_md_table(headers, rows)
        return

    for idx, chunk in enumerate(chunks):
        chunk_keys = dict(chunk)
        chosen_key = _resolve_chosen(table_id, column_keys)
        t_suffix = f" (Part {idx+1}/{len(chunks)})" if len(chunks) > 1 else ""
        l_suffix = f"_{idx+1}" if len(chunks) > 1 else ""

        print(f"\n% --- {title}{t_suffix} ---")
        print(r"\begin{table*}[h]")
        print(r"  \centering")
        print(r"  \renewcommand{\arraystretch}{1.2}")
        print(r"  \setlength{\tabcolsep}{2pt}")
        print(f"  \\caption{{{title}{t_suffix}}}")
        print(f"  \\label{{{label}{l_suffix}}}")

        col_def = "@{}l" + "l" * len(chunk_keys) + "@{}"
        _begin_tabular(col_def)
        print(r"    \toprule")

        headers = [_label_with_seeds(k, v) + ("$^{\\dagger}$" if k == chosen_key else "") for k, v in chunk_keys.items()]
        if bold_headers:
            print(r"    \textbf{Graph} & \textbf{" + "} & \\textbf{".join(headers) + r"} \\")
        else:
            print(r"    \textbf{Graph} & " + " & ".join(headers) + r" \\")

        chunk_start_idx = idx * split_threshold

        def _avg_row(label, col_avgs, best_avg, second_avg, ds_list):
            row = f"    \\textbf{{{label}}}"
            for i, k in enumerate(chunk_keys.keys()):
                val = col_avgs[chunk_start_idx + i]
                if val is not None:
                    valid_ds = [d for d in ds_list if d in std_df.columns]
                    s = std_df.loc[k, valid_ds].mean() if k in std_df.index else 0
                    is_best = (best_avg is not None and abs(val - best_avg) < 1e-6)
                    is_second = (not is_best and second_avg is not None and abs(val - second_avg) < 1e-6)
                    row += f" & {format_cell(val, s, is_best, is_second, chosen_key == k)}"
                else:
                    row += " & -"
            print(row + r" \\")

        group_avgs = ({
            id(UNDIRECTED_GRAPHS): (undirected_col_avgs, undirected_best_avg, undirected_second_avg, "Undirected AVG"),
            id(DIRECTED_GRAPHS):   (directed_col_avgs,   directed_best_avg,   directed_second_avg,   "Directed AVG"),
        } if split_directed else {
            id(SOCIAL_NETWORKS): (social_col_avgs, social_best_avg, social_second_avg, "Real-World AVG"),
        })

        for group_name, dataset_list in groups:
            print(r"    \midrule")
            if split_directed:
                print(fr"    \multicolumn{{{len(chunk_keys) + 1}}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

            for i, ds in enumerate(dataset_list):
                display_ds = ds.replace("_", r"\_")
                row = f"    \\hspace{{0.5em}}{display_ds}"

                for k in chunk_keys.keys():
                    if k in mean_df.index and ds in mean_df.columns:
                        m = mean_df.loc[k, ds]
                        s = std_df.loc[k, ds]
                        is_best = (row_bests[ds] is not None and abs(m - row_bests[ds]) < 1e-6)
                        is_second = (not is_best and row_seconds[ds] is not None and abs(m - row_seconds[ds]) < 1e-6)
                        row += f" & {format_cell(m, s, is_best, is_second, chosen_key == k)}"
                    else:
                        row += " & -"
                print(row + r" \\")

                if dotted and i < len(dataset_list) - 1:
                    print(r"    \hdashline")

            if show_group_avgs and id(dataset_list) in group_avgs:
                col_avgs, best_avg, second_avg, label = group_avgs[id(dataset_list)]
                _avg_row(label, col_avgs, best_avg, second_avg, dataset_list)

        print(r"    \midrule")
        _avg_row("AVG", all_col_avgs, global_best_avg, global_second_avg, DATASETS)
        print(r"    \bottomrule")
        _end_tabular()
        print(r"\end{table*}")

def generate_comparison_table(mean_df, std_df, higher_is_better=True, dotted=False, fmt="latex", split_directed=False):
    cols = {
        get_algo_name(train=BEST_TRAIN_TYPE, init=r["init_type"],
                      layers=r["num_layers"], nhid=r["nhid"], dropout=r["dropout"]): label.capitalize()
        for label, r in BEST_RUNS.items()
    }
    generate_transposed_latex_table(
        "Best Model", "tab:comparison",
        mean_df, std_df, cols, table_id="cmp",
        higher_is_better=higher_is_better, dotted=dotted, show_group_avgs=True, fmt=fmt,
        split_directed=split_directed,
    )


def _dxw_grid(mean_df, std_df, layers, nhid, pins):
    """Returns (mean_grid, std_grid) over (L, H), each cell = avg over all test graphs."""
    valid_ds = [d for d in DATASETS if d in mean_df.columns]
    mean_grid = np.full((len(layers), len(nhid)), np.nan)
    std_grid  = np.full((len(layers), len(nhid)), np.nan)
    for i, L in enumerate(layers):
        for j, H in enumerate(nhid):
            key = get_algo_name(layers=L, nhid=H, **pins)
            if key in mean_df.index:
                mean_grid[i, j] = mean_df.loc[key, valid_ds].mean()
            if key in std_df.index:
                std_grid[i, j]  = std_df.loc[key, valid_ds].mean()
    return mean_grid, std_grid


def generate_depth_width_table(mean_df, std_df, layers, nhid, pins, fmt="latex"):
    """Depth × width table: rows = num_layers, cols = nhid; cell = mean KT over all test graphs."""
    mean_grid, std_grid = _dxw_grid(mean_df, std_df, layers, nhid, pins)
    flat = [(v, (i, j)) for i in range(len(layers))
                       for j in range(len(nhid))
                       if not np.isnan(v := mean_grid[i, j])]
    flat.sort(key=lambda x: x[0], reverse=True)
    best_ij   = flat[0][1] if len(flat) > 0 else None
    second_ij = flat[1][1] if len(flat) > 1 else None

    if fmt == "markdown":
        print("\n### Depth × Width — KT Accuracy (↑), mean over all test graphs\n")
        headers = ["L \\ nhid"] + [str(H) for H in nhid]
        rows = []
        for i, L in enumerate(layers):
            row = [f"**L={L}**"]
            for j in range(len(nhid)):
                rank = 0 if (i, j) == best_ij else (1 if (i, j) == second_ij else -1)
                row.append(_fmt_md(mean_grid[i, j], std_grid[i, j], rank))
            rows.append(row)
        _print_md_table(headers, rows)
        return

    note = _caption_pin_note(pins, varied=("layers", "nhid"))
    print(r"\begin{table*}[t]")
    print(r"  \setlength{\tabcolsep}{4pt}")
    print(fr"  \caption{{Ablation: Depth $\times$ Width (accuracy).{note}}}")
    print(r"  \label{tab:depth_width}")
    print(r"  \centering")
    print(r"  \small")
    col_spec = "@{}l" + "l" * len(nhid) + "@{}"
    _begin_tabular(col_spec)
    print(r"    \toprule")
    hdrs = " & ".join(fr"\textbf{{{H}}}" for H in nhid)
    print(fr"    \textbf{{$L$ \textbackslash{{}} $n_{{hid}}$}} & {hdrs} \\")
    print(r"    \midrule")
    for i, L in enumerate(layers):
        cells = []
        for j in range(len(nhid)):
            rank = 0 if (i, j) == best_ij else (1 if (i, j) == second_ij else -1)
            cells.append(format_val_std(mean_grid[i, j], std_grid[i, j], rank))
        print(fr"    \textbf{{{L}}} & " + " & ".join(cells) + r" \\")
    print(r"    \bottomrule")
    _end_tabular()
    print(r"\end{table*}")

    missing = [(layers[i], nhid[j])
               for i in range(len(layers)) for j in range(len(nhid))
               if np.isnan(mean_grid[i, j])]
    if missing:
        print(f"% Missing cells (no rows in {RESULTS_KT}): {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="all", help="Options: main, split, large_main, dxw, all, 1, 2, 3, 4 (=dxw), 5")
    parser.add_argument("--mode", default="accuracy", choices=["accuracy", "inference", "training", "comparison"],
                        help="Select metric.")
    parser.add_argument("--dotted", action="store_true", help="Add dotted lines between rows (requires arydshln)")
    parser.add_argument("--format", default="latex", choices=["latex", "markdown"],
                        help="Output format: latex (default) or markdown")
    parser.add_argument("--no-stdev", action="store_true", help="Suppress standard deviation in all cells")
    parser.add_argument("--no-improvement", action="store_true", help="Suppress the %% Imp./Speedup column")
    parser.add_argument("--topk", type=int, choices=[1, 5, 10], default=None,
                        help="Use Top-K KT (1, 5, or 10 percent) from all_results_topk.csv instead of full KT. "
                             "Affects 'main' and 'large_main' tables; time/inference tables are unchanged.")
    parser.add_argument("--to-file", action="store_true", help="Write output to ./table.txt")
    parser.add_argument("--x100", action=argparse.BooleanOptionalAction, default=True, help="Multiply KT values by 100 (default: on; pass --no-x100 to disable)")
    parser.add_argument("--round", type=int, default=1, metavar="N", help="Decimal places (default 1)")
    parser.add_argument("--show-seeds", action="store_true", help="Append (N) seed count after each column name")
    parser.add_argument("--resizebox", action="store_true", help="Wrap each LaTeX tabular in \\resizebox{\\textwidth}{!}{...}")
    parser.add_argument("--split-directed", dest="split_directed", action="store_true", default=True,
                        help="Split rows into Undirected vs Directed groups with per-group and overall AVG rows (default on).")
    parser.add_argument("--unsplit-directed", dest="split_directed", action="store_false",
                        help="Disable the Undirected/Directed split — render datasets as a single flat list.")
    args = parser.parse_args()
    fmt = args.format
    global NO_STDEV, NO_IMPROVEMENT, DECIMALS, SHOW_SEEDS, RESIZEBOX
    NO_STDEV = args.no_stdev
    NO_IMPROVEMENT = args.no_improvement
    SHOW_SEEDS = args.show_seeds
    RESIZEBOX = args.resizebox
    DECIMALS = args.round
    kt_scale = 100.0 if args.x100 else 1.0

    outfile = "table.md" if fmt == "markdown" else "table.txt"
    if args.to_file:
        sys.stdout = open(outfile, "w")

    if args.mode == "comparison":
        mean_df, std_df = load_and_aggregate(RESULTS_KT, scale=kt_scale)
        if mean_df is not None:
            generate_comparison_table(mean_df, std_df, higher_is_better=True, dotted=args.dotted, fmt=fmt,
                                      split_directed=args.split_directed)
        if args.to_file:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            print(f"Output written to {outfile}")
        return

    if args.mode == "inference":
        target_file = RESULTS_TIME
        higher_is_better = False
    elif args.mode == "training":
        target_file = RESULTS_TRAINING
        higher_is_better = False
    else:
        target_file = RESULTS_KT
        higher_is_better = True

    primary_mean, primary_std = load_and_aggregate(target_file, scale=kt_scale if args.mode == "accuracy" else 1.0)
    if primary_mean is None: return

    # T1: chose -HY Sym
    T1_PINS = dict(
        init="degree_mix_mass_3",
        layers=MANUAL_LAYERS,
        nhid=MANUAL_NHID,
        dropout=MANUAL_DROPOUT,
    )
    t1_cols = {
        # Singles
        get_algo_name(train="SF_10_Dir", **T1_PINS): "SF Dir",
        get_algo_name(train="SF_10_Sym", **T1_PINS): "SF Sym",
        get_algo_name(train="HY_10_Dir", **T1_PINS): "HY Dir",
        get_algo_name(train="HY_10_Sym", **T1_PINS): "HY Sym",
        # Leave-one-out
        get_algo_name(train="SF_10_Sym+HY_10_Dir+HY_10_Sym", **T1_PINS): "-SF Dir",
        get_algo_name(train="SF_10_Dir+HY_10_Dir+HY_10_Sym", **T1_PINS): "-SF Sym",
        get_algo_name(train="SF_10_Dir+SF_10_Sym+HY_10_Sym", **T1_PINS): "-HY Dir",
        get_algo_name(train="SF_10_Dir+SF_10_Sym+HY_10_Dir", **T1_PINS): "-HY Sym",
        # Halves
        get_algo_name(train="SF_10_Dir+HY_10_Dir", **T1_PINS): "Dir-only",
        get_algo_name(train="SF_10_Sym+HY_10_Sym", **T1_PINS): "Sym-only",
        get_algo_name(train="SF_10_Dir+HY_10_Sym", **T1_PINS): "SF+HY*",
        # All four
        get_algo_name(train="SF_10_Dir+SF_10_Sym+HY_10_Dir+HY_10_Sym", **T1_PINS): "All-4",
        # Swapped with mNN
        get_algo_name(train="SF_10_Dir+SF_10_Sym+HY_10_mNN+HY_10_Sym", **T1_PINS): "mNN-swap",
        get_algo_name(train="SF_10_Dir+SF_10_Sym+HY_10_mNN",            **T1_PINS): "SF+mNN",
        # Headline three-regime mix with SF_Sym replaced by SF_SymBA (Barabási–Albert).
        get_algo_name(train="SF_10_Dir+SF_10_SymBA+HY_10_Dir",          **T1_PINS): "SymBA-swap",
    }

    if args.table in ["main", "combined", "split", "large_main", "all"]:
        if args.topk is not None:
            mean_kt, std_kt = load_and_aggregate(RESULTS_KT_TOPK, topk_pct=args.topk, scale=kt_scale, update_seeds=False)
        else:
            mean_kt, std_kt = load_and_aggregate(RESULTS_KT, scale=kt_scale, update_seeds=False)
        mean_time, std_time = load_and_aggregate(RESULTS_TIME, update_seeds=False)
        mean_flops, std_flops = load_and_aggregate(RESULTS_FLOPS, update_seeds=False)
        if mean_kt is not None and mean_time is not None:
            if args.table in ["main", "combined", "all"] and fmt == "latex":
                generate_combined_main_table(mean_kt, std_kt, mean_time, std_time, dotted=args.dotted,
                                             split_directed=args.split_directed)
            if args.table == "split" and fmt == "latex":
                brandes = load_brandes_time()
                if brandes is not None:
                    mean_time = pd.concat([mean_time, brandes])
                generate_split_main_table(mean_kt, std_kt, mean_time, std_time, dotted=args.dotted,
                                          split_directed=args.split_directed)
            # Superseded by the combined table above; uncomment to restore independent KT/Time/FLOPs tables.
            # if args.table in ["large_main", "all"]:
            #     generate_large_main_table(mean_kt, std_kt, mean_time, std_time,
            #                               mean_flops=mean_flops, std_flops=std_flops,
            #                               dotted=args.dotted, fmt=fmt,
            #                               split_directed=args.split_directed)

    # T2: chose degree_mix_mass_6
    T2_PINS = dict(
        train=MANUAL_TRAIN_TYPE,
        layers=MANUAL_LAYERS,
        nhid=MANUAL_NHID,
        dropout=MANUAL_DROPOUT,
    )

    t2_cols = {
        get_algo_name(init="degree_mix_mass_1",        **T2_PINS): r"[$d^{(0)}$]",
        get_algo_name(init="degree_mix_mass_2",        **T2_PINS): r"$[d^{(0)} d^{(1)}]$",
        get_algo_name(init="degree_mix_mass_3",        **T2_PINS): r"$[\dots d^{(2)}]$",
        get_algo_name(init="degree_mix_mass_4",        **T2_PINS): r"$[\dots d^{(3)}]$",
        get_algo_name(init="degree_mix_mass_5",        **T2_PINS): r"$[\dots d^{(4)}]$",
        get_algo_name(init="degree_mix_mass_6",        **T2_PINS): r"$[\dots d^{(5)}]$",
        get_algo_name(init="degree_mix_mass_12",       **T2_PINS): r"$[\dots d^{(11)}]$",
        get_algo_name(init="degree_mix_independent_6", **T2_PINS): r"$[d^{(0)}, Ad^{(0)}, \dots, A^5d^{(0)}]$",
        get_algo_name(init="random_6",                 **T2_PINS): r"\textbf{random-6}",
    }

    # T3: chose default (shared encoders, mul fusion)
    T3_PINS = dict(
        train=MANUAL_TRAIN_TYPE,
        init=MANUAL_DEGREE_CODE,
        layers=MANUAL_LAYERS,
        nhid=MANUAL_NHID,
        dropout=MANUAL_DROPOUT,
    )
    t3_cols = {
        get_algo_name(**T3_PINS):                            "BRAVA",
        get_algo_name(unshared_encoders=True, **T3_PINS):    "BRAVA+Unshared",
        get_algo_name(fusion="add",           **T3_PINS):    "BRAVA+FusionAdd",
        get_algo_name(fusion="cat_mlp",       **T3_PINS):    "BRAVA+FusionCat-MLP",
    }

    # T4: chose L=2, nhid=12. Grid table varying both.
    T4_LAYERS = [0, 1, 2, 4, 6]
    T4_NHID   = [8, 12, 16, 32]
    T4_PINS = dict(
        train=MANUAL_TRAIN_TYPE,
        init=MANUAL_DEGREE_CODE,
        dropout=MANUAL_DROPOUT,
    )

    # T5: chose dropout=0.3
    T5_PINS = dict(
        train=MANUAL_TRAIN_TYPE,
        init=MANUAL_DEGREE_CODE,
        layers=MANUAL_LAYERS,
        nhid=MANUAL_NHID,
    )
    t5_cols = {
        get_algo_name(dropout=d, **T5_PINS): f"{d:.1f}"
        for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    }

    print("\n\n\n\n\n")
    print(f"% Ablation | Mode: {args.mode}")

    ablation_tables = [
        ("1", f"Ablation: Training Distribution ({args.mode}).{_caption_pin_note(T1_PINS, varied=('train',))}",  "tab:train",  t1_cols, 7),
        ("2", f"Ablation: Feature Initialization ({args.mode}).{_caption_pin_note(T2_PINS, varied=('init',))}",   "tab:init",   t2_cols, 10),
        ("3", f"Ablation: Fusion / Encoder Sharing ({args.mode}).{_caption_pin_note(T3_PINS)}",                   "tab:fusion", t3_cols, 5),
        ("5", f"Ablation: Dropout ({args.mode}).{_caption_pin_note(T5_PINS, varied=('dropout',))}",               "tab:drop",   t5_cols, 7),
    ]

    for table_id, title, label, cols, split_thresh in ablation_tables:
        # T4 uses its own generator; emit just before T5 to keep T-order.
        if table_id == "5" and args.table in ["dxw", "4", "all"] and args.mode == "accuracy":
            generate_depth_width_table(primary_mean, primary_std, T4_LAYERS, T4_NHID, T4_PINS, fmt=fmt)
        if args.table in [table_id, "all"]:
            generate_transposed_latex_table(title, label, primary_mean, primary_std, cols, table_id, split_thresh, higher_is_better, args.dotted,
                                            split_directed=args.split_directed,
                                            bold_headers=(table_id != "2"))

    if args.to_file:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print("Output written to table.txt")

if __name__ == "__main__":
    main()
