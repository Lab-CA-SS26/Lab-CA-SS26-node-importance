#!/usr/bin/env python3
"""
update_report_from_rerun.py --- write the table bodies and figures that the stopping-coordination
rerun (results/rerun_fix/, see its README) changes, straight into Report/.

Each table body is the output of the script that owns it, spliced between the table's \\midrule
and \\bottomrule; captions and prose are not touched (they quote ranges the same scripts print, and
are edited by hand from that output). Each figure is written by make_plots.py.

  Report/tables/cpp_vs_julia_tight_table.tex   summarize_reproduce.py tight   rerun_fix/tight
  Report/tables/tight_accuracy_table.tex       summarize_reproduce.py tightx  rerun_fix/tight + rerun_fix/bvk
  Report/tables/brava_vs_kadabra_table.tex     summarize_seeds.py             rerun_fix/bvk (+ kadabra_seeds),
                                                                              BRAVA-GNN seed log unchanged
  Report/tables/topk_allocation_table.tex      summarize_topk.py              rerun_fix/topk (4 small graphs only)
  Report/tables/cpp_julia_quality_table.tex    summarize_cpp_quality.py       --julia-from rerun_fix
  Report/figures/thread_scaling.{pdf,png}      make_plots.py threads          rerun_fix/ts
  Report/figures/topk_allocation.{pdf,png}     make_plots.py topk             rerun_fix/topk
  Report/figures/brava_vs_kadabra.{pdf,png}    make_plots.py bvk              rerun_fix/bvk (+ kadabra_seeds, BRAVA log)

Usage:  python3 update_report_from_rerun.py [--check] [--results DIR] [--report DIR]
        --check    only report which files would change
        --results  a results tree laid out like benchmark/results/ (rerun_fix/, cpp_quality/,
                   brava_retrained/logs/eval_seeds.log); default benchmark/results
        --report   where tables/ and figures/ go; default ../Report. A table whose .tex does
                   not exist there is written as its bare body (reproduce_all.sh on a checkout
                   without the Report repo)
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

B = os.path.dirname(os.path.abspath(__file__))


def opt(name, default):
    if name in sys.argv:
        return os.path.abspath(sys.argv[sys.argv.index(name) + 1])
    return default


R = opt("--results", os.path.join(B, "results"))
RF = os.path.join(R, "rerun_fix")
REPORT = opt("--report", os.path.join(B, "..", "Report"))
BRAVA_LOG = os.path.join(R, "brava_retrained", "logs", "eval_seeds.log")


def run(*args):
    out = subprocess.run([sys.executable, *args], cwd=B, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"!! {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def body_after(text, marker):
    """The '&' rows following the line containing `marker`, up to the next blank line."""
    lines = text.splitlines()
    i = next(n for n, l in enumerate(lines) if marker in l)
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        if "&" in l:
            rows.append(re.sub(r"\s+% KADABRA loses.*$", "", l.rstrip()))
    return rows


def splice(table, rows, check):
    path = os.path.join(REPORT, "tables", table)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not check:
            open(path, "w").write("\n".join(rows) + "\n")
        print(f"  {'would write' if check else 'wrote'} body only: tables/{table}")
        return
    src = open(path).read()
    m = re.search(r"(\\midrule\n)(.*?)(\n\s*\\bottomrule)", src, re.S)
    new = src[:m.start(2)] + "\n".join(rows) + src[m.end(2):]
    changed = new != src
    print(f"  {'would change' if check else 'wrote' if changed else 'unchanged'}: tables/{table}"
          if changed or not check else f"  unchanged: tables/{table}")
    if changed and not check:
        open(path, "w").write(new)


def link_dir(tmp, name, patterns):
    d = os.path.join(tmp, name)
    os.makedirs(d)
    for pat in patterns:
        for f in glob.glob(pat):
            os.symlink(f, os.path.join(d, os.path.basename(f)))
    return d


def main():
    check = "--check" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        # --- tables ---
        out = run("summarize_reproduce.py", "tight", os.path.join(RF, "tight"))
        splice("cpp_vs_julia_tight_table.tex", body_after(out, "cpp_vs_julia_tight_table"), check)

        tx = link_dir(tmp, "tightx", [os.path.join(RF, "tight", "tight_julia_*.json"),
                                      os.path.join(RF, "bvk", "bvk_*.json")])
        out = run("summarize_reproduce.py", "tightx", tx)
        splice("tight_accuracy_table.tex", body_after(out, "tight_accuracy_table"), check)

        out = run("summarize_seeds.py", os.path.join(RF, "bvk", "kadabra_seeds"), BRAVA_LOG,
                  os.path.join(RF, "bvk"))
        splice("brava_vs_kadabra_table.tex", body_after(out, "brava_vs_kadabra_table"), check)

        # A.5: the table carries the four graphs run under every allocation; amazon and dblp,
        # run under the repaired one only, appear in the figure.
        out = run("summarize_topk.py", os.path.join(RF, "topk"))
        lines = out.splitlines()
        i = next(n for n, l in enumerate(lines) if "topk_allocation_table" in l)
        blocks, cur = [], []
        for l in lines[i + 1:]:
            if not l.strip():
                break
            if l.strip() == "\\midrule":
                blocks.append(cur); cur = []
            else:
                cur.append(l.rstrip())
        blocks.append(cur)
        keep = [b for b in blocks if not any(g in b[0] for g in ("{amazon}", "{dblp}"))]
        rows = []
        for n, b in enumerate(keep):
            rows += b + (["            \\midrule"] if n < len(keep) - 1 else [])
        splice("topk_allocation_table.tex", rows, check)

        out = run("summarize_cpp_quality.py", R, "--julia-from", "rerun_fix")
        splice("cpp_julia_quality_table.tex", body_after(out, "cpp_julia_quality_table"), check)

        # --- figures ---
        figdir = os.path.join(tmp, "fig")
        run("make_plots.py", "threads", os.path.join(RF, "ts"), figdir)
        bvk = link_dir(tmp, "bvk", [os.path.join(RF, "bvk", "bvk_*.json")])
        os.symlink(os.path.join(RF, "bvk", "kadabra_seeds"), os.path.join(bvk, "kadabra_seeds"))
        os.makedirs(os.path.join(bvk, "logs"))
        os.symlink(BRAVA_LOG, os.path.join(bvk, "logs", "eval_seeds.log"))
        run("make_plots.py", "bvk", bvk, figdir)
        run("make_plots.py", "topk", os.path.join(RF, "topk"), figdir)
        os.makedirs(os.path.join(REPORT, "figures"), exist_ok=True)
        for f in sorted(os.listdir(figdir)):
            dst = os.path.join(REPORT, "figures", f)
            if check:
                print(f"  would write: figures/{f}")
            else:
                shutil.copyfile(os.path.join(figdir, f), dst)
                print(f"  wrote: figures/{f}")


if __name__ == "__main__":
    main()
