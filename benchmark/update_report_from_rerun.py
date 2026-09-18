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
  Report/figures/thread_scaling.{pdf,png}      make_plots.py threads          rerun_fix/ts
  Report/figures/brava_vs_kadabra.{pdf,png}    make_plots.py bvk              rerun_fix/bvk (+ kadabra_seeds, BRAVA log)

Usage:  python3 update_report_from_rerun.py [--check]
        --check  only report which files would change
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

B = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(B, "results")
RF = os.path.join(R, "rerun_fix")
REPORT = os.path.join(B, "..", "Report")
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

        # --- figures ---
        figdir = os.path.join(tmp, "fig")
        run("make_plots.py", "threads", os.path.join(RF, "ts"), figdir)
        bvk = link_dir(tmp, "bvk", [os.path.join(RF, "bvk", "bvk_*.json")])
        os.symlink(os.path.join(RF, "bvk", "kadabra_seeds"), os.path.join(bvk, "kadabra_seeds"))
        os.makedirs(os.path.join(bvk, "logs"))
        os.symlink(BRAVA_LOG, os.path.join(bvk, "logs", "eval_seeds.log"))
        run("make_plots.py", "bvk", bvk, figdir)
        for f in sorted(os.listdir(figdir)):
            dst = os.path.join(REPORT, "figures", f)
            if check:
                print(f"  would write: figures/{f}")
            else:
                shutil.copyfile(os.path.join(figdir, f), dst)
                print(f"  wrote: figures/{f}")


if __name__ == "__main__":
    main()
