import re, sys, glob, os
from collections import defaultdict
rows = defaultdict(dict)   # (graph, seed, k) -> {variant: stop}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "out_*_s*.txt"))):
    base = os.path.basename(f)[4:-4]
    graph, seed = base.rsplit("_s", 1)
    k = None
    for line in open(f):
        m = re.match(r"#+ k = (\d+)", line)
        if m: k = int(m.group(1)); continue
        m = re.match(r"\s+(code|paper|cpp)\s+(\d+) samples", line)
        if m and k is not None:
            rows[(graph, int(seed), k)][m.group(1)] = int(m.group(2))
        m = re.match(r"\s+(code|paper|cpp)\s+does NOT stop", line)
        if m and k is not None:
            rows[(graph, int(seed), k)][m.group(1)] = None

graphs = sorted({g for g, _, _ in rows})
ks = sorted({k for _, _, k in rows})
print(f"{'graph':<17}{'k':>5} | " + "  ".join(f"{v+'/code':>12}" for v in ("paper","cpp")) + f"  {'code/k=0':>10}  {'paper/k=0':>10}")
print("-"*80)
for g in graphs:
    seeds = sorted({s for gg,s,_ in rows if gg==g})
    for k in ks:
        cells=[]
        for v in ("paper","cpp"):
            r=[rows[(g,s,k)][v]/rows[(g,s,k)]["code"] for s in seeds
               if (g,s,k) in rows and rows[(g,s,k)].get(v) and rows[(g,s,k)].get("code")]
            cells.append(f"{sum(r)/len(r):.3f}" if r else "--")
        base=[]; pbase=[]
        for s in seeds:
            z=rows.get((g,s,0),{}).get("code")
            if z and (g,s,k) in rows:
                if rows[(g,s,k)].get("code"): base.append(rows[(g,s,k)]["code"]/z)
                if rows[(g,s,k)].get("paper"): pbase.append(rows[(g,s,k)]["paper"]/z)
        b = f"{sum(base)/len(base):.3f}" if base else "--"
        pb= f"{sum(pbase)/len(pbase):.3f}" if pbase else "--"
        print(f"{g:<17}{k:>5} | {cells[0]:>12}  {cells[1]:>12}  {b:>10}  {pb:>10}")
    print()
