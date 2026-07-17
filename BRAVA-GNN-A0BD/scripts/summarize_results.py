"""Split all_results.csv into separate regular and filtered files.

Reads results/betweenness/all_results.csv and writes:
  - results/betweenness/all_results_regular.csv  (rows whose name does NOT end in _filtered)
  - results/betweenness/all_results_filtered.csv (rows whose name ends in _filtered)
"""
import argparse
import csv
import os
import re
from collections import defaultdict


def aggregate_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row:
            continue
        base_name = re.sub(r'_S\d+', '', row[0])
        groups[base_name].append(row)
        
    aggregated =[]
    for base_name, group_rows in groups.items():
        n_seeds = len(group_rows)
        orig_name = group_rows[0][0]
        
        if n_seeds > 1:
            new_name = re.sub(r'_S\d+', f'_{n_seeds}_seeds', orig_name, count=1)
        else:
            new_name = orig_name
            
        new_row = [new_name]
        for col_idx in range(1, len(group_rows[0])):
            vals =[]
            for r in group_rows:
                if col_idx < len(r):
                    try:
                        vals.append(float(r[col_idx]))
                    except ValueError:
                        pass
            
            if vals:
                new_row.append(f"{sum(vals)/len(vals):.4f}")
            else:
                new_row.append("-")
                
        aggregated.append(new_row)
        
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/betweenness")
    args = parser.parse_args()

    src = os.path.join(args.results_dir, "all_results.csv")
    out_reg = os.path.join(args.results_dir, "all_results_regular.csv")
    out_filt = os.path.join(args.results_dir, "all_results_filtered.csv")

    with open(src) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]

    body = aggregate_seeds(body)

    regular = [r for r in body if not r[0].endswith("_filtered")]
    filtered = [r for r in body if r[0].endswith("_filtered")]

    for path, rows_to_write in [(out_reg, regular), (out_filt, filtered)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_to_write)

    print(f"Wrote {out_reg} ({len(regular)} rows) and {out_filt} ({len(filtered)} rows)")


if __name__ == "__main__":
    main()