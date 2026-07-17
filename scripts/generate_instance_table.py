import os
import glob

def process_graph_file(filepath):
    nodes = set()
    edges = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('%') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
                edges += 1
    return len(nodes), edges

def main():
    dirs = ['Instances/ABCDE', 'Instances/TestInstances']
    results = []
    
    for d in dirs:
        for filepath in sorted(glob.glob(os.path.join(d, '*.txt'))):
            if '-score.txt' in filepath:
                continue
            
            filename = os.path.basename(filepath)
            instance_name = os.path.splitext(filename)[0]
            print(f"Processing {instance_name}...")
            
            n, m = process_graph_file(filepath)
            results.append((instance_name, n, m))
            
    # sort by number of nodes
    results.sort(key=lambda x: x[1])
    
    tex = []
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(r"instance & $n$ & $m$ & $m/n$ \\")
    tex.append(r"\midrule")
    
    for instance, n, m in results:
        density = m / n if n > 0 else 0
        # Format with thin spaces for thousands
        n_str = f"{n:,}".replace(",", r"\,")
        m_str = f"{m:,}".replace(",", r"\,")
        # Escape underscores in instance names for LaTeX
        clean_name = instance.replace("_", r"\_")
        tex.append(f"{clean_name} & {n_str} & {m_str} & {density:.2f} \\\\")
        
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    
    output_path = "Report/tables/instance_table_new.tex"
    with open(output_path, 'w') as f:
        f.write("\n".join(tex) + "\n")
        
    print(f"Table successfully written to {output_path}")

if __name__ == "__main__":
    main()
