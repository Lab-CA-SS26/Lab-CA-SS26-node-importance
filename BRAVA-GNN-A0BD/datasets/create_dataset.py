import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
import os
import argparse
import sys
random.seed(10)

def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data, num_copies, min_adj_size=10000):

    max_nodes = 0
    for g_data in list_data:
        max_nodes = max(max_nodes, g_data[0].number_of_nodes())
    
    adj_size = max(min_adj_size, max_nodes)

    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num),dtype=np.float32)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    for g_data in list_data:

        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1


    if total_num > 0:
        serial_list = [i for i in range(total_num)]
        random.shuffle(serial_list)

        list_graph = reorder_list(list_graph,serial_list)
        list_n_sequence = reorder_list(list_n_sequence,serial_list)
        list_node_num = reorder_list(list_node_num,serial_list)
        cent_mat_tmp = cent_mat[:,np.array(serial_list)]
        cent_mat = cent_mat_tmp

    return list_graph, list_n_sequence, list_node_num, cent_mat


def get_split(source_file,num_train,num_test,num_copies,adj_size,save_path):

    with open(source_file,"rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    print(f"Total number of graphs in {source_file} file:{num_graph}")

    assert num_train+num_test == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train], num_copies=num_copies, min_adj_size=adj_size)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test], num_copies=1, min_adj_size=adj_size)

    with open(save_path+"test.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)


def bundle_real_graphs(names, bundle_name, base_dir="./datasets"):
    """Combine per-graph source pickles from datasets/graphs/ into a single training bundle.

    Writes to datasets/data_splits/{bundle_name}/{centrality}/training.pickle.
    Skips missing graphs; skips writing if output already exists.
    """
    for suffix, folder in [("bet", "betweenness")]:
        save_dir = os.path.join(base_dir, "data_splits", bundle_name, folder)
        out_file = os.path.join(save_dir, "training.pickle")
        if os.path.exists(out_file):
            print(f"[bundle] {bundle_name}/{folder}/training.pickle already exists, skipping.")
            continue
        combined = []
        for name in names:
            src = os.path.join(base_dir, "graphs", f"{name}_{suffix}.pickle")
            if not os.path.exists(src):
                print(f"[bundle] {name}_{suffix}.pickle not found, skipping.")
                continue
            with open(src, "rb") as f:
                combined.extend(pickle.load(f))
        if not combined:
            print(f"[bundle] No data for {bundle_name}/{folder} — skipping.")
            continue
        os.makedirs(save_dir, exist_ok=True)
        lg, lns, lnn, cm = create_dataset(combined, num_copies=1, min_adj_size=adj_size)
        with open(out_file, "wb") as f:
            pickle.dump([lg, lns, lnn, cm], f)
        print(f"[bundle] {bundle_name}/{folder}/training.pickle written ({len(combined)} graphs).")


adj_size = 10000

if __name__ == "__main__":
    num_copies = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Wiki-Vote", "soc-Epinions1", "soc-Slashdot0811","p2p-Gnutella31", "web-Google"])
    # Manually extract --datasets values before argparse to handle names starting with '-'
    # (e.g. negative lat/lon coordinates like -0.3560_117.2740_r195239_n102157)
    _datasets = None
    if '--datasets' in sys.argv:
        _start = sys.argv.index('--datasets')
        _vals, _i = [], _start + 1
        while _i < len(sys.argv) and not sys.argv[_i].startswith('--'):
            _vals.append(sys.argv[_i])
            _i += 1
        _datasets = _vals
        sys.argv = sys.argv[:_start] + sys.argv[_i:]
    args = parser.parse_args()
    if _datasets is not None:
        args.datasets = _datasets

    output_split_dir = "./datasets/data_splits/"
    if not os.path.exists(output_split_dir):
        os.makedirs(output_split_dir)

    for data in args.datasets:
        is_synthetic = data in ["SF", "ER", "GRP"] or data.startswith("HY") or data.startswith("SF_")

        if is_synthetic:
            print(f"Processing {data}...")
            bet_source_file = "./datasets/graphs/"+ data + "_data_bet.pickle"

            cur_num_train = 5
            cur_num_test = 0

            if data.startswith(("HY_", "SF_")):
                try:
                    cur_num_train = int(data.split("_")[1])
                except:
                    pass

            save_path_bet = "./datasets/data_splits/"+data+"/betweenness/"
            if not os.path.exists(save_path_bet): os.makedirs(save_path_bet)

            if not (os.path.exists(save_path_bet+"training.pickle") and os.path.exists(save_path_bet+"test.pickle")):
                if os.path.exists(bet_source_file):
                    get_split(bet_source_file,cur_num_train,cur_num_test,num_copies,adj_size,save_path_bet)
                    print(f" {data} Bet Data split saved.")

        else:
            source_file = "./datasets/graphs/"+data+"_bet.pickle"
            dest_file = os.path.join(output_split_dir, data+"_bet.pickle")

            if os.path.exists(source_file) and not os.path.exists(dest_file):
                print(f"Loading and processing {data} bet graph...")
                with open(source_file,"rb") as fopen:
                    list_data = pickle.load(fopen)
                list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data,num_copies = 1, min_adj_size=adj_size)
                with open(dest_file,"wb") as fopen:
                    pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
                print(f"{data} bet graph saved.")

    print("End.")