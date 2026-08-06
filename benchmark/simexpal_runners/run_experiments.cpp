#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <omp.h>
#include <nlohmann/json.hpp>

#include "Probabilistic.h"
#include "KendallTau.hpp"
#include <sstream>

#include "CommandLineParser.hpp"

using json = nlohmann::json;
using namespace std;

void print_usage() {
    cerr << "Usage: run_experiments -i <input_file> -o <output_file> -threads <int> -k <int> -delta <float> -epsilon <float> -a <algorithm> -v <implementation_language> [-s <seed>] [-d]" << endl;
    cerr << "  -d: treat graph as directed (optional)" << endl;
}

int main(int argc, char* argv[]) {
    CommandLineParser clp(argc, argv);
    
    const auto input_file = clp.value<std::string>("i", "");
    const auto output_file = clp.value<std::string>("o", "");
    
    if (input_file.empty() || output_file.empty()) {
        cerr << "Error: Input (-i) and output (-o) files are required." << endl;
        print_usage();
        return 1;
    }

    const auto threads = clp.value<int>("threads", 1);
    const auto k = clp.value<int>("k", 0);
    const auto delta = clp.value<double>("delta", 0.1);
    const auto epsilon = clp.value<double>("epsilon", 0.01);
    const auto algorithm = clp.value<std::string>("a", "kadabra");
    const auto version = clp.value<std::string>("v", "cpp");
    const auto directed = clp.value<bool>("d", false);
    if (algorithm != "kadabra") {
        cerr << "Error: C++ version only supports 'kadabra' algorithm." << endl;
        return 1;
    }
    const auto seed = clp.value<int>("s", 0);

    // Set threads
    omp_set_num_threads(threads);

    // Load graph
    auto io_start = chrono::high_resolution_clock::now();
    Probabilistic G(input_file, directed, 0); // verb = 0 (silent)
    auto io_end = chrono::high_resolution_clock::now();
    chrono::duration<double> io_diff = io_end - io_start;
    double io_time = io_diff.count();

    // Run KADABRA and measure time
    auto start = chrono::high_resolution_clock::now();
    
    if (seed != 0) {
        G.set_seed((uint32_t)seed);
    }
    
    G.run((uint32_t) k, delta, epsilon);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    double execution_time = diff.count();

    // Prepare JSON output
    json j;
    j["parameters"]["input_file"] = input_file;
    j["parameters"]["algorithm"] = algorithm;
    j["parameters"]["version"] = version;
    j["parameters"]["threads"] = threads;
    j["parameters"]["k"] = k;
    j["parameters"]["delta"] = delta;
    j["parameters"]["epsilon"] = epsilon;
    j["parameters"]["directed"] = directed;
    j["parameters"]["seed"] = seed;
    j["io_time_seconds"] = io_time;

    j["execution_time_seconds"] = execution_time;
    j["num_samples"] = G.get_n_pairs();

    // Helper to get graph name
    std::string graph_name;
    size_t last_slash = input_file.find_last_of('/');
    std::string filename = (last_slash == std::string::npos) ? input_file : input_file.substr(last_slash + 1);
    size_t last_dot = filename.find_last_of('.');
    if (last_dot != std::string::npos) {
        filename = filename.substr(0, last_dot);
    }
    graph_name = filename;

    std::string gt_path;
    size_t inst_pos = input_file.find("Instances/");
    if (inst_pos != std::string::npos) {
        gt_path = input_file.substr(0, inst_pos + 10) + "ground_truth/test_instances/" + graph_name + "_bet.csv";
    } else {
        gt_path = "Instances/ground_truth/test_instances/" + graph_name + "_bet.csv";
    }

    double tau_overall = NAN;
    double tau_topk = NAN;
    long long overlap = 0;
    double max_ae = NAN;
    double mae = NAN;
    double ndcg_topk = NAN;

    std::ifstream gt_file(gt_path);
    uint32_t nn = G.get_nn();
    
    if (gt_file.is_open()) {
        std::string line;
        std::getline(gt_file, line); // Skip header
        
        std::vector<std::pair<int, double>> exact_entries;
        int shift = -1;
        uint32_t max_node = 0;
        
        while (std::getline(gt_file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string item;
            if (std::getline(ss, item, ',')) {
                int u = std::stoi(item);
                if (std::getline(ss, item, ',')) {
                    double val = std::stod(item);
                    exact_entries.push_back({u, val});
                    if (shift == -1) {
                        shift = (u == 0) ? 1 : 0;
                    }
                    if ((uint32_t)(u + shift) > max_node) {
                        max_node = u + shift;
                    }
                }
            }
        }
        gt_file.close();

        std::vector<double> bt_exact(max_node + 1, 0.0);
        for (const auto& entry : exact_entries) {
            bt_exact[entry.first + shift] = entry.second;
        }

        std::vector<double> bt_approx(max_node + 1, 0.0);
        for (uint32_t v = 0; v < nn; ++v) {
            double cent = G.get_centrality(v);
            if (v <= max_node) {
                bt_approx[v] = cent;
            }
        }

        tau_overall = kendall_tau(bt_exact, bt_approx);

        max_ae = 0.0;
        double sum_ae = 0.0;
        for (int v = 0; v <= (int)max_node; v++) {
            double ae = std::abs(bt_exact[v] - bt_approx[v]);
            if (ae > max_ae) max_ae = ae;
            sum_ae += ae;
        }
        mae = (max_node + 1 > 0) ? sum_ae / (max_node + 1) : 0.0;

        int eval_k = (k == 0) ? 100 : k;
        if (eval_k > (int)max_node + 1) eval_k = max_node + 1;

        std::vector<int> p_exact(max_node + 1);
        std::vector<int> p_approx(max_node + 1);
        for (int i = 0; i <= (int)max_node; i++) {
            p_exact[i] = i;
            p_approx[i] = i;
        }

        std::sort(p_exact.begin(), p_exact.end(), [&](int a, int b) {
            if (bt_exact[a] != bt_exact[b]) return bt_exact[a] > bt_exact[b];
            return a < b;
        });
        std::sort(p_approx.begin(), p_approx.end(), [&](int a, int b) {
            if (bt_approx[a] != bt_approx[b]) return bt_approx[a] > bt_approx[b];
            return a < b;
        });

        std::vector<double> topk_exact_vals(eval_k);
        std::vector<double> topk_approx_vals(eval_k);
        std::vector<int> topk_exact_nodes(eval_k);
        std::vector<int> topk_approx_nodes(eval_k);
        
        for (int i = 0; i < eval_k; i++) {
            int u = p_exact[i];
            topk_exact_vals[i] = bt_exact[u];
            topk_approx_vals[i] = bt_approx[u];
            topk_exact_nodes[i] = u;
            
            int v = p_approx[i];
            topk_approx_nodes[i] = v;
        }
        
        tau_topk = kendall_tau(topk_exact_vals, topk_approx_vals);
        
        std::sort(topk_exact_nodes.begin(), topk_exact_nodes.end());
        std::sort(topk_approx_nodes.begin(), topk_approx_nodes.end());
        std::vector<int> intersection;
        std::set_intersection(topk_exact_nodes.begin(), topk_exact_nodes.end(),
                              topk_approx_nodes.begin(), topk_approx_nodes.end(),
                              std::back_inserter(intersection));
        overlap = intersection.size();

        double dcg = 0.0;
        double idcg = 0.0;
        for (int i = 0; i < eval_k; i++) {
            int u_approx = topk_approx_nodes[i];
            int u_exact = topk_exact_nodes[i];
            dcg += bt_exact[u_approx] / std::log2(i + 2.0);
            idcg += bt_exact[u_exact] / std::log2(i + 2.0);
        }
        ndcg_topk = (idcg > 0.0) ? dcg / idcg : 1.0;
    } else {
        std::cerr << "Ground truth file not found: " << gt_path << std::endl;
    }

    if (!std::isnan(tau_overall)) j["tau_overall"] = tau_overall;
    else j["tau_overall"] = nullptr;
    
    if (!std::isnan(tau_topk)) j["tau_topk"] = tau_topk;
    else j["tau_topk"] = nullptr;
    
    j["overlap_topk"] = overlap;

    if (!std::isnan(max_ae)) j["max_ae"] = max_ae;
    else j["max_ae"] = nullptr;

    if (!std::isnan(mae)) j["mae"] = mae;
    else j["mae"] = nullptr;

    if (!std::isnan(ndcg_topk)) j["ndcg_topk"] = ndcg_topk;
    else j["ndcg_topk"] = nullptr;

    // Write to output file
    ofstream o(output_file);
    if (!o.is_open()) {
        cerr << "Error: Could not open output file " << output_file << endl;
        return 1;
    }
    o << j.dump(4) << endl;
    o.close();

    cout << "Successfully wrote results to " << output_file << endl;
    return 0;
}
