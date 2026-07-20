#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <omp.h>
#include <nlohmann/json.hpp>

#include "Probabilistic.h"

#include "CommandLineParser.hpp"

using json = nlohmann::json;
using namespace std;

void print_usage() {
    cerr << "Usage: run_experiments -i <input_file> -o <output_file> -threads <int> -k <int> -delta <float> -epsilon <float> -a <algorithm> -v <implementation_language> [-d]" << endl;
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

    // Set threads
    omp_set_num_threads(threads);

    // Load graph
    Probabilistic G(input_file, directed, 0); // verb = 0 (silent)

    // Run KADABRA and measure time
    auto start = chrono::high_resolution_clock::now();
    
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

    j["execution_time_seconds"] = execution_time;
    j["num_samples"] = G.get_n_pairs();

    // Gather centralities
    uint32_t nn = G.get_nn();
    json centralities = json::object();
    for (uint32_t v = 0; v < nn; ++v) {
        double cent = G.get_centrality(v);
        if (cent > 0.0) {
            centralities[to_string(v)] = cent;
        }
    }
    j["centralities"] = centralities;

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
