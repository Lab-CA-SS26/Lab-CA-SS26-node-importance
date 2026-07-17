/** @file exact.cpp
 * Driver for the exact computation of betweenness centrality.
 *
 * @author Matteo Riondato
 * @date 2020 04 25
 *
 * @copyright
 *
 * Copyright 2020 Matteo Riondato <rionda@acm.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <unistd.h>
extern int optind;

#include <omp.h>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/centrality/Betweenness.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>

static const std::string VERBOSE_HEADER{"INFO: "};

void usage(char *cmd) {
  std::cout << "usage: " << cmd << " [-dfhjs] [-v[v[...]]] file" << std::endl;
  std::cout << "\t-d: consider the graph as directed" << std::endl;
  std::cout << "\t-f: print additional info at the end of execution"
            << std::endl;
  std::cout << "\t-h: print this help message and exit" << std::endl;
  std::cout << "\t-j: print the output in JSON format" << std::endl;
  std::cout << "\t-s: disable OpenMP parallelism" << std::endl;
  std::cout << "\t-v: print additional info during execution (can be "
               "specified up to 3 times for more verbosity)"
            << std::endl;
}

class Settings {
public:
  Settings() {}

  Settings(int argc, char **argv) : graph_file{argv[argc - 1]} {
    int opt;
    while ((opt = getopt(argc, argv, "dfhjsv")) != -1 && !asked_for_help_) {
      switch (opt) {
      case 'd':
        directed_ = true;
        break;
      case 'f':
        fullprint_ = true;
        break;
      case 'h':
        asked_for_help_ = true;
        break;
      case 'j':
        json_ = true;
        break;
      case 's':
        openmp_ = false;
        break;
      case 'v':
        loglevel_++;
        break;
      case '?':
        throw std::invalid_argument("illegal option");
      }
    }
    constexpr int arg_num{1}; // number of required arguments.
    if (!asked_for_help_) {
      if (optind != argc - arg_num) {
        std::stringstream ss;
        ss << "Wrong number of arguments (" << (argc - optind) << " instead of "
           << arg_num << ")";
        throw std::invalid_argument(ss.str());
      }
      if (!std::filesystem::exists(graph_file) ||
          !std::filesystem::is_regular_file(graph_file) ||
          std::filesystem::is_empty(graph_file)) {
        std::stringstream ss;
        ss << "graph file '" << graph_file << "' does not exist or "
           << "can't be read or is empty.";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  bool asked_for_help() const { return asked_for_help_; }

  bool directed() const { return directed_; }

  bool fullprint() const { return fullprint_; }

  const std::string &graph() const { return graph_file; }

  bool json() const { return json_; }

  int loglevel() const { return loglevel_; }

  bool openmp() const { return openmp_; }

private:
  std::string graph_file{};
  int loglevel_{0};
  bool asked_for_help_{false};
  bool directed_{false};
  bool fullprint_{false};
  bool json_{false};
  bool openmp_{true};
};

std::ostream &operator<<(std::ostream &os, const Settings &s) {
  std::string c{","};
  std::string q{"\""};
  if (!s.fullprint())
    return os;
  if (!s.json())
    q = c = "";
  else
    os << "{" << std::endl;
  os << q << "fullprint" << q << " : " << s.fullprint() << c << std::endl;
  os << q << "graph" << q << "     : " << q << s.graph() << q << c << std::endl;
  os << q << "directed" << q << "  : " << s.directed() << c << std::endl;
  os << q << "json" << q << "      : " << s.json() << c << std::endl;
  os << q << "openmp" << q << "    : " << s.openmp() << c << std::endl;
  os << q << "loglevel" << q << "  : " << s.loglevel() << std::endl;
  if (s.json())
    os << "}";
  return os;
}

int main(int argc, char **argv) {
  const auto start{
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
  Settings s;
  try {
    s = Settings(argc, argv);
  } catch (std::invalid_argument &e) {
    std::cerr << "Error parsing the command line: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::out_of_range &e) {
    std::cerr << "Error parsing the command line: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (s.asked_for_help()) {
    usage(*argv);
    return EXIT_SUCCESS;
  }
  switch (s.loglevel()) {
  case 0:
    Aux::Log::setLogLevel("WARN");
    break;
  case 1:
    Aux::Log::setLogLevel("WARN");
    break;
  case 2:
    Aux::Log::setLogLevel("INFO");
    break;
  case 3:
    Aux::Log::setLogLevel("DEBUG");
    break;
  default:
    Aux::Log::setLogLevel("TRACE");
    break;
  }
  if (s.loglevel() > 0)
    std::cerr << VERBOSE_HEADER << "Reading graph from " << s.graph() << "...";
  NetworKit::EdgeListReader reader('\t', 1, "#", false, s.directed());
  NetworKit::Graph graph;
  const auto graphread_start{std::chrono::steady_clock::now()};
  try {
    graph = reader.read(s.graph());
  } catch (std::runtime_error &e) {
    std::cerr << "Error reading the graph file: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  const auto graphread_end{std::chrono::steady_clock::now()};
  const auto graphread{std::chrono::duration_cast<std::chrono::milliseconds>(
                           graphread_end - graphread_start)
                           .count()};
  if (s.loglevel() > 0)
    std::cerr << "done (" << graphread << " millisecs)" << std::endl;
  if (!s.openmp()) {
    if (s.loglevel() > 0)
      std::cerr << VERBOSE_HEADER << "Disabling OpenMP parallelism...";
    omp_set_num_threads(1);
    if (s.loglevel() > 0)
      std::cerr << "done" << std::endl;
  }
  if (s.loglevel() > 0)
    std::cerr << VERBOSE_HEADER << "Building algorithm object...";
  const auto constr_time_start{std::chrono::steady_clock::now()};
  // We use 'false' for normalized (currently the default, but we prefer to be
  // explicit), because the normalization done by NetworKit is not the same as
  // the one we do, as it considers a different number of pairs.
  NetworKit::Betweenness algo(graph, false);
  const auto constr_time_end{std::chrono::steady_clock::now()};
  const auto constr_time{std::chrono::duration_cast<std::chrono::milliseconds>(
                             constr_time_end - constr_time_start)
                             .count()};
  if (s.loglevel() > 0)
    std::cerr << "done (" << constr_time << " millisecs)" << std::endl
              << VERBOSE_HEADER << "Running algorithm...";
  const auto algo_time_start{std::chrono::steady_clock::now()};
  algo.run();
  const auto algo_time_end{std::chrono::steady_clock::now()};
  const auto algo_time{std::chrono::duration_cast<std::chrono::milliseconds>(
                           algo_time_end - algo_time_start)
                           .count()};
  if (s.loglevel() > 0)
    std::cerr << "done (" << algo_time << " millisecs)" << std::endl
              << VERBOSE_HEADER << "Getting scores...";
  const auto scores{algo.scores()};
  if (s.loglevel() > 0)
    std::cerr << "done" << std::endl;
  std::cerr.flush();
  char timestr[100];
  std::strftime(timestr, sizeof(timestr), "%F %T", std::localtime(&start));
  std::string c{","};
  std::string q{"\""};
  if (!s.json())
    q = c = "";
  else
    std::cout << "{" << std::endl; // global json start
  if (s.fullprint()) {
    if (s.json()) {
      std::cout << "\t\"algo\" : \"Exact\"," << std::endl;
      std::cout << "\t\"date\" : \"" << timestr << "\"," << std::endl;
      std::cout << "\t\"settings\" : ";
      std::cout << s << "," << std::endl;
      std::cout << "\t\"total_time\" : " << (constr_time + algo_time) << c
                << std::endl;
      std::cout << "\t\"constr_time\" : " << constr_time << c << std::endl;
      std::cout << "\t\"algo_time\" : " << algo_time << c << std::endl;
      std::cout << "\t\"properties\" : {" << std::endl;
      std::cout << "\t\t\"nodes\" : " << graph.numberOfNodes() << c
                << std::endl;
      std::cout << "\t\t\"edges\" : " << graph.numberOfEdges() << std::endl;
      std::cout << "\t}," << std::endl;
      std::cout << "\t\"results\" : {" << std::endl;
    } else {
      std::cout << "# Exact" << std::endl << std::endl;
      std::cout << "date : " << timestr << std::endl;
      std::cout << std::endl << "## Settings" << std::endl << std::endl;
      std::cout << s;
      std::cout << std::endl << "## Properties" << std::endl << std::endl;
      std::cout << "Nodes : " << graph.numberOfNodes() << std::endl;
      std::cout << "Edges : " << graph.numberOfEdges() << std::endl;
      std::cout << std::endl << "## Runtime" << std::endl << std::endl;
      std::cout << "total time  : " << (constr_time + algo_time) << " millisecs"
                << std::endl;
      std::cout << "constr time : " << constr_time << " millisecs" << std::endl;
      std::cout << "algo time   : " << algo_time << " millisecs" << std::endl;
      std::cout << std::endl << "## Results" << std::endl << std::endl;
    }
  }
  std::stringstream ss;
  const auto denom{graph.numberOfNodes() * (graph.numberOfNodes() - 1)};
  for (const auto &el : reader.getNodeMap())
    ss << q << el.first << q << " : " << scores.at(el.second) / denom << c
       << std::endl;
  if (s.json()) {
    ss.seekp(-2, ss.cur);   // remove last two chars (comma and newline)
    ss << " " << std::endl; // must put two chars in place.
  }
  std::cout << ss.str();
  if (s.json()) {
    if (s.fullprint()) {
      std::cout << "}" << std::endl; // closing "results"
    }
    std::cout << "}" << std::endl; // closing the whole thing
  }
  return EXIT_SUCCESS;
}
