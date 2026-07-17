/** @file bavarian.hpp
 * Where the real code is.
 *
 * @author Matteo Riondato
 * @date 2020 04 20
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

#ifndef BAVARIAN_HPP
#define BAVARIAN_HPP

#include <algorithm>
#include <array>
#include <cerrno>
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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <omp.h>
extern char *optarg;
extern int optind;

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/centrality/Bavarian.hpp>
#include <networkit/centrality/ProgrBavarian.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>

namespace Bavarian {
static const std::string VERBOSE_HEADER = "INFO: ";

enum class Sampling : bool { progressive = true, statix = false };

template <Sampling type> void usage(char *cmd);

template <> void usage<Sampling::statix>(char *cmd) {
  std::cout << "usage: " << cmd
            << " [-dfhjos] [-t timeout] [-v[v[...]]] "
               "samples mctrials approx_method delta file"
            << std::endl;
  std::cout << "\t-d: consider the graph as directed" << std::endl;
  std::cout << "\t-f: print additional info at the end of execution"
            << std::endl;
  std::cout << "\t-h: print this help message and exit" << std::endl;
  std::cout << "\t-j: print the output in JSON format" << std::endl;
  std::cout << "\t-o: also compute the approximation quality with a non-MC "
               "approach (when possible)"
            << std::endl;
  std::cout << "\t-s: disable OpenMP parallelism" << std::endl;
  std::cout << "\t-t timeout: run for 'timeout' seconds. 'timeout' must be "
               "positive, and may have a decimal part."
            << std::endl;
  std::cout << "\t-v: print additional info during execution (can be "
               "specified up to 3 times for more verbosity)"
            << std::endl;
  std::cout << std::endl;
  std::cout << "\tapprox_method must be one of 'ab', 'bp', and 'rk'"
            << std::endl;
  std::cout << "\tmctrials and samples must be positive. samples is ignored "
               "when '-m timeout' is specified"
            << std::endl;
  std::cout << "\tdelta must be in (0,1)" << std::endl;
}

template <> void usage<Sampling::progressive>(char *cmd) {
  std::cout << "usage: " << cmd
            << " [-dfhjos] [-v[v[...]]] "
               "iterations epsilon mctrials approx_method delta file"
            << std::endl;
  std::cout << "\t-d: consider the graph as directed" << std::endl;
  std::cout << "\t-f: print additional info at the end of execution"
            << std::endl;
  std::cout << "\t-h: print this help message and exit" << std::endl;
  std::cout << "\t-j: print the output in JSON format" << std::endl;
  std::cout << "\t-m multiplier: use 'multiplier' as the scaling factor for "
               "the sample sizes. 'multiplier' must be greater than 1.0."
            << std::endl;
  std::cout << "\t-o: also compute the approximation quality with a non-MC "
               "approach(when possible)."
            << std::endl;
  std::cout << "\t-s: disable OpenMP parallelism" << std::endl;
  std::cout << "\t-v: print additional info during execution (can be "
               "specified up to 3 times for more verbosity)"
            << std::endl;
  std::cout << std::endl;
  std::cout << "\tapprox_method must be one of 'ab', 'bp', and 'rk'"
            << std::endl;
  std::cout << "\tmctrials and iterations must be positive. iterations is "
               "ignored when '-m muliplier' is specified"
            << std::endl;
  std::cout << "\tdelta and epsilon must be in (0,1)" << std::endl;
}

template <Sampling type> class Settings {
public:
  Settings() {}

  Settings(int argc, char **argv) : graph_file{argv[argc - 1]} {
    int opt;
    std::string options;
    if constexpr (type == Sampling::progressive)
      options = "dfhjm:osv";
    else
      options = "dfhjost:v";
    while ((opt = getopt(argc, argv, options.c_str())) != -1 &&
           !asked_for_help_) {
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
      case 'm':
        errno = 0;
        multiplier_ = std::strtod(optarg, NULL);
        if (errno == ERANGE || multiplier_ <= 1) {
          std::stringstream ss;
          ss << "Passed argument '" << *optarg
             << "' for multiplier not a number greater than 1.0.";
          throw std::out_of_range(ss.str());
        }
        break;
      case 'o':
        nomc_eps_ = true;
        break;
      case 's':
        openmp_ = false;
        break;
      case 't': { // Braces needed because we create a variable
        errno = 0;
        const auto t{std::strtod(optarg, NULL)};
        if (errno == ERANGE || t <= 0) {
          std::stringstream ss;
          ss << "Passed argument '" << *optarg
             << "' for timeout not a positive number.";
          throw std::out_of_range(ss.str());
        }
        if (t >= std::numeric_limits<double>::max() / 1000) {
          std::stringstream ss;
          ss << "Passed argument '" << *optarg
             << "' for timeout too large to be converted to milliseconds.";
          throw std::out_of_range(ss.str());
        }
        timeout_ = std::chrono::milliseconds(
            static_cast<long long int>(std::ceil(t * 1000)));
      } break;
      case 'v':
        loglevel_++;
        break;
      case '?':
        throw std::invalid_argument("illegal option");
      }
    }
    int arg_num; // number of required arguments.
    if constexpr (type == Sampling::progressive) {
      arg_num = 6;
    } else {
      arg_num = 5;
    }
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
      if constexpr (type == Sampling::progressive) {
        if (multiplier_ == 0) {
          errno = 0;
          const auto i{std::strtol(argv[argc - 6], NULL, 10)};
          if (errno == ERANGE || i <= 0) {
            std::stringstream ss;
            ss << "Passed argument '" << argv[argc - 6]
               << "' for "
                  "the number of iterations not a number or not "
                  "positive.";
            throw std::out_of_range(ss.str());
          }
          iterations_ = static_cast<decltype(iterations_)>(i);
        }
        errno = 0;
        epsilon_ = std::strtod(argv[argc - 5], NULL);
        if (errno == ERANGE || epsilon_ <= 0 || epsilon_ >= 1) {
          std::stringstream ss;
          ss << "Passed argument '" << argv[argc - 5]
             << "' for "
                "epsilon not a number or not between 0 and 1.";
          throw std::out_of_range(ss.str());
        }
      } else {
        if (timeout_ == std::chrono::milliseconds::zero()) {
          errno = 0;
          const auto s{std::strtol(argv[argc - 5], NULL, 10)};
          if (errno == ERANGE || s <= 0) {
            std::stringstream ss;
            ss << "Passed argument '" << argv[argc - 5]
               << "' for "
                  "the number of samples not a number or not "
                  "positive.";
            throw std::out_of_range(ss.str());
          }
          samples_ = static_cast<decltype(samples_)>(s);
        }
      }
      errno = 0;
      const auto mcs{std::strtol(argv[argc - 4], NULL, 10)};
      if (errno == ERANGE || mcs <= 0) {
        std::stringstream ss;
        ss << "Passed argument '" << argv[argc - 4]
           << "' for "
              "the number of Monte-Carlo samples not a number or not "
              "positive.";
        throw std::out_of_range(ss.str());
      }
      mctrials_ = static_cast<decltype(mctrials_)>(mcs);
      method_ = std::string(argv[argc - 3]);
      if (!(method_.compare("ab") == 0 || method_.compare("bp") == 0 ||
            method_.compare("rk") == 0)) {
        std::stringstream ss;
        ss << "Passed argument '" << argv[argc - 3]
           << "' not one "
              "of 'ab' or 'bp' or 'rk'";
        throw std::out_of_range(ss.str());
      }
      errno = 0;
      delta_ = std::strtod(argv[argc - 2], NULL);
      if (errno == ERANGE || delta_ <= 0 || delta_ >= 1) {
        std::stringstream ss;
        ss << "Passed argument '" << argv[argc - 2]
           << "' for "
              "delta not a number or not between 0 and 1.";
        throw std::out_of_range(ss.str());
      }
    }
  }

  auto asked_for_help() const { return asked_for_help_; }

  auto delta() const { return delta_; }

  auto directed() const { return directed_; }

  auto epsilon() const { return epsilon_; }

  auto fullprint() const { return fullprint_; }

  const std::string &graph() const { return graph_file; }

  auto iterations() const { return iterations_; }

  auto json() const { return json_; }

  auto loglevel() const { return loglevel_; }

  auto mctrials() const { return mctrials_; }

  const std::string &method() const { return method_; }

  auto multiplier() const { return multiplier_; }

  auto openmp() const { return openmp_; }

  auto nomc_eps() const { return nomc_eps_; }

  auto samples() const { return samples_; }

  auto timeout() const { return timeout_; }

private:
  std::string graph_file{};
  std::string method_{};
  double delta_{0};
  double epsilon_{0};
  double multiplier_{0};
  int loglevel_{0};
  NetworKit::count iterations_{0};
  NetworKit::count mctrials_{0};
  NetworKit::count samples_{0};
  std::chrono::milliseconds timeout_{std::chrono::milliseconds::zero()};
  bool asked_for_help_{false};
  bool directed_{false};
  bool fullprint_{false};
  bool json_{false};
  bool nomc_eps_{false};
  bool openmp_{true};
}; // namespace Bavarian

class Results {
public:
  double epsilon;
  double nomc_epsilon;
  NetworKit::count iterations;
  NetworKit::count max_iterations;
  NetworKit::count samples;
  NetworKit::count nomc_samples;
  NetworKit::count nomc_samples_usereps;
  unsigned long constr_time;
  unsigned long algo_time;
  std::array<double, 5> sdbintvals;
  std::vector<double> scores;
};

template <typename T, Sampling type>
static Results run_and_get_results(T &algo, const Settings<type> &s,
                                   const NetworKit::Graph &graph) {
  const auto algo_time_start{std::chrono::steady_clock::now()};
  algo.run();
  const auto algo_time_end{std::chrono::steady_clock::now()};
  Results r;
  r.algo_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    algo_time_end - algo_time_start)
                    .count();
  if (s.loglevel() > 0)
    std::cerr << "done (" << r.algo_time << " millisecs)" << std::endl
              << VERBOSE_HEADER << "Getting results...";
  r.epsilon = algo.epsilon();
  r.samples = algo.samples();
  if constexpr (type == Sampling::progressive) {
    r.iterations = algo.iterations();
    r.max_iterations = algo.maxiterations();
  }
  r.scores = algo.scores();
  r.sdbintvals = algo.sdbintvals();
  if (s.nomc_eps()) {
    if constexpr (std::is_same<typename T::algo, NetworKit::BA::ABRA>::value)
      // For ABRA, we need to pass the empirical wimpy variance (i.e., the max
      // Euclidean norm)
      r.nomc_epsilon =
          T::algo::nomc_epsilon(graph, s.delta(), r.samples, r.sdbintvals[0]);
    else
      r.nomc_epsilon = T::algo::nomc_epsilon(graph, s.delta(), r.samples);
    if constexpr (std::is_same<typename T::algo, NetworKit::BA::ABRA>::value)
      // For ABRA, we need to pass the empirical wimpy variance (i.e., the max
      // Euclidean norm)
      r.nomc_samples = T::algo::nomc_samplesize(graph, s.delta(), r.epsilon,
                                                r.sdbintvals[0]);
    else
      r.nomc_samples = T::algo::nomc_samplesize(graph, s.delta(), r.epsilon);
    if constexpr (type == Sampling::progressive) {
      if constexpr (std::is_same<typename T::algo, NetworKit::BA::ABRA>::value)
        // For ABRA, we need to pass the empirical wimpy variance (i.e., the max
        // Euclidean norm)
        r.nomc_samples_usereps = T::algo::nomc_samplesize(
            graph, s.delta(), s.epsilon(), r.sdbintvals[0]);
      else
        r.nomc_samples_usereps =
            T::algo::nomc_samplesize(graph, s.delta(), s.epsilon());
    }
  }
  if (s.loglevel() > 0)
    std::cerr << "done" << std::endl;
  return r;
}

template <typename T, Sampling type>
static Results run(const Settings<type> &s, const NetworKit::Graph &graph) {
  Results r;
  if (s.loglevel() > 0)
    std::cerr << VERBOSE_HEADER << "Building algorithm object...";
  auto constr_time_start{std::chrono::steady_clock::now()};
  decltype(constr_time_start) constr_time_end;
  if constexpr (type == Sampling::progressive) {
    if (s.multiplier() != 0) {
      NetworKit::ProgrBavarian<T> algo(graph, s.delta(), s.mctrials(),
                                       s.epsilon(), s.multiplier());
      constr_time_end = std::chrono::steady_clock::now();
      if (s.loglevel() > 0)
        std::cerr << "done" << std::endl
                  << VERBOSE_HEADER << "Running algorithm...";
      r = run_and_get_results(algo, s, graph);
    } else {
      NetworKit::ProgrBavarian<T> algo(graph, s.delta(), s.mctrials(),
                                       s.epsilon(), s.iterations());
      constr_time_end = std::chrono::steady_clock::now();
      if (s.loglevel() > 0)
        std::cerr << "done" << std::endl
                  << VERBOSE_HEADER << "Running algorithm...";
      r = run_and_get_results(algo, s, graph);
    }
  } else {
    if (s.timeout() == std::chrono::seconds::zero()) {
      NetworKit::Bavarian<T> algo(graph, s.delta(), s.mctrials(), s.samples());
      constr_time_end = std::chrono::steady_clock::now();
      if (s.loglevel() > 0)
        std::cerr << "done" << std::endl
                  << VERBOSE_HEADER << "Running algorithm...";
      r = run_and_get_results(algo, s, graph);
    } else {
      NetworKit::Bavarian<T> algo(graph, s.delta(), s.mctrials(), s.timeout());
      constr_time_end = std::chrono::steady_clock::now();
      if (s.loglevel() > 0)
        std::cerr << "done" << std::endl
                  << VERBOSE_HEADER << "Running algorithm...";
      r = run_and_get_results(algo, s, graph);
    }
  }
  r.constr_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      constr_time_end - constr_time_start)
                      .count();
  return r;
}

} // namespace Bavarian

template <Bavarian::Sampling type>
std::ostream &operator<<(std::ostream &os, const Bavarian::Settings<type> &s) {
  std::string c = ",";
  std::string q = "\"";
  if (!s.fullprint())
    return os;
  if (!s.json())
    q = c = "";
  else
    os << "{" << std::endl;
  os << q << "fullprint" << q << "  : " << s.fullprint() << c << std::endl;
  os << q << "graph" << q << "      : " << q << s.graph() << q << c
     << std::endl;
  os << q << "directed" << q << "   : " << s.directed() << c << std::endl;
  os << q << "json" << q << "       : " << s.json() << c << std::endl;
  os << q << "openmp" << q << "     : " << s.openmp() << c << std::endl;
  os << q << "17execut" << q << "   : "
#ifdef BAVARIAN_WITH_EXECUTION
     << 1
#else
     << 0
#endif
     << c << std::endl;
  os << q << "loglevel" << q << "   : " << s.loglevel() << c << std::endl;
  os << q << "mctrials" << q << "   : " << s.mctrials() << c << std::endl;
  if constexpr (type == Bavarian::Sampling::progressive) {
    os << q << "multiplier" << q << " : " << s.multiplier() << c << std::endl;
    os << q << "iterations" << q << " : " << s.iterations() << c << std::endl;
    os << q << "epsilon" << q << "    : " << s.epsilon() << c << std::endl;
  } else {
    os << q << "samples" << q << "    : " << s.samples() << c << std::endl;
    os << q << "timeout" << q << "    : " << s.timeout().count() << c
       << std::endl;
  }
  os << q << "method" << q << "     : " << q << s.method() << q << c
     << std::endl;
  os << q << "delta" << q << "      : " << s.delta() << c << std::endl;
  os << q << "nomc_eps" << q << "   : " << s.nomc_eps() << std::endl;
  if (s.json())
    os << "}";
  return os;
}

template <Bavarian::Sampling type> int doMain(int argc, char **argv) {
  const auto start{
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
  Bavarian::Settings<type> s;
  try {
    s = Bavarian::Settings<type>(argc, argv);
  } catch (std::invalid_argument &e) {
    std::cerr << "Error parsing the command line: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::out_of_range &e) {
    std::cerr << "Error parsing the command line: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (s.asked_for_help()) {
    Bavarian::usage<type>(*argv);
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
    std::cerr << Bavarian::VERBOSE_HEADER << "Reading graph from " << s.graph()
              << "...";
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
      std::cerr << Bavarian::VERBOSE_HEADER
                << "Disabling OpenMP parallelism...";
    omp_set_num_threads(1);
    if (s.loglevel() > 0)
      std::cerr << "done" << std::endl;
  }
  Bavarian::Results res;
  if (s.method().compare("ab") == 0) {
    res = Bavarian::run<NetworKit::BA::ABRA, type>(s, graph);
  } else if (s.method().compare("bp") == 0) {
    res = Bavarian::run<NetworKit::BA::BP, type>(s, graph);
  } else if (s.method().compare("rk") == 0) {
    res = Bavarian::run<NetworKit::BA::RK, type>(s, graph);
  } else {
    std::cerr << "Reached code supposed to be unreacheable" << std::endl;
    return EXIT_FAILURE;
  }
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
      if constexpr (type == Bavarian::Sampling::progressive)
        std::cout << "\t\"algo\" : \"Bavarian-Progressive\"," << std::endl;
      else
        std::cout << "\t\"algo\" : \"Bavarian-Static\"," << std::endl;
      std::cout << "\t\"date\" : \"" << timestr << "\"," << std::endl;
      std::cout << "\t\"settings\" : ";
      std::cout << s << "," << std::endl;
      std::cout << "\t\"total_time\"  : " << (res.constr_time + res.algo_time)
                << c << std::endl;
      std::cout << "\t\"constr_time\" : " << res.constr_time << c << std::endl;
      std::cout << "\t\"algo_time\"   : " << res.algo_time << c << std::endl;
      std::cout << "\t\"properties\"  : {" << std::endl;
      std::cout << "\t\t\"nodes\" : " << graph.numberOfNodes() << c
                << std::endl;
      std::cout << "\t\t\"edges\" : " << graph.numberOfEdges() << std::endl;
      std::cout << "\t}," << std::endl;
      std::cout << "\t\"results\" : {" << std::endl;
      std::cout << "\t\t\"epsilon\"    : " << res.epsilon << c << std::endl;
      if (s.nomc_eps()) {
        std::cout << "\t\t\"nomc_epsilon\" : " << res.nomc_epsilon << c
                  << std::endl;
        std::cout << "\t\t\"nomc_samples\" : " << res.nomc_samples << c
                  << std::endl;
        if constexpr (type == Bavarian::Sampling::progressive)
          std::cout << "\t\t\"nomc_samples_usereps\" : "
                    << res.nomc_samples_usereps << c << std::endl;
      }
      std::cout << "\t\t\"ewimpy\"     : " << res.sdbintvals[0] << c
                << std::endl;
      std::cout << "\t\t\"wimpybound\" : " << res.sdbintvals[1] << c
                << std::endl;
      std::cout << "\t\t\"mcera\"      : " << res.sdbintvals[2] << c
                << std::endl;
      std::cout << "\t\t\"erabound\"   : " << res.sdbintvals[3] << c
                << std::endl;
      std::cout << "\t\t\"rabound\"    : " << res.sdbintvals[4] << c
                << std::endl;
      std::cout << "\t\t\"samples\"    : " << res.samples << c << std::endl;
      if constexpr (type == Bavarian::Sampling::progressive) {
        std::cout << "\t\t\"iterations\" : " << res.iterations << c
                  << std::endl;
        std::cout << "\t\t\"max_iterations\" : " << res.max_iterations << c
                  << std::endl;
      }
      std::cout << "\t\t\"approximations\" : {" << std::endl;
    } else {
      std::cout << "# Bavarian" << std::endl << std::endl;
      std::cout << "date : " << timestr << std::endl;
      std::cout << std::endl << "## Settings" << std::endl << std::endl;
      std::cout << s;
      std::cout << std::endl << "## Properties" << std::endl << std::endl;
      std::cout << "Nodes : " << graph.numberOfNodes() << std::endl;
      std::cout << "Edges : " << graph.numberOfEdges() << std::endl;
      std::cout << std::endl << "## Runtime" << std::endl << std::endl;
      std::cout << "total time  : " << (res.constr_time + res.algo_time)
                << " millisecs" << std::endl;
      std::cout << "constr time : " << res.constr_time << " millisecs"
                << std::endl;
      std::cout << "algo time   : " << res.algo_time << " millisecs"
                << std::endl;
      std::cout << std::endl << "## Results" << std::endl << std::endl;
      std::cout << "epsilon      : " << res.epsilon << std::endl;
      if (s.nomc_eps())
        std::cout << "nomc_epsilon : " << res.nomc_epsilon << std::endl;
      std::cout << "ewimpy       : " << res.sdbintvals[0] << std::endl;
      std::cout << "wimpybound   : " << res.sdbintvals[1] << std::endl;
      std::cout << "mcera        : " << res.sdbintvals[2] << std::endl;
      std::cout << "erabound     : " << res.sdbintvals[3] << std::endl;
      std::cout << "rabound      : " << res.sdbintvals[4] << std::endl;
      std::cout << "samples      : " << res.samples << std::endl;
      if constexpr (type == Bavarian::Sampling::progressive)
        std::cout << "iterations    : " << res.iterations << std::endl;
      std::cout << std::endl << "### Approximations" << std::endl << std::endl;
    }
  }
  std::stringstream ss;
  for (const auto &el : reader.getNodeMap())
    ss << q << el.first << q << " : " << res.scores.at(el.second) << c
       << std::endl;
  if (s.json()) {
    ss.seekp(-2, ss.cur);   // remove last two chars (comma and newline)
    ss << " " << std::endl; // must put two chars in place.
  }
  std::cout << ss.str();
  if (s.json()) {
    if (s.fullprint()) {
      std::cout << "}" << std::endl; // closing "approximations"
      std::cout << "}" << std::endl; // closing "results"
    }
    std::cout << "}" << std::endl; // closing the whole thing
  }
  return EXIT_SUCCESS;
}

#endif // BAVARIAN_HPP
