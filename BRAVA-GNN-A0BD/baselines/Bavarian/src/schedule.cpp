/** @file schedule.cpp
 * Utility to get information about the sample schedule for the progressive
 * sampling algorithm.
 *
 * @author Matteo Riondato
 * @date 2020 04 24j
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/centrality/ProgrBavarianSchedule.hpp>

#include <unistd.h>
extern char *optarg;
extern int optind;
extern int optopt;
extern int opterr;
extern int optreset;

static constexpr double DEFAULT_TOLERANCE{0.02};

void usage(char *cmd) {
  std::cout << "usage: " << cmd
            << " [-f | -j | -l] [-h] [-m multiplier] "
               "[-t tolerance] [-v[v[...]]] vertexDiameterBound iterations "
               "mcsamples[,mcsamples2[,...]] epsilon delta"
            << std::endl;
  std::cout << "\t-f: only print the first sample size (only one of '-f' "
               "'-j', and '-l' can be specified)"
            << std ::endl;
  std::cout << "\t-l: only print the last sample size (only one of '-f' "
               "'-j', and '-l' can be specified)"
            << std ::endl;
  std::cout << "\t-j: print the whole schedule in JSON format (only one of "
               "'-f' '-j', and '-l' can be specified)"
            << std::endl;
  std::cout << "\t-h: print this help message and exit" << std::endl;
  std::cout << "\t-m multiplier: use 'multiplier' as the scaling factor for "
               "the sample sizes. 'multiplier' must be greater than 1.0."
            << std::endl;
  std::cout
      << "\t-t tolerance: Use 'tolerance' as the relative tolerance to "
         "stop the binary search(es) for the first and/or last sample size(s). "
         "'tolerance' must be in (0,1). The default value for the tolerance, "
         "when this option is not specified, is "
      << DEFAULT_TOLERANCE << std::endl;
  std::cout << "\t-v: print additional info during execution (can be "
               "specified up to 3 times for more verbosity)"
            << std::endl;
  std::cout << std::endl;
  std::cout
      << "\tvertexDiameterBound, iterations, and mcsamples must be "
         "positive. vertexDiameterBound is ignored when '-f' is specified. "
         "iterations is ignored when '-m multiplier' is specified."
      << std::endl;
  std::cout << "\t epsilon and delta must be in (0,1)" << std::endl;
}

class Settings {
public:
  Settings() {}

  Settings(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "fhjlm:t:v")) != -1 && !asked_for_help_) {
      switch (opt) {
      case 'f':
        first_ = true;
        break;
      case 'h':
        asked_for_help_ = true;
        break;
      case 'j':
        json_ = true;
        break;
      case 'l':
        last_ = true;
        break;
      case 'm':
        errno = 0;
        multiplier_ = std::strtod(optarg, NULL);
        if (errno == ERANGE || multiplier_ <= 1) {
          std::stringstream ss;
          ss << "Passed argument '" << *optarg
             << "' for "
                "multiplier not a number greater than 1.0.";
          throw std::out_of_range(ss.str());
        }
        break;
      case 't':
        errno = 0;
        tolerance_ = std::strtod(optarg, NULL);
        if (errno == ERANGE || tolerance_ <= 1) {
          std::stringstream ss;
          ss << "Passed argument '" << *optarg
             << "' for "
                "tolerance not a number greater than 1.0.";
          throw std::out_of_range(ss.str());
        }
        break;
      case 'v':
        loglevel_++;
        break;
      case '?':
        throw std::invalid_argument("illegal option");
      }
    }
    constexpr int arg_num{5}; // number of required arguments.
    if (!asked_for_help_) {
      if ((first_ && (last_ || json_)) || (last_ && json_))
        throw std::invalid_argument("Only one of '-f', '-j', and '-l' "
                                    "can be specified");
      if (optind != argc - arg_num) {
        std::stringstream ss;
        ss << "Wrong number of arguments (" << (argc - optind) << " instead of "
           << arg_num << ")";
        throw std::invalid_argument(ss.str());
      }
      if (!first_) {
        errno = 0;
        const auto vd{std::strtol(argv[argc - 5], NULL, 10)};
        if (errno == ERANGE || vd <= 0) {
          std::stringstream ss;
          ss << "Passed argument '" << argv[argc - 5]
             << "' for "
                "the vertex diameter bound not a positive number.";
          throw std::out_of_range(ss.str());
        }
        vdbound_ = static_cast<NetworKit::count>(vd);
      }
      if (multiplier_ == 0) {
        errno = 0;
        const auto its{std::strtol(argv[argc - 4], NULL, 10)};
        if (errno == ERANGE || its <= 0) {
          std::stringstream ss;
          ss << "Passed argument '" << argv[argc - 4]
             << "' for "
                "the number of iterations not a positive number.";
          throw std::out_of_range(ss.str());
        }
        iterations_ = static_cast<NetworKit::count>(its);
      }
      auto token = std::strtok(argv[argc - 3], ",");
      do {
        errno = 0;
        const auto v{std::strtol(token, NULL, 10)};
        if (errno == ERANGE || v <= 0) {
          std::stringstream ss;
          ss << "Passed argument '" << argv[argc - 3]
             << "' for "
                "the number of Monte-Carlo samples not a "
                "(comma-separated list of) positive number(s).";
          throw std::out_of_range(ss.str());
        }
        mcsamples_.emplace(static_cast<NetworKit::count>(v));
        token = std::strtok(NULL, ",");
      } while (token != NULL);
      errno = 0;
      epsilon_ = std::strtod(argv[argc - 2], NULL);
      if (errno == ERANGE || epsilon_ <= 0 || epsilon_ >= 1) {
        std::stringstream ss;
        ss << "Passed argument '" << argv[argc - 1]
           << "' for "
              "epsilon not a number or not between 0 and 1.";
        throw std::out_of_range(ss.str());
      }
      errno = 0;
      delta_ = std::strtod(argv[argc - 1], NULL);
      if (errno == ERANGE || delta_ <= 0 || delta_ >= 1) {
        std::stringstream ss;
        ss << "Passed argument '" << argv[argc - 1]
           << "' for "
              "delta not a number or not between 0 and 1.";
        throw std::out_of_range(ss.str());
      }
    }
  }

  auto asked_for_help() const { return asked_for_help_; }

  auto delta() const { return delta_; }

  auto epsilon() const { return epsilon_; }

  auto first() const { return first_; }

  auto iterations() const { return iterations_; }

  auto json() const { return json_; }

  auto last() const { return last_; }

  auto loglevel() const { return loglevel_; }

  auto mcsamples() const { return mcsamples_; }

  auto multiplier() const { return multiplier_; }

  auto tolerance() const { return tolerance_; }

  auto vdbound() const { return vdbound_; }

private:
  bool asked_for_help_{false};
  bool first_{false};
  bool last_{false};
  bool json_{false};
  double delta_{0};
  double epsilon_{1.0};
  double multiplier_{0};
  double tolerance_{DEFAULT_TOLERANCE};
  int loglevel_{0};
  NetworKit::count iterations_{0};
  NetworKit::count vdbound_{0};
  std::set<NetworKit::count> mcsamples_;
};

int main(int argc, char **argv) {
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

  if (s.asked_for_help()) {
    usage(*argv);
    return EXIT_SUCCESS;
  }

  std::map<int, std::vector<NetworKit::count>> scheds;
  NetworKit::ProgrBavarianSchedule sched;
  for (const auto mcs : s.mcsamples()) {
    if (s.multiplier() == 0)
      sched = NetworKit::ProgrBavarianSchedule(s.vdbound(), mcs, s.epsilon(),
                                               s.delta(), s.iterations(),
                                               s.tolerance());
    else
      sched = NetworKit::ProgrBavarianSchedule(s.vdbound(), mcs, s.epsilon(),
                                               s.delta(), s.multiplier(),
                                               s.tolerance());
    if (s.first()) {
      scheds.emplace(mcs, std::vector<NetworKit::count>(1, sched.first()));
    } else if (s.last()) {
      scheds.emplace(mcs, std::vector<NetworKit::count>(1, sched.last()));
    } else {
      std::vector<NetworKit::count> szs;
      for (szs.push_back(sched.next()); szs.back() < sched.last();
           szs.push_back(sched.next())) {
        // empty loop
      }
      if (szs.size() != sched.maxiterations()) {
        std::cerr << "The number of iterations (" << szs.size()
                  << ") is different than what was expected ("
                  << sched.maxiterations() << ")!" << std::endl;
        return EXIT_FAILURE;
      }
      scheds.emplace(mcs, std::move(szs));
    }
  }

  std::string c = ",";
  std::string q = "\"";
  if (!s.json())
    q = c = "";
  else
    std::cout << "{" << std::endl; // global json start
  std::stringstream ss;
  for (const auto &p : scheds) {
    ss << p.first << ": ";
    if (s.json())
      ss << " [ ";
    std::for_each(p.second.cbegin(), p.second.cend(),
                  [&ss](const NetworKit::count p) { ss << p << ", "; });
    ss.seekp(-2, ss.cur); // remove last two chars (comma and space)
    if (s.json())
      ss << "],";
    else
      ss << " "; // must put two chars in place to compensate deletion
                 // (one being the following endl)
    ss << std::endl;
  }
  if (s.json()) {
    ss.seekp(-2, ss.cur);          // remove last two chars (comma and newline)
    ss << " " << std::endl;        // must put two chars in place.
    std::cout << "}" << std::endl; // closing the whole thing
  }
  std::cout << ss.str();
  return EXIT_SUCCESS;
}
