# Bavarian: Betweenness Centrality Approximation with Variance-Aware Rademacher Averages

This repository contains the code for the paper *Bavarian: Betweenness
Centrality Approximation with Variance-Aware Rademacher Averages*
([PDF](CousinsEtAl-Bavarian-TKDD23.pdf)), by Cyrus Cousins, Chloe Wohlgemuth,
and Matteo Riondato, appearing in the proceedings of KDD'21, and in ACM
Transactions on Knowledge Discovery from Data.

An [Amherst College Data* Mammoths](https://acdmammoths.github.io) project. This
work was funded, in part, by [NSF award
IIS-2006765](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2006765&HistoricalAwards=false).

## Software Prerequisites

* For the build process: CMake, at least 3.8.0;
* To compile Bavarian and its dependencies: a C++ compiler that supports the
    main features of C++20 and OpenMP; recent clang and g++ satisfy these
    requirements;
* To run the experiments, analyze their results, and generate the figures:
    - Python 3.8 or successive, with the NumPy, Matplotlib, and pandas libraries;
    - LaTeX (Alternatively you can comment out the relative lines in
      `staticres.py`).

We did not test the code on Windows, but we did test it on macOS and various
*NIX flavors (GNU/Linux and FreeBSD).

## Instructions

1. Run the `run_everything.sh` script in this directory. It will first checkout
   the necessary submodules ([NetworKit](https://github.com/networkit/networkit)
   and [FindTBB](https://github.com/justusc/FindTBB)), copy the necessary code
   in the correct locations, and then compile the code, run the exact algorithm
   on the graphs, and finally run Bavarian on each graph with the same set of
   parameters used for the experiments reported in the paper, and finally it
   will generate the figures.
2. Wait :-). Running everything will take a while (potentially a few days,
   depending on your machine). The reasons are that (1) the exact experiments
   take a long time as some graphs are large; and (2) we run every Bavarian
   experiment multiple times for each combination of parameters, and there are
   many combinations of the parameters (sample_size, mc-trials, epsilons, …).
3. Once the script has completed, you can find the figures in `../res/`.

## License

Copyright 2021-2022 Cyrus Cousins and Matteo Riondato

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
