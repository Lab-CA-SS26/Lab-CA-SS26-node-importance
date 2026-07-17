/*
 * ProgrBavarian.hpp
 *
 * Created on: 20200425
 * Author: Matteo Riondato
 *
 * Copyright 2020 Matteo Riondato and Cyrus Cousins
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef NETWORKIT_CENTRALITY_PROGRBAVARIAN_HPP_
#define NETWORKIT_CENTRALITY_PROGRBAVARIAN_HPP_

#include <networkit/Globals.hpp>
#include <networkit/centrality/BavarianBA.hpp>
#include <networkit/centrality/Centrality.hpp>
#include <networkit/centrality/ProgrBavarianSchedule.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/graph/Graph.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace NetworKit {

template <typename BCAlg>
class ProgrBavarian : public Centrality {
public:
    using algo = BCAlg;

    ProgrBavarian(const Graph &G, const double delta, const count mcsamples, const double epsilon,
                  const double multiplier, const double tolerance = 0.02)
        : Centrality{G, true}, eps{epsilon}, delta{delta}, mcs{mcsamples},
          sched{G, mcsamples, epsilon, delta, multiplier, tolerance} {}

    ProgrBavarian(const Graph &G, const double delta, const count mcsamples, const double epsilon,
                  const count iterations, const double tolerance = 0.02)
        : Centrality{G, true}, eps{epsilon}, delta{delta}, mcs{mcsamples},
          sched{G, mcsamples, epsilon, delta, iterations, tolerance} {}

    /**
     * @return the computed approximation quality in the last run (or -1 if we
     * never run).
     */
    double epsilon() const { return eps; }

    /**
     * @return the number of iterations in the last run (or 0 if we never run).
     */
    count iterations() const { return iters; }

    /**
     * @return the maximum number of iterations in the last run.
     */
    count maxiterations() const { return sched.maxiterations(); }

    /**
     * @return the number of samples in the last run (or 0 if we never run).
     */
    count samples() const { return sam; }

    /**
     * @return the intermediate values for the computation of the supdev bound
     * epsilon.
     */
    const auto &sdbintvals() const { return sdbintvals_; }

    /**
     * Computes betweenness approximation on the graph passed in constructor.
     */
    void run() final override { G.isWeighted() ? doRun<Dijkstra>() : doRun<BFS>(); }

private:
    double eps;         // accuracy guarantee
    const double delta; // failure probability
    const count mcs;    // number of MonteCarlo samples
    count iters{0};     // number of iterations
    count sam{0};       // number of samples
    std::array<double, 5>
        sdbintvals_; // intermediate values for the computation of the supdev bound
    ProgrBavarianSchedule sched;

    template <typename SSSPAlg>
    void doRun() {
        Aux::SignalHandler handler;
        scoreData.clear();
        scoreData.resize(G.upperNodeIdBound());
        BA::impl::Sums sums{scoreData.size(), mcs};
        BA::impl::Sups sups{mcs};
        double currEps{1};
        do {
            ++iters;
            const auto oldSam{sam};
            sam = sched.next();
            const auto toSample{sam - oldSam};
            INFO("Iteration ", iters, ": taking ", sam, " samples with ", mcs,
                 " MonteCarlo samples");
            sums.populate<BCAlg, SSSPAlg>(G, toSample);
            // XXX MR: Theoretically, there is no need to get estimations until
            // the stopping condition is satisfied, but since 1) we are using
            // the same code also in the static-sampling algorithm; 2) computing
            // the estimations is actually relatively quick; and 3) we have to
            // compute them anyway at some point, let's just forget about it and
            // live happy.
            INFO("Getting suprema and estimations");
            sups.populateAndComputeScores(scoreData, sums, G, sam);
            INFO("Computing epsilon");
            currEps =
                BA::impl::epsilon(delta, mcs, sam, sups.mcera(sam), sups.wimpy(sam), sdbintvals_);
        } while (currEps > eps);
        eps = currEps;
        hasRun = true;
    }
};

} // namespace NetworKit
#endif // NETWORKIT_CENTRALITY_PROGRBAVARIAN_HPP_
