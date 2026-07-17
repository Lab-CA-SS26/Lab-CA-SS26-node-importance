/*
 * Bavarian.hpp
 *
 * Created on: 20200409
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

#ifndef NETWORKIT_CENTRALITY_BAVARIAN_HPP_
#define NETWORKIT_CENTRALITY_BAVARIAN_HPP_

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/centrality/BavarianBA.hpp>
#include <networkit/centrality/Centrality.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

#include <array>
#include <chrono>

namespace NetworKit {

template <typename BCAlg>
class Bavarian : public Centrality {

public:
    using algo = BCAlg;

    Bavarian(const Graph &G, const double delta, const count mctrials, const count samples)
        : Centrality{G, true}, delta{delta}, mct{mctrials}, sam{samples},
          tmout{std::chrono::milliseconds::zero()} {}

    Bavarian(const Graph &G, const double delta, const count mctrials,
             const std::chrono::milliseconds &timeout)
        : Centrality{G, true}, delta{delta}, mct{mctrials}, sam{0}, tmout{timeout} {}

    /**
     * @return the computed approximation quality in the last run (or -1 if we
     * never run).
     */
    double epsilon() const { return eps; }

    /**
     * @return the number of samples in the last run (or 0 if we never run).
     */
    double samples() const { return sam; }

    /**
     * @return the intermediate values for the computation of the supdev bound
     * epsilon.
     */
    const auto &sdbintvals() const { return sdbintvals_; }

    /**
     * Computes betweenness approximation on the graph passed in constructor.
     */
    void run() final override {
        Aux::SignalHandler handler;
        scoreData.clear();
        scoreData.resize(G.upperNodeIdBound());
        // We look at the algorithm as if its main purpose was to
        // populate the following data structure, which indeed will
        // contain all the information that are needed to compute the quality
        // approximation and the estimates.
        BA::impl::Sums sums{scoreData.size(), mct};
        INFO("taking ", sam, " samples with ", mct, " MonteCarlo trials");
        // Choose the right variant of the function to call.
        if (G.isWeighted()) {
            if (tmout == std::chrono::milliseconds::zero())
                sums.populate<BCAlg, Dijkstra>(G, sam);
            else
                sam = sums.populate<BCAlg, Dijkstra>(G, tmout);
        } else {
            if (tmout == std::chrono::milliseconds::zero())
                sums.populate<BCAlg, BFS>(G, sam);
            else
                sam = sums.populate<BCAlg, BFS>(G, tmout);
        }
        handler.assureRunning();
        INFO("Getting suprema and estimations");
        // Bavarian "suprema": to store the maxima, over the nodes, of the first
        // mct+1 elements of the sums.
        BA::impl::Sups sups{mct};
        // The following function also populate the scoreData vector, to avoid
        // another pass over the sums data structure.
        sups.populateAndComputeScores(scoreData, sums, G, sam);
        handler.assureRunning();
        INFO("Computing epsilon");
        eps = BA::impl::epsilon(delta, mct, sam, sups.mcera(sam), sups.wimpy(sam), sdbintvals_);
        hasRun = true;
    }

private:
    double eps{-1};                        // accuracy guarantee
    const double delta;                    // failure probability
    const count mct;                       // number of MonteCarlo trials
    count sam;                             // number of samples
    const std::chrono::milliseconds tmout; // number of milliseconds to run
    std::array<double, 5>
        sdbintvals_; // intermediate values for the computation of the supdev bound.
                     //
                     // index 0: empirical wimpy variance
                     //       1: upper bound to the wimpy variance
                     //       2: MC-ERA
                     //       3: upper bound to the ERA
                     //       4: upper bound to the RA
};

} // namespace NetworKit
#endif // NETWORKIT_CENTRALITY_BAVARIAN_HPP_
