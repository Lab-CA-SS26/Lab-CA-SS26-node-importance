/*
 * BavarianBA.hpp
 *
 * Created on: 20200421
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

#ifndef NETWORKIT_CENTRALITY_BAVARIANBA_HPP_
#define NETWORKIT_CENTRALITY_BAVARIANBA_HPP_

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
// XXX MR: As of 20200422, libc++ (used by clang++) does not have <execution>.
// libstdc++ (used by g++) does, but it relies on the external Intel library
// ThreadsBuldingBlocks (tbb).
#ifdef BAVARIAN_WITH_EXECUTION
#include <execution>
#endif
#include <queue>
#include <ratio>
#include <unordered_map>
#include <utility>
#include <vector>

namespace NetworKit::BA::impl {
// Bavarian "sums": a vector of (at least) as many elements as there are
// nodes, and each element of this vector being a vector of mctrials+2
// elements such that the elements from index 0 to mctrials-1 store the
// "Rademacher" sums (used in the mctrials-MCERA), the element at index mctrials
// stores the sums of squares of function values (used in the tail bound), and
// the last element stores the sum of function values (used for the BC
// estimation).
class Sums {
public:
    Sums(const count size, const count mctrials)
        : mctrials{mctrials}, sums(size, std::vector<double>(mctrials + 2)) {}

    const std::vector<double> &at(const node v) const { return sums.at(v); }

    static void increase(std::vector<std::vector<double>> &sums, const node v,
                         const std::vector<int> &rv, const double f) {
        // Going through the vector backwards
        auto it{sums.at(v).rbegin()};
        *(it++) += f;
        *(it++) += f * f;
        auto itOut{it};
        // This transform cannot be parallelized because
        // it modifies one of the sequences.
        std::transform(rv.crbegin(), rv.crend(), it, itOut,
                       [f](const int l, double s) { return s + l * f; });
    }

    static void merge(std::vector<std::vector<double>> &sums,
                      const std::vector<std::vector<double>> &oth) {
        // This transform cannot be parallelized because it modifies one
        // of the sequences.
        std::transform(sums.begin(), sums.end(), oth.cbegin(), sums.begin(),
                       [](std::vector<double> &a, const std::vector<double> &b) {
                           // This transform cannot be parallelized because
                           // it modifies one of the sequences.
                           std::transform(a.cbegin(), a.cend(), b.cbegin(), a.begin(),
                                          std::plus<double>());
                           return std::move(a);
                       });
    }

// Custom reduction for aggregating the sums
// XXX MR: One may want to consider switching from using an OpenMP reduction to
// using some kind of locking mechanism or atomics, as there may not be much
// contention anyway, but the amount of refactoring needed is not trivial.
#pragma omp declare reduction  \
            (sumsReduction: std::vector<std::vector<double>>: \
                Sums::merge(omp_out, omp_in)) \
            initializer (omp_priv = decltype(omp_orig)(omp_orig.size(), \
                        std::vector<double>(omp_orig.at(0).size())))

    template <typename BCAlg, typename SSSPAlg>
    void populate(const Graph &G, const count samples) {
        Aux::SignalHandler handler;
        handler.assureRunning();
        // XXX MR: We move the data member because it looks like OpenMP
        // reductions on data members are not supported.
        // See http://forum.openmp.org/forum/viewtopic.php?f=3&t=79&start=0.
        // The cost of this move should be negligible.
        std::vector<std::vector<double>> os{std::move(sums)};
        // Use the nonmonotonic:dynamic schedule because each iteration
        // may take a different time (dynamic), and we do not care about
        // monotonicity.
        const auto s{static_cast<omp_index>(samples)};
#pragma omp parallel for schedule(nonmonotonic : dynamic) reduction(sumsReduction : os)
        for (omp_index i = 1; i <= s; ++i) {
            if (!handler.isRunning())
                continue;
            DEBUG("sample ", i);
            // Sample the Rademacher vector
            std::vector<int> rv(mctrials);
            std::generate(
#ifdef BAVARIAN_WITH_EXECUTION
                std::execution::par_unseq,
#endif
                rv.begin(), rv.end(), []() { return (Aux::Random::probability() < 0.5) ? -1 : 1; });
            // Sample and get the values of the functions on the sample
            // as prescribed by BCAlg, and update the sums data structure
            // using those values and the Rademacher vector rv.
            BCAlg::template sample<SSSPAlg>(G, rv, os);
        }
        handler.assureRunning();
        sums = std::move(os);
    }

    template <typename BCAlg, typename SSSPAlg>
    count populate(const Graph &G, const std::chrono::milliseconds &timeout) {
        Aux::SignalHandler handler;
        handler.assureRunning();
        // XXX MR: We move the data member because it looks like OpenMP
        // reductions on data members are not supported.
        // See http://forum.openmp.org/forum/viewtopic.php?f=3&t=79&start=0.
        // Since we are moving, the cost should be negligible.
        std::vector<std::vector<double>> os{std::move(sums)};
        count samples{1};
        count toSample{1};
        auto elapsed{std::chrono::milliseconds::zero()};
        do {
            const auto s{static_cast<omp_index>(toSample)};
            const auto start{std::chrono::steady_clock::now()};
// Use the nonmonotonic:dynamic schedule because each iteration
// may take a different time (dynamic), and we do not care about
// monotonicity.
#pragma omp parallel for schedule(nonmonotonic : dynamic) reduction(sumsReduction : os)
            for (omp_index i = 1; i <= s; ++i) {
                if (!handler.isRunning())
                    continue;
                DEBUG("sample ", i);
                // Sample the Rademacher vector
                std::vector<int> rv(mctrials);
                std::generate(
#ifdef BAVARIAN_WITH_EXECUTION
                    std::execution::par_unseq,
#endif
                    rv.begin(), rv.end(),
                    []() { return (Aux::Random::probability() < 0.5) ? -1 : 1; });
                // Sample and get the values of the functions on the sample
                // as prescribed by BCAlg, and update the sums data structure
                // using those values and the Rademacher vector lv.
                BCAlg().template sample<SSSPAlg>(G, rv, os);
            }
            elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            handler.assureRunning();
            if (elapsed > timeout)
                break;
            // Get the average time per sample so far, and the leftover time
            // before the timeout, and decide to sample, in the next
            // iteration of the do-while loop, 75% of the samples that would
            // fit in the leftover interval, if each of them took the
            // average time. The idea is to roughly account for possible
            // variance in the time needed to take a sample.
            const auto timePerSample{elapsed / samples};
            const std::chrono::duration<double, std::milli> leftover{timeout - elapsed};
            toSample = static_cast<count>(leftover.count() * 0.75 / timePerSample.count());
            if (toSample == 0)
                break;
            samples += toSample;
            DEBUG("leftover time: ", leftover.count(), "ms, sampling ", toSample, " more samples");
        } while (true);

        sums = std::move(os);
        return samples;
    }

private:
    const count mctrials;
    std::vector<std::vector<double>> sums;
};

// Used to compute the suprema in a custom OpenMP reduction.
class Sups {
public:
    Sups(const count mctrials) : sups(mctrials + 1) {}

    double mcera(const count samples) const {
        return std::reduce(
#ifdef BAVARIAN_WITH_EXECUTION
                   std::execution::par_unseq,
#endif
                   sups.cbegin(), std::prev(sups.cend()))
               / ((sups.size() - 1) * samples);
    }

    static void merge(std::vector<double> &sups, const std::vector<double> &oth) {
        // "Merge" oth with sups by keeping, for each index, the maximum
        // entry between the two vectors.
        //
        // This transform cannot be parallelized because it modifies one of
        // the sequences.
        std::transform(sups.cbegin(), sups.cend(), oth.cbegin(), sups.begin(),
                       [](const double a, const double b) { return std::max(a, b); });
    }

// Custom reduction to compute the suprema.
// XXX MR: Perhaps the initializer expression can also use "omp_priv =
// omp_orig", but it is not entirely clear to me whether that's true.
#pragma omp declare reduction (supsReduction : std::vector<double>: \
                Sups::merge(omp_out, omp_in)) \
            initializer (omp_priv = decltype(omp_orig)(omp_orig.size(), std::numeric_limits<double>::lowest()))

    void populateAndComputeScores(std::vector<double> &scores, const Sums &sums, const Graph &G,
                                  const count samples) {
        const auto nodes{static_cast<omp_index>(G.upperNodeIdBound())};
        Aux::SignalHandler handler;
        // XXX MR: We move the data member because it looks like OpenMP
        // reductions on data members are not supported.
        // See http://forum.openmp.org/forum/viewtopic.php?f=3&t=79&start=0.
        // The cost of this move should be negligible.
        std::vector<double> s{std::move(sups)};
// Use the nonmonotonic:dynamic schedule because each iteration may take a
// different time (dynamic), and we do not care about monotonicity.
#pragma omp parallel for schedule(nonmonotonic : dynamic) reduction(supsReduction : s)
        for (omp_index v = 0; v < nodes; ++v) {
            // Checking for node existence is *necessary*: one or more of the
            // mct summands of the MCERA may be negative (because it may be
            // the supremum of all negative quantities), while non-existing
            // nodes have corresponding value zero, so they must be skipped
            // for *correctness* (and efficiency will also benefit).
            if (handler.isRunning() && G.hasNode(v)) {
                const auto &sumsV{sums.at(v)};
                // Update the suprema
                merge(s, sumsV);
                // Obtain the BC estimation.
                scores[v] = sumsV.back() / samples;
            }
        }
        handler.assureRunning();
        sups = std::move(s);
    }

    // Wimpy variance
    double wimpy(const count samples) const { return sups.back() / samples; }

private:
    std::vector<double> sups;
};

static inline std::pair<node, node> samplePair(const Graph &G) {
    // Sample two distinct nodes that are not neighbors.
    // Restricting to non-neighbors is a slight variant of the original
    // algorithms (RK and ABRA), but the analysis is still correct.
    node u{GraphTools::randomNode(G)};
    while (G.degree(u) >= G.numberOfNodes() - 1) {
        u = GraphTools::randomNode(G);
    }
    node v{GraphTools::randomNode(G)};
    while (v == u || G.hasEdge(u, v)) {
        v = GraphTools::randomNode(G);
    }
    TRACE("sampled nodes ", u, " and ", v);
    return {u, v};
}

static inline double epsilon(const double delta, const count mctrials, const count samples,
                             const double mcera, const double ewimpy,
                             std::array<double, 5> &intermediate) {
    const auto logFiveOverDeltaOverM{(std::log(5) - std::log(delta)) / samples};
    const auto wimpybound{ewimpy + logFiveOverDeltaOverM * (2.0 / 3)
                          + std::sqrt(std::pow(logFiveOverDeltaOverM / std::sqrt(3), 2)
                                      + 2 * ewimpy * logFiveOverDeltaOverM)};
    const auto erabound{mcera + (logFiveOverDeltaOverM / mctrials) * (2.0 / 3)
                        + std::sqrt(4 * ewimpy * logFiveOverDeltaOverM / mctrials)};
    const auto rabound{erabound + logFiveOverDeltaOverM / 3
                       + std::sqrt(std::pow(logFiveOverDeltaOverM / (2 * std::sqrt(3)), 2)
                                   + erabound * logFiveOverDeltaOverM)};
    const double eps{2 * rabound + std::sqrt(2 * logFiveOverDeltaOverM * (wimpybound + 4 * rabound))
                     + logFiveOverDeltaOverM / 3};
    DEBUG("mcera: ", mcera, ", ewimpy: ", ewimpy, ", wimpybound: ", wimpybound,
          ", erabound: ", erabound, ", rabound: ", rabound, ", eps: ", eps);
    intermediate[0] = ewimpy;
    intermediate[1] = wimpybound;
    intermediate[2] = mcera;
    intermediate[3] = erabound;
    intermediate[4] = rabound;
    return eps;
}
} // namespace NetworKit::BA::impl

namespace NetworKit::BA {
class RK {
public:
    // Return the quality guarantee from the original RK paper.
    static double nomc_epsilon(const Graph &G, const double delta, const count samples,
                               edgeweight vd = 0) {
        if (vd == 0) {
            // Get the vertex-diameter approximation.
            Diameter diam{G, DiameterAlgo::estimatedPedantic};
            diam.run();
            vd = diam.getDiameter().first;
            DEBUG("Vertex diameter approx: ", vd);
        }
        return std::sqrt((std::floor(std::log2(vd - 2)) + 1 - std::log(delta)) / samples);
    }

    // Return the sample size needed to obtain a certain epsilon according to
    // the original RK paper.
    static count nomc_samplesize(const Graph &G, const double delta, const double epsilon,
                                 edgeweight vd = 0) {
        if (vd == 0) {
            // Get the vertex-diameter approximation.
            Diameter diam{G, DiameterAlgo::estimatedPedantic};
            diam.run();
            vd = diam.getDiameter().first;
            DEBUG("Vertex diameter approx: ", vd);
        }
        return (std::floor(std::log2(vd - 2)) + 1 - std::log(delta)) / (epsilon * epsilon);
    }

    // XXX MR: This friendship declaration is a little bit more extrovert
    // than I would like, but the C++ standard forbids partially-specified
    // template friend functions.
    template <typename BCAlg, typename SSSPAlg>
    friend void NetworKit::BA::impl::Sums::populate(const Graph &G, const count samples);
    template <typename BCAlg, typename SSSPAlg>
    friend count NetworKit::BA::impl::Sums::populate(const Graph &G,
                                                     const std::chrono::milliseconds &t);

private:
    template <typename SSSPAlg>
    static void sample(const Graph &G, const std::vector<int> &rv,
                       std::vector<std::vector<double>> &sums) {
        auto [u, v] = impl::samplePair(G);
        DEBUG("Running truncated SSSP from node ", u, " to node ", v);
        SSSPAlg sssp{G, u, true, false, v};
        sssp.run();
        if (sssp._numberOfPaths(v) > 0) {
            DEBUG("Sampling SP ", u, "->", v,
                  " and updating sums for nodes"
                  " on sampled SP");
            // Because samplePair returns two distinct nodes that are not
            // neighbors, we can be sure that the first predecessor that we
            // sample is not u.
            // From now on, v is the node that we are currently exploring in
            // the backtrack.
            v = samplePredecessor(sssp, v);
            do {
                impl::Sums::increase(sums, v, rv, 1);
                v = samplePredecessor(sssp, v);
            } while (v != u);
        }
    }

    template <typename SSSPAlg>
    static node samplePredecessor(const SSSPAlg &sssp, const node v) {
        // sample z in P_u(v) with probability sigma_uz / sigma_uv
        if (const auto &preds{sssp.getPredecessors(v)}; preds.size() > 1) {
            // Avoid going over all predecessors if we can help it.
            auto it{preds.cbegin()};
            auto d{Aux::Random::probability() * sssp._numberOfPaths(v)};
            do {
                d -= sssp._numberOfPaths(*(it++));
            } while (d >= 0);
            return *std::prev(it);
        } else
            return preds.at(0);
    }
};

class ABRA {
public:
    // Return the quality guarantee from Thm. 3.3 of the ABRA TKDD paper.
    static double nomc_epsilon(const Graph &G, const double delta, const count samples,
                               const double ewimpy) {
        // upper bound to the ERA with Massart's lemma.
        const auto eraBound{std::sqrt(ewimpy * 2 * std::log(G.numberOfNodes()) / samples)};
        const auto logTerm{std::log(3) - std::log(delta)};
        return 2 * eraBound
               + (logTerm + std::sqrt(logTerm * (logTerm + 4 * samples * eraBound))) / samples
               + std::sqrt(logTerm / (2 * samples));
    }

    // Return the sample size needed to obtain a certain epsilon according to
    // the ABRA TKDD paper
    static count nomc_samplesize(const Graph &G, const double delta, const double epsilon,
                                 const double ewimpy) {

        // Perform binary search to find the minimum sample size such that the
        // resulting epsilon is less than the desired epsilon.
        count lb{1};
        double eps = nomc_epsilon(G, delta, lb, ewimpy);
        while (eps > epsilon) {
            lb *= 2;
            eps = nomc_epsilon(G, delta, lb, ewimpy);
        }
        count ub{lb};
        lb /= 2;
        count m{lb + (ub - lb) / 2};
        for (eps = nomc_epsilon(G, delta, m, ewimpy); ub - lb > 1;
             eps = nomc_epsilon(G, delta, m, ewimpy)) {
            if (eps < epsilon)
                ub = m;
            else
                lb = m;
            m = lb + (ub - lb) / 2;
        }
        return m;
    }

    // XXX MR: This friendship declaration is a little bit more extrovert
    // than I would like, but the C++ standard forbids partially-specified
    // template friend functions.
    template <typename BCAlg, typename SSSPAlg>
    friend void NetworKit::BA::impl::Sums::populate(const Graph &G, const count samples);
    template <typename BCAlg, typename SSSPAlg>
    friend count NetworKit::BA::impl::Sums::populate(const Graph &G,
                                                     const std::chrono::milliseconds &t);

private:
    template <typename SSSPAlg>
    static void sample(const Graph &G, const std::vector<int> &rv,
                       std::vector<std::vector<double>> &sums) {
        auto [u, v] = impl::samplePair(G);
        DEBUG("Running truncated SSSP from node ", u, " to node ", v);
        SSSPAlg sssp{G, u, true, false, v};
        sssp.run();
        // XXX MR: NetworKit uses bigfloat as the return type of
        // sssp.numberOfPaths() (no underscore at beginning of function name),
        // but it really seems excessive.
        if (const auto pathsFromU{sssp._numberOfPaths(v)}; pathsFromU > 0) {
            DEBUG("Updating sums for nodes on SPs from ", u, " to ", v);
            // Traverse BFS DAG backwards, and update sums for encountered
            // nodes while traversing.
            std::unordered_map<node, double> pathsToTarget;
            // Priority queue to go through nodes in reverse order of
            // distance from u.
            const auto less_distance{[&s = std::as_const(sssp)](const node a, const node b) {
                return (s.distance(a) != s.distance(b)) ? s.distance(a) < s.distance(b) : a < b;
            }};
            std::priority_queue<node, std::vector<node>, decltype(less_distance)> q{less_distance};
            // Adding the predecessors of v to the maps to start the
            // backward traversal. Because v is not a neighbor of u (by
            // design of the sampling procedure), we have at least one set
            // of predecessors to handle.
            for (const node pred : sssp.getPredecessors(v)) {
                // There is always a single SP from a predecessor of v to v.
                pathsToTarget.emplace(pred, 1);
                q.push(pred);
            }
            // Start the backtracking, visiting nodes in reverse order of
            // distance from u.
            // From now on, v is the node that we are currently exploring in
            // the backtrack.
            v = q.top();
            q.pop();
            do {
                const double pathsToTargetFromV{pathsToTarget.at(v)};
                // XXX MR: NetworKit uses bigfloat as the return type of
                // sssp.numberOfPaths() (no underscore at beginning of function
                // name), but it really seems excessive.
                const double pathsThrough{sssp._numberOfPaths(v) * pathsToTargetFromV};
                impl::Sums::increase(sums, v, rv, pathsThrough / pathsFromU);
                // Add the predecessors of v to the relevant maps for
                // backtracking, or increase the running sum of the number
                // of SPs from v to the original target.
                for (const node pred : sssp.getPredecessors(v)) {
                    if (auto p{pathsToTarget.try_emplace(pred, pathsToTargetFromV)}; p.second)
                        q.push(pred);
                    else
                        p.first->second += pathsToTargetFromV;
                }
                v = q.top();
                q.pop();
            } while (v != u);
        } // End of block checking if there is at least one SP from u to v
    }
};

class BP {
public:
    // Return the quality guarantee from the original BP paper.
    static double nomc_epsilon(const Graph &G, const double delta, const count samples) {
        return std::sqrt((std::log(G.numberOfNodes()) + std::log(2) - std::log(delta))
                         / (2 * samples));
    }

    // Return the sample size from the original paper that would have been
    // needed to obtain a certain epsilon, according to the original BP paper.
    static count nomc_samplesize(const Graph &G, const double delta, const double epsilon) {
        return (std::log(G.numberOfNodes()) + std::log(2) - std::log(delta)) / (epsilon * epsilon);
    }

    // XXX MR: This friendship declaration is a little bit more extrovert
    // than I would like, but the C++ standard forbids partially-specified
    // template friend functions.
    template <typename BCAlg, typename SSSPAlg>
    friend void NetworKit::BA::impl::Sums::populate(const Graph &G, const count samples);
    template <typename BCAlg, typename SSSPAlg>
    friend count NetworKit::BA::impl::Sums::populate(const Graph &G,
                                                     const std::chrono::milliseconds &t);

private:
    template <typename SSSPAlg>
    static void sample(const Graph &G, const std::vector<int> &rv,
                       std::vector<std::vector<double>> &sums) {
        // Sample one node
        const auto u{GraphTools::randomNode(G)};
        DEBUG("running SSSP from node ", u);
        SSSPAlg sssp{G, u, true, true};
        sssp.run();
        DEBUG("Updating sums for nodes on SPs from ", u);
        const double denom{1.0 / (G.numberOfNodes() - 1)};
        // Dependencies, as in Brandes' original paper.
        std::vector<double> dep(G.upperNodeIdBound());
        const auto &nodesByDist{sssp.getNodesSortedByDistance()};
        // Traverse BFS DAG *backwards*, stopping just before u, which is
        // the first element of nodesByDist by construction.
        for (auto rit{std::prev(nodesByDist.cend())}; rit != nodesByDist.cbegin(); --rit) {
            // XXX MR: NetworKit uses bigfloat as the return type of
            // sssp.numberOfPaths() (no underscore at beginning of function
            // name), but it really seems excessive.
            const double mult{(1 + dep.at(*rit)) / sssp._numberOfPaths(*rit)};
            const std::vector<node> &preds{sssp.getPredecessors(*rit)};
            std::for_each(
#ifdef BAVARIAN_WITH_EXECUTION
                std::execution::par_unseq,
#endif
                preds.cbegin(), preds.cend(), [&s = std::as_const(sssp), &dep, mult](const node p) {
                    dep[p] += s._numberOfPaths(p) * mult;
                });
            impl::Sums::increase(sums, *rit, rv, dep.at(*rit) * denom);
        }
    }
};
} // namespace NetworKit::BA
#endif // NETWORKIT_CENTRALITY_BAVARIANBA_HPP_
