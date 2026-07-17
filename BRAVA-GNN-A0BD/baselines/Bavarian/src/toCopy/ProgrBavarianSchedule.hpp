/*
 * ProgrBavarianSchedule.hpp
 *
 * Created on: 20200424
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

#ifndef NETWORKIT_CENTRALITY_PROGRBAVARIANSCHEDULE_HPP_
#define NETWORKIT_CENTRALITY_PROGRBAVARIANSCHEDULE_HPP_

#include <networkit/Globals.hpp>
#include <networkit/centrality/BavarianBA.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/graph/Graph.hpp>

#include <cmath>
#include <functional>
#include <tuple>

namespace NetworKit {

class ProgrBavarianSchedule {
public:
    static count minSize(const double epsilon, const double delta, const count iterations,
                         const double tolerance, const count mctrials) {
        const auto eta{delta / iterations};
        std::array<double, 5> useless;
        const auto getEpsilon{[eta, mctrials, &useless](const count m) {
            //  The zeroes are for the mcera and the empirical wimpy variance
            return BA::impl::epsilon(eta, mctrials, m, 0, 0, useless);
        }};
        return minSampleSizeForBound(epsilon, getEpsilon, tolerance);
    }

    static count maxSize(const double epsilon, const double delta, const count iterations,
                         const double tolerance, const count vcbound) {
        // Using the "simple" McDiarmid-bound approach
        const auto getEpsilon{[delta, vcbound, iterations](const count m) {
            constexpr double C{16.1864140562};
            const double radeBound{std::sqrt(C * vcbound / m)};
            return 2 * radeBound
                   + 3 * std::sqrt((std::log(2 * iterations) - std::log(delta)) / (2 * m));
        }};
        return minSampleSizeForBound(epsilon, getEpsilon, tolerance);
    }

    ProgrBavarianSchedule() {} // Creates a meaningless "zero" schedule

    // Schedule with passed iterations and passed vertex-diameter bound
    ProgrBavarianSchedule(const count vertexDiameterBound, const count mctrials,
                          const double epsilon, const double delta, const count iterations,
                          const double tolerance)
        : f{minSize(epsilon, delta, iterations, tolerance, mctrials)},
          l{maxSize(epsilon, delta, iterations, tolerance,
                    static_cast<count>(std::floor(std::log2(vertexDiameterBound - 2)) + 1))},
          mult{std::pow(static_cast<double>(l) / f, 1. / (iterations - 1))}, curr{(iterations == 1)
                                                                                      ? l
                                                                                      : f} {}

    // Schedule with passed iterations and passed graph from which to compute the
    // vertex-diameter bound
    ProgrBavarianSchedule(const Graph &G, const count mctrials, const double epsilon,
                          const double delta, const count iterations, const double tolerance)
        : ProgrBavarianSchedule{
            getVertexDiameterBound(G), mctrials, epsilon, delta, iterations, tolerance} {}

    // Schedule with passed multiplier and passed vertex-diameter bound
    ProgrBavarianSchedule(const count vertexDiameterBound, const count mctrials,
                          const double epsilon, const double delta, const double multiplier,
                          const double tolerance)
        : f{0}, l{0}, mult{multiplier}, curr{0} {

        const auto logm{std::log(mult)};
        DEBUG("logm: ", logm);
        const auto vcdimBound{
            static_cast<count>(std::floor(std::log2(vertexDiameterBound - 2)) + 1)};
        DEBUG("vcdimBound: ", vcdimBound);
        const auto func{[vcdimBound, mctrials, epsilon, delta, tolerance, logm,
                         multiplier](const int i) -> std::tuple<count, count, count> {
            // As the first sample size, we use the minimal one multiplied by
            // mult, to get something realistical.
            const auto lf{multiplier * minSize(epsilon, delta, i, tolerance, mctrials)};
            const auto ll{maxSize(epsilon, delta, i, tolerance, vcdimBound)};
            return {lf, ll,
                    static_cast<count>(std::ceil(std::log(static_cast<double>(ll) / lf) / logm))};
        }};
        count candidate{1};
        count actual;
        std::tie(f, l, actual) = func(candidate);
        DEBUG("f: ", f, ", l: ", l, ", actual: ", actual);
        while (actual > candidate) {
            candidate *= 2;
            DEBUG("candidate: ", candidate);
            std::tie(f, l, actual) = func(candidate);
            DEBUG("f: ", f, ", l: ", l, ", actual: ", actual);
        }
        count ub{candidate};
        count lb{candidate / 2};
        candidate = lb + (ub - lb) / 2;
        DEBUG("candidate: ", candidate);
        for (std::tie(f, l, actual) = func(candidate); candidate != actual;
             std::tie(f, l, actual) = func(candidate)) {
            DEBUG("f: ", f, ", l: ", l, ", actual: ", actual);
            if (actual < candidate)
                ub = candidate;
            else if (lb != ub - 1) // should be the same as lb != candidate
                lb = candidate;
            else // if the gap between ub and lb is 1, and candidate is lb,
                // make the bounds the same, so candidate becomes the bound and
                // we'll terminate.
                lb = ub;
            DEBUG("lb: ", lb, ", ub: ", ub);
            candidate = lb + (ub - lb) / 2;
            DEBUG("candidate: ", candidate);
        }
        if (candidate == 1)
            curr = l;
        else
            curr = f;
    }

    // Schedule with passed multiplier and passed graph from which to compute
    // the vertex-diameter bound
    ProgrBavarianSchedule(const Graph &G, const count mctrials, const double epsilon,
                          const double delta, const double multiplier, const double tolerance)
        : ProgrBavarianSchedule{
            getVertexDiameterBound(G), mctrials, epsilon, delta, multiplier, tolerance} {}

    // Return the first sample size
    count first() const { return f; }

    // Compute and return the next sample size
    count next() {
        const auto oldCurr{curr};
        if (curr < l) {
            curr = static_cast<count>(std::ceil(curr * mult));
            if (curr > l)
                curr = l;
        }
        return oldCurr;
    }

    // Return the last sample size
    count last() const { return l; }

    // Return the number of iterations
    count maxiterations() const {
        return static_cast<count>(std::floor((std::log(l) - std::log(f)) / std::log(mult)) + 2);
    }

private:
    count f;     // first sample size
    count l;     // last sample size
    double mult; // multiplier between sample sizes, either computed or passed in the constructor
    count curr;  //  current sample size

    static count minSampleSizeForBound(const double bound,
                                       const std::function<double(const count)> &getEpsilon,
                                       const double tolerance) {
        // Perform binary search to find the minimum sample size such that the
        // resulting epsilon is less than bound.
        count lb{1};
        double eps{getEpsilon(lb)};
        while (eps > bound) {
            lb *= 2;
            eps = getEpsilon(lb);
        }
        count ub{lb};
        lb /= 2;
        count m{lb + (ub - lb) / 2};
        for (eps = getEpsilon(m); ub - lb > 1 && (ub - lb) / static_cast<double>(lb) > tolerance;
             eps = getEpsilon(m)) {
            if (eps < bound)
                ub = m;
            else
                lb = m;
            m = lb + (ub - lb) / 2;
        }
        return m;
    }

    static count getVertexDiameterBound(const Graph &G) {
        Diameter diam{G, DiameterAlgo::estimatedPedantic};
        diam.run();
        const auto vd{diam.getDiameter().first};
        DEBUG("Vertex diameter approx: ", vd);
        return vd;
    }
};

} // namespace NetworKit
#endif // NETWORKIT_CENTRALITY_PROGRBAVARIANSCHEDULE_HPP_
