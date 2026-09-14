#!/usr/bin/env julia
#
# diagnose_topk_oracle.jl --- how much could a *predicted* ranking save KADABRA's
# top-k mode, if the prediction were perfect?
#
# Top-k mode spends its confidence budget per vertex from the rank gaps it sees in the
# burn-in (`compute_delta_guess!`). That allocation is a heuristic: any allocation fixed
# before Phase 2 keeps the (eps, delta) guarantee, and only the sample count moves. So a
# learned model could supply the gaps instead of the burn-in. This script asks what the
# best case would be, by feeding in the *exact* betweenness as the prediction.
#
# Like diagnose_topk_delta.jl it bisects the sample count at which `check_finished` would
# first fire, with the betweenness estimates held fixed --- deterministic, no Phase 2
# sampling. It crosses two allocation sources with two idealisations of the estimates:
#
#   allocation   burn   the burn-in estimates (what KADABRA does)
#                orc    exact betweenness (a perfect prediction)
#   estimates    burn   the burn-in estimates, scaled (diagnose_topk_delta.jl's model)
#                exact  exact betweenness (Phase 2 fully converged)
#
# The exact-rank allocation is an idealised prediction, not a proven optimum: the
# allocation rule itself is a heuristic, so some other ranking could in principle do
# better. Treat `orc` as "a very good model", not as a hard bound.
#
# Usage:
#   julia --project=.. diagnose_topk_oracle.jl <graph.txt> <exact_bc.jld2> [--directed]
#         [-k 3,5,10,100] [--variants paper_bd,paper] [--epsilon 1e-4] [--delta 0.1]
#         [--seeds 1,2,3]
#
# Exact caches are Graphs.jl `betweenness_centrality(g, normalize = true)`, i.e. divided by
# (n-1)(n-2); KADABRA's scale is the fraction of ordered pairs, so multiply by (n-2)/n.
# `union_sample` depends on the thread count: run with JULIA_NUM_THREADS=8 to match the
# stored top-k runs.

using Printf
using Random
using JLD2

include(joinpath(@__DIR__, "..", "src", "kadabra.jl"))
include(joinpath(@__DIR__, "scripts", "BenchmarkUtils.jl"))
using .BenchmarkUtils

function parse_args(args)
    o = Dict{String,Any}("directed" => false, "k" => [3, 5, 10, 100], "epsilon" => 1e-4,
                         "delta" => 0.1, "seeds" => [1, 2, 3],
                         "variants" => [:paper_bd, :paper], "files" => String[])
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--directed"
            o["directed"] = true; i += 1
        elseif a in ("-k", "--k")
            o["k"] = [parse(Int, x) for x in split(args[i+1], ",")]; i += 2
        elseif a in ("-e", "--epsilon")
            o["epsilon"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--delta"
            o["delta"] = parse(Float64, args[i+1]); i += 2
        elseif a in ("-s", "--seeds")
            o["seeds"] = [parse(Int, x) for x in split(args[i+1], ",")]; i += 2
        elseif a == "--variants"
            o["variants"] = [Symbol(x) for x in split(args[i+1], ",")]; i += 2
        else
            push!(o["files"], a); i += 1
        end
    end
    length(o["files"]) == 2 ||
        error("usage: diagnose_topk_oracle.jl <graph.txt> <exact_bc.jld2> [--directed] [-k K,..]")
    return o
end

"""Draw the burn-in phase sequentially and return the raw hit counts."""
function burn_in(g, tau::Int, seed::Int)
    n = Int(nv(g))
    counts = zeros(Int, n)
    ws = KadabraWorkspace(g)
    rng = Random.Xoshiro(seed)
    T = eltype(g)
    for _ = 1:tau
        s = rand(rng, 1:n)
        t = rand(rng, 1:n)
        while s == t
            t = rand(rng, 1:n)
        end
        sample_shortest_path!(counts, ws, g, rng, T(s), T(t); endpoints = false)
    end
    return counts
end

"""
Smallest sample count at which `check_finished` fires when every vertex's estimate is
held at `rate[v]` (a fraction of pairs) and its tracking set is `tracked`. Monotone in the
sample count, so the bisection is exact up to its resolution. `nothing` if it never fires
below omega.
"""
function predicted_stop(rate, tracked, k, err, dl, du, omega, absolute, variant, lo0)
    n = length(rate)
    us = length(tracked)
    bet_buf, el_buf, eu_buf = zeros(us), zeros(us), zeros(us)
    scaled = zeros(Int, n)
    fires(it) = begin
        @inbounds for v = 1:n
            scaled[v] = round(Int, rate[v] * it)
        end
        check_finished(scaled, tracked, it, k, err, dl, du, omega, absolute,
                       bet_buf, el_buf, eu_buf; variant = variant)
    end
    hi = Int(floor(omega))
    fires(hi) || return nothing
    lo = lo0
    while hi - lo > max(1000, hi ÷ 10_000)
        mid = (lo + hi) ÷ 2
        fires(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

function top_set(score, m)
    order = collect(1:length(score))
    partialsort!(order, 1:m, by = x -> score[x], rev = true)
    return order[1:m]
end

fmt(s) = s === nothing ? "   no stop" : @sprintf("%10d", s)
ratio(a, b) = (a === nothing || b === nothing) ? "   n/a" : @sprintf("%6.3f", a / b)

function main()
    o = parse_args(ARGS)
    gfile, bcfile = o["files"]
    g = BenchmarkUtils.load_graph_from_edgelist(gfile, o["directed"])
    n = Int(nv(g))
    err, delta = o["epsilon"], o["delta"]
    start_factor = 100

    exact = JLD2.load(bcfile, "exact_bc")
    length(exact) == n || error("exact cache has $(length(exact)) entries, graph has $n vertices")
    p = exact .* ((n - 2) / n)           # fraction of ordered pairs, KADABRA's scale

    diam_est = max(estimate_diameter(g), 2.0)
    omega = 0.5 / err^2 * (log2(diam_est - 1.0) + 1.0 + log(0.5 / delta))
    tau = max(round(Int, omega / start_factor), 1)

    @printf("graph    %s  (n=%d, m=%d, directed=%s)\n", basename(gfile), n, ne(g), o["directed"])
    @printf("eps=%g delta=%g threads=%d  omega=%.0f  burn-in tau=%d\n\n",
            err, delta, Threads.nthreads(), omega, tau)

    rows = Tuple[]
    for seed in o["seeds"]
        t0 = time()
        counts = burn_in(g, tau, seed)
        rate_b = counts ./ tau
        @printf("seed %d: burn-in %.1f s\n", seed, time() - t0)

        # Sanity: burn-in and exact must agree on scale and orientation. A directedness or
        # normalisation mismatch shows up here as a ratio far from 1.
        t5 = top_set(p, 5)
        @printf("  sanity, top-5 exact vs burn-in: %s\n",
                join([@sprintf("%.3e/%.3e", p[v], rate_b[v]) for v in t5], "  "))

        for k_req in o["k"], variant in o["variants"]
            k = min(k_req, n)
            union_target = max(2.0 * sqrt(ne(g)) / Threads.nthreads(), Float64(k_req) + 20.0)
            us = min(n, Int(floor(union_target)))
            T_b = top_set(rate_b, us)
            T_o = top_set(p, us)

            dl_b, du_b = zeros(n), zeros(n)
            compute_delta_guess!(dl_b, du_b, T_b, counts, tau, n, k, false, err, delta,
                                 start_factor; variant = variant)
            dl_o, du_o = zeros(n), zeros(n)
            # compute_delta_guess! divides by n_pairs; exact fractions go in with n_pairs = 1.
            compute_delta_guess!(dl_o, du_o, T_o, p, 1, n, k, false, err, delta,
                                 start_factor; variant = variant)

            s = Dict{Tuple{Symbol,Symbol},Any}()
            for (alloc, dl, du) in ((:burn, dl_b, du_b), (:orc, dl_o, du_o))
                s[(alloc, :burn)] = predicted_stop(rate_b, T_b, k, err, dl, du, omega, false, variant, tau)
                s[(alloc, :exact)] = predicted_stop(p, T_o, k, err, dl, du, omega, false, variant, tau)
            end
            overlap_b = length(intersect(T_b[1:k], T_o[1:k]))
            push!(rows, (seed, k_req, variant, s, overlap_b))
        end
    end

    println()
    println("predicted stopping sample counts; ratio = oracle allocation / burn-in allocation")
    println("burn-top-k = how many of the exact top k the burn-in ranks in its own top k")
    println()
    @printf("%-4s %-4s %-9s | %-10s %-10s %-6s | %-10s %-10s %-6s | %s\n",
            "seed", "k", "variant", "est=burn", "", "", "est=exact", "", "", "burn-top-k")
    @printf("%-4s %-4s %-9s | %-10s %-10s %-6s | %-10s %-10s %-6s |\n",
            "", "", "", "alloc=burn", "alloc=orc", "ratio", "alloc=burn", "alloc=orc", "ratio")
    for (seed, k, variant, s, ov) in rows
        @printf("%-4d %-4d %-9s | %s %s %s | %s %s %s | %d/%d\n",
                seed, k, variant,
                fmt(s[(:burn, :burn)]), fmt(s[(:orc, :burn)]), ratio(s[(:orc, :burn)], s[(:burn, :burn)]),
                fmt(s[(:burn, :exact)]), fmt(s[(:orc, :exact)]), ratio(s[(:orc, :exact)], s[(:burn, :exact)]),
                ov, k)
    end

    println()
    println("mean +- std of the ratio over seeds")
    for k in o["k"], variant in o["variants"]
        sel = [r for r in rows if r[2] == k && r[3] === variant]
        for est in (:burn, :exact)
            rs = [r[4][(:orc, est)] / r[4][(:burn, est)] for r in sel
                  if r[4][(:orc, est)] !== nothing && r[4][(:burn, est)] !== nothing]
            isempty(rs) && continue
            m = sum(rs) / length(rs)
            sd = length(rs) > 1 ? sqrt(sum((rs .- m) .^ 2) / (length(rs) - 1)) : 0.0
            @printf("  k=%-4d %-9s est=%-5s  %.3f +- %.3f  (n=%d)\n", k, variant, est, m, sd, length(rs))
        end
    end
end

main()
