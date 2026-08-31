#!/usr/bin/env julia
#
# diagnose_topk_delta.jl --- why the paper's and the C++ reference's top-k
# confidence-budget allocations differ, and what it costs.
#
# The reference implementation sizes each vertex's *lower* deviation budget by the
# rank gap ABOVE it and its *upper* budget by the gap BELOW it; its own stopping
# test needs exactly the opposite pairing (v_i's lower bound is what has to clear
# v_{i+1}'s upper bound). Borassi & Natale's Section 5.2 pairs them the other way
# round, i.e. consistently with the test.
#
# This script isolates that difference from sampling noise. It draws the burn-in
# phase ONCE, then --- holding those betweenness estimates fixed --- asks, for each
# variant, at what sample count the stopping condition would first fire and which
# vertex is the last to be resolved. No Phase 2 sampling is involved, so the answer
# is deterministic given the seed and costs ~1% of a real run.
#
# Usage:
#   julia --project=.. diagnose_topk_delta.jl <graph.txt> [--directed] [-k 10]
#                                             [--epsilon 1e-4] [--delta 0.1] [--seed 1]

using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "kadabra.jl"))
include(joinpath(@__DIR__, "scripts", "BenchmarkUtils.jl"))
using .BenchmarkUtils

const VARIANTS = (:code, :paper, :cpp)

function parse_args(args)
    o = Dict{String,Any}("directed" => false, "k" => [10], "epsilon" => 1e-4,
                         "delta" => 0.1, "seed" => 1, "file" => "")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--directed"
            o["directed"] = true; i += 1
        elseif a in ("-k", "--k")
            o["k"] = [parse(Int, x) for x in split(args[i+1], ",")]; i += 2
        elseif a in ("-e", "--epsilon")
            o["epsilon"] = parse(Float64, args[i+1]); i += 2
        elseif a in ("-d", "--delta")
            o["delta"] = parse(Float64, args[i+1]); i += 2
        elseif a in ("-s", "--seed")
            o["seed"] = parse(Int, args[i+1]); i += 2
        else
            o["file"] = a; i += 1
        end
    end
    isempty(o["file"]) && error("usage: diagnose_topk_delta.jl <graph.txt> [--directed] [-k K] [--epsilon E]")
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
Smallest `n_pairs` at which `check_finished` fires, holding the burn-in estimates
fixed and scaling the hit counts with the sample size. Monotone in `n_pairs`, so a
bisection is exact up to the resolution asked for.
"""
function predicted_stop(counts, top_k_nodes, tau, k, err, dl, du, omega, absolute, variant)
    n = length(counts)
    us = length(top_k_nodes)
    bet_buf, el_buf, eu_buf = zeros(us), zeros(us), zeros(us)
    scaled = zeros(Int, n)

    fires(it) = begin
        f = it / tau
        @inbounds for v = 1:n
            scaled[v] = round(Int, counts[v] * f)
        end
        check_finished(scaled, top_k_nodes, it, k, err, dl, du, omega, absolute,
                       bet_buf, el_buf, eu_buf; variant = variant)
    end

    hi = Int(floor(omega))
    fires(hi) || return (nothing, bet_buf, el_buf, eu_buf)
    lo = tau
    while hi - lo > max(1000, hi ÷ 10_000)
        mid = (lo + hi) ÷ 2
        fires(mid) ? (hi = mid) : (lo = mid)
    end
    fires(hi)
    return (hi, copy(bet_buf), copy(el_buf), copy(eu_buf))
end

"""Which vertices are still unresolved at `it` --- i.e. what the run is waiting for."""
function unfinished_at(counts, top_k_nodes, tau, it, k, err, dl, du, omega, absolute, variant)
    n = length(counts)
    us = length(top_k_nodes)
    bet, el, eu = zeros(us), zeros(us), zeros(us)
    f = it / tau
    scaled = [round(Int, counts[v] * f) for v = 1:n]
    for i = 1:us
        v = top_k_nodes[i]
        bet[i] = clamp(scaled[v] / it, 0.0, 1.0)
        el[i] = compute_f(bet[i], it, dl[v], omega)
        eu[i] = compute_g(bet[i], it, du[v], omega)
    end
    stuck = Int[]
    for i = 1:us
        ok = if i == 1
            us > 1 ? (bet[1] - el[1]) > (bet[2] + eu[2]) : true
        elseif i < k
            ((bet[i-1] - el[i-1]) > (bet[i] + eu[i])) && ((bet[i] - el[i]) > (bet[i+1] + eu[i+1]))
        elseif i == k
            k < us ? (((bet[k-1] - el[k-1]) > (bet[k] + eu[k])) && ((bet[k] - el[k]) > (bet[k+1] + eu[k+1]))) :
                     ((bet[k-1] - el[k-1]) > (bet[k] + eu[k]))
        else
            variant === :cpp ? (bet[k] - eu[k]) > (bet[i] + eu[i]) : (bet[k] - el[k]) > (bet[i] + eu[i])
        end
        ok = ok || ((el[i] < err) && (eu[i] < err))
        ok || push!(stuck, i)
    end
    return stuck, bet, el, eu
end

function main()
    o = parse_args(ARGS)
    g = BenchmarkUtils.load_graph_from_edgelist(o["file"], o["directed"])
    n = Int(nv(g))
    err, delta, seed = o["epsilon"], o["delta"], o["seed"]
    start_factor = 100

    diam_est = max(estimate_diameter(g), 2.0)
    omega = 0.5 / err^2 * (log2(diam_est - 1.0) + 1.0 + log(0.5 / delta))
    tau = max(round(Int, omega / start_factor), 1)

    @printf("graph          %s  (n=%d, m=%d, directed=%s)\n", basename(o["file"]), n, ne(g), o["directed"])
    @printf("eps=%g delta=%g seed=%d threads=%d\n", err, delta, seed, Threads.nthreads())
    @printf("omega=%.0f  burn-in tau=%d\n", omega, tau)
    @printf("tie-collapse threshold sqrt(start_factor)*eps/4 = %.3e\n\n", sqrt(start_factor) * err / 4)

    print("drawing the burn-in phase ... "); flush(stdout)
    t0 = time()
    counts = burn_in(g, tau, seed)
    @printf("%.1f s\n", time() - t0)

    for k_req in o["k"]
        analyse(counts, g, n, tau, omega, err, delta, k_req, start_factor)
    end
end

function analyse(counts, g, n, tau, omega, err, delta, k_req, start_factor)
    absolute = (k_req == 0)
    k = Int(k_req == 0 ? n : min(k_req, n))
    union_target = max(2.0 * sqrt(ne(g)) / Threads.nthreads(), Float64(k_req) + 20.0)
    union_sample = min(n, Int(floor(union_target)))
    @printf("\n############################## k = %d  (tracking set %d) ##############################\n\n",
            k_req, union_sample)

    order = collect(1:n)
    partialsort!(order, 1:union_sample, by = x -> counts[x], rev = true)
    top_k_nodes = view(order, 1:union_sample)

    results = Dict{Symbol,Any}()
    for variant in VARIANTS
        dl, du = zeros(n), zeros(n)
        compute_delta_guess!(dl, du, top_k_nodes, counts, tau, n, k, absolute,
                             err, delta, start_factor; variant = variant)
        stop, _, _, _ = predicted_stop(counts, top_k_nodes, tau, k, err, dl, du,
                                       omega, absolute, variant)
        results[variant] = (dl = dl, du = du, stop = stop)
    end

    println("=== predicted stopping point (burn-in estimates held fixed) ===")
    base = results[:code].stop
    for variant in VARIANTS
        s = results[variant].stop
        rel = (s === nothing || base === nothing) ? "" : @sprintf("  (%.3fx code)", s / base)
        @printf("  %-6s  %s%s\n", variant,
                s === nothing ? "does NOT stop below omega" : @sprintf("%12d samples", s), rel)
    end
    println()

    # What each variant is waiting for just BEFORE it stops: the binding constraint.
    ref = minimum(x.stop for x in values(results) if x.stop !== nothing; init = typemax(Int))
    ref == typemax(Int) && return
    ref = max(tau + 1, round(Int, 0.97 * ref))
    println("=== still-unresolved vertices at $(ref) samples (3% short of the cheapest stop) ===")
    println("  rank = position in the descending-centrality tracking set; rank 1 is the most central vertex.")
    for variant in VARIANTS
        r = results[variant]
        stuck, bet, el, eu = unfinished_at(counts, top_k_nodes, tau, ref, k, err,
                                           r.dl, r.du, omega, absolute, variant)
        @printf("  %-6s  %d unresolved", variant, length(stuck))
        if isempty(stuck)
            println("  (finished)")
        else
            println("   ranks: ", join(first(stuck, 8), ", "), length(stuck) > 8 ? ", ..." : "")
            for i in first(stuck, 5)
                @printf("           rank %-5d bet=%.3e  f=%.3e (delta_L=%.2e)  g=%.3e (delta_U=%.2e)\n",
                        i, bet[i], el[i], r.dl[top_k_nodes[i]], eu[i], r.du[top_k_nodes[i]])
            end
        end
    end
    println()

    absolute && return
    println("=== budget targets from the burn-in ===")
    bet0 = [counts[top_k_nodes[i]] / tau for i = 1:union_sample]
    el, eu = zeros(union_sample), zeros(union_sample)
    println("  rank      bet        gap-above    gap-below | code: err_l   err_u | paper: err_l   err_u")
    tbl = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    for variant in (:code, :paper)
        compute_bet_err!(copy(bet0), el, eu, tau, k, absolute, err, start_factor; variant = variant)
        tbl[variant] = (copy(el), copy(eu))
    end
    for i = 1:min(k + 3, 12, union_sample)
        ga = i == 1 ? NaN : bet0[i-1] - bet0[i]
        gb = i == union_sample ? NaN : bet0[i] - bet0[i+1]
        @printf("  %-9d %.3e  %.3e  %.3e | %.3e  %.3e | %.3e  %.3e\n",
                i, bet0[i], ga, gb,
                tbl[:code][1][i], tbl[:code][2][i], tbl[:paper][1][i], tbl[:paper][2][i])
    end
end

main()
