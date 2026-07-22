# kadabra.jl
using Graphs
using Random

"""
    estimate_diameter(g::AbstractGraph)

Computes an upper bound on the diameter of the graph using the AllCCUpperBound technique
(Borassi et al. 2015). Highly efficient for both directed and undirected graphs.
"""

mutable struct KadabraWorkspace{T<:Integer}
    ball_indicator::Vector{UInt8}
    n_paths::Vector{Float64}
    dist::Vector{Int}

    preds_data::Vector{T}
    preds_count::Vector{Int}
    preds_offset::Vector{Int}

    cur_s::Vector{T}
    next_s::Vector{T}
    cur_t::Vector{T}
    next_t::Vector{T}

    sp_edges::Vector{Tuple{T,T}}
    visited_nodes::Vector{T}
    local_top_k::Vector{T}
    min_count_local::Base.RefValue{Int}
end

function KadabraWorkspace(g::AbstractGraph{T}) where {T}
    n = nv(g)

    preds_offset = zeros(Int, n + 1)
    offset = 1
    for v = 1:n
        preds_offset[v] = offset
        # maximum predecessors a node can have in a shortest path BFS is bounded by its degree
        offset += max(indegree(g, v), outdegree(g, v))
    end
    preds_offset[n+1] = offset

    preds_data = zeros(T, max(1, offset - 1))
    preds_count = zeros(Int, n)

    cur_s = zeros(T, n)
    next_s = zeros(T, n)
    cur_t = zeros(T, n)
    next_t = zeros(T, n)

    sp_edges = fill((zero(T), zero(T)), max(1, ne(g)))
    visited_nodes = zeros(T, n + 2)

    return KadabraWorkspace{T}(
        zeros(UInt8, n),
        zeros(Float64, n),
        fill(typemax(Int), n),
        preds_data,
        preds_count,
        preds_offset,
        cur_s,
        next_s,
        cur_t,
        next_t,
        sp_edges,
        visited_nodes,
        zeros(T, 0), # initialized later in kadabra_centrality
        Ref(0)
    )
end

function estimate_diameter(g::AbstractGraph)
    n = nv(g)
    n == 0 && return 0.0
    n == 1 && return 0.0

    # 1. Compute strongly/connected components
    sccs = is_directed(g) ? strongly_connected_components(g) : connected_components(g)
    n_components = length(sccs)

    # Map each vertex to its component index
    cc = zeros(Int, n)
    for (i, component) in enumerate(sccs)
        for v in component
            cc[v] = i
        end
    end

    # 2. Compute pivots for each component
    # The pivot vertex is the vertex maximizing the sum of the out-degree and the in-degree.
    pivots = zeros(Int, n_components)
    for i = 1:n_components
        component = sccs[i]
        best_v = component[1]
        best_deg = outdegree(g, best_v) + indegree(g, best_v)
        for v in component
            deg = outdegree(g, v) + indegree(g, v)
            if deg > best_deg
                best_v = v
                best_deg = deg
            end
        end
        pivots[i] = best_v
    end

    # 3. Compute SCC adjacency graph (DAG of components)
    cc_adj = [Set{Int}() for _ = 1:n_components]
    for u = 1:n
        for v in outneighbors(g, u)
            if cc[u] != cc[v]
                push!(cc_adj[cc[u]], cc[v])
            end
        end
    end

    # 4. Helper BFS to compute eccentricity of pivot in its SCC.
    ecc_dist = fill(-1, n)

    function compute_ecc_in_scc(start::Int, backward::Bool)
        fill!(ecc_dist, -1)
        q = Int[]
        push!(q, start)
        ecc_dist[start] = 0

        head = 1
        while head <= length(q)
            u = q[head]
            head += 1

            neighbors = backward ? inneighbors(g, u) : outneighbors(g, u)
            for v in neighbors
                if ecc_dist[v] == -1 && cc[v] == cc[u]
                    ecc_dist[v] = ecc_dist[u] + 1
                    push!(q, v)
                end
            end
        end

        return isempty(q) ? 0 : ecc_dist[q[end]]
    end

    # 5. Compute forward and backward eccentricities of pivots in their SCCs
    ecc_f_pivots_scc = zeros(Float64, n_components)
    ecc_b_pivots_scc = zeros(Float64, n_components)
    for i = 1:n_components
        ecc_f_pivots_scc[i] = compute_ecc_in_scc(pivots[i], false)
        ecc_b_pivots_scc[i] = compute_ecc_in_scc(pivots[i], true)
    end

    # 6. DP to compute bounds across components (memoized DFS).
    ecc_f_pivots = fill(-1.0, n_components)

    function get_ecc_f_pivot(i::Int)
        ecc_f_pivots[i] != -1.0 && return ecc_f_pivots[i]

        val = ecc_f_pivots_scc[i]
        for cc_dest in cc_adj[i]
            val = max(
                val,
                ecc_f_pivots_scc[i] +
                1 +
                ecc_b_pivots_scc[cc_dest] +
                get_ecc_f_pivot(cc_dest),
            )
        end
        ecc_f_pivots[i] = val
        return val
    end

    diam = 0.0
    for i = 1:n_components
        diam = max(diam, get_ecc_f_pivot(i) + ecc_b_pivots_scc[i])
    end

    return max(diam, 1.0)
end

function zero_alloc_top_k!(
    top_k_nodes::AbstractVector{Int},
    values::AbstractVector{Int},
    k::Int,
)
    n = length(values)
    for i = 1:k
        top_k_nodes[i] = i
    end
    for i = 2:k
        curr = top_k_nodes[i]
        val = values[curr]
        j = i - 1
        while j > 0 && values[top_k_nodes[j]] < val
            top_k_nodes[j+1] = top_k_nodes[j]
            j -= 1
        end
        top_k_nodes[j+1] = curr
    end

    min_val = values[top_k_nodes[k]]
    for i = (k+1):n
        val = values[i]
        if val > min_val
            j = k - 1
            while j > 0 && values[top_k_nodes[j]] < val
                top_k_nodes[j+1] = top_k_nodes[j]
                j -= 1
            end
            top_k_nodes[j+1] = i
            min_val = values[top_k_nodes[k]]
        end
    end
end

"""
    kadabra_centrality(g::AbstractGraph, k::Int, err::Float64, delta::Float64; start_factor::Int=100)

Estimates the betweenness centrality of vertices in graph `g` using the KADABRA algorithm 
(Borassi & Natale, 2016). KADABRA is an adaptive sampling algorithm that guarantees the 
estimated betweenness is within an additive error bound with high probability.

# Arguments
- `g::AbstractGraph`: The input graph (directed or undirected).
- `k::Int`: If `k = 0`, guarantees absolute error `err` for all vertices. If `k > 0`, 
  guarantees the relative ranking of the top `k` vertices is correct within the error bound.
- `err::Float64`: The maximum additive error tolerance (e.g., 0.01).
- `delta::Float64`: The confidence parameter. Results are guaranteed with probability `1 - delta`.
- `start_factor::Int`: Keyword argument to scale the initial burn-in phase duration (default: 100).
- `endpoints::Bool`: If true, include the endpoints of the sampled shortest paths in the centrality counts (default: false).
- `normalize::Symbol`: How to normalize the output. `:graphs` matches Graphs.jl, `:kadabra` matches the raw KADABRA paper output, `:none` returns unnormalized counts (default: `:graphs`).

# Returns
- `NamedTuple`: A named tuple containing three `Vector{Float64}`: `centralities`, `lower_bounds`, 
  and `upper_bounds` for each vertex.
"""
function _kadabra_worker_task!(
    g::AbstractGraph{T},
    n::Int,
    seed::UInt64,
    tid::Int,
    n_pairs::Threads.Atomic{Int},
    tau_per_thread::Int,
    omega::Float64,
    stop_flag::Threads.Atomic{Bool},
    check_lock::Threads.SpinLock,
    global_approx::Vector{Int},
    top_k_nodes::Vector{Int},
    union_sample::Int,
    absolute::Bool,
    k::Int,
    err::Float64,
    delta_l_guess::Vector{Float64},
    delta_u_guess::Vector{Float64},
    bet_buf::Vector{Float64},
    err_l_buf::Vector{Float64},
    err_u_buf::Vector{Float64},
    endpoints::Bool,
    workspaces::Vector{KadabraWorkspace{T}}
) where {T}
    local ws = workspaces[tid]
    local counts = global_approx
    local t_rng = Random.Xoshiro(seed)

    # --- PHASE 1 ---
    for _ = 1:tau_per_thread
        s = rand(t_rng, 1:n)
        t = rand(t_rng, 1:n)
        while s == t
            t = rand(t_rng, 1:n)
        end
        sample_shortest_path!(counts, ws, g, t_rng, T(s), T(t); endpoints = endpoints)
    end
    
    Threads.atomic_add!(n_pairs, tau_per_thread)

    # --- PHASE 2 ---
    local_pairs = 0
    while !stop_flag[] && n_pairs[] < omega
        check_interval_dynamic = max(500, min(10000, cld(ceil(Int, omega - n_pairs[]), 2 * Threads.nthreads())))
        for _ = 1:check_interval_dynamic
            s = rand(t_rng, 1:n)
            t = rand(t_rng, 1:n)
            while s == t
                t = rand(t_rng, 1:n)
            end
            sample_shortest_path!(counts, ws, g, t_rng, T(s), T(t); endpoints = endpoints)
            local_pairs += 1
        end

        Threads.atomic_add!(n_pairs, local_pairs)
        local_pairs = 0

        if trylock(check_lock)
            try
                if stop_flag[]
                    continue
                end

                if absolute
                    @inbounds for j = 1:union_sample; top_k_nodes[j] = j; end
                else
                    # Aggregate local top Ks
                    # We can use top_k_nodes as a buffer initially since it has size union_sample
                    # We collect all candidates
                    candidates = Int[]
                    for i = 1:Threads.nthreads()
                        tk = workspaces[i].local_top_k
                        for j = 1:length(tk)
                            if tk[j] > 0
                                push!(candidates, tk[j])
                            end
                        end
                    end
                    unique!(candidates)
                    
                    num_active = length(candidates)
                    if num_active < union_sample
                        copyto!(top_k_nodes, 1, candidates, 1, num_active)
                        idx = num_active + 1
                        v = 1
                        while idx <= union_sample && v <= n
                            if global_approx[v] == 0
                                top_k_nodes[idx] = v
                                idx += 1
                            end
                            v += 1
                        end
                    else
                        partialsort!(candidates, 1:union_sample, by = x -> global_approx[x], rev = true)
                        copyto!(top_k_nodes, 1, candidates, 1, union_sample)
                    end
                end

                if check_finished(
                    global_approx, view(top_k_nodes, 1:union_sample),
                    n_pairs[], k, err, delta_l_guess, delta_u_guess,
                    omega, absolute, bet_buf, err_l_buf, err_u_buf
                )
                    Threads.atomic_xchg!(stop_flag, true)
                end
            finally
                unlock(check_lock)
            end
        end
    end
end

function kadabra_centrality(
    g::AbstractGraph{T},
    k::Int,
    err::Float64,
    delta::Float64;
    start_factor::Int = 100,
    endpoints::Bool = false,
    normalize::Symbol = :graphs,
    parallel::Bool = true,
    rng::Union{AbstractRNG,Nothing} = nothing,
) where {T}
    nv(g) >= 2 || throw(ArgumentError("Graph must have at least 2 vertices (got $(nv(g)))"))
    err > 0 || throw(ArgumentError("err must be positive (got $err)"))
    0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1) (got $delta)"))

    n = nv(g)
    absolute = (k == 0)
    k = Int(k == 0 ? n : min(k, n))

    diam_est = max(estimate_diameter(g), 2.0)

    omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
    tau = max(round(Int, omega / start_factor), 1)

    global_approx = zeros(Int, n)
    union_sample = Int(absolute ? k : min(n, k + 20))

    check_interval = max(10, tau ÷ 100)
    delta_l_guess = fill(delta / (4 * n), n)
    delta_u_guess = fill(delta / (4 * n), n)

    bet_buf = zeros(Float64, union_sample)
    err_l_buf = zeros(Float64, union_sample)
    err_u_buf = zeros(Float64, union_sample)
    top_k_nodes = collect(1:n)

    base_rng = rng === nothing ? Random.default_rng() : rng
    final_n_pairs = 0

    if parallel
        nthreads = Threads.nthreads()
        # println("[MAIN] Starting parallel KADABRA with $nthreads tasks.")
        
        global_approx = zeros(Int, n)
        workspaces = [KadabraWorkspace(g) for _ = 1:nthreads]
        for w in workspaces
            w.local_top_k = zeros(Int, union_sample)
        end
        
        # Buffer arrays for evaluating convergence (allocated per worker task? 
        # No, these are used inside check_lock, so shared is fine)
        bet_buf = zeros(Float64, union_sample)
        err_l_buf = zeros(Float64, union_sample)
        err_u_buf = zeros(Float64, union_sample)

        stop_flag = Threads.Atomic{Bool}(false)
        check_lock = Threads.SpinLock()
        check_interval = max(1000, tau ÷ 10)

        n_pairs = Threads.Atomic{Int}(0)
        
        tau_per_thread = cld(tau, nthreads)
        # println("[MAIN] Target omega = $omega, tau = $tau. Each task will do $tau_per_thread pairs in Phase 1.")

        # Generate seeds beforehand to avoid base_rng thread-safety issues
        thread_seeds = [rand(base_rng, UInt64) for _ in 1:nthreads]
        tasks = Vector{Task}(undef, nthreads)
        
        for i = 1:nthreads
            tasks[i] = Threads.@spawn _kadabra_worker_task!(
                g, n, thread_seeds[i], i, n_pairs, tau_per_thread,
                omega, stop_flag, check_lock, global_approx,
                top_k_nodes, union_sample, absolute, k, err, delta_l_guess,
                delta_u_guess, bet_buf, err_l_buf, err_u_buf, endpoints, workspaces
            )
        end

        # Wait for all worker tasks to finish safely
        wait.(tasks)
        final_n_pairs = n_pairs[]
    else
        # Single-threaded fallback
        s_rng = rng === nothing ? Random.default_rng() : rng
        counts = global_approx
        ws = KadabraWorkspace(g)
        ws.local_top_k = zeros(Int, union_sample)

        # --- PHASE 1 ---
        for _ = 1:tau
            s = rand(s_rng, 1:n)
            t = rand(s_rng, 1:n)
            while s == t
                t = rand(s_rng, 1:n)
            end
            sample_shortest_path!(counts, ws, g, s_rng, T(s), T(t); endpoints = endpoints)
        end
        final_n_pairs = tau
        
        stop_flag_seq = false
        check_interval = max(1000, tau ÷ 10)

        while !stop_flag_seq && final_n_pairs < omega
            for _ = 1:check_interval
                s = rand(s_rng, 1:n)
                t = rand(s_rng, 1:n)
                while s == t
                    t = rand(s_rng, 1:n)
                end
                sample_shortest_path!(counts, ws, g, s_rng, T(s), T(t); endpoints = endpoints)
                final_n_pairs += 1
            end

            if absolute
                for i = 1:union_sample; top_k_nodes[i] = i; end
            else
                copyto!(top_k_nodes, 1:n)
                partialsort!(top_k_nodes, 1:union_sample, by = x -> counts[x], rev = true)
            end

            if check_finished(
                counts, view(top_k_nodes, 1:union_sample), final_n_pairs,
                k, err, delta_l_guess, delta_u_guess, omega, absolute,
                bet_buf, err_l_buf, err_u_buf
            )
                stop_flag_seq = true
            end
        end
    end

    res = [global_approx[v] / final_n_pairs for v = 1:n]
    lower_bounds = Float64[max(0.0, res[v] - compute_f(res[v], final_n_pairs, delta_l_guess[v], omega)) for v = 1:n]
    upper_bounds = Float64[min(1.0, res[v] + compute_g(res[v], final_n_pairs, delta_u_guess[v], omega)) for v = 1:n]

    scale = 1.0
    if normalize == :graphs
        scale = n > 2 ? (n * (n - 1.0)) / ((n - 1.0) * (n - 2.0)) : 0.0
    elseif normalize == :none
        scale = is_directed(g) ? (n * (n - 1.0)) : (n * (n - 1.0)) / 2.0
    elseif normalize == :kadabra
        scale = 1.0
    else
        throw(ArgumentError("Unknown normalize option: $normalize. Use :graphs, :kadabra, or :none."))
    end

    if scale != 1.0
        res .*= scale
        lower_bounds .*= scale
        upper_bounds .*= scale
    end

    return (centralities = res, lower_bounds = lower_bounds, upper_bounds = upper_bounds, n_samples = final_n_pairs)
end

function kadabra_centrality(
    g::AbstractGraph{T},
    k::Int,
    err::Float64,
    delta::Float64,
    distmx::AbstractMatrix;
    kwargs...,
) where {T}
    throw(
        ArgumentError(
            "KADABRA centrality does not support weighted graphs. Please do not provide a distmx argument.",
        ),
    )
end

"""
    kadabra_top_k(g::AbstractGraph, k::Int, err::Float64, delta::Float64; kwargs...)

Convenience wrapper that runs KADABRA and efficiently extracts the top `k` most central nodes.
Due to adaptive sampling, nodes with overlapping confidence intervals near the k-th rank cannot 
be strictly ordered. Therefore, this function returns a candidate set of size `k' >= k` that 
contains the true top-k nodes with high probability.
Returns a vector of `NamedTuple`s containing the `node` ID, its `centrality` score, 
`lower_bound`, and `upper_bound`, sorted in descending order of centrality.
"""
function kadabra_top_k(
    g::AbstractGraph{T},
    k::Int,
    err::Float64,
    delta::Float64;
    kwargs...,
) where {T}
    k > 0 || throw(ArgumentError("k must be greater than 0 to extract top k nodes"))

    # Run the standard Kadabra algorithm
    res = kadabra_centrality(g, k, err, delta; kwargs...)
    centralities = res.centralities
    lower_bounds = res.lower_bounds
    upper_bounds = res.upper_bounds

    # Find the candidate threshold by determining the k-th highest lower bound
    k_th_lower_bound = partialsort(lower_bounds, k, rev = true)

    # Filter candidate nodes where upper_bound >= k_th_lower_bound
    candidate_nodes = findall(u -> u >= k_th_lower_bound, upper_bounds)

    # Sort candidate nodes by centrality in descending order
    sort!(candidate_nodes, by = v -> centralities[v], rev = true)

    return [
        (
            node = v,
            centrality = centralities[v],
            lower_bound = lower_bounds[v],
            upper_bound = upper_bounds[v],
        ) for v in candidate_nodes
    ]
end


"""
    compute_f(btilde, iter_num, delta_l, omega)

Computes the Chernoff bound error function `f` that bounds the betweenness of a vertex from below.
Evaluated dynamically during Phase 2 to determine if the lower bound of the confidence interval
satisfies the stopping condition.
"""
function compute_f(btilde::Float64, iter_num::Int, delta_l::Float64, omega::Float64)
    tmp = (omega / iter_num) - (1.0 / 3.0)
    err_chern =
        (log(1.0 / delta_l) / iter_num) *
        (-tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_l)))
    return min(err_chern, btilde)
end

"""
    compute_g(btilde, iter_num, delta_u, omega)

Computes the Chernoff bound error function `g` that bounds the betweenness of a vertex from above.
Evaluated dynamically during Phase 2 to determine if the upper bound of the confidence interval
satisfies the stopping condition.
"""
function compute_g(btilde::Float64, iter_num::Int, delta_u::Float64, omega::Float64)
    tmp = (omega / iter_num) + (1.0 / 3.0)
    err_chern =
        (log(1.0 / delta_u) / iter_num) *
        (tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_u)))
    return min(err_chern, 1.0 - btilde)
end

"""
    check_finished(approx_counts, top_k_nodes, n_pairs, k, err, delta_l_guess, delta_u_guess, omega, absolute)

Evaluates whether the KADABRA algorithm has met the stopping criteria based on the current samples.
Supports both absolute error mode (all vertices have error < err) and relative mode (the gap between 
the top-k rankings is strictly larger than their overlapping error bounds).
"""
function check_finished(
    approx_counts::Vector{Int},
    top_k_nodes::AbstractVector{Int},
    n_pairs::Int,
    k::Int,
    err::Float64,
    delta_l_guess::Vector{Float64},
    delta_u_guess::Vector{Float64},
    omega::Float64,
    absolute::Bool,
    bet::Vector{Float64},
    err_l::Vector{Float64},
    err_u::Vector{Float64},
)
    n_tracked = length(top_k_nodes)

    @inbounds for i = 1:n_tracked
        v = top_k_nodes[i]
        # clamp fängt asynchrone Auslesefehler ab und schützt vor DomainErrors
        bet[i] = clamp(approx_counts[v] / n_pairs, 0.0, 1.0)
    end

    @inbounds for i = 1:n_tracked
        v = top_k_nodes[i]
        err_l[i] = compute_f(bet[i], n_pairs, delta_l_guess[v], omega)
        err_u[i] = compute_g(bet[i], n_pairs, delta_u_guess[v], omega)
    end

    all_finished = true

    if absolute
        @inbounds for i = 1:k
            finished = (err_l[i] < err) && (err_u[i] < err)
            all_finished = all_finished && finished
        end
    else
        @inbounds for i = 1:n_tracked
            if i == 1
                if n_tracked > 1
                    finished = (bet[1] - err_l[1]) > (bet[2] + err_u[2])
                else
                    finished = true
                end
            elseif i < k
                finished =
                    ((bet[i-1] - err_l[i-1]) > (bet[i] + err_u[i])) &&
                    ((bet[i] - err_l[i]) > (bet[i+1] + err_u[i+1]))
            elseif i == k
                if k < n_tracked
                    finished =
                        ((bet[k-1] - err_l[k-1]) > (bet[k] + err_u[k])) &&
                        ((bet[k] - err_l[k]) > (bet[k+1] + err_u[k+1]))
                else
                    finished = (bet[k-1] - err_l[k-1]) > (bet[k] + err_u[k])
                end
            else
                finished = (bet[k] - err_l[k]) > (bet[i] + err_u[i])
            end

            finished = finished || ((err_l[i] < err) && (err_u[i] < err))
            all_finished = all_finished && finished
        end
    end

    return all_finished
end


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------



"""
    sample_shortest_path!(counts::Vector{Int}, ws::KadabraWorkspace{T}, g, s, t[; endpoints=false])

Sample a single shortest path uniformly at random using pre-allocated workspace memory,
and directly increment `counts` for every vertex on the path. No heap allocations occur.
"""
function sample_shortest_path!(
    counts::Vector{Int},
    ws::KadabraWorkspace{T},
    g::AbstractGraph{T},
    rng::AbstractRNG,
    s::T,
    t::T;
    endpoints::Bool = false,
) where {T}
    s == t && return
    _sample_shortest_path!(g, rng, s, t, counts, ws, outneighbors, inneighbors, outdegree, indegree, endpoints)
end

"""
    _bb_bfs_sample!(counts, ws, g, s, t, neighborfn_s, neighborfn_t, endpoints)

Internal helper function that performs a Balanced Bidirectional BFS from source `s` and target `t`.
The search expands the frontier with the smallest sum of out-degrees to minimize edge traversals.
When the frontiers intersect, it selects a single bridge edge uniformly at random (weighted by 
the number of shortest paths crossing it) and backtracks to construct the sampled path.
The nodes on the resulting path are incremented directly in the `counts` array without allocating memory.
"""
@inline function _sample_shortest_path!(
    g::AbstractGraph{T},
    rng::AbstractRNG,
    s::T,
    t::T,
    counts::Vector{Int},
    ws::KadabraWorkspace{T},
    neighborfn_s::F1,
    neighborfn_t::F2,
    degreefn_s::F3,
    degreefn_t::F4,
    endpoints::Bool,
) where {T<:Integer,F1,F2,F3,F4}
    ball_indicator = ws.ball_indicator
    n_paths = ws.n_paths
    dist = ws.dist
    preds_data = ws.preds_data
    preds_count = ws.preds_count
    preds_offset = ws.preds_offset
    cur_s = ws.cur_s
    next_s = ws.next_s
    cur_t = ws.cur_t
    next_t = ws.next_t
    sp_edges = ws.sp_edges
    visited_nodes = ws.visited_nodes

    ball_indicator[s] = 0x01
    n_paths[s] = 1.0
    dist[s] = 0
    cur_s_len = 1
    cur_s[1] = s
    visited_len = 1
    visited_nodes[1] = s
    sum_degs_s = degreefn_s(g, s)

    ball_indicator[t] = 0x02
    n_paths[t] = 1.0
    dist[t] = 0
    cur_t_len = 1
    cur_t[1] = t
    visited_len += 1
    visited_nodes[visited_len] = t
    sum_degs_t = degreefn_t(g, t)

    sp_edges_len = 0
    have_to_stop = false

    begin
        iter_count = 0
        while !have_to_stop && (cur_s_len > 0 && cur_t_len > 0)
            iter_count += 1
            if sum_degs_s <= sum_degs_t
                sum_degs_s = 0
                next_s_len = 0
                @inbounds for i = 1:cur_s_len
                    x = cur_s[i]
                    @inbounds for y in neighborfn_s(g, x)
                        if ball_indicator[y] == 0x00
                            ball_indicator[y] = 0x01
                            n_paths[y] = n_paths[x]
                            dist[y] = dist[x] + 1

                            count = preds_count[y]
                            idx = preds_offset[y] + count
                            preds_data[idx] = x
                            preds_count[y] = count + 1

                            next_s_len += 1
                            next_s[next_s_len] = y

                            visited_len += 1
                            visited_nodes[visited_len] = y
                            sum_degs_s += degreefn_s(g, y)

                        elseif ball_indicator[y] == 0x02
                            have_to_stop = true
                            sp_edges_len += 1
                            sp_edges[sp_edges_len] = (x, y)

                        elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x01
                            n_paths[y] += n_paths[x]
                            count = preds_count[y]
                            idx = preds_offset[y] + count
                            preds_data[idx] = x
                            preds_count[y] = count + 1
                        end
                    end
                end
                cur_s_len = next_s_len
                cur_s, next_s = next_s, cur_s

            else
                sum_degs_t = 0
                next_t_len = 0
                @inbounds for i = 1:cur_t_len
                    x = cur_t[i]
                    @inbounds for y in neighborfn_t(g, x)
                        if ball_indicator[y] == 0x00
                            ball_indicator[y] = 0x02
                            n_paths[y] = n_paths[x]
                            dist[y] = dist[x] + 1

                            count = preds_count[y]
                            idx = preds_offset[y] + count
                            preds_data[idx] = x
                            preds_count[y] = count + 1

                            next_t_len += 1
                            next_t[next_t_len] = y

                            visited_len += 1
                            visited_nodes[visited_len] = y
                            sum_degs_t += degreefn_t(g, y)

                        elseif ball_indicator[y] == 0x01
                            have_to_stop = true
                            sp_edges_len += 1
                            sp_edges[sp_edges_len] = (y, x)

                        elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x02
                            n_paths[y] += n_paths[x]
                            count = preds_count[y]
                            idx = preds_offset[y] + count
                            preds_data[idx] = x
                            preds_count[y] = count + 1
                        end
                    end
                end
                cur_t_len = next_t_len
                cur_t, next_t = next_t, cur_t
            end
        end

        if sp_edges_len > 0
            tot_weight = 0.0
            for i = 1:sp_edges_len
                (u_bridge, v_bridge) = sp_edges[i]
                tot_weight += n_paths[u_bridge] * n_paths[v_bridge]
            end

            rand_val = rand(rng) * tot_weight
            cur_weight = 0.0
            selected_edge = sp_edges[sp_edges_len] 

            for i = 1:sp_edges_len
                (u_bridge, v_bridge) = sp_edges[i]
                cur_weight += n_paths[u_bridge] * n_paths[v_bridge]
                if cur_weight >= rand_val
                    selected_edge = (u_bridge, v_bridge)
                    break
                end
            end

            _backtrack!(
                selected_edge[1],
                s,
                ws,
                counts,
                rng,
                endpoints,
            )
            _backtrack!(
                selected_edge[2],
                t,
                ws,
                counts,
                rng,
                endpoints,
            )
        end

        for i = 1:visited_len
            v = visited_nodes[i]
            ball_indicator[v] = 0x00
            n_paths[v] = 0.0
            dist[v] = typemax(Int)
            preds_count[v] = 0
        end
    end

    return
end

"""
    _backtrack!(curr, target, ws, counts, changed_nodes, rng, endpoints)

Internal helper that reconstructs a single shortest path by backtracking from the `curr` node 
towards the `target` node using the predecessor map `preds` generated during the BFS.
If multiple optimal predecessors exist, one is selected randomly weighted by the number of 
shortest paths `n_paths` arriving through that predecessor.
The thread-local `counts` buffer is incremented in-place for every node visited.
"""
function update_local_top_k!(
    local_top_k::Vector{T},
    v::T,
    c::Int,
    global_approx::Vector{Int},
    min_count_local::Base.RefValue{Int}
) where {T}
    k = length(local_top_k)
    # Check if already in top K
    @inbounds for i = 1:k
        if local_top_k[i] == v
            return
        end
    end

    # Replace the minimum
    min_idx = 1
    min_val = typemax(Int)
    @inbounds for i = 1:k
        val = local_top_k[i] == 0 ? 0 : global_approx[local_top_k[i]]
        if val < min_val
            min_val = val
            min_idx = i
        end
    end
    
    @inbounds local_top_k[min_idx] = v

    # Recompute new minimum
    new_min_val = typemax(Int)
    @inbounds for i = 1:k
        val = local_top_k[i] == 0 ? 0 : global_approx[local_top_k[i]]
        if val < new_min_val
            new_min_val = val
        end
    end
    min_count_local[] = new_min_val
    return
end

@inline function _backtrack!(
    curr::T,
    target::T,
    ws::KadabraWorkspace{T},
    counts::Vector{Int},
    rng::AbstractRNG,
    endpoints::Bool,
) where {T}
    preds_data = ws.preds_data
    preds_count = ws.preds_count
    preds_offset = ws.preds_offset
    n_paths = ws.n_paths
    local_top_k = ws.local_top_k
    min_count_local = ws.min_count_local

    @inbounds while curr != target
        counts[curr] += 1
        c = counts[curr]
        if c > min_count_local[] && length(local_top_k) > 0
            update_local_top_k!(local_top_k, curr, c, counts, min_count_local)
        end

        count = preds_count[curr]
        offset = preds_offset[curr]

        if count == 0
            break
        elseif count == 1
            curr = preds_data[offset]
        else
            tot = 0.0
            for i = 1:count
                p = preds_data[offset+i-1]
                tot += n_paths[p]
            end

            r = rand(rng) * tot
            c = 0.0
            
            curr = preds_data[offset+count-1] 
            
            for i = 1:count
                p = preds_data[offset+i-1]
                c += n_paths[p]
                if c >= r
                    curr = p
                    break
                end
            end
        end
    end
    if endpoints
        counts[target] += 1
        c = counts[target]
        if c > min_count_local[] && length(local_top_k) > 0
            update_local_top_k!(local_top_k, target, c, counts, min_count_local)
        end
    end
end
