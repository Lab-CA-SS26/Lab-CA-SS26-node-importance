# kadabra.jl
using Graphs

"""
    estimate_diameter(g::AbstractGraph)

Computes an upper bound on the diameter of the graph using the AllCCUpperBound technique
(Borassi et al. 2015). Highly efficient for both directed and undirected graphs.
"""
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
    for i in 1:n_components
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
    cc_adj = [Set{Int}() for _ in 1:n_components]
    for u in 1:n
        for v in outneighbors(g, u)
            if cc[u] != cc[v]
                push!(cc_adj[cc[u]], cc[v])
            end
        end
    end
    
    # 4. Helper BFS to compute eccentricity of pivot in its SCC.
    # Fix: pre-allocate a single dist buffer and reuse it across all calls
    # to avoid O(n_components * n) allocations (was: fill(-1, n) per call).
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
    for i in 1:n_components
        ecc_f_pivots_scc[i] = compute_ecc_in_scc(pivots[i], false)
        ecc_b_pivots_scc[i] = compute_ecc_in_scc(pivots[i], true)
    end
    
    # 6. DP to compute bounds across components (memoized DFS).
    # Note: cc_adj is a DAG for directed graphs (by construction from SCCs).
    # For undirected graphs each CC is a single component node, so cc_adj is
    # also acyclic. The recursion therefore always terminates.
    ecc_f_pivots = fill(-1.0, n_components)
    
    function get_ecc_f_pivot(i::Int)
        ecc_f_pivots[i] != -1.0 && return ecc_f_pivots[i]
        
        val = ecc_f_pivots_scc[i]
        for cc_dest in cc_adj[i]
            val = max(val, ecc_f_pivots_scc[i] + 1 + ecc_b_pivots_scc[cc_dest] + get_ecc_f_pivot(cc_dest))
        end
        ecc_f_pivots[i] = val
        return val
    end
    
    diam = 0.0
    for i in 1:n_components
        diam = max(diam, get_ecc_f_pivot(i) + ecc_b_pivots_scc[i])
    end
    
    return max(diam, 1.0)
end

function kadabra_centrality(g::AbstractGraph, k::Int, err::Float64, delta::Float64; start_factor::Int=100)
    # --- Input validation ---
    nv(g) >= 2    || throw(ArgumentError("Graph must have at least 2 vertices (got $(nv(g)))"))
    err  > 0      || throw(ArgumentError("err must be positive (got $err)"))
    0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1) (got $delta)"))

    n = nv(g)
    absolute = (k == 0)
    k = Int(k == 0 ? n : min(k, n))
    
    # Estimate diameter using AllCCUpperBound
    diam_est = max(estimate_diameter(g), 2.0)
    
    # Sample-count upper bound (Borassi & Natale 2016, Theorem 3.2):
    #   ω = (1 / 2ε²) * (log₂(D − 1) + log₂(2/δ))
    #     = (1 / 2ε²) * (log₂(D − 1) + 1 + log₂(1/δ))
    # Fix: previous code used ln(0.5/δ) (natural log) where log₂(1/δ) is required,
    # causing omega to be underestimated by up to ~2× for typical δ values.
    omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
    tau = max(round(Int, omega / start_factor), 1)
    
    # Thread-local storage to avoid locking when updating centralities
    nthreads = Threads.nthreads()
    approx_local = [zeros(Int, n) for _ in 1:nthreads]
    workspaces = [KadabraWorkspace(g) for _ in 1:nthreads]
    n_pairs = Threads.Atomic{Int}(0)
    
    # Per-vertex confidence budget (uniform allocation).
    # Note: the paper (§5) describes an adaptive scheme that updates these weights
    # based on running estimates, which would improve early termination. That
    # refinement is not implemented here; the uniform baseline is correct but
    # may run slightly more samples than strictly necessary.
    delta_l_guess = fill(delta / (4 * n), n)
    delta_u_guess = fill(delta / (4 * n), n)
    
    # ---------------------------------------------------------
    # PHASE 1: Initial burn-in sampling (exactly tau samples, race-free).
    # Fix: the previous loop read n_pairs[] non-atomically in the while condition,
    # allowing up to (nthreads − 1) extra samples beyond tau. Now each thread
    # atomically claims a slot before doing any work; it exits immediately if
    # the quota is already met, so n_pairs ends up at exactly tau.
    # ---------------------------------------------------------
    phase1_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for tid in 1:nthreads
        ws = workspaces[tid]
        while true
            prev = Threads.atomic_add!(phase1_claimed, 1)  # returns old value
            prev >= tau && break                            # quota met, stop
            s, t = rand(1:n), rand(1:n)
            while s == t; t = rand(1:n); end
            path = sample_shortest_path!(ws, g, s, t)
            for v in path
                approx_local[tid][v] += 1
            end
            Threads.atomic_add!(n_pairs, 1)
        end
    end
    
    # ---------------------------------------------------------
    # PHASE 2: Main loop until stopping condition is met.
    # Each thread works in batches of 10 samples before the stopping condition
    # is rechecked. This amortises the cost of aggregation; the tradeoff is an
    # overshoot of at most 10 * nthreads samples past the convergence point.
    # ---------------------------------------------------------
    stop_flag = Threads.Atomic{Bool}(false)
    
    Threads.@threads for tid in 1:nthreads
        ws = workspaces[tid]
        
        while !stop_flag[] && n_pairs[] < omega
            # Do a small batch of work before checking status
            for _ in 1:10
                s, t = rand(1:n), rand(1:n)
                while s == t; t = rand(1:n); end
                
                path = sample_shortest_path!(ws, g, s, t)
                
                for v in path
                    approx_local[tid][v] += 1
                end
                Threads.atomic_add!(n_pairs, 1)
            end
            
            # Only thread 1 checks the stopping condition to avoid redundant work.
            # Thread-safety note: thread 1 reads approx_local[tid] for all tid
            # without a synchronisation fence. Each other thread writes only to its
            # own disjoint approx_local[tid] array. On x86-64 (TSO) aligned Int
            # reads/writes are naturally coherent; on weakly-ordered architectures
            # a fence (e.g. Base.Threads.fence()) would be needed for full safety.
            if tid == 1
                global_approx = zeros(Int, n)
                for t_approx in approx_local
                    global_approx .+= t_approx
                end
                
                # Sort to find tracked top nodes
                union_sample = Int(absolute ? k : min(n, k + 20))
                top_k_nodes = sortperm(global_approx, rev=true)[1:union_sample]
                
                if check_finished(global_approx, top_k_nodes, n_pairs[], k, err, delta_l_guess, delta_u_guess, omega, absolute)
                    Threads.atomic_xchg!(stop_flag, true)
                end
            end
        end
    end
    
    # Final aggregation
    final_approx = zeros(Int, n)
    for t_approx in approx_local
        final_approx .+= t_approx
    end
    
    return [final_approx[v] / n_pairs[] for v in 1:n]
end


"""
    compute_f(btilde, iter_num, delta_l, omega)

Computes the function f that bounds the betweenness of a vertex from below.
"""
function compute_f(btilde::Float64, iter_num::Int, delta_l::Float64, omega::Float64)
    tmp = (omega / iter_num) - (1.0 / 3.0)
    # Chernoff bound error
    err_chern = (log(1.0 / delta_l) / iter_num) * (-tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_l)))
    return min(err_chern, btilde)
end

"""
    compute_g(btilde, iter_num, delta_u, omega)

Computes the function g that bounds the betweenness of a vertex from above.
"""
function compute_g(btilde::Float64, iter_num::Int, delta_u::Float64, omega::Float64)
    tmp = (omega / iter_num) + (1.0 / 3.0)
    # Chernoff bound error
    err_chern = (log(1.0 / delta_u) / iter_num) * (tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_u)))
    return min(err_chern, 1.0 - btilde)
end

"""
    check_finished(approx, top_k_nodes, n_pairs, k, err, delta_l_guess, delta_u_guess, omega, absolute)

Evaluates whether the algorithm has met the stopping criteria.
"""
function check_finished(
    approx_counts::Vector{Int}, 
    top_k_nodes::Vector{Int}, 
    n_pairs::Int, 
    k::Int, 
    err::Float64, 
    delta_l_guess::Vector{Float64}, 
    delta_u_guess::Vector{Float64}, 
    omega::Float64, 
    absolute::Bool
)
    # Get current betweenness approximations for the tracked nodes
    bet = [approx_counts[v] / n_pairs for v in top_k_nodes]
    
    n_tracked = length(top_k_nodes)
    err_l = zeros(Float64, n_tracked)
    err_u = zeros(Float64, n_tracked)
    
    for i in 1:n_tracked
        v = top_k_nodes[i]
        err_l[i] = compute_f(bet[i], n_pairs, delta_l_guess[v], omega)
        err_u[i] = compute_g(bet[i], n_pairs, delta_u_guess[v], omega)
    end
    
    all_finished = true
    
    if absolute
        # Absolute error mode: all k nodes must have error < err
        for i in 1:k
            finished = (err_l[i] < err) && (err_u[i] < err)
            all_finished = all_finished && finished
        end
    else
        # Relative Top-K ranking mode
        for i in 1:n_tracked
            if i == 1
                if n_tracked > 1
                    finished = (bet[1] - err_l[1]) > (bet[2] + err_u[2])
                else
                    finished = true
                end
            elseif i < k
                finished = ((bet[i-1] - err_l[i-1]) > (bet[i] + err_u[i])) && 
                           ((bet[i] - err_l[i]) > (bet[i+1] + err_u[i+1]))
            elseif i == k
                if k < n_tracked
                    finished = ((bet[k-1] - err_l[k-1]) > (bet[k] + err_u[k])) &&
                               ((bet[k] - err_l[k]) > (bet[k+1] + err_u[k+1]))
                else
                    finished = (bet[k-1] - err_l[k-1]) > (bet[k] + err_u[k])
                end
            else
                finished = (bet[k] - err_l[k]) > (bet[i] + err_u[i])
            end
            
            # Also valid if error strictly falls below threshold
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
    KadabraWorkspace{T}

A memory workspace to prevent allocations during KADABRA's hot sampling loop.
Instantiate this once per graph and reuse it across all `sample_shortest_path!` calls.

Type parameters:
- `T`: vertex index integer type (e.g. `Int64`, `UInt8`), inferred from the graph.

Fix notes:
- `dist` uses `Vector{Int}` (not `Vector{T}`) to avoid silent BFS-layer wrap-around
  for small integer types: a `UInt8` graph could have diameter > 255.
- `n_paths` uses `Vector{Float64}` (not `Vector{UInt64}`) to prevent silent overflow
  of path counts on graphs with exponentially many shortest paths (e.g. grids).
  Floating-point arithmetic provides exact results up to 2^53 paths and saturates
  gracefully beyond that, preserving the correctness of proportional sampling.
"""
struct KadabraWorkspace{T<:Integer}
    ball_indicator::Vector{UInt8}
    n_paths::Vector{Float64}   # Float64: avoids UInt64 overflow for exponential path counts
    dist::Vector{Int}          # Int: avoids wrap-around for small T (e.g. UInt8 diameter > 255)
    preds::Vector{Vector{T}}
    
    cur_s::Vector{T}
    next_s::Vector{T}
    cur_t::Vector{T}
    next_t::Vector{T}
    
    sp_edges::Vector{Tuple{T, T}}
    visited_nodes::Vector{T} # Tracks exactly which nodes to reset
end

function KadabraWorkspace(g::AbstractGraph{T}) where {T}
    n = nv(g)
    
    preds = [Vector{T}() for _ in 1:n]
    for p in preds; sizehint!(p, 5); end
    
    cur_s = Vector{T}(); sizehint!(cur_s, n ÷ 4)
    next_s = Vector{T}(); sizehint!(next_s, n ÷ 4)
    cur_t = Vector{T}(); sizehint!(cur_t, n ÷ 4)
    next_t = Vector{T}(); sizehint!(next_t, n ÷ 4)
    
    sp_edges = Vector{Tuple{T, T}}(); sizehint!(sp_edges, 100)
    visited_nodes = Vector{T}(); sizehint!(visited_nodes, n ÷ 2)
    
    return KadabraWorkspace{T}(
        zeros(UInt8, n),
        zeros(Float64, n),   # n_paths: Float64 (overflow-safe)
        fill(typemax(Int), n), # dist: Int sentinel (no small-type wrap)
        preds,
        cur_s, next_s, cur_t, next_t,
        sp_edges,
        visited_nodes
    )
end

"""
    sample_shortest_path!(ws::KadabraWorkspace{T}, g, s, t[; dir=:out])

Sample a single shortest path uniformly at random using pre-allocated workspace memory.
Returns an empty path (not contributing to betweenness) when `s == t`.
"""
function sample_shortest_path!(ws::KadabraWorkspace{T}, g::AbstractGraph, s::Integer, t::Integer; dir=:out) where T
    # s == t contributes nothing to betweenness; return empty rather than [s]
    # to avoid a heap allocation on this dead-code branch.
    s == t && return T[]
    return if (dir == :out)
        _bb_bfs_sample!(ws, g, s, t, outneighbors, inneighbors)
    else
        _bb_bfs_sample!(ws, g, s, t, inneighbors, outneighbors)
    end
end

# Fix: parameterise neighbour functions as F1/F2 instead of the abstract `Function`
# type, so Julia can specialise _bb_bfs_sample! on the concrete function types
# (outneighbors / inneighbors). This enables inlining of the inner BFS loop,
# which is the hottest code path in the entire algorithm.
function _bb_bfs_sample!(
    ws::KadabraWorkspace{T}, 
    g::AbstractGraph{T}, 
    s::Integer, 
    t::Integer, 
    neighborfn_s::F1,
    neighborfn_t::F2
) where {T, F1, F2}
    s = T(s)
    t = T(t)
    
    # Extract arrays from workspace
    ball_indicator = ws.ball_indicator
    n_paths = ws.n_paths
    dist = ws.dist
    preds = ws.preds
    cur_s = ws.cur_s
    next_s = ws.next_s
    cur_t = ws.cur_t
    next_t = ws.next_t
    sp_edges = ws.sp_edges
    visited_nodes = ws.visited_nodes

    # Initialize S
    ball_indicator[s] = 0x01
    n_paths[s] = 1.0
    dist[s] = 0
    push!(cur_s, s)
    push!(visited_nodes, s)
    sum_degs_s = length(neighborfn_s(g, s))

    # Initialize T
    ball_indicator[t] = 0x02
    n_paths[t] = 1.0
    dist[t] = 0
    push!(cur_t, t)
    push!(visited_nodes, t)
    sum_degs_t = length(neighborfn_t(g, t))

    have_to_stop = false

    # Balanced Bidirectional BFS
    while !have_to_stop && (!isempty(cur_s) && !isempty(cur_t))
        
        # Decide which ball to expand based on frontier weight
        if sum_degs_s <= sum_degs_t 
            sum_degs_s = 0
            @inbounds for x in cur_s
                @inbounds for y in neighborfn_s(g, x)
                    if ball_indicator[y] == 0x00
                        ball_indicator[y] = 0x01
                        n_paths[y] = n_paths[x]
                        dist[y] = dist[x] + 1
                        push!(preds[y], x)
                        push!(next_s, y)
                        push!(visited_nodes, y) # Log for O(visited) cleanup
                        sum_degs_s += length(neighborfn_s(g, y))
                        
                    elseif ball_indicator[y] == 0x02
                        have_to_stop = true
                        push!(sp_edges, (x, y)) # (S-side, T-side)
                        
                    elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x01
                        n_paths[y] += n_paths[x]
                        push!(preds[y], x)
                    end
                end
            end
            empty!(cur_s)
            # Local pointer swap (does NOT update workspace fields; cleanup below
            # always calls empty! on both ws.cur_s and ws.next_s directly).
            cur_s, next_s = next_s, cur_s
            
        else
            sum_degs_t = 0
            @inbounds for x in cur_t
                @inbounds for y in neighborfn_t(g, x)
                    if ball_indicator[y] == 0x00
                        ball_indicator[y] = 0x02
                        n_paths[y] = n_paths[x]
                        dist[y] = dist[x] + 1
                        push!(preds[y], x)
                        push!(next_t, y)
                        push!(visited_nodes, y) # Log for O(visited) cleanup
                        sum_degs_t += length(neighborfn_t(g, y))
                        
                    elseif ball_indicator[y] == 0x01
                        have_to_stop = true
                        push!(sp_edges, (y, x)) # Note edge direction: y is on S-side, x is on T-side
                        
                    elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x02
                        n_paths[y] += n_paths[x]
                        push!(preds[y], x)
                    end
                end
            end
            empty!(cur_t)
            cur_t, next_t = next_t, cur_t # Local pointer swap
        end
    end

    # Default: disconnected pair → empty path
    path_s = Vector{T}()

    if !isempty(sp_edges)
        # 1. Weight the bridges and select one proportionally at random.
        # Fix: use Float64 arithmetic throughout to avoid UInt64 overflow on graphs
        # with exponentially many shortest paths (e.g. grids, dense random graphs).
        # The product n_paths[u] * n_paths[v] can easily exceed 2^64 for UInt64.
        tot_weight = 0.0
        @inbounds for (u_bridge, v_bridge) in sp_edges
            tot_weight += n_paths[u_bridge] * n_paths[v_bridge]
        end

        # Continuous uniform draw in [0, tot_weight); equivalent to rand(1:N) for integers.
        rand_val = rand() * tot_weight
        cur_weight = 0.0
        selected_edge = sp_edges[1]
        
        @inbounds for (u_bridge, v_bridge) in sp_edges
            cur_weight += n_paths[u_bridge] * n_paths[v_bridge]
            if cur_weight >= rand_val
                selected_edge = (u_bridge, v_bridge)
                break
            end
        end

        # 2. Backtrack to construct the single sample path
        _backtrack!(path_s, selected_edge[1], s, preds, n_paths)
        
        path_t = Vector{T}()
        _backtrack!(path_t, selected_edge[2], t, preds, n_paths)
        
        reverse!(path_s)
        append!(path_s, path_t)
    end

    # 3. O(visited) Cleanup - reset state exactly for the nodes we touched.
    # Both ws.cur_s/ws.next_s are cleared directly (not through the local aliases
    # cur_s/next_s, which may have been swapped) to ensure a clean workspace
    # regardless of how many swap iterations the BFS performed.
    @inbounds for v in visited_nodes
        ball_indicator[v] = 0x00
        n_paths[v] = 0.0
        dist[v] = typemax(Int)
        empty!(preds[v])
    end
    
    empty!(visited_nodes)
    empty!(ws.cur_s)
    empty!(ws.next_s)
    empty!(ws.cur_t)
    empty!(ws.next_t)
    empty!(sp_edges)

    return path_s
end

# Internal backtracking helper.
# Fix: n_paths changed from Vector{UInt64} to Vector{Float64} (consistent with
# KadabraWorkspace). Random selection uses rand() * tot (continuous) instead of
# rand(1:tot) (integer), which avoids UInt64 overflow for large path counts.
function _backtrack!(path::Vector{T}, curr::T, target::T, preds::Vector{Vector{T}}, n_paths::Vector{Float64}) where {T}
    @inbounds while curr != target
        push!(path, curr)
        parents = preds[curr]
        
        if length(parents) == 1
            curr = parents[1]
        else
            tot = sum(p -> n_paths[p], parents; init=0.0)
            r = rand() * tot          # continuous draw in [0, tot)
            c = 0.0
            for p in parents
                c += n_paths[p]
                if c >= r
                    curr = p
                    break
                end
            end
        end
    end
    push!(path, target)
end