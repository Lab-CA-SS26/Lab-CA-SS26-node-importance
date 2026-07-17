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
- `Vector{Float64}`: A vector of length `nv(g)` containing the estimated betweenness centrality 
  for each vertex.
"""
function kadabra_centrality(g::AbstractGraph{T}, k::Int, err::Float64, delta::Float64; start_factor::Int=100, endpoints::Bool=false, normalize::Symbol=:graphs) where T
    # --- Input validation ---
    nv(g) >= 2    || throw(ArgumentError("Graph must have at least 2 vertices (got $(nv(g)))"))
    err  > 0      || throw(ArgumentError("err must be positive (got $err)"))
    0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1) (got $delta)"))

    n = nv(g)
    absolute = (k == 0)
    k = Int(k == 0 ? n : min(k, n))
    
    # Estimate diameter using AllCCUpperBound
    diam_est = max(estimate_diameter(g), 2.0)
    
    # Sample-count upper bound
    omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log2(1.0 / delta))
    tau = max(round(Int, omega / start_factor), 1)
    
    # Thread-local storage to avoid locking when updating centralities
    nthreads = Threads.nthreads()
    approx_local = [zeros(Int, n) for _ in 1:nthreads]
    workspaces = [KadabraWorkspace(g) for _ in 1:nthreads]
    n_pairs = Threads.Atomic{Int}(0)
    
    # Pre-allocate Thread 1's aggregation buffer for Phase 2 checks
    global_approx = zeros(Int, n)
    union_sample = Int(absolute ? k : min(n, k + 20))
    
    # Adaptive check interval scales with burn-in tau to prevent hanging
    check_interval = max(10, tau ÷ 100)
    
    delta_l_guess = fill(delta / (4 * n), n)
    delta_u_guess = fill(delta / (4 * n), n)
    
    # ---------------------------------------------------------
    # PHASE 1: Initial burn-in sampling (exactly tau samples)
    # ---------------------------------------------------------
    phase1_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for tid in 1:nthreads
        ws = workspaces[tid]
        counts = approx_local[tid]
        while true
            prev = Threads.atomic_add!(phase1_claimed, 1)
            prev >= tau && break                            
            s, t = rand(1:n), rand(1:n)
            while s == t; t = rand(1:n); end
            
            # Zero-allocation increment
            sample_shortest_path!(counts, ws, g, s, t; endpoints=endpoints)
            Threads.atomic_add!(n_pairs, 1)
        end
    end
    
    # ---------------------------------------------------------
    # PHASE 2: Main loop until stopping condition is met
    # ---------------------------------------------------------
    stop_flag = Threads.Atomic{Bool}(false)
    
    Threads.@threads for tid in 1:nthreads
        ws = workspaces[tid]
        counts = approx_local[tid]
        
        while !stop_flag[] && n_pairs[] < omega
            # Small batch before status check
            for _ in 1:check_interval
                s, t = rand(1:n), rand(1:n)
                while s == t; t = rand(1:n); end
                
                sample_shortest_path!(counts, ws, g, s, t; endpoints=endpoints)
                Threads.atomic_add!(n_pairs, 1)
            end
            
            # Only thread 1 handles the heavy stopping calculation
            if tid == 1
                fill!(global_approx, 0)
                for t_approx in approx_local
                    global_approx .+= t_approx
                end
                
                # O(N log K) partial sort instead of O(N log N) full sort
                top_k_nodes = partialsortperm(global_approx, 1:union_sample, rev=true)
                
                if check_finished(global_approx, top_k_nodes, n_pairs[], k, err, delta_l_guess, delta_u_guess, omega, absolute)
                    Threads.atomic_xchg!(stop_flag, true)
                end
            end
        end
    end
    
    # Final aggregation
    fill!(global_approx, 0)
    for t_approx in approx_local
        global_approx .+= t_approx
    end
    
    res = [global_approx[v] / n_pairs[] for v in 1:n]
    
    scale = 1.0
    if normalize == :graphs
        if n > 2
            # Graphs.jl normalizes by (N-1)(N-2) for directed, and (N-1)(N-2)/2 for undirected
            graphs_norm = is_directed(g) ? ((n - 1.0) * (n - 2.0)) : ((n - 1.0) * (n - 2.0)) / 2.0
            scale = (n * (n - 1.0)) / graphs_norm
        else
            scale = 0.0
        end
    elseif normalize == :none
        scale = is_directed(g) ? (n * (n - 1.0)) : (n * (n - 1.0)) / 2.0
    elseif normalize == :kadabra
        scale = 1.0 # KADABRA inherently outputs the fraction of pairs, which is already normalized by N(N-1)
    else
        throw(ArgumentError("Unknown normalize option: $normalize. Use :graphs, :kadabra, or :none."))
    end
    
    if scale != 1.0
        res .*= scale
    end
    
    return res
end

function kadabra_centrality(g::AbstractGraph{T}, k::Int, err::Float64, delta::Float64, distmx::AbstractMatrix; kwargs...) where T
    throw(ArgumentError("KADABRA centrality does not support weighted graphs. Please do not provide a distmx argument."))
end

"""
    kadabra_top_k(g::AbstractGraph, k::Int, err::Float64, delta::Float64; kwargs...)

Convenience wrapper that runs KADABRA and efficiently extracts only the top `k` most central nodes.
Returns a vector of `NamedTuple`s containing the `node` ID and its `centrality` score, sorted in descending order.
"""
function kadabra_top_k(g::AbstractGraph{T}, k::Int, err::Float64, delta::Float64; kwargs...) where T
    k > 0 || throw(ArgumentError("k must be greater than 0 to extract top k nodes"))
    
    # Run the standard Kadabra algorithm
    centralities = kadabra_centrality(g, k, err, delta; kwargs...)
    
    # Efficiently find the indices of the top k nodes without doing a full sort
    top_nodes = partialsortperm(centralities, 1:k, rev=true)
    
    return [(node = v, centrality = centralities[v]) for v in top_nodes]
end


"""
    compute_f(btilde, iter_num, delta_l, omega)

Computes the Chernoff bound error function `f` that bounds the betweenness of a vertex from below.
Evaluated dynamically during Phase 2 to determine if the lower bound of the confidence interval
satisfies the stopping condition.
"""
function compute_f(btilde::Float64, iter_num::Int, delta_l::Float64, omega::Float64)
    tmp = (omega / iter_num) - (1.0 / 3.0)
    err_chern = (log(1.0 / delta_l) / iter_num) * (-tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_l)))
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
    err_chern = (log(1.0 / delta_u) / iter_num) * (tmp + sqrt(tmp^2 + 2.0 * btilde * omega / log(1.0 / delta_u)))
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
    absolute::Bool
)
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
        for i in 1:k
            finished = (err_l[i] < err) && (err_u[i] < err)
            all_finished = all_finished && finished
        end
    else
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
            
            finished = finished || ((err_l[i] < err) && (err_u[i] < err))
            all_finished = all_finished && finished
        end
    end
    
    return all_finished
end


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

struct KadabraWorkspace{T<:Integer}
    ball_indicator::Vector{UInt8}
    n_paths::Vector{Float64}   
    dist::Vector{Int}          
    preds::Vector{Vector{T}}
    
    cur_s::Vector{T}
    next_s::Vector{T}
    cur_t::Vector{T}
    next_t::Vector{T}
    
    sp_edges::Vector{Tuple{T, T}}
    visited_nodes::Vector{T} 
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
        zeros(Float64, n),   
        fill(typemax(Int), n), 
        preds,
        cur_s, next_s, cur_t, next_t,
        sp_edges,
        visited_nodes
    )
end

"""
    sample_shortest_path!(counts::Vector{Int}, ws::KadabraWorkspace{T}, g, s, t[; dir=:out, endpoints=false])

Sample a single shortest path uniformly at random using pre-allocated workspace memory,
and directly increment `counts` for every vertex on the path. No heap allocations occur.
"""
function sample_shortest_path!(counts::Vector{Int}, ws::KadabraWorkspace{T}, g::AbstractGraph{T}, s::Integer, t::Integer; dir=:out, endpoints::Bool=false) where T
    s == t && return
    if (dir == :out)
        _bb_bfs_sample!(counts, ws, g, s, t, outneighbors, inneighbors, endpoints)
    else
        _bb_bfs_sample!(counts, ws, g, s, t, inneighbors, outneighbors, endpoints)
    end
end

"""
    _bb_bfs_sample!(counts, ws, g, s, t, neighborfn_s, neighborfn_t, endpoints)

Internal helper function that performs a Balanced Bidirectional BFS from source `s` and target `t`.
The search expands the frontier with the smallest sum of out-degrees to minimize edge traversals.
When the frontiers intersect, it selects a single bridge edge uniformly at random (weighted by 
the number of shortest paths crossing it) and backtracks to construct the sampled path.
The nodes on the resulting path are incremented directly in the `counts` array without allocating memory.
"""
function _bb_bfs_sample!(
    counts::Vector{Int},
    ws::KadabraWorkspace{T}, 
    g::AbstractGraph{T}, 
    s::Integer, 
    t::Integer, 
    neighborfn_s::F1,
    neighborfn_t::F2,
    endpoints::Bool
) where {T, F1, F2}
    s = T(s)
    t = T(t)
    
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

    ball_indicator[s] = 0x01
    n_paths[s] = 1.0
    dist[s] = 0
    push!(cur_s, s)
    push!(visited_nodes, s)
    sum_degs_s = length(neighborfn_s(g, s))

    ball_indicator[t] = 0x02
    n_paths[t] = 1.0
    dist[t] = 0
    push!(cur_t, t)
    push!(visited_nodes, t)
    sum_degs_t = length(neighborfn_t(g, t))

    have_to_stop = false

    while !have_to_stop && (!isempty(cur_s) && !isempty(cur_t))
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
                        push!(visited_nodes, y) 
                        sum_degs_s += length(neighborfn_s(g, y))
                        
                    elseif ball_indicator[y] == 0x02
                        have_to_stop = true
                        push!(sp_edges, (x, y)) 
                        
                    elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x01
                        n_paths[y] += n_paths[x]
                        push!(preds[y], x)
                    end
                end
            end
            empty!(cur_s)
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
                        push!(visited_nodes, y) 
                        sum_degs_t += length(neighborfn_t(g, y))
                        
                    elseif ball_indicator[y] == 0x01
                        have_to_stop = true
                        push!(sp_edges, (y, x)) 
                        
                    elseif dist[y] == dist[x] + 1 && ball_indicator[y] == 0x02
                        n_paths[y] += n_paths[x]
                        push!(preds[y], x)
                    end
                end
            end
            empty!(cur_t)
            cur_t, next_t = next_t, cur_t 
        end
    end

    if !isempty(sp_edges)
        tot_weight = 0.0
        @inbounds for (u_bridge, v_bridge) in sp_edges
            tot_weight += n_paths[u_bridge] * n_paths[v_bridge]
        end

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

        _backtrack!(counts, selected_edge[1], s, preds, n_paths, endpoints)
        _backtrack!(counts, selected_edge[2], t, preds, n_paths, endpoints)
    end

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

    return
end

"""
    _backtrack!(counts, curr, target, preds, n_paths, endpoints)

Internal helper that reconstructs a single shortest path by backtracking from the `curr` node 
towards the `target` node using the predecessor map `preds` generated during the BFS.
If multiple optimal predecessors exist, one is selected randomly weighted by the number of 
shortest paths `n_paths` arriving through that predecessor.
The thread-local `counts` buffer is incremented in-place for every node visited.
"""
function _backtrack!(counts::Vector{Int}, curr::T, target::T, preds::Vector{Vector{T}}, n_paths::Vector{Float64}, endpoints::Bool) where {T}
    @inbounds while curr != target
        counts[curr] += 1
        parents = preds[curr]
        
        if length(parents) == 1
            curr = parents[1]
        else
            tot = sum(p -> n_paths[p], parents; init=0.0)
            r = rand() * tot          
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
    if endpoints
        @inbounds counts[target] += 1
    end
end