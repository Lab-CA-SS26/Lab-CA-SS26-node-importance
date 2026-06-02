# kadabra.jl
using Graphs

function kadabra_centrality(g::AbstractGraph, k::Int, err::Float64, delta::Float64; start_factor::Int=100)
    n = nv(g)
    absolute = (k == 0)
    k = k == 0 ? n : min(k, n)
    
    # Estimate diameter (In C++, they use estimate_diameter() with AllCCUpperBound)
    # For now, placeholder to a known upper bound or basic estimate
    diam_est = 5.0 
    
    # Calculate initial omega
    omega = 0.5 / (err^2) * (log2(diam_est - 1.0) + 1.0 + log(0.5 / delta))
    tau = round(Int, omega / start_factor)
    
    # Thread-local storage to avoid locking when updating centralities
    nthreads = max(Threads.nthreads(), 4)
    approx_local = [zeros(Int, n) for _ in 1:nthreads]
    workspaces = [KadabraWorkspace(g) for _ in 1:nthreads]
    for i in 1:nthreads
        workspaces[i] = KadabraWorkspace(g)
    end
    n_pairs = Threads.Atomic{Int}(0)
    
    # Arrays for delta optimization (using uniform delta as baseline)
    delta_l_guess = fill(delta / (4 * n), n)
    delta_u_guess = fill(delta / (4 * n), n)
    
    # ---------------------------------------------------------
    # PHASE 1: Initial burn-in sampling (Tau iterations)
    # ---------------------------------------------------------
    Threads.@threads for _ in 1:tau
        tid = min(Threads.threadid(), nthreads)
        ws = workspaces[tid]
        
        # Pick random distinct s, t
        s, t = rand(1:n), rand(1:n)
        while s == t; t = rand(1:n); end
        
        path = sample_shortest_path!(ws, g, s, t)
        
        # Update thread-local path counts
        for v in path
            approx_local[tid][v] += 1
        end
        Threads.atomic_add!(n_pairs, 1)
    end
    
    # NOTE: In C++, compute_delta_guess() runs here to optimize the delta arrays.
    # You can plug that heuristic in here later.
    
    # ---------------------------------------------------------
    # PHASE 2: Main loop until stopping condition is met
    # ---------------------------------------------------------
    stop_flag = Threads.Atomic{Bool}(false)
    
    Threads.@threads for _ in 1:typemax(Int)
        if stop_flag[] || n_pairs[] >= omega
            break
        end
        
        tid = min(Threads.threadid(), nthreads)
        ws = workspaces[tid]
        
        # Do a small batch of work before checking status (C++ uses a batch of 10)
        for _ in 1:10
            s, t = rand(1:n), rand(1:n)
            while s == t; t = rand(1:n); end
            
            path = sample_shortest_path!(ws, g, s, t)
            
            for v in path
                approx_local[tid][v] += 1
            end
            Threads.atomic_add!(n_pairs, 1)
        end
        
        # Only thread 1 handles the heavy stopping calculation to avoid overhead
        if tid == 1
            # Aggregate centralities across threads
            global_approx = zeros(Int, n)
            for t_approx in approx_local
                global_approx .+= t_approx
            end
            
            # Sort to find top k
            top_k_nodes = sortperm(global_approx, rev=true)[1:min(k, n)]
            
            if check_finished(global_approx, top_k_nodes, n_pairs[], k, err, delta_l_guess, delta_u_guess, omega, absolute)
                Threads.atomic_xchg!(stop_flag, true)
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
    # Get current betweenness approximations for the top k
    bet = [approx_counts[v] / n_pairs for v in top_k_nodes]
    
    err_l = zeros(Float64, length(top_k_nodes))
    err_u = zeros(Float64, length(top_k_nodes))
    
    all_finished = true
    
    # Calculate bounds
    for i in 1:length(top_k_nodes)
        v = top_k_nodes[i]
        err_l[i] = compute_f(bet[i], n_pairs, delta_l_guess[v], omega)
        err_u[i] = compute_g(bet[i], n_pairs, delta_u_guess[v], omega)
    end
    
    if absolute
        # Absolute error mode (k=0 in C++)
        for i in 1:k
            finished = (err_l[i] < err) && (err_u[i] < err)
            all_finished = all_finished && finished
        end
    else
        # Relative Top-K ranking mode
        for i in 1:k
            if i == 1
                finished = (bet[i] - err_l[i]) > (bet[i+1] + err_u[i+1])
            elseif i < k
                finished = ((bet[i-1] - err_l[i-1]) > (bet[i] + err_u[i])) && 
                           ((bet[i] - err_l[i]) > (bet[i+1] + err_u[i+1]))
            else
                finished = (bet[k-1] - err_u[k-1]) > (bet[i] + err_u[i])
            end
            
            # Also valid if error strictly falls below threshold
            finished = finished || ((err_l[i] < err) && (err_u[i] < err))
            all_finished = all_finished && finished
        end
    end
    
    return all_finished
end



"""

Sampling Things
"""


"""
    KadabraWorkspace{T}

A memory workspace to prevent allocations during KADABRA's hot sampling loop.
Instantiate this once per graph and reuse it across all `sample_shortest_path!` calls.
"""
struct KadabraWorkspace{T<:Integer}
    ball_indicator::Vector{UInt8}
    n_paths::Vector{UInt64}
    dist::Vector{T}
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
        zeros(UInt64, n),
        fill(typemax(T), n),
        preds,
        cur_s, next_s, cur_t, next_t,
        sp_edges,
        visited_nodes
    )
end

"""
    sample_shortest_path!(ws::KadabraWorkspace, g, s, t[; dir=:out])

Sample a single shortest path uniformly at random using pre-allocated workspace memory.
"""
function sample_shortest_path!(ws::KadabraWorkspace, g::AbstractGraph, s::Integer, t::Integer; dir=:out)
    s == t && return [s]
    return if (dir == :out)
        _bb_bfs_sample!(ws, g, s, t, outneighbors, inneighbors)
    else
        _bb_bfs_sample!(ws, g, s, t, inneighbors, outneighbors)
    end
end

function _bb_bfs_sample!(
    ws::KadabraWorkspace{T}, 
    g::AbstractGraph{T}, 
    s::Integer, 
    t::Integer, 
    neighborfn_s::Function, 
    neighborfn_t::Function
) where {T}
    
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
    n_paths[s] = 1
    dist[s] = 0
    push!(cur_s, s)
    push!(visited_nodes, s)
    sum_degs_s = length(neighborfn_s(g, s))

    # Initialize T
    ball_indicator[t] = 0x02
    n_paths[t] = 1
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
            cur_s, next_s = next_s, cur_s # Local pointer swap
            
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

    # Default disconnected path
    path_s = Vector{T}()

    if !isempty(sp_edges)
        # 1. Weight the bridges and select one uniformly at random
        tot_weight = zero(UInt64)
        @inbounds for (u_bridge, v_bridge) in sp_edges
            tot_weight += n_paths[u_bridge] * n_paths[v_bridge]
        end

        rand_val = rand(1:tot_weight)
        cur_weight = zero(UInt64)
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

    # 3. O(visited) Cleanup - reset state exactly for the nodes we touched
    @inbounds for v in visited_nodes
        ball_indicator[v] = 0x00
        n_paths[v] = 0
        dist[v] = typemax(T)
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

# Internal backtracking helper
function _backtrack!(path::Vector{T}, curr::T, target::T, preds::Vector{Vector{T}}, n_paths::Vector{UInt64}) where {T}
    @inbounds while curr != target
        push!(path, curr)
        parents = preds[curr]
        
        if length(parents) == 1
            curr = parents[1]
        else
            tot = sum(p -> n_paths[p], parents; init=zero(UInt64))            
            r = rand(1:tot)
            c = zero(UInt64)
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