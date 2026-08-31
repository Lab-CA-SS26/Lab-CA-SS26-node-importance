import re

with open("src/kadabra.jl", "r") as f:
    content = f.read()

# 1. Replace Threads.@spawn closure with a function call
spawn_match = re.search(r'Threads\.@spawn begin(.*?)^\s*end', content, re.MULTILINE | re.DOTALL)
if spawn_match:
    body = spawn_match.group(1)
    # The body has locals like `local ws = KadabraWorkspace(g)`
    # We will extract this into `_kadabra_worker_task!`
    worker_fn = f"""
function _kadabra_worker_task!(
    g::AbstractGraph{{T}},
    n::Integer,
    seed::UInt64,
    tid::Integer,
    n_pairs::Threads.Atomic{{Int}},
    tau_per_thread::Integer,
    omega::Float64,
    stop_flag::Threads.Atomic{{Bool}},
    check_lock::Threads.SpinLock,
    approx_local::Vector{{Vector{{Int}}}},
    global_approx::Vector{{Int}},
    top_k_nodes::Vector{{Int}},
    union_sample::Integer,
    absolute::Bool,
    k::Integer,
    err::Float64,
    delta_l_guess::Vector{{Float64}},
    delta_u_guess::Vector{{Float64}},
    bet_buf::Vector{{Float64}},
    err_l_buf::Vector{{Float64}},
    err_u_buf::Vector{{Float64}},
    endpoints::Bool
) where {{T}}
{body}
end
"""
    # Replace the spawn block with a call to the new function
    call = """
                Threads.@spawn _kadabra_worker_task!(
                    g, n, thread_seeds[i], i, n_pairs, tau_per_thread,
                    omega, stop_flag, check_lock, approx_local, global_approx,
                    top_k_nodes, union_sample, absolute, k, err, delta_l_guess,
                    delta_u_guess, bet_buf, err_l_buf, err_u_buf, endpoints
                )
    """
    content = content[:spawn_match.start()] + call.strip() + "\n" + content[spawn_match.end():]
    
    # Prepend the worker function right before kadabra_centrality
    idx = content.find("function kadabra_centrality")
    content = content[:idx] + worker_fn + "\n\n" + content[idx:]

# 2. Add @inline to BFS functions
content = content.replace("function _sample_shortest_path!(", "@inline function _sample_shortest_path!(")
content = content.replace("function _backtrack!(", "@inline function _backtrack!(")

# 3. Remove GC.safepoint()
content = content.replace("GC.safepoint()", "")

# 4. Add @inbounds in inner BFS loops
content = re.sub(
    r'(for y in neighborfn_s\(g, x\))(.*?)(?=^\s*end)',
    r'\1\n                            @inbounds begin\2\n                            end',
    content,
    flags=re.MULTILINE | re.DOTALL
)
content = re.sub(
    r'(for y in neighborfn_t\(g, x\))(.*?)(?=^\s*end)',
    r'\1\n                            @inbounds begin\2\n                            end',
    content,
    flags=re.MULTILINE | re.DOTALL
)

with open("src/kadabra_opt.jl", "w") as f:
    f.write(content)
