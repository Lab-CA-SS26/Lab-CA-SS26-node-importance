using Pkg
Pkg.activate(".")
Pkg.add([
    "Graphs",
    "SparseArrays",
    "DataFrames",
    "CSV",
    "Flux",
    "JLD2",
    "Random",
    "Statistics",
    "Optimisers",
    "CUDA",
    "cuDNN",
])

using Graphs
using SparseArrays
using CSV
using DataFrames
using Flux
using CUDA
using cuDNN
using JLD2
using Random
using Statistics
using Optimisers
using Dates

include("BRAVAGNN.jl")
using .BRAVAGNN

const TRAINING_DIR = joinpath(dirname(@__DIR__), "Instances", "Training")
const BATCH_SIZE = 1024
const EPOCHS = 10
const LEARNING_RATE = 5e-3

# The PageRank input channel is OFF by default: it is our own addition, the BRAVA-GNN
# paper never mentions it, and exactly 1 of the 846 configurations in the authors'
# `all_results.csv` carries the `_pr` suffix. Enabling it costs ~12.6 tau_b points on
# average and collapses graphs that the clique mask prunes heavily (email-EuAll: 0.99
# -> 0.46), because PageRank is not zeroed by the mask and so hands every pruned
# vertex a distinct score instead of leaving the block tied at the bottom.
const USE_PR = ("--pr" in ARGS)

# Pair sampling for the margin ranking loss.
#   :authors    -- upstream `utils.loss_cal`: draw two RANK positions uniformly and
#                  label by which rank is higher, keeping ties (they get an arbitrary
#                  but fixed label). This is what produced the paper's numbers.
#   :drop-ties  -- draw two VERTICES uniformly and discard pairs of equal betweenness.
const LOSS = let a = findfirst(s -> startswith(s, "--loss="), ARGS)
    a === nothing ? :authors : Symbol(split(ARGS[a], "=")[2])
end
LOSS in (:authors, Symbol("drop-ties")) || error("--loss= must be 'authors' or 'drop-ties'")

# Seed 1 is the canonical model. Training is not bit-reproducible even so --- cuDNN
# kernels and the sparse matmul reduce in nondeterministic order --- but seeding fixes
# the initialisation and the pair sampling, which is what separates one run from
# another by more than rounding.
const SEED = let a = findfirst(s -> startswith(s, "--seed="), ARGS)
    a === nothing ? 1 : parse(Int, split(ARGS[a], "=")[2])
end

# Parse the graph depending on whether it is BO/UDHY (Directed) or SBO (Undirected)
function load_training_graph(filepath::String, is_directed::Bool)
    edges_list = Tuple{Int,Int}[]
    max_node = 0
    for line in eachline(filepath)
        parts = split(strip(line))
        if length(parts) >= 2
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            push!(edges_list, (u, v))
            max_node = max(max_node, u, v)
        end
    end

    # 0-indexed adjustment
    min_node = isempty(edges_list) ? 0 : minimum(min(u, v) for (u, v) in edges_list)
    shift = min_node == 0 ? 1 : 0

    g = is_directed ? SimpleDiGraph(max_node + shift) : SimpleGraph(max_node + shift)
    for (u, v) in edges_list
        add_edge!(g, u + shift, v + shift)
    end
    return g
end

function train()
    println("--- BRAVA-GNN Training Script ---")
    println("  config     : degree_mix_mass_6, nhid=12, L=2, dropout=0.3, " *
            "epochs=$EPOCHS, Adam lr=$LEARNING_RATE")
    println("  use_pr     : $USE_PR")
    println("  loss       : $LOSS")
    println("  seed       : $SEED")
    println("  cuda       : $(CUDA.functional())")

    Random.seed!(SEED)
    CUDA.functional() && CUDA.seed!(SEED)

    # 1. Load Data
    graph_files = filter(f -> endswith(f, ".txt"), readdir(TRAINING_DIR))
    println("Found $(length(graph_files)) training graphs.")
    for fam in unique(map(f -> replace(f, r"_\d+\.txt$" => ""), graph_files))
        println("    $fam: $(count(f -> startswith(f, fam * "_"), graph_files))")
    end

    training_data = []

    for gf in graph_files
        name = replace(gf, ".txt" => "")
        is_directed = occursin("Dir", name)

        # Load graph
        g = load_training_graph(joinpath(TRAINING_DIR, gf), is_directed)

        # Load scores
        scores_df = CSV.read(joinpath(TRAINING_DIR, "$(name)_scores.csv"), DataFrame)
        scores = scores_df.betweenness

        # Precompute PageRank feature once for the graph
        A = Float32.(sparse(g))
        A_t = A'

        # Apply preprocessing heuristic
        mask = brava_clique_mask(g)
        A = spdiagm(mask) * A
        A_t = spdiagm(mask) * A_t

        pr_feat = compute_pagerank_feature(A)

        X_out = compute_degree_masses(A, pr_feat, 6; use_pr = USE_PR)
        X_in = compute_degree_masses(A_t, pr_feat, 6; use_pr = USE_PR)

        # GPU transfer if available
        device = CUDA.functional() ? gpu : cpu

        A_gpu = device(A)
        A_t_gpu = device(A_t)
        X_in_gpu = device(X_in)
        X_out_gpu = device(X_out)

        # Vertices in descending true betweenness; the :authors loss samples rank
        # positions in this order rather than vertices.
        order = sortperm(scores, rev = true)

        push!(
            training_data,
            (
                A = A_gpu,
                A_t = A_t_gpu,
                X_in = X_in_gpu,
                X_out = X_out_gpu,
                scores = scores,
                order = order,
            ),
        )
    end

    device = CUDA.functional() ? gpu : cpu

    # 2. Initialize Model
    model = BRAVAModel(m_hops = 6, hidden_dim = 12, num_layers = 2, use_pr = USE_PR) |> device
    opt_state = Flux.setup(Flux.Adam(LEARNING_RATE), model)

    # 3. Training Loop
    for epoch = 1:EPOCHS
        epoch_loss = 0.0
        n_batches = 0

        for virtual_copy = 1:50
            Random.shuffle!(training_data)

            for data in training_data
                N = length(data.scores)
                k = N * 20

                if LOSS === :authors
                    # utils.loss_cal: sample two rank positions, label by rank order.
                    # Ties keep a label (arbitrary but fixed by the sort) rather than
                    # being dropped -- faithful to the run that produced the paper.
                    i1 = rand(1:N, k)
                    i2 = rand(1:N, k)
                    Y = sign.(i2 .- i1)
                    U = data.order[i1]
                    V = data.order[i2]
                else
                    U = rand(1:N, k)
                    V = rand(1:N, k)
                    diffs = data.scores[U] .- data.scores[V]
                    Y = sign.(diffs)
                    valid_idx = findall(!iszero, diffs)
                    U = U[valid_idx]
                    V = V[valid_idx]
                    Y = Y[valid_idx]
                end

                U_dev = device(U)
                V_dev = device(V)
                Y_dev = device(Float32.(Y))

                loss_val, grads = Flux.withgradient(model) do m
                    preds = m(data.A, data.A_t, data.X_in, data.X_out)
                    margin_ranking_loss(preds, U_dev, V_dev, Y_dev)
                end

                Flux.update!(opt_state, model, grads[1])

                epoch_loss += loss_val
                n_batches += 1
            end
        end

        println(
            "Epoch $epoch / $EPOCHS - Avg Loss: $(round(epoch_loss / max(n_batches, 1), digits=4))",
        )
    end

    # 4. Save Model (Move back to CPU before saving)
    model = model |> cpu
    # The default configuration -- no PageRank channel, upstream's pair sampling -- is
    # the one the paper reports, so it claims the canonical filename that
    # run_experiments.jl and brava_centrality load. Variants get a suffix.
    suffix = (USE_PR ? "_pr" : "") * (LOSS === :authors ? "" : "_droptie") *
             (SEED == 1 ? "" : "_S$SEED")
    cache_dir = joinpath(dirname(@__DIR__), "benchmark", "cache")
    save_path = joinpath(cache_dir, "bravagnn_weights$(suffix).jld2")
    @save save_path model
    open(joinpath(cache_dir, "bravagnn_weights$(suffix).config.txt"), "w") do io
        println(io, "trained      : $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "init_type    : degree_mix_mass_6$(USE_PR ? "_pr" : "")")
        println(io, "nhid         : 12")
        println(io, "num_layers   : 2")
        println(io, "dropout      : 0.3")
        println(io, "epochs       : $EPOCHS")
        println(io, "optimizer    : Adam(lr=$LEARNING_RATE)")
        println(io, "loss         : $LOSS")
        println(io, "seed         : $SEED")
        ngraphs = count(f -> endswith(f, ".txt"), readdir(TRAINING_DIR))
        println(io, "training_set : $ngraphs graphs from $TRAINING_DIR")
    end
    println("Training complete. Model saved to $save_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
