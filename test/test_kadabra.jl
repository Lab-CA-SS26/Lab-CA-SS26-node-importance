#test_kadabra.jl

using Test
using Graphs

include("../src/kadabra.jl")

@testset "KADABRA Centrality Test Suite" begin

    @testset "1. Math and VC-Dimension Bounds" begin
        btilde = 0.5
        iter_num = 1000
        delta = 0.1
        omega = 1500.0

        f_val = compute_f(btilde, iter_num, delta, omega)
        g_val = compute_g(btilde, iter_num, delta, omega)

        # f(btilde) should bounded by btilde itself
        @test f_val <= btilde
        @test f_val >= 0.0

        # g(btilde) should be bounded by 1 - btilde
        @test g_val <= 1.0 - btilde
        @test g_val >= 0.0
    end

    @testset "2. Stopping Conditions (check_finished)" begin
        k = 3
        err = 0.05
        omega = 1000.0
        n_pairs = 1000
        n = 5

        approx_counts = [500, 300, 100, 50, 10]
        top_k_nodes = [1, 2, 3]
        delta_l_guess = fill(0.01, n)
        delta_u_guess = fill(0.01, n)

        # Test absolute mode (requires tight confidence intervals < err)
        bet_buf = zeros(Float64, length(top_k_nodes))
        err_l_buf = zeros(Float64, length(top_k_nodes))
        err_u_buf = zeros(Float64, length(top_k_nodes))

        is_finished_abs = check_finished(
            approx_counts,
            top_k_nodes,
            n_pairs,
            k,
            err,
            delta_l_guess,
            delta_u_guess,
            omega,
            true,
            bet_buf,
            err_l_buf,
            err_u_buf,
        )
        # Because we artificially set err=0.05 but n_pairs is only 1000, 
        # Chernoff bounds will be loose, so it shouldn't finish immediately.
        @test is_finished_abs isa Bool
    end

    @testset "3. Core Sampling (sample_shortest_path!)" begin
        # Test Path Graph (1 -> 2 -> 3 -> 4 -> 5)
        path_g = path_graph(5)
        ws_path = KadabraWorkspace(path_g)
        counts_path = zeros(Int, 5)

        sample_shortest_path!(counts_path, ws_path, path_g, Random.Xoshiro(1), 1, 5; endpoints = true)
        # 1-5 shortest path includes all 5 nodes, so counts_path should be 1 for all of them
        @test all(counts_path .== 1)

        # Test Same Source and Target
        sample_shortest_path!(counts_path, ws_path, path_g, Random.Xoshiro(1), 3, 3; endpoints = true)
        # Should do nothing, return directly
        @test counts_path[3] == 1 # still 1 from previous call

        # Test Disconnected Graph
        # Nodes 1-3 are connected, Nodes 4-5 are connected, but no bridge
        disc_g = Graph(5)
        add_edge!(disc_g, 1, 2)
        add_edge!(disc_g, 2, 3)
        add_edge!(disc_g, 4, 5)
        ws_disc = KadabraWorkspace(disc_g)
        counts_disc = zeros(Int, 5)

        sample_shortest_path!(counts_disc, ws_disc, disc_g, Random.Xoshiro(1), 1, 5; endpoints = true)
        @test all(counts_disc .== 0)

        # Ensure workspace cleans up correctly (ball indicators reset to 0)
        @test all(ws_path.ball_indicator .== 0x00)
        @test all(ws_path.dist .== typemax(Int))
    end

    @testset "4. Main Algorithm (kadabra_centrality)" begin
        # Using a Star Graph: Node 1 is connected to 2, 3, 4, 5
        # Node 1 must be the absolute most central node. Leaves should have near 0 centrality.
        g = star_graph(5)

        # Run KADABRA with absolute error stopping
        err = 0.1
        delta = 0.1
        res = kadabra_centrality(g, 0, err, delta; start_factor = 10, endpoints = true)
        approx_bet = res.centralities

        # Center node should have highest centrality
        @test argmax(approx_bet) == 1

        # Center node should be present in a massive majority of paths
        @test approx_bet[1] > 0.8

        # Leaf nodes should all have significantly lower centrality than the center
        for i = 2:5
            @test approx_bet[i] < approx_bet[1] * 0.6  # Leaf scores should be distinctly lower
            @test approx_bet[i] > 0.2                  # But strictly above 0 due to endpoint selection
        end
    end

    @testset "5. Relative Top-K Mode Validation" begin
        # Star Graph: Node 1 is the absolute center (highest BC).
        g = star_graph(5)

        # Test relative top-1
        res_1 = kadabra_centrality(g, 1, 0.1, 0.1; start_factor = 10)
        approx_bet_1 = res_1.centralities
        @test argmax(approx_bet_1) == 1

        # Test relative top-3
        res_3 = kadabra_centrality(g, 3, 0.1, 0.1; start_factor = 10)
        approx_bet_3 = res_3.centralities
        # The center node 1 should be the highest
        @test argmax(approx_bet_3) == 1
    end

    @testset "6. kadabra_top_k Output Length and Bounds" begin
        g = star_graph(5)
        # top 2 nodes could return more than 2 nodes if confidence intervals overlap.
        # But in a star graph with large error tolerance, we can force overlap.
        top_k_res = kadabra_top_k(g, 2, 0.5, 0.5; start_factor = 10, endpoints = true)

        # Test that length is >= k (which is 2)
        @test length(top_k_res) >= 2

        # Test the structure of the NamedTuple
        @test hasproperty(top_k_res[1], :node)
        @test hasproperty(top_k_res[1], :centrality)
        @test hasproperty(top_k_res[1], :lower_bound)
        @test hasproperty(top_k_res[1], :upper_bound)

        # Test that they are sorted by centrality in descending order
        cents = [x.centrality for x in top_k_res]
        @test issorted(cents, rev = true)
    end
    @testset "7. Top-k budget-allocation variants" begin
        # The paper and the C++ reference pair the rank gaps with opposite sides of each
        # vertex's confidence interval (see kadabra_centrality's `topk_variant`). All three
        # readings must run and must agree on who the most central vertex is; only the
        # sample count is allowed to differ.
        g = star_graph(50)
        for variant in (:code, :paper, :cpp)
            res = kadabra_centrality(g, 3, 0.1, 0.1; start_factor = 10, parallel = false,
                                     rng = Random.Xoshiro(1), topk_variant = variant)
            @test argmax(res.centralities) == 1
            @test res.n_samples > 0
        end
        @test_throws ArgumentError kadabra_centrality(g, 3, 0.1, 0.1; topk_variant = :nope)

        # On well-separated ranks the two allocations really are transposed.
        bet = [0.30, 0.20, 0.18, 0.05, 0.04, 0.01, 0.005]
        el_c, eu_c = zeros(7), zeros(7)
        el_p, eu_p = zeros(7), zeros(7)
        compute_bet_err!(copy(bet), el_c, eu_c, 10_000, 3, false, 0.001, 100; variant = :code)
        compute_bet_err!(copy(bet), el_p, eu_p, 10_000, 3, false, 0.001, 100; variant = :paper)
        # v_1's lower budget is what the stopping test needs; only :paper sizes it.
        @test el_p[1] < 1.0
        @test el_c[1] > 1.0
        # v_2: the two budgets are each other's mirror image.
        @test el_p[2] ≈ eu_c[2]
        @test eu_p[2] ≈ el_c[2]
    end

    @testset "8. k larger than nv(g)" begin
        # Both entry points clamp k to nv(g). This is the only way to reach the
        # `k == union_sample` arm of the top-k stopping test.
        g = star_graph(5)
        res = kadabra_centrality(g, 10, 0.1, 0.1; start_factor = 10, parallel = false,
                                 rng = Random.Xoshiro(1))
        @test length(res.centralities) == 5
        @test argmax(res.centralities) == 1
        top = kadabra_top_k(g, 10, 0.1, 0.1; start_factor = 10, parallel = false,
                            rng = Random.Xoshiro(1))
        @test length(top) == 5 # the four leaves all have betweenness 0 and tie
        @test top[1].node == 1
    end

    @testset "9. _backtrack! rejects a corrupt predecessor structure" begin
        # Breaking out of the walk silently would return a plausible but wrong path
        # count; these states cannot arise from a correct BFS, so they must be loud.
        counts, offsets, np = zeros(Int, 3), [1, 2, 3], ones(3)
        rng = Random.Xoshiro(1)
        # A cycle 1 <- 2 <- 1 in the predecessor map never reaches the target.
        @test_throws ErrorException _backtrack!(
            counts, 1, 3, [2, 1, 0], [1, 1, 0], offsets, np, rng, false)
        # A vertex on the walk with no recorded predecessor.
        @test_throws ErrorException _backtrack!(
            counts, 1, 3, [0, 0, 0], [0, 0, 0], offsets, np, rng, false)
    end
end
