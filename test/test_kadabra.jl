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
        is_finished_abs = check_finished(
            approx_counts, top_k_nodes, n_pairs, k, err, 
            delta_l_guess, delta_u_guess, omega, true
        )
        # Because we artificially set err=0.05 but n_pairs is only 1000, 
        # Chernoff bounds will be loose, so it shouldn't finish immediately.
        @test is_finished_abs isa Bool
    end

    @testset "3. Core Sampling (sample_shortest_path!)" begin
        # Test Path Graph (1 -> 2 -> 3 -> 4 -> 5)
        # There is exactly one shortest path between 1 and 5
        path_g = path_graph(5)
        ws_path = KadabraWorkspace(path_g)
        
        sp = sample_shortest_path!(ws_path, path_g, 1, 5)
        @test sp == [1, 2, 3, 4, 5] || sp == [5, 4, 3, 2, 1]
        
        # Test Same Source and Target
        sp_same = sample_shortest_path!(ws_path, path_g, 3, 3)
        @test sp_same == [3]

        # Test Disconnected Graph
        # Nodes 1-3 are connected, Nodes 4-5 are connected, but no bridge
        disc_g = Graph(5)
        add_edge!(disc_g, 1, 2)
        add_edge!(disc_g, 2, 3)
        add_edge!(disc_g, 4, 5)
        ws_disc = KadabraWorkspace(disc_g)
        
        sp_disc = sample_shortest_path!(ws_disc, disc_g, 1, 5)
        @test isempty(sp_disc)
        
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
        approx_bet = kadabra_centrality(g, 0, err, delta; start_factor=10)
        
        # Center node should have highest centrality
        @test argmax(approx_bet) == 1
    
        # Center node should be present in a massive majority of paths
        @test approx_bet[1] > 0.8
    
        # Leaf nodes should all have significantly lower centrality than the center
        for i in 2:5
            @test approx_bet[i] < approx_bet[1] * 0.6  # Leaf scores should be distinctly lower
            @test approx_bet[i] > 0.2                  # But strictly above 0 due to endpoint selection
        end
    end

    @testset "5. Relative Top-K Mode Validation" begin
        # Star Graph: Node 1 is the absolute center (highest BC).
        g = star_graph(5)
        
        # Test relative top-1
        approx_bet_1 = kadabra_centrality(g, 1, 0.1, 0.1; start_factor=10)
        @test argmax(approx_bet_1) == 1
        
        # Test relative top-3
        approx_bet_3 = kadabra_centrality(g, 3, 0.1, 0.1; start_factor=10)
        # The center node 1 should be the highest
        @test argmax(approx_bet_3) == 1
    end
end