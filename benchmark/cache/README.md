# Benchmark Cache

This directory stores serialized ground truth Betweenness Centrality scores (`*_exact_bc.jld2`). 

Since calculating exact Betweenness Centrality via Brandes' algorithm is extremely computationally expensive ($O(|V||E|)$), the benchmark scripts cache the results upon first calculation. Subsequent runs of the benchmarks will automatically load the exact scores from this cache to drastically speed up evaluation loops.

If you modify the underlying dataset, you should delete the corresponding `.jld2` cache file to force a recalculation.
