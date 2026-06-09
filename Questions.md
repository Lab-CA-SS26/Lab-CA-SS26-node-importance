# Open Questions for Kadabra Integration

## Normalization Modes

Right now, our `normalize=true`/`false` flag perfectly mimics `Graphs.jl`. But `Graphs.jl` excludes the endpoints from its scaling denominator `(V-1)(V-2)`. 

The original Borassi paper, however, mathematically defines centrality purely as a **probability** ($\pi_v \in [0, 1]$), which effectively uses `(V)(V-1)` as the denominator. 

We could expand the `normalize` argument to accept a `Symbol` instead of just a boolean. For example:

- `normalize=:graphs` (default): Returns the `Graphs.jl` `(V-1)(V-2)` scaling (matches exact algorithm).
- `normalize=:graphs_unraw`: Returns the `Graphs.jl` absolute path counts.
- `normalize=:probabilistic`: Returns the native Borassi $\pi_v$ probabilities.

**Question:** Do researchers using this repository find the native probabilistic output useful, or is it better to keep it strictly aligned with standard `Graphs.jl` `true/false`?
