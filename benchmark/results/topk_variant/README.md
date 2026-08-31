# Top-k confidence-budget allocation: paper vs C++ reference

KADABRA's top-k mode fixes each vertex's confidence budget from its rank gaps before
sampling starts (`compute_bet_err!`, called once by `compute_delta_guess!`). Borassi &
Natale's Section 5.2 and the authors' C++ reference disagree about which gap sizes which
side of the interval, and the reference is inconsistent with its own stopping test.

| | paper (Section 5.2 / Algorithm 2) | C++ reference (`Probabilistic.cpp`) |
| --- | --- | --- |
| `lambda_U(v_i)`, i <= k | gap **above** v_i | gap **below** |
| `lambda_L(v_i)`, i <= k | gap **below** v_i | gap **above** |
| top vertex v_1 | `lambda_L` = gap, `lambda_U` unconstrained | `lambda_L` **unconstrained**, `lambda_U` = gap |
| `lambda_U(v_i)`, i > k | `b(v_k)` **-** `lambda_L(v_k)` - `b(v_i)` | `b(v_k)` **+** ... |
| external exclusion test | `b(v_k) - f(v_k)` | `b(v_k) - `**`g`**`(v_k)` |

Algorithm 2 line 4 separates v_i from v_{i+1} by requiring
`b(v_i) - f(v_i) > b(v_{i+1}) + g(v_{i+1})`, so it is v_i's *lower* deviation that must
clear the gap *below* it: the paper's pairing is the self-consistent one.

NetworKit's `networkit/cpp/centrality/KadabraBetweenness.cpp` reproduces the reference
verbatim, including the `g(v_k)` substitution. Its logic is untouched since 2021 (only
clang-format / clang-tidy commits), and `k` is a public constructor argument.

## Variants

| `topk_variant` | allocation | external test |
| --- | --- | --- |
| `:paper` (default) | paper | paper |
| `:cpp` | reference | reference |
| `:code` | reference | paper — the hybrid this port carried until 2026-08-30 |

All three are valid `(eps, delta)`-approximations: the allocation decides *where* the
confidence budget is spent, not what is guaranteed.

## Results

`measured/` — 156 runs from `reproduce_report.sh --stage kx` and `--stage kxs` on
`coan-wrk-01`: 4 graphs x k in {0,3,5,10,100} x 3 variants x 3 seeds, eps=1e-4, delta=0.1,
t=8. Summarised in `measured_summary.txt` (`summarize_topk.py`), which pairs every ratio
against the *same seed's* k=0 run so the shared denominator noise cancels.

Samples relative to `:code`, pooled over 48 paired cells:

| variant | ratio |
| --- | --- |
| `:paper` | **0.802 +- 0.175** (min 0.452, max 1.022) |
| `:cpp` | 0.998 +- 0.020 (min 0.881, max 1.026) |

Top-k accuracy is **identical** across the three — same overlap with the exact ranking,
same `tau_topk` to three decimals — so this is purely a sample-cost difference. Global
`tau_b` is at most 0.016 lower for `:paper`, which is what drawing up to 55% fewer samples
buys; top-k mode does not promise a global ranking.

The one cell where `:paper` is (marginally) dearer is `email-EuAll` at k=5, 1.021 +- 0.002.

`predicted/` — `diagnose_topk_delta.jl` output for the same graphs at 3 seeds. It draws the
burn-in once and then bisects the sample count at which each variant *would* stop, holding
those estimates fixed; deterministic, and it costs ~1% of a real run. Useful for seeing
*which vertex* is binding and why. Run with `JULIA_NUM_THREADS=8`: `union_sample` is
derived from the thread count, so a different count is not comparable.

## Why the effect hides at k=10 and k=100 on some graphs

The tie-collapse safeguard resets both budgets to the uniform `eps` for any adjacent pair
closer than `sqrt(start_factor)*eps/4` (= 2.5 eps). On graphs whose top ranks are tightly
packed that catches precisely the vertices that end up binding, so the two allocations
coincide where it matters. On `email-EuAll`, ranks 1-2 stay separated and the effect
survives to k=100.
