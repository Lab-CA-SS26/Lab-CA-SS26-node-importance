# Two discrepancies in KADABRA's top-*k* confidence-budget allocation

**Status:** draft note for Aurora Rossi and, if she thinks it useful, Emanuele Natale.
**Date:** 2026-08-31
**Context:** Lab course (SS26), reimplementing KADABRA and BRAVA-GNN in Julia.

---

## Summary

While porting KADABRA to Julia for the lab, we found two places where the reference C++
implementation departs from Borassi & Natale (2019), both inside `computeDelta`'s top-*k*
budget allocation. Neither affects correctness — the allocation is a heuristic for *where*
to spend the confidence budget δ, not part of the (λ, δ) guarantee, and absolute-error mode
(`k = 0`) never enters this code path. Both cost samples, and the second one causes a
counter-intuitive effect we could reproduce across graphs and seeds: **asking for the top *k*
vertices can take more samples than computing every centrality.**

Following the paper on both points removes that effect entirely and needs, on average, 30%
fewer samples for an identical top-*k* answer.

We would like to know whether the reference's choices were deliberate. If they were not,
NetworKit is affected too — `KadabraBetweenness.cpp` reproduces all of them verbatim.

---

## Finding 1 — λ_L and λ_U are paired with the opposite rank gaps

Section 5.2 of the paper sets, for the first *k* vertices,

```
λ_U(v_i) = (b̃(v_{i-1}) − b̃(v_i)) / 2        # gap ABOVE v_i
λ_L(v_i) = (b̃(v_i) − b̃(v_{i+1})) / 2        # gap BELOW v_i
```

which is what Algorithm 2 goes on to consult: separating v_i from v_{i+1} requires
`b̃(v_i) − f(v_i) ≥ b̃(v_{i+1}) + g(v_{i+1})`, so it is v_i's *lower* deviation that must clear
the gap *below* it, and its upper deviation that must be cleared from above.

`Probabilistic.cpp` pairs them the other way round, and three further consequences follow:

| | paper (§5.2 / Algorithm 2) | reference implementation |
| --- | --- | --- |
| λ_U(v_i), i ≤ k | gap **above** v_i | gap **below** |
| λ_L(v_i), i ≤ k | gap **below** v_i | gap **above** |
| top vertex v₁ | λ_L = gap, λ_U unconstrained | λ_L **unconstrained**, λ_U = gap |
| λ_U(v_i), i > k | `b̃(v_k) − λ_L(v_k) − b̃(v_i)` | `b̃(v_k)` **+** `λ_L(v_k)` − `b̃(v_i)` |
| external exclusion test | `b̃(v_k) − f(v_k)` | `b̃(v_k) − ` **`g`** `(v_k)` |

The v₁ case is the sharpest: it genuinely has no rank neighbour above it, so one of its two
budgets is unconstrained — but the code leaves λ_L(v₁) unconstrained rather than λ_U(v₁).
Since `computeDelta` allocates δ in proportion to `exp(−C·λ²/b̃)`, an unconstrained λ drives
that vertex's δ to its numerical floor and its deviation bound to a maximum. The one bound
v₁'s own stopping test depends on ends up the loosest in the graph.

### What it costs

Four SNAP graphs × *k* ∈ {3, 5, 10, 100} × 3 seeds, ε = 1e-4, δ = 0.1, 8 threads. Each ratio
is formed against the *same seed's* own run, then averaged, so the shared sampling noise
cancels.

| allocation | samples, relative to the reference's |
| --- | --- |
| paper (§5.2) | **0.80 ± 0.17** (min 0.45, max 1.02) |
| reference, but with the paper's `f(v_k)` in the exclusion test | 1.00 ± 0.02 |

**Top-*k* accuracy is identical** — same overlap with the exact ranking, same Kendall τ over
the top *k* to three decimals. Only the sample count moves.

The `g(v_k)`-for-`f(v_k)` substitution turns out to be inert (1.00 ± 0.02), because the
vertices that finish last are almost always resolved by the `f, g ≤ λ` fallback rather than by
exclusion, so which deviation is subtracted is rarely consulted.

---

## Finding 2 — the tie-collapse rule skips the pair the exclusion test is built around

The paper's rule:

> Finally, if b̃(v_i) − b̃(v_{i+1}) is small, we simply set
> λ_L(v_i) = λ_U(v_i) = λ_L(v_{i+1}) = λ_U(v_{i+1}) = λ, because we do not know if
> bc(v_{i+1}) > bc(v_i), or vice versa.

No restriction on *i*. The reference implements it as two loops — adjacent pairs
(v₁,v₂)…(v_{k−1},v_k), and (v_{k+1}, v_i) for i ≥ k+2 — with the threshold
`√start_factor · λ / 4` (= 2.5λ at the default). **The pair (v_k, v_{k+1}) falls between the
two loops and is never collapsed**, and it is the one pair the external-exclusion test is
built around.

### Why that deadlocks

Whenever `b̃(v_k) − b̃(v_{k+1}) < 2λ`, exclusion is *arithmetically* unsatisfiable: it needs
`f(v_k) + g(v_{k+1})` to fall below that gap, and neither is ever budgeted below λ. So
v_{k+1} must use the `f, g ≤ λ` fallback — but the allocation has written off its lower bound
(λ_L = ∞ for every vertex outside the top *k*), so `f(v_{k+1})` is pinned near its maximum and
the fallback is out of reach too. v_k is stuck symmetrically: it fails the same separation,
and its own λ_U was sized by the wider gap above it, so its fallback fails on `g`.

Both vertices are therefore unresolvable, and the run samples far past where either criterion
was budgeted for. Collapsing the pair — exactly the paper's rule, applied at i = k — gives all
four bounds a real budget and breaks the deadlock. We verified that collapsing only half a
pair changes nothing: it takes all four.

Concretely, on `soc-Epinions1` at k = 5: gap = 1.43λ, exclusion misses by 8e-6,
δ_L(v₆) = 3.3e-10 against a required `f < λ`.

### What it costs

Samples relative to the same seed's `k = 0` run — values above 1.00 mean the top-*k* query was
*dearer* than computing every centrality:

| graph | *k* | reference | + gap pairing | + boundary pair |
| --- | --- | --- | --- | --- |
| `email-EuAll` | 3 | 1.72 ± 0.01 | 1.24 ± 0.10 | **0.86 ± 0.06** |
| `email-EuAll` | 5 | 1.62 ± 0.01 | 1.65 ± 0.01 | **0.92 ± 0.03** |
| `soc-Epinions1` | 5 | 1.77 ± 0.01 | 1.42 ± 0.16 | **0.99 ± 0.09** |
| `soc-Epinions1` | 10 | 1.28 ± 0.31 | 1.00 ± 0.09 | 1.00 ± 0.09 |
| `soc-Epinions1` | 100 | 1.30 ± 0.31 | 1.01 ± 0.09 | 1.01 ± 0.09 |

The columns are cumulative, and both corrections are the paper's own. Over all 16 (graph, *k*)
configurations: **three sit above `k = 0` beyond one standard deviation under the reference,
the same three still do once only the gap pairing is corrected, and none do once the boundary
pair is collapsed as well.** Finding 1 on its own makes top-*k* uniformly cheaper but does not
remove the effect — it takes both.
The mean falls from 0.79 to 0.69, and the top-*k* answer is unchanged in 15 of the 16
(identical overlap and τ over the top *k*; the exception differs by 0.001 in the latter).

---

## Why this would not have shown up in the original evaluation

We looked at whether the paper's own experiments could have caught either of these, and we
think not — for a structural reason rather than an oversight.

Section 9 evaluates top-*k*: the IMDB actor-collaboration snapshots from 1940 to 2014 and the
DBPedia 3.7 Wikipedia citation network, at λ = 0.0002 and δ = 0.1, with wall-clock reported
("all the graphs were processed in less than 1 hour, apart from the Wikipedia graph"). But
those are *standalone* timings. The comparison against RK and ABRA — the experiment with a
baseline to measure against — is run in absolute-error mode, over
λ ∈ {0.03, …, 0.005} with every algorithm required to approximate bc(v) for **every** v.

So the two halves of the evaluation never meet: top-*k* is timed without a reference point,
and the reference point is only ever measured at k = 0. Both effects here are **ratios** —
"top-*k* costs 1.7× what k = 0 costs on the same graph at the same λ" — and that ratio is
never formed. An absolute runtime of "under an hour" on IMDB looks entirely reasonable
whether or not the run drew twice the samples it needed to.

There is a second reason the top-*k* runs might not have shown it even if a baseline had
existed. The deadlock needs `b̃(v_k) − b̃(v_{k+1})` to fall below 2λ, which depends on where
the centrality distribution happens to sit at the cut. At λ = 0.0002 on two graph families it
may simply not have arisen; in our grid it appears on two of four graphs, and only at some *k*.

This is also, we suspect, why it survived into NetworKit: the port is faithful to the source,
and nothing in the usual way of benchmarking betweenness approximation — absolute error
against a competitor — exercises the top-*k* branch at all.

---

## How the repaired version performs

Both fixes together, against the reference implementation's behaviour:

- **~30% fewer samples on average** in top-*k* mode, up to 2.2× fewer at small *k*.
- **No configuration where top-*k* costs more than `k = 0`** — the effect disappears on every
  graph and every *k* we tested.
- **Identical top-*k* output**: same vertices, same order.
- **`k = 0` is untouched**, bit-for-bit: absolute mode takes a different branch entirely.
- Global Kendall τ over *all* vertices drops by at most 0.016, which is simply what drawing
  fewer samples costs on the vertices top-*k* mode was never asked to rank.

Our Julia port implements both, and both are in the pull request we have open against
`JuliaGraphs/Graphs.jl` (#518), each in its own commit with the reasoning.

---

## Questions

1. Was the reference's λ_L/λ_U pairing deliberate — is §5.2 a typesetting slip, or is the code
   transposed? The paper's ordering is the one consistent with Algorithm 2, which is why we
   followed it, but only you can settle it.
2. Same question for the boundary pair: is its omission from the tie-collapse loops intended,
   or an off-by-one? The paper states the rule without restricting *i*.
3. If both are unintended, is it worth reporting to NetworKit? Its `KadabraBetweenness`
   reproduces all of it verbatim and exposes `k` publicly; the file's algorithmic content has
   not changed since 2021. We have not contacted them, and would rather you decide.

---

## Reproducing

Everything is in `Lab-CA-SS26-node-importance/`:

```bash
# the measurements above (4 graphs x k in {3,5,10,100} x 3 seeds, eps=1e-4)
benchmark/reproduce_report.sh --stage kx    # allocation variants
benchmark/reproduce_report.sh --stage kxs   # small k
benchmark/reproduce_report.sh --stage kxb   # the boundary-pair repair
python3 benchmark/summarize_topk.py <outdir>
```

`benchmark/diagnose_topk_delta.jl` is the cheaper diagnostic: it draws the burn-in once and
then, holding those estimates fixed, bisects the sample count at which each variant would
stop, reporting which vertex is binding and why. It costs about 1% of a real run and is
deterministic given the seed.

Raw runs (252 JSON files) and both summaries are under
`benchmark/results/topk_variant/`. `src/kadabra.jl` selects the behaviour with
`topk_variant = :paper_bd | :paper | :cpp | :code | :paper_ex`, defaulting to the repaired
version; `:cpp` reproduces the reference verbatim.
