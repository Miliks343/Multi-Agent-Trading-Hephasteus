# Latency experiment — pre-registration

**Written and committed 2026-09-05, before the run started.** Findings go in
`notes/latency_results.md`.

## The question

grid2 found **no viable region**: all twelve cells of informed flow x
competition lose, with spread capture at 0.8-1.4% of |P&L| throughout. So the
question is no longer "where is the boundary" but **"does an edge create one at
all"**. Speed is the canonical market-maker edge, and ABIDES has had it wired
in and unused the whole time.

## An uncontrolled variable, found while wiring this

`config_add_agents` regenerates the latency model once our agents are appended,
placing every agent at a uniformly random point on a Seattle-to-NYC line and
setting pairwise latency to the light-speed travel time. **Our latency to the
exchange has therefore been uncontrolled in every run this project has done —
redrawn on every seed.**

Measured: the background median is **3.2 ms**, spanning 1 µs to 6.5 ms. So
`--latency-edge 1.0` — pinning us to the median — is a *control*, not a no-op,
and this suite is the first run in the project where latency is held fixed.

## What a latency edge can and cannot buy here

Stated before running, because it bounds what the result can mean.

Our wake-up interval is 1 s and the median latency is 3.2 ms. **The edge cannot
buy faster reaction** — we cannot re-quote until our next wake-up regardless of
how fast our messages travel. What it buys is **queue priority**: our order
reaches the book ahead of a competitor's identical-price order, and our cancel
lands sooner.

A real HFT speed edge works through microsecond re-quoting, which this decision
cadence cannot express. That is a **limitation of the experiment, not a
finding**, and it belongs in any writeup of the result. Testing the full effect
would need `wakeup_interval` cut by orders of magnitude, which changes episode
length and cost by the same factor.

## Design

- `latency_edge` in {0.0 (co-located), 0.25, 1.0 (median control), 4.0
  (deliberate disadvantage)}
- 20 seeds/cell = **80 runs**, seed-major
- `num_value_agents=50`, `n_agents=1`, `--agent-id-width 4`, 50k env steps,
  eval on seeds 42/43/44

v50 is chosen deliberately: it was the **least unprofitable** n=1 cell in grid2
(−$1,820, against −$2,649 at v10), so it is where viability is closest and an
edge has its best chance of mattering. n=1 keeps competition out of it.

## Predictions

1. **Fill rate rises as latency falls.** This is the one mechanism the cadence
   permits. It is also the manipulation check: if fills do not move, the edge
   was never applied and nothing else here is interpretable.
2. **Δequity does not improve, and plausibly gets worse.** grid2 measured
   capture at ~1% of |P&L| with the rest adverse selection, so winning *more of
   the same fills* means losing more. **A speed edge that makes things worse is
   the interesting outcome, and it is the one predicted here.**
3. **No cell becomes profitable.** The edge does not create a viable region.
4. **Quoted spread stays pinned at the 2c floor.** Nothing here gives the agent
   a reason to widen.

## Falsification

- 1 fails if fills are flat across all four latency levels — then the treatment
  did not take.
- 2 fails if Δequity improves monotonically with speed. That would be the
  genuinely surprising result and would mean queue priority is worth more than
  the adverse selection it attracts.
- 3 fails if any cell's 95% CI on Δequity excludes zero from above.

## Analysis, committed in advance

- Contrasts paired by training seed, sign test alongside t, n/sd/interval on
  every number, effect sizes reported next to p-values.
- Primary contrast: `lat0.0 − lat1.0` (co-located vs median control) on fills
  and on Δequity.
- Monotonicity across all four levels reported, not just the endpoints — a
  non-monotonic pattern would suggest noise rather than a mechanism.
- Markout decomposition per cell: if fills rise and capture stays ~1% of |P&L|,
  that *is* the mechanism for prediction 2 and should be shown directly rather
  than inferred.

## Known limitations

- The 1 s cadence bounds what latency can do, as above. This is the big one.
- 20 seeds settles signs, not magnitudes.
- One informed-flow level (v50) and one agent count (n=1).
- Latency is applied symmetrically on our pair with the exchange only; the
  background population's own structure is untouched, deliberately, so this is
  a faster *us* rather than a different market.
