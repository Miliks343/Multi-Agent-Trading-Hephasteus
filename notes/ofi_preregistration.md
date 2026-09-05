# OFI experiment — pre-registration

**Written and committed 2026-09-05, before the run started.** Findings go in
`notes/ofi_results.md`; this file is not to be edited after the first job.

## The gap this closes

grid2 (`notes/grid2_results.md`) found P1 unsupported — withdrawal did not rise
with informed flow — and P2 unsupported with a **degenerate dependent
variable**: quoted spread was 1.985-2.032c at all twelve cells, pinned to the
2c floor, with raw pre-quantisation offsets of 0.18-0.29c. The policy was not
trying to widen at all.

**But the observation could not have supported either result.** It is entirely
a snapshot — the book as it stands, plus inventory, cash, spread and time. The
agent has no memory and therefore cannot see the book *change*. Adverse
selection is a statement about **flow**, not about levels. An agent with no
flow signal cannot condition on informed flow, and would look exactly like this
whether or not the effect is real.

So P1's failure is currently a fact about our observation, not about market
making. This experiment separates the two.

## What was added

Two features (`--ofi`), following Cont, Kukanov & Stoikov (2014) on L1:
last-step order-flow imbalance and its EWMA (alpha 0.05, ~13.5s half-life at
1s wake-ups). Positive is net buying pressure.

Both are squashed by **measurement, not by convention** — the scaling took
three attempts and the record is in `src/marl_lob/observation_extractor.py`.
Final health over a 900-step rollout: `ofi_norm` std 0.494 with 1.8%
saturation, `ofi_ewma_norm` std 0.475 with 0.8%. Those are the best-behaved
features in the observation; for scale, `bid/ask_size_L1_norm` saturate 23%.

## Design

- `num_value_agents` in {10, 50, 150, 400} x {`ofi_off`, `ofi_on`} = 8 cells
- 20 seeds/cell = **160 runs**, seed-major
- `n_agents=1` so competition cannot confound — grid2 showed competition does
  nothing, and n=1 halves the cost
- `--agent-id-width 4` pinned, so these cells stay directly comparable to
  grid2's `v*_n1` cells
- 50k env steps per agent, eval on seeds 42/43/44

## Predictions

1. **Without OFI, withdrawal stays flat across informed flow** — reproducing
   grid2's P1 null on n=1 cells. This is the control. If it fails, the whole
   comparison is void and nothing else here is interpretable.
2. **With OFI, withdrawal rises with informed flow.** The real test.
3. **With OFI, quoted spread rises above the 2c floor at high informed flow.**
   The floor binds at 2c so the only available direction is up, and grid2
   showed the policy was not pushing against it.
4. **Δequity stays negative everywhere.** Seeing informed flow is not the same
   as being able to profit despite it. Nothing here gives the agent an edge —
   that is the latency experiment, which comes next.

## Falsification

Prediction 2 fails if the v400 − v10 withdrawal contrast with OFI is within
noise, **or is no larger than the same contrast without OFI**. The claim is the
*interaction*, not the main effect: OFI must change how the agent responds to
informed flow, not merely shift its behaviour.

A null on 2 with a clean control on 1 is still a real result — it would say the
agent has the signal and does not use it, which points at the reward or the
optimiser rather than at the observation, and would be worth knowing.

## Analysis, committed in advance

- Primary: `withdrawn%` and `spread(c)` vs `num_value_agents`, split by arm.
- The headline is a **difference-in-differences**: (v400 − v10 | OFI on) minus
  (v400 − v10 | OFI off), paired by training seed.
- Contrasts paired by training seed, sign test alongside t, n/sd/interval on
  every number. Report effect sizes next to p-values — grid2 produced
  "significant" sign tests on spread differences of 0.002c because the metric
  was pinned against a floor.
- Feature-health check on the `ofi_on` arm: if the two OFI channels are not
  moving in the saved observations, the arm did not receive its treatment and
  its result is void, exactly as `quote ident%` gates the competition arms.

## Known limitations

- 20 seeds settles signs, not magnitudes.
- OFI is computed on L1 only. Deeper-book flow and trade-level signed volume
  are not included.
- `num_value_agents` remains a coarse integer x-axis; `val_lambda_a` is the
  continuous dial.
- One simulated hour, no latency edge, PPO only.
