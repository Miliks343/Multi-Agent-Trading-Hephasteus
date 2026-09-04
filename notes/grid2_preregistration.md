# grid2 — pre-registration

**Written and committed 2026-09-04, before the run started.** Nothing in this
file may be edited after the first job completes; findings go in
`notes/grid2_results.md`.

## Why re-run the grid at all

The 2026-09 grid (`notes/grid_results.md`) tested competition x informed flow
and found nothing on all three predictions. The reason was not that the effects
are absent — it is that **the treatment was never applied**, for two
independent reasons since fixed:

1. The agent posted a genuine two-sided market on ~2% of steps. Its offsets and
   sizes rounded to zero, so "post nothing" was a large, flat, zero-gradient
   region that the policy initialised inside and could not feel its way out of.
2. Its "competing" agents were the same bot — quoting identically on 97-99.9%
   of steps across 4,350 pairs (`notes/agent_divergence_results.md`). Varying
   `n_agents` varied copies of one policy, not competition.

Both are fixed and verified at 10 seeds (`notes/gate_results.md`): gated
actions raise two-sided quoting on 10/10 seeds, and the identity one-hot drops
quote agreement from 99.6% to 53.1% on 10/10 seeds.

## Design

- `num_value_agents` in {10, 50, 150, 400} x `n_agents` in {1, 2, 4} = 12 cells
- 20 seeds/cell = **240 runs**, seed-major so an interrupted run still leaves a
  complete balanced design
- 50k env steps **per agent** (E4), eval on seeds 42/43/44
- Gated action space, identity one-hot, `--agent-id-width 4` pinned so every
  cell shares one observation width regardless of `n_agents`

Informed fraction by cell, against ~2000 noise + 10 momentum agents:
v10 ~ 0.5%, v50 ~ 2.4% (the default), v150 ~ 6.9%, v400 ~ 16.6%.

## The decisive cell is v10_n1

A properly quoting agent already **loses $1,373 per agent at v50**
(`notes/gate_results.md`). v10 is the only cell in the design with *less*
informed flow than that, and n=1 the only one with no competition.

**So if v10_n1 is not profitable, there is no viable region anywhere in this
parameterisation.** That is a result, not a failure — it says the environment
as configured does not support market making, which is a statement about the
environment rather than about PPO.

## Predictions

Committed before running. Prediction 3 is the one worth being wrong about.

1. **Withdrawal rises with informed flow.** `withdrawn%` increases
   monotonically across v10 -> v400. This is the Glosten-Milgrom prediction and
   it is directly testable for the first time: `withdrawn%` is now a gate the
   policy sets, not a rounding artifact.
2. **Quoted spread widens with informed flow.** Mean `spread(c)` on two-sided
   steps increases across v10 -> v400.
3. **Delta-equity is negative at every cell, v10_n1 included.** No viable
   region.
4. **Competition effects stay small.** The n=4 minus n=1 contrast on Δequity
   and on quoted spread is small relative to seed variance. Unlike 2026-09 this
   is a genuine test rather than a null by construction.

## What would falsify each

1. Flat or non-monotonic `withdrawn%` across informed flow. Note the gate
   already measured 22.7-32.5% withdrawal at v50, so there is headroom in both
   directions.
2. Flat or narrowing spread. The floor is 1 tick, so the measurable range is
   bounded below at ~2c (one tick each side).
3. Any cell with a positive mean Δequity across 20 seeds. v10_n1 is where to
   look.
4. A contrast exceeding roughly 2 standard errors on 20 seeds.

## Analysis committed in advance

- Primary: `withdrawn%` and `spread(c)` vs `num_value_agents`, from
  `scripts/quote_stats.py`.
- Contrasts **paired by training seed**, reported with a sign test alongside t.
  Between-seed variance has swamped every unpaired comparison in this project;
  pairing and the sign test are what made `notes/gate_results.md` readable.
- Report n, sd and an interval for every number. No point estimates.
- `scripts/agent_divergence.py` on every n>=2 cell as an acceptance check: if
  `quote ident%` is not well below 99%, the competition treatment did not take
  in that cell and its competition result is void regardless of what it says.

## Known limitations, stated in advance

- 20 seeds settles signs, not magnitudes. This project has already seen a
  contrast read +448 (t=1.31) at 15 seeds and -88 at 50.
- `num_value_agents` is a coarse, integer x-axis. `val_lambda_a` is the
  continuous informed-arrival dial and would locate a threshold properly rather
  than bracketing it between four settings; deferred deliberately to keep this
  comparable with the 2026-09 grid.
- One simulated hour, one fundamental-value process, no latency edge, PPO only.
- Seeds are contiguous from 0 so the pool can be extended to 20-49 later
  without re-running anything.
