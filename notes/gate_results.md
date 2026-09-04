# Acceptance gate — do the two fixes actually take? (2026-09-04)

Three cells, one seed, 10k env steps per agent, eval seed 42, on the desktop.
`runs/gate/{gated_id,gated_noid,legacy}/seed0`.

**This is a gate, not a result.** One seed at a fifth of the grid's training
budget. Seed variance in this project has been enormous (sd 15.1k on a mean of
-14.5k in `min_size_10`), so nothing below supports a magnitude. It supports
*signs* and *mechanisms*, which is what a gate is for.

The three cells isolate the two changes:

| cell | action space | agent-id one-hot |
|---|---|---|
| `legacy` | 4-tuple (pre-2026-09) | no |
| `gated_noid` | gated 6-tuple | no |
| `gated_id` | gated 6-tuple | yes |

## Did it start making markets? Yes.

`scripts/quote_stats.py gate`

| cell | gated | two-sided% | any quote% | withdrawn% | crossed% | spread(c) | fills |
|---|---|---|---|---|---|---|---|
| `gated_id` | yes | **47.0** | 98.6 | 1.4 | 0.0 | 2.00 | 1694 |
| `gated_noid` | yes | **52.9** | 83.6 | 16.4 | 0.0 | 2.00 | 1356 |
| `legacy` | no | **0.0** | 10.9 | 89.1 | 0.0 | 0.00 | 116 |

`legacy` reproduces the September grid's degenerate behaviour on this budget:
a genuine two-sided market on **0.0%** of steps. Gated, that becomes 47-53%,
at a quoted spread of 2.00c, with 12-15x the fills.

`crossed%` is 0.0 in every gated cell, as the floors guarantee.

**Skewing appeared, unprompted.** In `gated_id` the policy quotes *something*
on 98.6% of steps but two-sided on only 47.0% - so on roughly half of all
steps it deliberately quotes one side only. That is inventory skewing, and the
legacy action space could not express it as a choice.

**Withdrawal is now a measurement rather than an artifact,** and it is low:
1.4% in `gated_id`. At 2.4% informed flow a maker choosing to quote almost
always is the *expected* direction for Glosten-Milgrom. The prediction to test
is that withdrawal rises with informed flow - which is the grid, and is now
answerable because withdrawal is a gate the policy sets rather than a rounding
outcome.

## Did the agents stop being the same bot? Only with the one-hot.

`scripts/agent_divergence.py`

| cell | raw ident% | **quote ident%** | inventory corr | obs dims≠ | fills a/b |
|---|---|---|---|---|---|
| `gated_id` | 0.00 | **42.31** | **-0.477** | 4/46 | 1989 / 1400 |
| `gated_noid` | 4.56 | **99.94** | 0.99986 | 2/44 | 1355 / 1357 |
| `legacy` | 77.86 | **100.00** | 1.00000 | 1/44 | 116 / 116 |

**Gating alone does not break the symmetry.** `gated_noid` trades 1,356 times
per eval - twelve times the legacy cell - and its two agents still post
identical quotes on 99.94% of steps with inventory correlation 0.99986. This
settles a question left open in `notes/agent_divergence_results.md`: more
trading was *not* enough to differentiate parameter-shared agents. The
observation simply cannot distinguish them.

**One identity bit does break it, hard.** `gated_id` drops quote agreement to
42.31%, and the inventory correlation goes **negative**: the two agents run
*opposite* positions, finishing at -86 and +651. Fills diverge 1989 vs 1400.

That is not merely "distinguishable". It is specialisation - the same network,
handed one bit saying which agent it is, learning two different strategies.

## P&L, for completeness and not to be quoted

Eval prints Δequity in cents.

| cell | mm_0 Δeq | mm_1 Δeq | MaxDD | final inv |
|---|---|---|---|---|
| `gated_id` | -204,764c (-$2,048) | +418,841c (+$4,188) | 2.95% / 5.62% | -86 / +651 |
| `gated_noid` | -21,760c (-$218) | -20,012c (-$200) | 1.12% / 1.12% | -74 / -79 |
| `legacy` | -1,083c (-$11) | -1,063c (-$11) | 0.13% / 0.13% | +84 / +84 |

`legacy` barely loses because it barely trades - the frozen-policy pattern.
The `gated_id` split (one agent well up, the other down, on a single seed) is
the most interesting and least trustworthy number here. Do not carry it
anywhere without seeds.

## What this changes

1. **The grid re-run is unblocked.** Both prerequisites are met and measured:
   the agent makes markets, and the agents are distinct. `n_agents` can now
   vary competition rather than copies.
2. **`notes/current_state.md`'s blocker is resolved** - though not by the
   `--min-offset-cents` flag it prescribed. The floor is applied inside the
   gated branch of `translate_action`, exposed as `--min-offset-ticks`.
3. **A possible new beat for act 7**, unearned until it replicates: give two
   identical networks one bit of identity, and they specialise into opposite
   strategies.

## Immediate next step

Seeds, before anything else. The three cells at N seeds, to see whether the
negative inventory correlation and the P&L asymmetry survive replication. On
the evidence of the 15-vs-50-seed reversal in `notes/grid_results.md`, three
seeds will not be enough to state anything but a sign.
