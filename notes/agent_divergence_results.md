# Agent divergence — are the parameter-shared agents distinct? (2026-09-04)

**No. Across 4,350 agent pairs the effective quotes are identical on 97–99.9%
of steps.** The `n_agents` axis does not vary competition between market
makers; it varies how many copies of one policy stand in the book.

Measured with `scripts/agent_divergence.py` on the existing desktop artifacts —
no new training. `runs/retrain` (6 cells x 3 train seeds x 3 eval seeds) and
`runs/grid` (12 cells x 50 seeds x 3 eval seeds).

## Why this was worth measuring

`scripts/train.py:81` wraps the env in `ss.pettingzoo_env_to_vec_env_v1`, which
presents the N PettingZoo agents to SB3 as N parallel copies of a single-agent
env. One `MlpPolicy` is trained on the pooled experience and every agent
evaluates the same weights. Whether that yields distinguishable behaviour is an
empirical question, and 40 of the 44 observation features are the shared order
book — only `inventory_norm` and `cash_norm` can differentiate one agent from
another.

## Retrain suite

| cell | pairs | raw ident% | quote ident% | act corr | inv corr | obs dims≠ | fills a/b |
|---|---|---|---|---|---|---|---|
| `noop` | 9 | 100.00 | 100.00 | 1.00000 | — | 0.0 | 0/0 |
| `invpen0_only` | 9 | 100.00 | 100.00 | 1.00000 | — | 0.0 | 0/0 |
| `vecnorm_only` | 9 | 29.76 | 99.96 | 1.00000 | 0.99978 | 1.9 | 599/605 |
| `exp1` | 9 | 14.98 | 98.90 | 0.99991 | 0.99882 | 2.0 | 967/971 |
| `min_size_10` | 9 | 11.94 | 97.19 | 0.97989 | 0.99528 | 2.0 | 1160/1202 |
| `both_hatches` | 9 | 11.94 | 97.19 | 0.97989 | 0.99528 | 2.0 | 1160/1202 |

## Grid (n=1 cells omitted — a single agent has no pair)

| cell | pairs | raw ident% | quote ident% | act corr | inv corr | obs dims≠ | fills a/b |
|---|---|---|---|---|---|---|---|
| `v10_n2` | 150 | 24.31 | 99.91 | 0.99999 | 0.99968 | 1.8 | 808/815 |
| `v10_n4` | 900 | 20.24 | 99.78 | 0.99988 | 0.99873 | 1.9 | 811/809 |
| `v50_n2` | 150 | 31.23 | 99.81 | 0.99984 | 0.99821 | 1.8 | 642/646 |
| `v50_n4` | 900 | 26.80 | 99.77 | 0.99974 | 0.99729 | 1.9 | 567/565 |
| `v150_n2` | 150 | 27.03 | 99.82 | 0.99990 | 0.99846 | 1.9 | 606/611 |
| `v150_n4` | 900 | 24.34 | 99.84 | 0.99994 | 0.99869 | 1.9 | 657/655 |
| `v400_n2` | 150 | 33.13 | 99.92 | 0.99999 | 0.99952 | 1.8 | 584/588 |
| `v400_n4` | 900 | 26.83 | 99.91 | 0.99998 | 0.99952 | 1.9 | 604/603 |

## Reading the two identity columns

The gap between them is the whole result.

**`raw ident%` is low (12–33%).** In continuous action space the agents *do*
differ. Inventory diverges slightly, that feeds `inventory_norm` and
`cash_norm`, and the shared network emits slightly different floats.

**`quote ident%` is 97–100%.** After `translate_action` quantises offsets to
whole ticks and sizes to integers, the agents post a bit-identical quote on
essentially every step. **The differences are real but sub-tick** — smaller
than the exchange's resolution, so they never reach the book.

Everything else agrees. Action correlation 0.980–1.000. Inventory correlation
0.995–0.9998, i.e. the same position curve. `obs dims≠` is exactly 2.0 of 44 —
only `inventory_norm` and `cash_norm`, as the mechanism predicts. Fill counts
match to within ~3%.

The two frozen cells are the degenerate confirmation: `100.00 / 100.00 / 0
obs dims`. With zero fills nothing can distinguish the agents at all, so the
symmetry is exact.

## Consequence

This is a **second and independent** reason the grid found nothing, and it
survives the fix for the first one. `grid_results.md` attributes the null to
the agent never making markets, which is correct. But fixing the action space
does not by itself restore the competition treatment: `min_size_10` trades
1,160 times per eval and still posts identical quotes on 97% of steps. Two
identical makers quoting the identical price is not competition; it is one
maker with double the size.

**The multi-agent axis was never multi-agent.**

## What would fix it, cheapest first

1. **Agent-ID one-hot in the observation.** ~10 lines. The network stays
   shared, but it can condition on *which* agent it is and so learn to
   differentiate. Standard in the parameter-sharing MARL literature precisely
   because full separation is expensive. Necessary but possibly not sufficient.
2. **`--min-offset-cents`** (the bug-4 fix) helps indirectly: the agents' only
   current source of differentiation is trading history, so an agent that
   barely quotes can barely differentiate. Not sufficient alone — see
   `min_size_10` above.
3. **Separate networks per agent.** Drops SuperSuit; needs N PPO instances
   driven against one env in lockstep. Real work, and N times the samples.
4. **Heterogeneous populations** — different risk appetites or size limits, so
   the competition has structure rather than being symmetric by construction.

## Acceptance criterion this gives us

`quote ident%` is now a direct instrument for "are these agents distinct",
alongside `quote_stats.py`'s "is this agent making markets at all". Any future
multi-agent claim should report both. A grid re-run that does not move
`quote ident%` well below 99% has not applied its treatment.

## Reproducing

    python scripts/agent_divergence.py runs/retrain/min_size_10/seed0/eval

Pure helpers are unit-tested in `tests/test_agent_divergence.py` (16 tests).
