# Gate results — the gated action space and the identity one-hot (2026-09-04)

30 runs, `experiments/run.sh gate -j 6` on the desktop, **42m24s, 30/30 ok,
zero failures**. Three cells x 10 training seeds, 50k env steps per agent
(matching `suite_grid` and `suite_retrain`), `n_agents=2`, eval on seeds
42/43/44. Raw artifacts under `runs/gate/`.

**This supersedes the single-seed version of this file.** Its participation
numbers were measured at 10k steps/agent and did not survive the budget
increase; its headline specialisation result did not survive replication.

| cell | action space | identity one-hot |
|---|---|---|
| `legacy` | 4-tuple (pre-2026-09) | no |
| `gated_noid` | gated 6-tuple | no |
| `gated_id` | gated 6-tuple | yes |

## Paired contrasts

Pairing by **training seed** is what makes these readable: between-seed
variance has swamped every unpaired comparison in this project. The sign test
is reported alongside t because one or two runaway seeds per cell would
otherwise dominate the mean.

| metric | contrast | mean Δ | t | seeds + | sign p |
|---|---|---|---|---|---|
| two-sided% | `gated_noid − legacy` | **+14.04** | 2.85 | **10/10** | **0.002** |
| two-sided% | `gated_id − legacy` | +7.08 | 1.25 | 8/10 | 0.109 |
| two-sided% | `gated_id − gated_noid` | −6.96 | −0.95 | 3/10 | 0.344 |
| any-quote% | `gated_noid − legacy` | **+26.64** | 4.50 | **10/10** | **0.002** |
| any-quote% | `gated_id − legacy` | +30.13 | 3.91 | 8/10 | 0.109 |
| any-quote% | `gated_id − gated_noid` | +3.49 | 0.40 | 5/10 | 1.000 |
| crossed% | both gated cells − `legacy` | **−4.98** | −3.53 | **0/10** | **0.002** |
| quote ident% | `gated_noid − legacy` | −0.26 | −0.50 | 4/10 | 0.754 |
| quote ident% | `gated_id − legacy` | **−46.43** | −6.40 | **0/10** | **0.002** |
| quote ident% | `gated_id − gated_noid` | **−46.17** | −6.48 | **0/10** | **0.002** |
| inventory corr | `gated_noid − legacy` | −0.00 | −0.95 | 5/10 | 1.000 |
| inventory corr | `gated_id − legacy` | **−0.85** | −3.68 | **0/10** | **0.002** |
| fills | every contrast | +142 / +128 / −14 | ≤1.01 | — | ≥0.754 |

Aggregate (`scripts/quote_stats.py gate`):

| cell | two-sided% | any quote% | withdrawn% | crossed% | spread(c) | fills |
|---|---|---|---|---|---|---|
| `gated_id` | 19.9 | 77.3 | 22.7 | **0.0** | 2.03 | 1090 |
| `gated_noid` | 20.9 | 67.5 | 32.5 | **0.0** | 2.01 | 993 |
| `legacy` | 6.8 | 40.8 | 54.2 | 5.0 | 1.08 | 849 |

## 1. Gating makes it quote. Confirmed.

Two-sided +14.0pp and any-quote +26.6pp, **positive on all ten seeds**
(p = 0.002). The quoted spread also roughly doubles, 1.08c to ~2.02c, which is
the `min_offset_ticks` floor binding as designed.

Note `legacy`'s two-sided mean of 6.8% is carried almost entirely by seed0
(47.1%); the other nine seeds average **2.4%**. The effect is a move from
~2% to ~20%, not from 0%.

**A correction to the single-seed gate:** it reported `legacy` at 0.0%
two-sided with 116 fills. That was 10k steps/agent. At 50k the legacy action
space does learn to quote — 849 fills, 40.8% any-quote. Much of what looked
like a broken action space at the smaller budget was undertraining. The action
space is still worse, but by 5-8x, not infinitely.

## 2. Self-crossing is gone. Structurally, not statistically.

`legacy` crosses on 4.98% of steps; both gated cells are **exactly 0.00% on
every seed and every eval**. With both gates open and offsets floored at one
tick, `bid_price <= mid-1 < mid+1 <= ask_price`, so there is nothing to
measure. This is the one result that needs no seeds.

## 3. Only the identity bit differentiates the agents.

This is the cleanest causal isolation in the project.

- `gated_id − legacy`: quote agreement **−46.4pp** (99.6% → 53.1%), 10/10
  seeds, p = 0.002.
- `gated_noid − legacy`: **−0.26pp, 4/10 seeds, p = 0.75.** A clean null.

Gating alone does nothing for distinctness even though it triples quoting.
`gated_noid` trades more than `legacy` on 10/10 seeds and its agents remain
99.3% identical. **More trading was never going to differentiate
parameter-shared agents** — the observation could not tell them apart, and one
bit fixes it.

## 4. The *direction* of differentiation is a coin flip.

Inventory correlation per training seed, mean over the three eval seeds:

| seed | `legacy` | `gated_noid` | `gated_id` |
|---|---|---|---|
| 0 | 0.997 | 1.000 | **−0.764** |
| 1 | 1.000 | 1.000 | +0.898 |
| 2 | 0.999 | 1.000 | +0.913 |
| 3 | 1.000 | 0.992 | **−0.122** |
| 4 | 1.000 | 0.997 | **−0.651** |
| 5 | 1.000 | 0.997 | **−0.587** |
| 6 | 1.000 | 1.000 | +0.832 |
| 7 | 1.000 | 1.000 | **−0.450** |
| 8 | 1.000 | 1.000 | +0.465 |
| 9 | 0.999 | 1.000 | +0.957 |

**Exactly 5 negative and 5 positive**, mean +0.149, sd 0.731. Within a training
seed the sign is stable across eval seeds, so it is a property of the learned
policy rather than evaluation noise — but across seeds it is as close to a coin
flip as a measurement can get.

**The single-seed result is withdrawn.** "Give two identical networks one bit
of identity and they take opposite positions" was one draw of a bimodal
distribution. What survives, and is significant at 10/10 seeds, is the
*magnitude*: every seed's inventory correlation falls from ~1.00 to somewhere
in [−0.97, +0.99]. The symmetry breaks reliably; which way it breaks is
arbitrary, set by initialisation.

That is a more interesting statement than the original, and it is true.

## 5. It makes markets, and it still loses money.

Δequity (`final_pnl`, cents; mean over 2 agents x 3 eval seeds, then over the
10 training seeds):

| cell | mean | USD | sd | MaxDD |
|---|---|---|---|---|
| `legacy` | −280,486c | −$2,805 | 205,158 | 5.8% |
| `gated_noid` | −137,315c | −$1,373 | 124,513 | 4.8% |
| `gated_id` | −148,867c | −$1,489 | 138,957 | 4.8% |

Gating roughly halves the loss, but **not significantly**: `gated_noid −
legacy` = +143,171c at 7/10 seeds, p = 0.34. All three cells lose. **One of 30
seed-cells is meaningfully positive** (`gated_id` seed9, +102,010c).

This is the outcome `notes/grid_results.md` anticipated: *"`both_hatches` is
the one configuration measured so far that produced actual market making, and
it lost heavily. That is a publishable result on its own — it suggests no edge
exists at these parameters."* It now has ten seeds and a properly specified
action space behind it. An agent that genuinely quotes, at ~2c spread, into
2.4% informed flow, loses money.

Fills are flat across all three cells (every contrast p >= 0.75). More quoting
did not produce more fills — it produced *better* quoting, spread ~2c rather
than sub-tick noise.

## 6. Withdrawal is now measurable, and it is high

`withdrawn%` is 22.7% (`gated_id`) and 32.5% (`gated_noid`) — the share of
steps on which the policy set both gates off. Under `legacy` the comparable
54.2% is mostly rounding, not choice.

This is the dependent variable the phase diagram needs, and it is now a gate
the policy sets rather than an artifact. **Untested here:** whether withdrawal
rises with informed flow. That is the grid, and it is now answerable.

## Reconciling `legacy` against the September grid

`legacy` is configured identically to the grid's `v50_n2` (default 50 value
agents, `n_agents=2`, 50k steps/agent), so it should reproduce it. It appeared
not to. Re-running `participation_stats` over the grid's **own saved
artifacts** settles it:

| metric | grid artifacts, recomputed | `grid_results.md` reported |
|---|---|---|
| two-sided% | 1.79 | 1.9 |
| any-quote% | 37.47 | 15.6 |
| crossed% | 3.92 | 60.4 |

Same data, different metric definitions. `two-sided` — defined identically in
both — agrees to rounding, and matches `legacy`'s 2.4% (ex-seed0). The other
two differ because `grid_results.md`'s table came from a one-off replay whose
"offsets cross" and "size rounds to 0" columns (62.9% and 62.8%) were computed
unconditionally, whereas `participation_stats` requires both sides to carry
positive size before a cross can exist.

**The `legacy` control reproduces the grid. No run is in question.** But
`grid_results.md`'s participation table is not comparable to `quote_stats.py`
output and should not be quoted next to it — see the correction added there.

## What is now true, in order of confidence

1. Gated sides cannot cross. Structural, 0.00% everywhere.
2. Gating increases quoting. 10/10 seeds, p = 0.002.
3. Only the identity bit differentiates agents; gating alone is a clean null.
   10/10 seeds vs p = 0.75.
4. The symmetry breaks in an arbitrary direction. 5 negative, 5 positive.
5. A genuinely quoting agent still loses money at these parameters. All three
   cells negative; the gating improvement is not significant.

## Next

The grid re-run is unblocked, and `withdrawal rate vs informed flow` is the
question it should now answer. Note from finding 5 that the honest hypothesis
is no longer "find the viability boundary" but "there may be no viable region
at all at these parameters" — which is itself a result, and one the grid can
support or refute.
