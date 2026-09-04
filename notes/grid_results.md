# Grid results — competition × informed flow (2026-09-04)

65 seeds per cell across the 4×3 grid (`num_value_agents ∈ {10,50,150,400}` ×
`n_agents ∈ {1,2,4}`), run on fixed code: `BoolDones` (E3), per-agent training
budget (E4), and one thread per job. 780 runs, zero failures.

**Headline: all three pre-registered predictions are unsupported, and the
reason is that the agent never made markets in the first place.**

## Provenance

| host | arch | seeds | runs |
|---|---|---|---|
| desktop (i7-9700K) | x86_64 | 15–64 | 600 |
| Pavels-MacBook-Air | arm64 | 0–14 | 180 |

Host effect tested before pooling, blocked by cell (12 differences, 11 df):
mean per-cell difference −347, t = −2.05 against a critical t(11) of 2.20 →
**not significant**. A naive pooled t-test over all 780 rows gives t = −2.01
and appears significant; that test is wrong because it treats correlated
within-cell observations as independent. Primary analysis below is
**desktop-only, 50 seeds, one architecture**; the pooled 65-seed figures agree.

## Prediction 1 — more informed flow → withdrawal

Not supported. Participation does not fall as `num_value_agents` rises; see
the participation table below, which is flat in `n_agents` and does not
show a withdrawal threshold anywhere in 10 → 400.

## Prediction 2 — more competitors → tighter quotes, less profit each

Both halves tested. Neither holds.

*Less profit each* (Δequity per agent, 50 seeds/cell, N=200 per group):

| `n_agents` | mean | 95% CI |
|---|---|---|
| 1 | −819 | [−1150, −489] |
| 2 | −1339 | [−1693, −985] |
| 4 | −908 | [−1236, −579] |

Contrast n=4 − n=1: **−88, SE 238, t = −0.37**. Pooled at 65 seeds: **+36,
95% CI [−355, +426]**. The effect is bounded to roughly ±400 of zero, not
merely unmeasured.

*Tighter quotes* — never previously tested:

| `n_agents` | quoted spread (¢) | quote size | fills/eval |
|---|---|---|---|
| 1 | 0.50 | 0.3 | 612 |
| 2 | 0.54 | 0.3 | 663 |
| 4 | 0.54 | 0.3 | 659 |

Contrasts n=4 − n=1: spread +0.04 (t=1.38), size −0.00 (t=−0.27), fills +47
(t=1.20). All null.

## Prediction 3 — withdrawal at lower informed fraction as competition rises

Cannot arise: there is no withdrawal threshold to move, because there is no
sustained market making at any cell.

## Why every test is null: the policy is degenerate

Two measurements, both at 50 seeds, explain the rest of this document.

**The P&L is not market making.** Spread capture is ~8¢ against adverse
selection of ~−900:

| `n_agents` | Δequity | capture | adverse | capture as % of \|P&L\| |
|---|---|---|---|---|
| 1 | −819 | 7.3 | −827 | 0.9% |
| 2 | −1339 | 8.2 | −1347 | 0.6% |
| 4 | −908 | 7.9 | −916 | 0.9% |

Under 1% of the P&L comes from earning the spread. The agent is not being paid
for liquidity; it is holding inventory and being marked to market.

**It is two-sided on 2% of steps.** Replaying the saved `actions` through the
quantisation in `actions.py` (offsets → whole ticks, sizes → `int(round())`,
both sides voided when `bid_price >= ask_price`):

| `n_agents` | quotes at all | two-sided | offsets cross | size rounds to 0 |
|---|---|---|---|---|
| 1 | 14.8% | 1.9% | 62.9% | 62.8% |
| 2 | 15.6% | 1.9% | 60.4% | 61.9% |
| 4 | 15.5% | 2.1% | 59.4% | 63.2% |

Participation contrast n=4 − n=1: +0.62pp, t = 0.92 — not significant.

The policy emits sub-tick offsets that round to zero on both sides — so the
quotes cross and are voided on ~60% of steps — and sizes averaging 0.3 against
a `max_size` of 100, which round to zero on ~62%. It posts a genuine two-sided
market on **about one step in fifty**.

Competition cannot be observed acting on market making that is not happening.
The grid's null is not evidence that competition is irrelevant; it is evidence
that the treatment was never applied.

## This is a broken refusal, not a correct one

The distinction the talk is built on. A *correct* refusal is an agent widening
and withdrawing because adverse selection makes quoting unprofitable — a
rational response to the environment. What the data shows is different: a
policy sitting in a degenerate optimum where it emits sub-tick crossed quotes
of near-zero size, collects almost no spread, and accumulates inventory it is
then marked against. It is not declining to trade because trading is
unprofitable; it never learned to quote.

## What would actually test the prediction

The competition question is not answerable until the agent makes markets at
all. In rough order of leverage:

1. **Make the action space unable to express a non-quote.** Offsets are
   currently continuous cents that round to zero, and sizes round to zero from
   below. A minimum tick offset and a minimum size would remove the degenerate
   region — this is the `both_hatches` idea from the cheap suite, which did
   force real quoting (capture rose ~30×, from 4–6 to 179) at the cost of
   catastrophic adverse selection.
2. **Re-run the grid on top of that**, since only then does `n_agents` vary
   competition between market makers rather than between non-participants.
3. Reward shaping for two-sided presence, if 1 proves insufficient.

`both_hatches` is the important precedent: it is the one configuration
measured so far that produced actual market making, and it lost heavily. That
is a publishable result on its own — it suggests no edge exists at these
parameters — but it needs the grid's seed count before it can be claimed.

## Sample size

Seed count mattered, and not in the direction more seeds usually matter:

| seeds/cell | contrast n=4 − n=1 | t |
|---|---|---|
| 3 | unresolvable (signal/noise 0.73) | — |
| 15 | +448 | 1.31 |
| 50 | −88 | −0.37 |

At 15 seeds the contrast looked like a trend in the *opposite* direction to
prediction 2. It was noise: going to 50 seeds did not shrink it, it changed
its sign. Between 3 and 15 seeds the measured within-cell sd also *rose*, from
1153 to 1786, because three draws had not sampled the tail. Any claim made off
the 3-seed grid would have been an artifact.
