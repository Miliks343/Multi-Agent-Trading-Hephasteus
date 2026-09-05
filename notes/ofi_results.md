# OFI results — the agent got the flow signal, and nothing changed

160 runs (8 cells x 20 seeds), `experiments/run.sh ofi -j 6` on the desktop,
**3h42m, 160/160 ok, zero failures**. `n_agents=1`, 50k env steps per agent,
eval on seeds 42/43/44. Predictions committed before the first job in
`notes/ofi_preregistration.md`.

## Headline

**grid2's P1 failure was not a measurement artifact — but it is not "the agent
ignores flow" either.** Adding order-flow imbalance changed nothing, and the
reason is a *second* zero-gradient region in the action space: the policy
cannot express a wider spread even when it has the signal to want one.

## Feature-health gate — passed

The pre-registered check: if the OFI channels are not moving in the saved
observations, the arm did not receive its treatment.

| cell | `ofi_norm` std | `ofi_ewma_norm` std | |
|---|---|---|---|
| `v10_ofi_on` | 0.4629 | 0.4179 | OK |
| `v50_ofi_on` | 0.4608 | 0.4080 | OK |
| `v150_ofi_on` | 0.4615 | 0.4117 | OK |
| `v400_ofi_on` | 0.4631 | 0.4140 | OK |

The features are live and are the **most active in the observation** — against
`ask_size_L1_norm` at 0.34 and `spread_norm` at 0.0001. The agent unambiguously
had the signal.

## P1 control — reproduced, so the comparison is valid

`withdrawn%`, 95% CI:

| arm | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| `ofi_off` | 27.9 ± 9.8 | 27.8 ± 7.9 | 28.5 ± 10.3 | 24.0 ± 7.8 |
| `ofi_on` | 24.4 ± 7.7 | 28.5 ± 7.3 | 28.0 ± 7.9 | 24.3 ± 5.9 |

v400 − v10, paired by seed: `ofi_off` **−3.83pp** (t = −0.58, 9/20, p = 0.82).
Flat, reproducing grid2's P1 null exactly. The control holds.

## P2 — unsupported. The interaction is not there.

v400 − v10 with OFI: **−0.03pp** (t = −0.01, 11/20, p = 0.82).

Difference-in-differences, (v400−v10 | on) − (v400−v10 | off), paired by seed:

| metric | DiD | sd | t | seeds + | sign p |
|---|---|---|---|---|---|
| `withdrawn%` | +3.800 | 33.90 | +0.50 | 9/20 | 0.824 |
| `spread(c)` | −0.027 | 0.091 | −1.31 | 7/20 | 0.263 |
| `two-sided%` | −0.808 | 29.97 | −0.12 | 9/20 | 0.824 |

All null. OFI does not change how the agent responds to informed flow.

## P3 — the spread never leaves the floor

| arm | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| `ofi_off` | 2.007c | 2.003c | 2.002c | 2.004c |
| `ofi_on` | 2.032c | 2.022c | 2.006c | 2.003c |

## P4 — supported. Still no profit anywhere.

| arm | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| `ofi_off` | −2649 ± 1859 | −1820 ± 1100 | −1988 ± 966 | −2337 ± 1006 |
| `ofi_on` | −2228 ± 1103 | −1985 ± 780 | −2078 ± 868 | −1578 ± 1445 |

OFI on − off, paired: +421 (p=0.50), −165 (p=1.00), −90 (p=0.82), +759
(p=0.50). All null. Capture stays at ~$22 and **~1% of |P&L| in all eight
cells**, exactly as in grid2.

## Why: a second dead zone, and it is the same bug as the first

The obvious reading of a null here is "the agent sees flow and ignores it,
so the problem is the reward or the optimiser". **That reading is wrong**, and
the action space says why.

Under gating, a side that is on resolves as

    off_ticks = max(round(raw_offset), min_offset_ticks)   # min_offset_ticks = 1

so **every raw offset in [0, 1.5) posts exactly one tick.** Identical quote,
identical fills, identical reward. Measured across the suite:

| cell | mean raw | p99 raw | max raw | % of steps raw > 1.5 |
|---|---|---|---|---|
| `v10_ofi_off` | 0.255 | 1.460 | 2.713 | **0.80%** |
| `v10_ofi_on` | 0.264 | 1.488 | 2.684 | **0.95%** |
| `v50_ofi_off` | 0.223 | 1.454 | 2.664 | **0.84%** |
| `v50_ofi_on` | 0.250 | 1.589 | 2.556 | **1.41%** |
| `v150_ofi_on` | 0.183 | 1.167 | 2.410 | **0.20%** |
| `v400_ofi_on` | 0.200 | 1.127 | 1.969 | **0.06%** |

**On 98.6-99.9% of steps the policy sits inside a region where changing its
offset changes nothing at all.** There is no local gradient pointing toward a
wider spread, so no observation — however informative — can produce one. The
signal has nowhere to go.

This is structurally the same defect as the original one. The gated action
space fixed *"cannot quote"*; it left *"cannot widen"*. The first dead zone was
at the origin of the whole action space; this one is between the floor and the
first tick above it, and only the floor's existence makes it visible.

## What this licenses saying, and what it does not

**Licensed:**
1. grid2's P1 null is **not** explained by a missing observation. The feature
   was added, verified live, and the null is unchanged in every metric.
2. A market maker with order-flow imbalance is still unprofitable at every
   informed-flow level tested. Capture stays ~1% of |P&L|.
3. The spread is pinned at the floor for a **mechanical** reason, now measured:
   the policy sits in a flat region of the action space on ~99% of steps.

**Not licensed:**
- "The agent sees informed flow and chooses not to respond." It cannot respond
  through the offset dimension at all. Attributing this to the reward or the
  optimiser would be a third wrong explanation in a row for the same class of
  bug.

## Next

The offset dimension needs the treatment the gate dimension already got: a
mapping on which the policy's usable output range spans more than one tick.
Until then the spread is not a free variable and no experiment can test whether
the agent would widen against informed flow.

That makes it a prerequisite for re-asking P1 — but **not** for the latency
experiment, whose predictions concern fills and P&L rather than spread. Its
prediction 4 ("spread stays pinned at the 2c floor") is now explained rather
than merely expected.
