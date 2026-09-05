# grid2 results — informed flow x competition, on the gated action space

240 runs (12 cells x 20 seeds), `experiments/run.sh grid2 -j 6` on the desktop,
**8h36m, 240/240 ok, zero failures**. 50k env steps per agent, eval on seeds
42/43/44. Predictions were committed before the first job in
`notes/grid2_preregistration.md`; analysis follows the plan stated there.

## Headline

**The treatment was applied this time, and there is still nothing there.**

Every cell loses money. Spread capture is **+$20 to +$26 per agent** against
adverse selection of **−$1,700 to −$3,200** — between **0.8% and 1.4% of |P&L|
at all twelve cells**. Neither informed flow nor competition moves
profitability, because there is no profitability anywhere in the design to move.

This is a different statement from the 2026-09 grid. That null meant "we never
applied the treatment". This one means "we applied it and the effect is not
there".

## Acceptance check — passed

The pre-registered gate: any n>=2 cell whose `quote ident%` is not well below
99% did not receive its competition treatment, and its result is void.

| cell | quote ident% | | cell | quote ident% |
|---|---|---|---|---|
| `v10_n2` | 60.07 ± 8.98 | | `v150_n2` | 52.07 ± 7.96 |
| `v10_n4` | 52.03 ± 11.77 | | `v150_n4` | 47.71 ± 10.38 |
| `v50_n2` | 48.99 ± 10.05 | | `v400_n2` | 43.41 ± 9.74 |
| `v50_n4` | 44.36 ± 8.62 | | `v400_n4` | 41.45 ± 10.79 |

41-60% everywhere against the 97-99.9% that voided the last grid. **This is the
first grid in the project whose competition axis actually varied competitors
rather than copies of one policy.**

## P3 — no viable region. Supported.

Δequity, USD per agent, 95% CI over 20 seeds:

| n_agents | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| 1 | **−2649 ± 1859** | −1820 ± 1100 | −1988 ± 966 | −2337 ± 1006 |
| 2 | −3193 ± 1044 | −2053 ± 1065 | −2092 ± 711 | −1676 ± 402 |
| 4 | −2809 ± 1486 | −2617 ± 729 | −2346 ± 460 | −1867 ± 610 |

**All twelve cells negative, and every interval excludes zero.** The decisive
cell `v10_n1` — least informed flow, no competition, the best case the design
contains — is −$2,649. There is no viable region in this parameterisation.

The markout says why, identically at every cell:

| cell | capture | adverse | Δequity | capture as % \|P&L\| |
|---|---|---|---|---|
| `v10_n1` | +23 | −2672 | −2649 | 0.9 |
| `v50_n1` | +22 | −1842 | −1820 | 1.2 |
| `v150_n1` | +23 | −2012 | −1988 | 1.1 |
| `v400_n1` | +23 | −2360 | −2337 | 1.0 |
| `v10_n2` | +26 | −3219 | −3193 | 0.8 |
| `v50_n2` | +21 | −2074 | −2053 | 1.0 |
| `v150_n2` | +22 | −2114 | −2092 | 1.1 |
| `v400_n2` | +23 | −1700 | −1676 | 1.4 |
| `v10_n4` | +23 | −2832 | −2809 | 0.8 |
| `v50_n4` | +22 | −2640 | −2617 | 0.8 |
| `v150_n4` | +20 | −2367 | −2346 | 0.9 |
| `v400_n4` | +20 | −1888 | −1867 | 1.1 |

**Capture is essentially constant at ~$22 across every cell**, while adverse
selection ranges over a factor of two. The agent earns the same trivial amount
of spread everywhere and the outcome is set entirely by how badly it is picked
off.

### What fixing the action space actually bought

Against the 2026-09 grid, pooled at n=1:

| | 2026-09 grid | grid2 |
|---|---|---|
| capture | +7.3 | +22.8 |
| adverse | −827 | −2221 |
| Δequity | −819 | −2198 |
| capture as % \|P&L\| | 0.9% | 1.0% |

Making the agent quote properly **tripled its spread capture and tripled its
adverse selection**. The ratio is unchanged to within a rounding error. It is
now a real market maker, and it is being run over at three times the scale.

## P1 — withdrawal rises with informed flow. Not supported.

`withdrawn%`, 95% CI:

| n_agents | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| 1 | 27.9 ± 9.8 | 27.8 ± 7.9 | 28.5 ± 10.3 | 24.0 ± 7.8 |
| 2 | 22.6 ± 6.2 | 27.2 ± 6.7 | 24.7 ± 6.5 | 25.1 ± 6.5 |
| 4 | 21.3 ± 6.3 | 22.8 ± 4.0 | 24.5 ± 5.6 | 23.4 ± 4.6 |

Paired v400 − v10: −3.83pp (p=0.82) at n=1, +2.42pp (p=0.50) at n=2, +2.04pp
(p=0.50) at n=4. Flat, and not even consistent in direction.

## P2 — spread widens with informed flow. Not supported, and the DV is degenerate.

Quoted spread is **1.985c to 2.032c at every one of the twelve cells**. The
floor is 2c (one tick each side), so the policy is pinned to the minimum
everywhere.

The raw pre-quantisation offsets say why: the policy emits **0.18-0.29c**
across all cells, i.e. it is *still trying to quote sub-tick* and the floor is
doing all the work. The gated action space made it post a real quote; it did
not make it want to.

**A methodological note worth carrying.** The v400−v10 sign tests at n=2 and
n=4 come out "significant" (p=0.041, p=0.012) on effect sizes of **−0.026c and
−0.002c**, in the *wrong* direction, and at n=150 the sign test reports 16/20
positive while the mean is negative. A metric pinned against a floor produces
tiny, consistent differences that a sign test will happily call significant.
These are not effects. Report effect sizes next to p-values or this kind of
thing walks straight into a slide.

## P4 — competition effects are small. Supported.

n=4 minus n=1, paired by training seed:

| metric | v10 | v50 | v150 | v400 |
|---|---|---|---|---|
| Δequity | −160 (p=1.00) | −797 (p=0.50) | −358 (p=0.26) | +470 (p=0.82) |
| withdrawn% | −6.55 (p=1.00) | −5.04 (p=0.12) | −4.04 (p=0.82) | −0.68 (p=0.12) |
| two-sided% | +6.93 (p=0.50) | +8.47 (p=0.12) | +1.69 (p=1.00) | −3.56 (p=0.26) |

Every contrast null. **Unlike 2026-09 this is a genuine null**: the acceptance
check confirms the agents were distinct, so competition was actually varied.
Adding competitors does not change what a market maker earns here, because what
it earns is ~$22 regardless.

## The caveat that matters most

**P1's failure may be a measurement problem rather than a finding, and the
project cannot currently tell the difference.**

Withdrawal is flat at ~25% and the quoted spread is pinned at the floor at
every level of informed flow. Two readings fit equally well:

1. Informed flow genuinely does not drive withdrawal in this environment.
2. **The agent cannot perceive informed flow at all.** The 44-dim observation
   is a book snapshot plus inventory, cash, spread and time. It contains **no
   order-flow imbalance**, no trade history, no signed flow — the single
   most-cited predictive feature in microstructure is absent
   (`notes/design_space.md`, "what is still genuinely missing"). An agent with
   no signal for adverse selection cannot condition its spread on it, and would
   look exactly like this.

Reading 2 is the more likely one, and it is testable: add order-flow imbalance
to the observation and re-run. Until that is done, **"the agent does not widen
against informed flow" is not a claim about market making** — it is a claim
about an agent that may be blind to the thing it is supposed to respond to.

The P3 result does not depend on this. Capture at ~1% of |P&L| holds however
the agent perceives its world.

## Other limitations, as pre-registered

- 20 seeds settles signs, not magnitudes.
- `num_value_agents` is a coarse integer x-axis; `val_lambda_a` is the
  continuous informed-arrival dial and would locate a threshold properly.
- One simulated hour, one fundamental process, no latency edge, PPO only.
- Seeds are contiguous from 0, so this extends to 20-49 with
  `MARL_GRID2_SEEDS=20-49` by adding files rather than re-running. Given how
  flat everything is, more seeds are unlikely to be the missing ingredient.

## What this licenses saying

1. On a properly specified action space, with agents verified distinct, a PPO
   market maker in RMSC03 **loses money at every combination of informed flow
   and competition tested**, and earns ~1% of its P&L from the spread.
2. Fixing the action space tripled both capture and adverse selection and left
   their ratio unchanged.
3. Competition has no measurable effect on profitability — a real null.
4. Whether informed flow drives withdrawal is **still untested**, because the
   agent has no observation channel for it.
