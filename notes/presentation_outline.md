# Presentation outline — working scaffold

Started 2026-09-04. **Supersedes the act structure in `notes/primer_structure.md`
and the section plan in `notes/presentation_plan.md`**, both of which were built
on results that have since been withdrawn.

**Deliberately not drafted prose.** Each act below lists the *evidence that
exists* and the *claim slot* that evidence has to support. The claims are
Pavel's to write — he defends them in Q&A. Claude fills in plumbing, checks
numbers against `notes/`, and attacks the claims once written.

Method template: `notes/primer_structure.md` (first real result by ~15% of
runtime, one variable at a time, let a visualisation fail before earning the
better one, close on what it does *not* mean).

Status key: **[solid]** measured on code with no known defect in the path ·
**[dead]** withdrawn, do not use · **[pending]** needs an experiment that has
not been run.

---

## The spine

**Pavel's north star (2026-09-04, his words):** show that we set up an
environment and tried a lot of things and hopefully got interesting behaviour;
and teach the room a bit about PPO and probably a second RL algorithm.

**Decided:** the teaching goal is load-bearing, not secondary. A talk that
teaches PPO properly to a room of AI students is deliverable *regardless* of
what any experiment shows, which makes it the safe spine. The environment is
the vehicle; PPO is the content; the bugs are the pedagogy.

> CLAIM SLOT — still open. Working candidate, proposed by Claude and not yet
> accepted: *"Here's what it actually takes to get PPO to do a real job — and
> every time we thought we were measuring the market, we were measuring our own
> setup."* The weakness in the north star as stated is "tried a lot of stuff and
> hopefully got interesting behaviour": that describes activity rather than
> making a claim, and "hopefully" is doing too much work, since as of today
> there is no interesting behaviour — the pending fixes are what might produce
> it.

## Act 1 — The world, shown before it is explained

**Evidence:** live ABIDES run, mid-price and spread over one simulated hour.
2,060 background traders, 50 of them informed. **[solid]**

No PettingZoo, no SB3, no architecture slide. Order book, ticking price, and
the fact that some traders know something the others don't.

> CLAIM SLOT — what question opens the talk?

## Act 2 — The naive baseline loses

**Evidence:** F sweep, `notes/experiments_results.md` Tier 2b. **[solid]** —
hand-coded, no PPO, no VecNormalize, untouched by all four defects.

| spread \ size | 10 | 100 |
|---|---|---|
| 10 ticks | −$997 / −$1,246, MaxDD ~1.9% | **−$16,674 / −$11,727, MaxDD 56.6% / 62.7%** (default) |
| 30 ticks | **+$590 / −$75, MaxDD ~5.4%** | −$20,102 / −$22,572, MaxDD ~33% |

The obvious strategy at obvious parameters bleeds. Tuned, it is profitable.
Caveat to carry: "bleeds ~−$14k" is the *mean of two agents*, not a measured
single figure.

> CLAIM SLOT — is this act "the problem is hard" or "baselines are a
> methodology trap"? It can carry either, not both.

## Act 3 — Hand it to RL and it refuses

**Evidence:** `noop` and `invpen0_only` cells, `notes/retrain_results.md`.
Zero quotes, zero fills, $0, on every seed, on fixed code. **[solid]**

State the expectation before revealing (Primer rule 4).

## Act 4 — Diagnose, one knob at a time

**Evidence:** `notes/retrain_results.md`, the 2x2 plus the constraint cells.
**[solid]** for everything below except where marked.

| cell | mean Δeq | fills | note |
|---|---|---|---|
| `noop` | 0 | 0 | frozen |
| `invpen0_only` | 0 | 0 | removing the penalty alone changes nothing |
| `vecnorm_only` | **+366** | 602 | the only profitable cell; positive 2/3 seeds |
| `exp1` | −3,971 | 969 | fix the scale, drop the penalty: trades and loses |
| `min_size_10` | −14,526 (sd 15,136) | 1,181 | forced to quote, gets run over |

The PPO teardown lives here: rewards ~1e7 → `value_loss` ~1e11 →
`explained_variance` negative → advantages are noise → `approx_kl` ~1e-5 →
policy frozen at initialisation. Every PPO component motivated by its own
failure mode. Diagnostics to teach as instruments: `value_loss`,
`approx_kl`, `explained_variance`, policy `std`.

**[dead] — do not use:** the specification-gaming beat. May's `min_size=10`
agent quoted ~50c off mid for 7 fills; retrained it quotes 0.287c and takes
1,181 fills. It does not route around the constraint.

**[dead] — do not use:** `both_hatches` as a distinct condition. It is
identical to `min_size_10` to the dollar; `max_offset_cents` is a ceiling the
policy never approaches.

> CLAIM SLOT — the most memorable beat of this act is gone. What replaces it?
> The measured alternative is "a constraint that forces participation and gets
> the agent killed, at a spread it cannot survive." Different lesson. Pavel's
> call, and it is decision 1 in `notes/current_state.md`.
> Sub-decision: cite May's 7-fills number at all, given its provenance?

## Act 5 — Let the visualisation fail, then earn the better one

**Evidence:** Δequity table alone explains nothing; the markout decomposition
explains all of it. **[solid]**

| n_agents | Δequity | capture | adverse | capture as % \|P&L\| |
|---|---|---|---|---|
| 1 | −819 | 7.3 | −827 | 0.9% |
| 2 | −1,339 | 8.2 | −1,347 | 0.6% |
| 4 | −908 | 7.9 | −916 | 0.9% |

Under 1% of P&L comes from earning the spread. The direct analogue of Primer's
ternary-plot detour; budget ~2 minutes and teach it properly.

## Act 6 — The honest correction

**Evidence:** the +$590 "profitable" baseline is −$3,310 at a 60s markout; it
is positive only on the closing mark. **[solid]**

Killing your own headline with your own measurement — the credibility beat.

## Act 7 — The agent was never playing the game

**[dead]** The phase diagram as originally designed. No viability boundary
exists on the existing grid because there is no market making anywhere on it.
Not fixable with more compute.

**[solid]** Three measured findings, in the order they should be told:

1. **The action space let the agent opt out by accident.** Offsets average
   ~0.3c and round to 0; sizes average 0.3 against `max_size=100` and round to
   0; `translate_action` voids both sides when the rounded bid crosses the
   rounded ask. Genuine two-sided market on **~2% of steps**.
   (`notes/grid_results.md`)

   The mechanism to explain, and the reason this is a *broken* refusal rather
   than an economic one: SB3's Gaussian policy initialises at mean ~0, and the
   action space floor is 0, so the policy **starts inside the no-quote region**.
   Within that region a parameter change produces no change in the posted quote
   and therefore no change in reward — zero gradient. There is no signal
   pointing out. That is an exploration failure, not a decision to withdraw.

2. **The agents were the same bot.** 4,350 pairs; effective quotes identical on
   **97–99.9%** of steps at every cell including n=4; inventory correlation
   >0.997; only 2 of 44 observation features ever differ.
   (`notes/agent_divergence_results.md`)

   **Decided 2026-09-04: this earns a slide, but as a preempt rather than a
   finding.** Parameter sharing producing near-identical agents is unsurprising
   to anyone who knows MARL, so it is not a discovery. It is worth showing
   because it is the first objection an informed audience will raise — Pavel's
   own reaction on being shown it was that this is what he would be wondering
   about from the audience. Showing the measurement answers the question before
   it is asked. **How we found it** is the interesting part: the raw actions
   differ on 70-88% of steps, and only collapse to identical once quantised to
   whole ticks. The agents disagree by less than the exchange's resolution.

3. **[demoted]** The seed-count reversal. **Decided 2026-09-04: this is a
   methods footnote, not an act.** Pavel's argument, which is correct: at 15
   seeds the contrast was t = 1.31, which does not clear significance anyway,
   so honest reporting with n, sd and a confidence interval would have said
   "not significant" at both seed counts. There is no scandal here — just an
   illustration of why you report intervals rather than point estimates.

Plus the methodological asset: 780 runs, 65 seeds/cell, zero failures, host
effect tested and rejected before pooling.

The distinction to hold onto: a **correct** refusal is an agent withdrawing
because adverse selection makes quoting unprofitable (Glosten-Milgrom, 1985).
What the data shows is a **broken** refusal — a policy that never learned to
quote at all.

**[solid, 2026-09-04] What happened when we fixed it.** The action space was
rebuilt so that withdrawal is an explicit gated decision rather than a rounding
artifact, and the agents were given an identity bit. 30 runs, 10 seeds, zero
failures (`notes/gate_results.md`). In confidence order:

- **Gated sides cannot cross.** 0.00% on every seed. Structural: with both
  gates open and offsets floored at a tick, `bid <= mid-1 < mid+1 <= ask`.
- **It quotes.** two-sided +14.0pp, any-quote +26.6pp, positive on 10/10
  seeds (p = 0.002), at a ~2c spread instead of sub-tick noise.
- **Only the identity bit differentiates the agents.** Quote agreement
  99.6% -> 53.1%, 10/10 seeds. Gating alone is a clean null (p = 0.75) even
  though it triples quoting. This is the tightest causal isolation in the
  project and it pairs directly with the preempt slide above.
- **And it still loses money.** legacy -$2,805, gated_noid -$1,373,
  gated_id -$1,489. The gating improvement is not significant (7/10,
  p = 0.34). One of 30 seed-cells is positive. An agent that genuinely makes
  markets at ~2c into 2.4% informed flow loses.

**[dead] The specialisation beat.** A single seed showed the two agents taking
*opposite* inventory positions (corr -0.477) and it looked like the best result
in the project. At 10 seeds it is **5 negative, 5 positive** - a coin flip. The
claim "one bit of identity makes them specialise into opposite strategies" is
withdrawn. What survives is the magnitude: every seed's correlation falls from
~1.00, so the symmetry breaks reliably and the *direction* is arbitrary.

This is the third time seed count has overturned an exciting single number
here. It belongs in the methods footnote with the others.

**[solid, 2026-09-05] The spine result — grid2.** 240 runs, 20 seeds, zero
failures, predictions pre-registered before the first job
(`notes/grid2_results.md`). The competition axis was verified live this time:
`quote ident%` 41-60% at every multi-agent cell, against the 97-99.9% that
voided the last grid.

- **No viable region.** All twelve cells of informed-flow x competition lose.
  Every 95% CI excludes zero. The best case in the whole design — least
  informed flow, no competition — is **−$2,649 per agent**.
- **The mechanism is identical everywhere.** Spread capture is ~$22 at every
  cell while adverse selection ranges over a factor of two, so **capture is
  0.8-1.4% of |P&L| throughout**. The agent earns the same trivial spread no
  matter what the market looks like; the outcome is set entirely by how badly
  it gets picked off.
- **The punchline for act 4's callback.** Fixing the action space tripled
  capture *and* tripled adverse selection, leaving the ratio unchanged. We
  turned it into a real market maker and it got run over at three times the
  scale.
- **Competition does nothing** — and this is a genuine null, not a null by
  construction, which is the difference the acceptance check buys.

**[pending] The honest gap, and it belongs on the "what this does not mean"
slide.** The agent never widens its spread: it is pinned at the 2c floor at
every cell, and its raw offsets are 0.18-0.29c, so it is still trying to quote
sub-tick. Two readings fit — informed flow genuinely does not drive withdrawal,
or **the agent cannot perceive informed flow at all**. The observation has no
order-flow imbalance, no trade history, no signed flow. An agent blind to
adverse selection would look exactly like this. The no-viable-region result does
not depend on which reading is right; the withdrawal question does, entirely.

> CLAIM SLOT — what this act concludes depends on whether the re-run produces
> a withdrawal curve. Do not write it until the result exists.

## Act 8 — What it does not mean

Seed variance · one simulated hour · no latency edge · 2–4 agents ·
independent learning with parameter sharing, not a centralised-critic method ·
and now: the agents were not distinguishable, so no claim about competition
between market makers was ever tested.

Have ready (will be asked): **why not "real" MARL** — MAPPO/QMIX/MADDPG use a
centralised critic for *cooperative* settings with a shared reward. These
makers have individual PnL and compete. Independent learning with parameter
sharing is the standard, appropriate baseline for non-cooperative settings.

---

## Depth beats for this audience (AI student club)

- Training diagnostics as instruments, not appendix plots.
- PPO explained *through* the bug, not as a prerequisite section.
- Stale on-policy data → importance ratio → clipping. Motivates why PPO exists
  at all, and why SAC being off-policy is nearly free in wall-clock here.
- "Constraints get routed around" is specification gaming — *if* act 4 keeps a
  version of that beat.

## Open decisions

1. **The spine sentence.** See the claim slot at the top. Still open.
2. **Act 4's replacement claim**, now that the specification-gaming beat is
   dead. Still open.
3. **Act 7's conclusion** — blocked on the gated-action-space result, correctly.
4. Whether to cite May's 7-fills number at all. Still open.

**Settled 2026-09-04:** the clone measurement is a preempt slide, not an act
(decision recorded in act 7). The seed reversal is a methods footnote, not an
act. The teaching goal is the spine, not a secondary objective.

## The build queue

Agreed with Pavel 2026-09-04. Ordered; each one gates the next.

1. **Gated action space.** Replace the 4-tuple with
   `(bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size)`. A gate
   that is on floors its offset at 1 tick and its size at 1 share, so a quote
   is a real quote; a gate that is off posts nothing on that side. Per-side
   gates also permit one-sided quoting, i.e. inventory skewing, which the
   current action space cannot express.

   Rationale: withdrawal stays possible — Pavel's realism requirement, and the
   behaviour Glosten-Milgrom actually predicts — but becomes a deliberate
   choice with a gradient rather than a consequence of rounding. **Accepted
   cost: this changes the action contract, so every existing PPO number becomes
   incomparable and must be retrained.** The hand-coded F baseline is
   unaffected and its numbers stand.

2. **Agent identity in the observation.** One-hot agent ID, so the shared
   network can condition on which agent it is and learn to differentiate.
   ~10 lines. Necessary for any claim about competition between makers.

3. **Cheap acceptance gate before committing to a full grid.** Two instruments,
   both already built: `scripts/quote_stats.py` answers "is it making markets
   at all", `scripts/agent_divergence.py` answers "are these agents distinct".
   A re-run that does not move `quote ident%` well below 99% has not applied
   its treatment and must not be scaled up.

4. **Grid re-run** on top of 1-3, with withdrawal rate as a first-class
   dependent variable.

5. **SAC as a control, not a horse race.** The question is "is the refusal a
   property of PPO or of the market", not "which algorithm wins" — a horse race
   is unresolvable at achievable seed counts and is not a finding anyway.

6. Separate networks per agent; heterogeneous populations; predator/prey. As
   time allows; none required for a complete talk.
