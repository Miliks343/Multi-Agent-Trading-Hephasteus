# Presentation method — Primer's instructional design

Source: structural breakdown of two Primer (Justin Helps) videos, obtained via
Gemini's YouTube integration 2026-08-19, because video isn't directly readable
here. Primer is Blender + Python simulations; Helps has a physics/materials
background and **worked at Khan Academy**, which is why the pedagogy is this
deliberate rather than incidental.

Pavel's intent: this is the quality bar and narrative template for the September
presentation — and it should shape **data gathering**, not just the slides,
because the structure dictates which experiments need to exist.

---

## The hard numbers worth stealing

- **Setup-to-first-result: ~15–16% of runtime, in both videos.** Natural
  Selection: rules explained by 1:12, first baseline result at 1:30 of 10:00.
  RPS: matrix explained by 1:55, first result at 2:27 of 15:00.
  For a 20-minute talk that means **the first real result lands by minute 3.**
- **A 90-second detour to teach a new visualisation** (the ternary plot), placed
  *after* the old visualisation visibly failed — not before.
- Deliberate failures get real screen time: 45s baseline, 38s extinction event,
  65s of inadequate bar graphs.

## The rules

1. **Open with a question, not an architecture.** Ground it in something real
   first (RPS opens on the side-blotched lizard, which actually plays RPS).
2. **Run a naive baseline before introducing the interesting mechanism.** In
   Natural Selection this proves the environment can support life at all and
   establishes carrying capacity (~95 blobs) so later numbers mean something.
3. **One variable at a time, each motivated by the real world or by the previous
   failure** — never "and now let's also try X". Speed → beat others to food.
   Size → creates an alternative food source. Sense → save energy. Mutations →
   dead strategies otherwise can't re-enter. Tie penalty → fighting wastes energy.
4. **State the expectation before revealing**, as a genuine multiple-choice, and
   repeat the prompt mid-talk. Verbatim: *"If we unpause this world with speed
   mutations turned on, what would you predict? … It's hard to say at this point,
   we'll just have to unpause and see."* And later: *"I know by this time in the
   video it's easy to just sit back and let it wash over you, but if you continue
   making predictions you will learn a bit more."*
5. **Let a visualisation fail, then earn the better one.** The bar graph swings
   wildly and pegs at 100%; only then does he spend 90 seconds teaching the
   ternary plot. The audience wants the new tool by the time it arrives.
6. **Show a failure, then rewind and fix the experiment, on camera.** Food
   100 → 10 abruptly kills everything; he rewinds and drops it gradually instead.
7. **Close on what it does *not* mean.** Natural Selection ends by rejecting the
   "evolution marches toward complexity" reading.

## Which video is the better template

**RPS, not Natural Selection.** Same shape as this project: competing strategies,
population dynamics, and the question of whether a stable equilibrium emerges.
Its 10-minute build showing that pure strategies *cannot* produce the coexistence
seen in nature — before introducing mixed strategies — is exactly the tension
structure available here.

---

## Mapping onto the MARL LoB talk

Act structure, using results that already exist unless marked:

1. **The world, shown before it's explained.** Animated order book: bids, asks,
   a ticking price, 2060 traders — 50 of whom know something. No PettingZoo, no
   SB3, no architecture slide.
2. **Naive baseline (rule 2).** Add one constant-spread market maker. It bleeds
   −$14k with ~60% drawdown. *The obvious strategy loses* — hook, before any RL.
3. **Hand the job to RL, and it refuses.** Zero fills, $0. State the expectation
   first (rule 4).
4. **Diagnose, then one knob at a time (rule 3).** Reward magnitudes ~10^7 →
   value_loss 10^11 → approx_kl 10^-5 → frozen policy. Fix scale, it trades and
   loses. Force min_size=10, it routes around by quoting ~50c off mid (7 fills
   vs 1181) — *the agent outsmarted the constraint*, the most memorable beat
   available. Then close both hatches. [needs the `cheap` suite]
5. **Let the visualisation fail, then earn the better one (rule 5).** Show the
   Δequity table: four configs, one mysteriously profitable, no mechanism
   visible. *Then* teach the markout decomposition — spread captured vs adverse
   selection — as the tool that explains all of it. This is the direct analogue
   of the ternary-plot detour and should get its own ~2 minutes.
6. **The honest correction.** The +$590 "profitable" baseline is −$3,310 at a 60s
   markout; it only ends positive on the closing mark. Killing your own headline
   with your own measurement is the credibility beat.
7. **Emergence.** Phase diagram: informed flow x number of competing makers, a
   viable region and a region where market making is impossible. The agent finds
   the boundary having never been told the theory — Glosten-Milgrom, 1985.
   [needs the `grid` suite]
8. **What it does not mean (rule 7).** Seed variance, single eval seed, one
   simulated hour, no latency edge, only 2-4 agents, and independent learning
   rather than a centralised-critic MARL method.

## Where to go deeper than Primer — this is an AI student club

Primer's audience wants none of this; this one does:

- **Training diagnostics as instruments**, not appendix plots: teach the room to
  read `value_loss`, `approx_kl`, `explained_variance`, policy `std`. Most of
  them have hit the reward-scale bug and misdiagnosed it as "PPO doesn't work on
  my problem."
- **Explain PPO through the bug.** Value head can't fit 10^7-magnitude returns →
  advantage estimates are garbage → the clipped objective sees no usable signal →
  approx_kl collapses → policy frozen at initialisation. Every component of PPO
  gets motivated by its own failure mode. That *is* the Primer method applied to
  the algorithm.
- **"Constraints get routed around" is specification gaming.** An AI audience
  will connect it to reward hacking immediately.
