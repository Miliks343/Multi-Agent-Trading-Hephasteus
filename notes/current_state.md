# Where the project stands

**Keep this file current.** It is the thing a fresh session should read first.
Everything else in `notes/` is a record of a moment; this is the running state.

Last updated: 2026-09-04 (grid2 running overnight).

## The one-line summary

The agent now makes markets, and still loses money. The action space was
rebuilt so withdrawal is a decision rather than a rounding artifact, and the
agents were given an identity bit so they stop being the same bot; both are
verified at 10 seeds. Acts 4 and 7 still need new claims.

## What is settled and quotable

- **The hand-coded F baseline.** No PPO, no VecNormalize, untouched by any of
  the four defects. `−$16,674 / −$11,727` default-F, the 56.6% / 62.7%
  drawdowns, and `+$590` at spread=30/size=10 all stand.
  (`notes/experiments_results.md`, Tier 2b.)
- **The frozen policy.** Default inventory penalty with no VecNormalize posts
  zero quotes and takes zero fills, on every seed, on fixed code.
- **VecNormalize is the load-bearing change**, and keeping the env's default
  inventory penalty beats removing it. Retrained: `vecnorm_only` +366 and
  positive on 2/3 seeds; `exp1` −3,971 and negative on 3/3.
- **The grid's null.** 780 runs, 65 seeds/cell, zero failures. All three
  pre-registered predictions unsupported, because there is no market making to
  compete over. (`notes/grid_results.md`.)

## What is broken or withdrawn

- **Act 7 (the phase diagram) is dead.** There is no viability boundary because
  there is no market making anywhere on the grid. Spread capture is under 1% of
  |P&L| at every cell. Not fixable with more compute.
- **Act 4's specification-gaming beat is gone.** May's `min_size=10` agent
  quoted ~50c off mid for 7 fills; retrained it quotes 0.287c and takes 1,181
  fills. It no longer routes around the constraint.
  (`notes/retrain_results.md`.)
- **Every PPO number in `experiments_results.md`** predates `808a6c0` and was
  trained against the E3 corrupted reward scale. Superseded by
  `notes/retrain_results.md`.

## Landed 2026-09-04 — the gated action space and the identity one-hot

See `notes/gate_results.md` (30 runs, 10 seeds, zero failures). In confidence
order:

1. **Gated sides cannot cross.** 0.00% on every seed; structural, not
   statistical.
2. **Gating increases quoting.** two-sided +14.0pp, any-quote +26.6pp, both
   positive on 10/10 seeds (p = 0.002).
3. **Only the identity bit differentiates the agents.** Quote agreement
   99.6% -> 53.1%, 10/10 seeds. Gating alone is a clean null (-0.26pp,
   p = 0.75) even though it triples quoting - more trading was never going to
   be enough.
4. **The direction of differentiation is a coin flip.** Inventory correlation
   across 10 seeds: 5 negative, 5 positive, mean +0.149. The single-seed
   "-0.477, they take opposite positions" result is **withdrawn**. What
   survives is the magnitude: every seed falls from ~1.00.
5. **A genuinely quoting agent still loses money.** legacy -$2,805,
   gated_noid -$1,373, gated_id -$1,489; the gating improvement is *not*
   significant (7/10, p = 0.34). One of 30 seed-cells is positive.

Cost accepted: every pre-2026-09 PPO checkpoint is incomparable. The
hand-coded F baseline is untouched and its numbers stand.

**`withdrawn%` is now a real dependent variable** (22.7% / 32.5% in the gated
cells) rather than a rounding artifact. Whether it rises with informed flow is
the grid, and is now answerable.

## Running now — grid2, the spine

Launched 2026-09-04 ~23:07 on the desktop, `setsid nohup experiments/run.sh
grid2 -j 6 > /tmp/grid2.log`. **240 runs** (12 cells x 20 seeds), ~6 hours.
Survives a Claude Code restart; it does not need a session attached.

Predictions were **pre-registered before the first job** in
`notes/grid2_preregistration.md`. Do not edit that file now.

The decisive cell is **v10_n1** — least informed flow, no competition. A
properly quoting agent already loses $1,373/agent at v50, and v10 is the only
cell with less informed flow than that. If v10_n1 is not profitable there is no
viable region anywhere in this parameterisation, which is a statement about the
environment rather than about PPO.

**To pick it up:**

    ssh pavel@100.92.153.77
    grep -E 'suite grid2 finished|NOTE:|FAIL' /tmp/grid2.log   # done?
    cd ~/marl-lob
    .work/venv/bin/python scripts/quote_stats.py grid2         # primary DV
    .work/venv/bin/python scripts/agent_divergence.py runs/grid2/v50_n2/seed0/eval

Analysis plan is in the pre-registration: contrasts **paired by training seed**
with a sign test alongside t, n/sd/interval on every number, and
`agent_divergence` as an acceptance check — any n>=2 cell whose `quote ident%`
is not well below 99% did not receive its competition treatment and its
competition result is void whatever it says.

Seeds are contiguous from 0 and jobs are seed-major, so this extends to 20-49
later by adding files rather than re-running (`MARL_GRID2_SEEDS=20-49`).

## Superseded — the old blocker on re-running the grid

`grid_results.md` prescribes "a minimum tick offset and a minimum size" to
remove the degenerate region, and names `both_hatches` as the precedent. **Half
that fix does not exist.** `--min-size` works, but `--max-offset-cents` is a
*ceiling*, and the policy never approaches it — which is why `both_hatches` is
identical to `min_size_10` to the dollar.

The action space floors offsets at zero and this is hardcoded (`env.py:113`):

    low  = [0.0, 0.0, min_size, min_size]
    high = [max_offset_cents, max_offset_cents, max_size, max_size]

With the floor at zero, offsets round to 0 on both sides, bid price equals ask
price, and `translate_action` voids both — the 54.7% self-void measured in
`retrain_results.md`. ~~**A `--min-offset-cents` flag, mirroring `--min-size`, is the prerequisite
for any grid re-run.**~~ **Resolved 2026-09-04, but not this way.** The floor
is applied inside the gated branch of `translate_action` and exposed as
`--min-offset-ticks`; a ceiling-style flag would not have helped, because the
problem was the *floor at zero*, not the ceiling. The grid re-run is
unblocked.

## Decisions waiting on Pavel

1. The spine sentence. North star as stated 2026-09-04: set up an environment,
   tried a lot of things, hopefully got interesting behaviour; and teach the
   room PPO plus probably a second algorithm. The teaching goal is agreed as
   load-bearing. See the claim slot in `notes/presentation_outline.md`.
2. What act 4 claims now that the evasion result is gone.
3. What act 7 concludes. It now has real material: the agent was never playing
   the game, three independent measurements say so, and once fixed it plays
   and still loses.
4. Whether to cite May's 7-fills number at all, given its provenance.

Settled 2026-09-04: the clone measurement is a preempt slide rather than an
act; the seed-count reversal is a methods footnote, not an act.

## Practical

- Deadline is not tight as of 2026-09-04; prefer doing the grid re-run properly
  over doing it soon.
- Machines: **desktop** (100.92.153.77, 8-core i7-9700K) is the runner — the
  retrain took 25m49s there at `-j 6`. It does **not** wake on LAN; someone has
  to press the button. **macmini** (4-core, shares the box with the media stack)
  has a verified env and is a slow fallback.
- `experiments/run.sh <suite>` **needs an explicit `-j`** on any machine older
  than `94016da`; before that fix `OMP_NUM_THREADS=1` made `nproc` report 1 core
  and the default collapsed to `-j 1`.
- `scripts/quote_stats.py <suite>` answers "was it actually making markets",
  which Δequity does not.
