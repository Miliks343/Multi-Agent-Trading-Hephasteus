# Where the project stands

**Keep this file current.** It is the thing a fresh session should read first.
Everything else in `notes/` is a record of a moment; this is the running state.

Last updated: 2026-09-04.

## The one-line summary

The agent does not make markets. Four defects were found and fixed, the May PPO
results have been retrained on the fixed code, and the presentation's two
payoff acts (4 and 7) both lost the result they were built on.

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

## The blocker on re-running the grid

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
`retrain_results.md`. **A `--min-offset-cents` flag, mirroring `--min-size`, is
the prerequisite for any grid re-run.** Not yet written. Re-running the grid
without it reproduces the same null for the same reason.

## Decisions waiting on Pavel

1. What act 4 claims now that the evasion result is gone.
2. What replaces act 7. One proposal on the table: "we looked for emergence and
   found a bug in ourselves," built on the contrast reading +448 (t=1.31) at 15
   seeds and −88 at 50 — it reversed sign. Not signed off.
3. Whether to cite May's 7-fills number at all, given its provenance.

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
