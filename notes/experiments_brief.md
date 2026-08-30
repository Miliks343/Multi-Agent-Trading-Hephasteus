# Experiments brief — pre-event MARL LoB runs

**Hand-off doc.** You are picking this up cold in the team repo (`Multi-Agent-Trading-Hephasteus`, branch `feat/pettingzoo-env`). Pavel will be running these with you. The goal is to harden a demo result and produce a few clean comparison points for a presentation in a few days.

## Context (one paragraph)

We have a working PettingZoo wrapper around ABIDES (RMSC03) and are training independent PPO with parameter sharing. First training attempt (cents reward, default action space, default inventory penalty) collapsed to a noop policy — PPO learned to quote size=0 because reward magnitudes (~10⁷ cents/episode) blew up `value_loss` and froze the policy (`approx_kl ~10⁻⁵`). Pavel ran two follow-ups today:

- **Exp 1 (validated):** `inventory_penalty=0.0` in `MarlLobEnv` + `VecNormalize(norm_reward=True, clip_reward=10.0)` on the SB3 vec env. PPO trained cleanly (value_loss → 0.006, approx_kl ~0.01, std → 0.92, explained_variance → 0.33). At seed 42, PPO Δequity ≈ −$6.7k / −$6.9k per agent, MaxDD 10%, ending inventory short ~380. F at default config (spread_ticks=10, quote_size=100, wake_up_freq=10s) Δequity ≈ −$16.7k / −$11.7k, MaxDD ~60%, ending inventory long ~2500. **PPO loses ~40% less than F with ~6× better drawdown — this is the demo number.**
- **Exp 2 (instructive failure):** added `min_size: int = 0` kwarg to `MarlLobEnv` raising `action_space.low` on size dims. At `min_size=10`, PPO routed around the constraint by quoting at max offset (~50¢ from mid) — only 7-8 fills per agent. Insight: PPO's attractor is "avoid adverse selection," not "size=0 specifically." Single-knob constraints get routed around.

This brief lays out the next 9 runs.

## Run matrix

All training runs: 50k steps, same architecture as Exp 1, same eval seed (42, matches Pavel's prior reports). Report per-agent results in the same table shape Pavel used.

### Tier 1 — seed robustness on Exp 1 (3 runs, training)

| # | Name | Training seed | Config diff vs Exp 1 |
|---|---|---|---|
| 1 | exp1_seed0 | 0 | none |
| 2 | exp1_seed1 | 1 | none |
| 3 | exp1_seed7 | 7 | none |

**Why:** Pavel's Exp 1 was a single-seed run. We need to know whether the demo number is reproducible or a cherry-pick. Acceptance: Δequity per agent within roughly 2× of −$6.7k across seeds (i.e. all in the −$3k to −$15k band, not one at $0 or +$5k). MaxDD ideally stays under 20% across seeds.

### Tier 2a — ablation of Exp 1 (2 runs, training, seed 42)

| # | Name | inventory_penalty | VecNormalize | Notes |
|---|---|---|---|---|
| 4 | abl_vecnorm_only | default (whatever it was before Exp 1) | on | isolates VecNormalize |
| 5 | abl_invpen0_only | 0.0 | off | isolates penalty removal |

**Why:** Exp 1 changed two things simultaneously. Before claiming "we diagnosed the reward-scale issue and fixed it," we need to know which lever was load-bearing. Strong prior: VecNormalize was the critical one (since the diagnosis was reward magnitude); penalty removal alone shouldn't help.

**Note for whoever runs this:** check what the *original* `inventory_penalty` value was before Exp 1 set it to 0 — that's what `abl_vecnorm_only` should restore. Probably non-zero in the env defaults. If you can't find a non-zero default in the code, ask Pavel what the original value was — don't guess.

### Tier 2b — F parameter sweep (4 runs, no training, just F + eval)

| # | spread_ticks | quote_size | wake_up_freq |
|---|---|---|---|
| 6 | 10 | 10 | 10s |
| 7 | 10 | 100 | 10s (this is Pavel's existing default — already have results, can skip if confirmed) |
| 8 | 30 | 10 | 10s |
| 9 | 30 | 100 | 10s |

**Why:** the demo claim is "PPO beats F." If F-at-fair-config (smaller size, wider spread) bleeds significantly less, we need to know before the talk and reframe. Acceptance: ideally F's best Δequity per agent across this 2×2 is still meaningfully worse than PPO's. If any F config gets to roughly −$7k or better, the demo claim weakens and we tell Pavel immediately so he can adjust the slide.

## Per-run reporting

For each training run, report (per agent — both agents in the env):

```
fills:     ___ / ___
Δequity:   $___ / $___
MaxDD:     ___%
final inv: ___ / ___
```

Plus training-health stats at end of training:
- final `value_loss`
- final `approx_kl`
- final policy `std`
- final `explained_variance`

For each F run, report the same eval block (no training stats).

Report to Pavel as you finish each run, not all at once. He's iterating on the presentation in parallel.

## Suggested execution order

If compute is sequential, run in this order — earliest results unblock the most decisions:

1. **F sweep (#6, #8, #9)** first — short runs, no training. If any F config blows up the demo claim, we want to know before sinking compute into seed runs. ~minutes each.
2. **Tier 1 seed robustness (#1, #2, #3)** next — most important for demo credibility.
3. **Tier 2a ablation (#4, #5)** last — important for the science slide but doesn't block the demo number.

If something can run in parallel (multiple cores / processes), prioritize getting one of {#1, #2, #3} done alongside the F sweep — at least one corroborating seed result early de-risks the most.

## Stop conditions / escalate to Pavel immediately

- Any seed run produces wildly different behavior (e.g. Δequity > $0, or fills < 100). Don't average it away — Pavel needs to see the divergence.
- Any F sweep config gets within ~$3k of PPO's −$6.7k. Demo framing changes.
- Training health breaks again (value_loss > 100, approx_kl < 10⁻⁴). Means VecNormalize alone isn't enough at that config — diagnose before next run.
- Anything else surprising. Surprises are more valuable than confirmations right now.

## Out of scope (do not run)

- Combined-constraint runs (min_size + max_offset clamp).
- Longer training (>50k).
- Turnover / fill reward shaping.

These are tier-3 and depend on what tier 1+2 show. Don't pre-empt.

## Repo notes (likely-relevant pointers)

- Train script: `scripts/train.py`
- Eval script: `scripts/eval.py`
- Baseline-only run: `scripts/run_baseline.py`
- Env: `src/marl_lob/env.py` (`MarlLobEnv`, `inventory_penalty` kwarg, `min_size` kwarg added in Exp 2 work)
- F: `src/marl_lob/baseline_traj.py` (`LoggingConstantSpreadMM`, params: spread_ticks, quote_size, wake_up_freq)

Verify these against the current branch state before relying on them — the file map in this brief is from memory and may be stale.
