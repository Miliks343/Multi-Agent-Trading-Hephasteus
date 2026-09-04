# Retrain results — the May-2026 PPO ablation on fixed code (2026-09-04)

18 runs (6 cells × 3 seeds), `experiments/run.sh retrain -j 6` on the desktop
(i7-9700K, 8 cores), git `8298886`, **25m49s, 18/18 ok, zero failures**.
50k env steps per agent, `n_agents=2`, eval on seeds 42/43/44.
Raw artifacts: `results/desktop-retrain-20260904T201240Z/`.

This is the replacement for `notes/experiments_results.md`, whose PPO numbers
were all trained under the E3 corrupted reward scale.

## Δequity

| cell | mean Δeq | sd | capture | adverse | capture as % \|P&L\| |
|---|---|---|---|---|---|
| `noop` | 0 | 0 | 0.0 | 0.0 | — |
| `invpen0_only` | 0 | 0 | 0.0 | 0.0 | — |
| `vecnorm_only` | **+366** | 555 | 6.7 | +359.3 | 1.8% |
| `exp1` | −3,971 | 2,747 | 17.0 | −3,989 | 0.4% |
| `min_size_10` | −14,526 | 15,136 | 135.8 | −14,675 | 0.9% |
| `both_hatches` | −14,526 | 15,136 | 135.8 | −14,675 | 0.9% |

Per seed, agent0/agent1:

| cell | seed0 | seed1 | seed7 |
|---|---|---|---|
| `noop` | +0 / +0 | +0 / +0 | +0 / +0 |
| `invpen0_only` | +0 / +0 | +0 / +0 | +0 / +0 |
| `vecnorm_only` | +659 / +687 | −415 / −411 | +835 / +844 |
| `exp1` | −4,320 / −4,394 | −428 / −436 | −7,033 / −7,217 |
| `min_size_10` | −384 / +654 | −41,460 / −27,342 | −9,349 / −9,274 |
| `both_hatches` | −384 / +654 | −41,460 / −27,342 | −9,349 / −9,274 |

## Quoting behaviour

`scripts/quote_stats.py retrain` — saved eval actions replayed through the same
quantisation `actions.py` applies before orders reach ABIDES.

| cell | two-sided% | any quote% | crossed% | spread (¢) | raw offset | raw size | fills |
|---|---|---|---|---|---|---|---|
| `noop` | 0.0 | 0.0 | 0.0 | 0.00 | 0.090 | 0.03 | 0 |
| `invpen0_only` | 0.0 | 0.0 | 0.0 | 0.00 | 0.088 | 0.03 | 0 |
| `vecnorm_only` | 0.3 | 37.1 | 2.2 | 0.97 | 0.257 | 0.25 | 602 |
| `exp1` | 17.5 | 46.4 | 4.6 | 1.28 | 0.349 | 0.53 | 969 |
| `min_size_10` | 45.3 | 45.3 | 54.7 | 1.10 | 0.287 | 10.00 | 1181 |
| `both_hatches` | 45.3 | 45.3 | 54.7 | 1.10 | 0.287 | 10.00 | 1181 |

## What survives from May

**The frozen policy reproduces exactly.** `noop` and `invpen0_only` post zero
quotes and take zero fills on all three seeds. This was expected — neither cell
wraps `VecNormalize`, so E3 could never have touched them — and it is now
confirmed on fixed code rather than assumed.

**VecNormalize is still the load-bearing change, and May's attribution still
holds.** `vecnorm_only` (VecNormalize + the env's default `inventory_penalty`
1e-4) is the only profitable cell, positive on two of three seeds. `exp1`
(VecNormalize, penalty removed) is negative on all three. So *"with
VecNormalize active, keeping the default inventory penalty is strictly better
than removing it"* survives, with the same sign and the same ordering as May's
+$117/+$113 vs −$6.7k.

## What does not survive

**The "agent routes around the constraint" result is gone.** May: forcing
`min_size=10` made the agent quote ~50¢ off mid and take **7** fills. Now:
forcing `min_size=10` gives offsets averaging 0.287 raw cents (max measured
1.5¢), sizes pinned at exactly 10.00, and **1,181** fills. The retrained policy
does not evade the constraint — it quotes tight and is run over, losing 14.5k on
average with a seed sd of 15.1k and one seed at −41,460/−27,342.

**`both_hatches` is identical to `min_size_10`, to the dollar, on every seed and
both agents.** Not a wiring bug — verified in `src/`. `max_offset_cents` only
sets the upper bound of the action-space `Box` (`env.py:115`) and SB3 clips
samples to it. The policy's offsets never exceed ~1.5¢ against a 5¢ clamp, so
the clamp binds on nothing. **The second hatch was never open**, which is only
visible now because `cheap` contained `both_hatches` but no `min_size`-only cell
to compare it against.

## Mechanism: why `min_size` forces quoting but half of it still self-voids

`translate_action` voids *both* sides when the rounded bid would cross the
rounded ask. With `min_size=10` the size floor guarantees `bid_qty > 0` and
`ask_qty > 0` on every step, so the cross test is always reached; and with
`TICK_SIZE_CENTS = 1`, an offset of 0.287¢ rounds to **0** on both sides, making
bid price equal ask price. That is the 54.7% crossed column. The other 45.3% is
a genuine two-sided market at a ~1.1¢ spread — which is what produces both the
1,181 fills and the losses, since capture is still only 0.9% of |P&L|.

So `min_size` alone is sufficient to force real quoting; the offset clamp adds
nothing at these parameters. The `both_hatches` precedent in
`notes/grid_results.md` — "the one configuration measured so far that produced
actual market making" — is more precisely a `min_size` result.

## Open for Pavel

1. **Act 4's most memorable beat is the specification-gaming one, and it no
   longer reproduces.** The measured replacement is arguably a different lesson
   (a constraint that forces participation and gets the agent killed, at a
   spread it cannot survive), but which story the act tells is a claim, not
   plumbing. Not drafted here on purpose.
2. Whether to quote the May `min_size` 7-fills number at all, and if so how to
   frame it against this. It was measured on a corrupted reward scale.
3. Seed variance in `min_size_10` is enormous (sd 15.1k on a mean of −14.5k,
   driven by seed1). Three seeds is not enough to state a magnitude; the grid
   needed 50. Stating a *sign* is defensible, a number is not.
