# Experiments results — pre-event runs (2026-05-11)

> ## ⚠️ Correction 2026-08-31 — every PPO number below is suspect
>
> These runs predate `808a6c0`. All of them used `VecNormalize(norm_reward=True)`,
> and E3 in `design_space.md` shows SuperSuit's `uint8` done flags made SB3
> corrupt the running return statistics that scale every reward. **The policies
> below were trained against a corrupted reward scale.** Re-evaluating the
> checkpoints does not help — it faithfully reproduces a policy trained on a bug.
> They need **retraining** on fixed code (~5 min/run on an M4).
>
> Specifically **not** to be quoted as-is: headline findings 1–3, the Tier 1 seed
> table, the "+$117 / +$113" best-PPO result, and the "−$6,874 to +$271" seed
> range. On corrected code, cells of this kind moved enough to flip sign
> (`v50_n2/seed0`: +1179 → −595).
>
> **Superseded 2026-09-04 — the retrain is done.** All six cells were re-run on
> fixed code; results are in `notes/retrain_results.md`, which is authoritative
> for every PPO number. Headline: the frozen-policy result and the
> "VecNormalize is load-bearing, keep the default inventory penalty" attribution
> both survive; the `min_size` specification-gaming result (7 fills, quoting
> ~50c off mid) does **not** — the retrained policy quotes tight and takes 1,181
> fills instead.
>
> **Unaffected and still quotable: the Tier 2b F sweep below.** F is hand-coded
> — no PPO, no VecNormalize — so the −$16,674 / −$11,727 default-F result, the
> 56.6% / 62.7% drawdowns, and the **+$590** at spread=30/size=10 all stand.
>
> Note the talk hook "the naive baseline bleeds ~−$14k" is the *mean of the two
> agents* (−$16,674 and −$11,727), not a single measured figure.

Executed per `notes/experiments_brief.md`. All training: 50k steps, 1 vec env, n_steps=512, PPO MlpPolicy defaults. All eval: seed 42, max_steps 3600, deterministic policy. **Two agents per run** (`mm_0`, `mm_1`) sharing parameters via SuperSuit.

Baselines compared against:
- **Exp 1 (prior session):** inv_penalty=0.0 + VecNormalize, train+eval seed 42. PPO Δeq −$6.7k / −$6.9k, MaxDD 10%, finalInv ~−380.
- **F default (#7, prior):** spread_ticks=10, quote_size=100, wake=10s, seed=42. Δeq −$16.7k / −$11.7k, MaxDD 56.6%/62.7%.

## Headline findings

1. **The "PPO beats F" claim is broken.** F at right-tuned params (spread=30, size=10) is *profitable* on seed 42; F at small quote_size (10) loses only ~$1k. PPO best run (seed 7) is competitive with F-best but PPO is not consistent across seeds.
2. **VecNormalize is the only load-bearing change.** Removing `inventory_penalty` does nothing on its own (collapses to noop). Worse — restoring the default inv_penalty=1e-4 *while keeping VecNormalize* gives the **best result of all PPO runs**: +$117 / +$113.
3. **PPO is highly seed-sensitive.** Across seeds {0, 1, 7, 42} the Δequity range is −$6,874 to +$271. The brief's "−$3k to −$15k acceptance band" fails — two of four seeds fall outside.

## Tier 2b — F parameter sweep (seed 42, no training)

| # | spread | size | wake | F[0] Δeq | F[1] Δeq | F[0] MaxDD | F[1] MaxDD | F[0] finalInv | F[1] finalInv |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 10 | 10 | 10s | −$997 | −$1,246 | 1.56% | 1.96% | −195 | −206 |
| 7 (default, prior) | 10 | 100 | 10s | −$16,674 | −$11,727 | 56.6% | 62.7% | +2305 | +2834 |
| 8 | 30 | 10 | 10s | **+$590** ✅ | −$75 | 5.25% | 5.55% | +145 | +135 |
| 9 | 30 | 100 | 10s | −$20,102 | −$22,572 | 31.5% | 34.6% | −2153 | −2192 |

**Interpretation.** Default-F is a strawman — `quote_size=100` gets adversely selected. Smaller quotes (size=10) dramatically reduce bleed regardless of spread. The combination spread=30, size=10 is even net-profitable on one agent. Trajectories saved to `runs/baseline_sweep/spread{10|30}_size{10|100}/`.

## Tier 1 — seed robustness on Exp 1 (50k train, eval seed 42)

Config: `inventory_penalty=0.0`, `VecNormalize(norm_reward=True, clip_reward=10.0)`. Train seed varies.

| Train seed | fills (avg) | mm_0 Δeq | mm_1 Δeq | MaxDD | finalInv | final value_loss | final approx_kl | final std | final ev |
|---|---|---|---|---|---|---|---|---|---|
| 42 (Exp 1) | 1186 | −$6,746 | −$6,864 | 10.0% | −380 | 0.006 | 0.010 | 0.92 | 0.33 |
| 0 | 391 | −$1,590 | −$1,604 | 5.7% | −375 | 0.0036 | 0.012 | 0.929 | −1.06 |
| 1 | 583 | −$6,786 | −$6,874 | 10.3% | −586 | 0.007 | 0.014 | 0.900 | 0.23 |
| 7 | 130 | **+$268** ✅ | **+$271** ✅ | 0.18% | +11 | 0.002 | 0.012 | 0.895 | −7.46 |

Δequity range across seeds: **−$6,874 → +$271** ($7k spread). Both fills count and final inventory direction vary wildly. Two of four seeds fall outside the brief's −$3k to −$15k acceptance band.

Checkpoints at `runs/marl_lob_ppo/exp1_seed{0,1,7}/ppo_marl_lob.zip`.

## Tier 2a — ablation (train+eval seed 42)

The Exp 1 setup changed two things at once (`inventory_penalty: 1e-4 → 0.0` AND added `VecNormalize`). The 2×2:

| inv_penalty | VecNormalize | result |
|---|---|---|
| 1e-4 (default) | OFF | (original failed run, prior session): noop, 0 fills, $0 |
| 0.0 | OFF | **Ablation #5: noop, 0 fills, $0** |
| 1e-4 (default) | ON | **Ablation #4: +$117 / +$113, MaxDD 0.66%, finalInv +65 / +65, 571 fills each** ✅ |
| 0.0 | ON | Exp 1 (prior): −$6,746 / −$6,864, MaxDD 10%, finalInv −380 |

**Conclusion.** VecNormalize is the only load-bearing change. Removing inventory penalty alone does *nothing*. Worse, with VecNormalize active, **keeping the default inventory penalty is strictly better** than removing it. The Exp 1 framing ("we removed the inv penalty and added VecNormalize") was wrong in attribution: the inv-penalty removal was actively counterproductive.

Training stats:
- **#4 (VecNorm only):** value_loss 0.00074, approx_kl 0.014, std 0.97, ev −5.21
- **#5 (inv_pen=0 only):** value_loss 4.5e5, approx_kl 0.006, std 0.982, ev 0.008 — value head can't fit raw-magnitude rewards, policy frozen exactly like the original noop run.

Checkpoints at `runs/marl_lob_ppo/abl_{vecnorm_only,invpen0_only}/ppo_marl_lob.zip`.

## Implications for the presentation

The original demo headline ("PPO beats F by ~40% with 6× better drawdown") was true only against a poorly-tuned F and from a single PPO seed. Three things changed materially:

1. **F-best (spread=30, size=10) is profitable.** Comparing PPO to default-F overstates PPO. Any honest comparison needs F-best.
2. **PPO's best result (Ablation #4: VecNorm + default inv_pen) is +$115/agent.** This is comparable to F-best (+$590 / −$75, mean ~+$250). PPO and F-best are roughly tied on seed 42.
3. **PPO variance across seeds is huge.** F is deterministic given seed; PPO is not. The honest claim is "PPO can find good policies but with high run-to-run variance."

Suggested reframings:
- Lead with the **reward-scale diagnosis** (cents → 10⁷ magnitudes → frozen policy) and **VecNormalize as the fix**. This is the actual scientific result and it's clean.
- Show the ablation 2×2 — it tells the value_loss-magnitude story crisply.
- For PPO-vs-F: show **distributions across seeds**, not single numbers. PPO best ≈ F-best; PPO worst is worse than F-default.
- Drop the "PPO beats F" headline; replace with "PPO learns market-making behavior when reward scale is normalized; matches well-tuned baseline on best seeds."

## Out-of-scope follow-ups (deferred, not run)

- Multi-seed eval of each PPO checkpoint (currently eval seed is always 42 — would tell us how the policy generalizes across market draws).
- Longer training (>50k) to reduce seed variance.
- Combined-constraint runs (min_size + max_offset clamp).
- F sweep at non-seed-42 simulator seeds.

## Artifacts

- TB logs: `runs/marl_lob_ppo/{exp1_seed0,exp1_seed1,exp1_seed7,abl_vecnorm_only,abl_invpen0_only}/tb/` (and prior runs in same parent).
- F trajectories: `runs/baseline_sweep/spread{10,30}_size{10,100}/trajectory_{0,1}_seed42.npz`.
- PPO eval trajectories: `runs/eval/ppo_trajectory_{0,1}_seed42.npz` (overwritten by latest eval; re-run any checkpoint to regenerate).
