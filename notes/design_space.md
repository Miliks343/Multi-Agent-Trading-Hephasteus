# Design space — everything we can turn

Compiled 2026-08-19 by reading the code, not from memory. Every default below is
the value actually in the source. Cost model: a 50k-step training run is
**~10–12 min**; an eval-only run (baseline or existing checkpoint) is
**seconds to a minute**. Compute is not the constraint — choosing is.

Notation: **[free]** = already a parameter, no code change. **[wire]** = the
parameter exists but isn't exposed through the CLI. **[code]** = needs a small
change.

---

## A. The world — market structure

The adverse-selection dial lives here. `src/marl_lob/configs/rmsc03_simple.py`
forwards these into upstream `rmsc03.build_config`, and `MarlLobEnv` forwards
anything you give it via `config_kwargs`.

| knob | default | what it tests | status |
|---|---|---|---|
| `num_value_agents` | 50 | **informed flow** — the Glosten–Milgrom dial | [free] `--num-value-agents` |
| `num_noise_agents` | 2000 | uninformed volume; denominator of informed fraction | [free] `--num-noise-agents` |
| `num_momentum_agents` | 10 | trend-followers — semi-informed, impact decays | [free] `--num-momentum-agents` |
| `execution_agents` | `False` | large directional orders (needs the POVExecutionAgent patch) | [wire] |
| background Adaptive MM | 1, fixed by rmsc03 | **a competitor we already have and ignore** | [code] |
| `start_time` / `end_time` | 09:30–10:30 | episode length (1h) | [free] |
| upstream `rmsc03` kwargs | — | fundamental-value process, volatility, arrival rates | [wire] |

Informed fraction today is **50 / 2060 ≈ 2.4%**. That's the x-axis of the phase
diagram. Note the noise count is also the volume driver — the module docstring
warns that below ~1000 the L1 barely moves, so sweeping the *ratio* means moving
value agents, not cutting noise.

## B. Our population — the multi-agent axis

| knob | default | what it tests | status |
|---|---|---|---|
| `n_agents` | 2 | **competition between market makers** | [free] `--n-agents` |
| parameter sharing | on (SuperSuit) | whether agents can differentiate at all | [code] |
| per-agent heterogeneity | uniform | asymmetric risk appetite / size limits competing | [code] |
| agent *role* | all market makers | flip a reward sign and you have a learning **informed trader** — predator/prey | [code] |

The last row is the one that's easy to miss: the rig has no concept of "market
maker." A `MarlChild` is a policy with inventory and order placement. Nothing
stops one of them from being an informed directional trader.

## C. Market-making mechanics — the action space

All on `MarlLobEnv.__init__`, all **[free]**.

| knob | default | notes |
|---|---|---|
| `max_size` | 100 | **the lethal variable.** Markout says 10× size scales capture ~9× and adverse selection ~93× |
| `min_size` | 0 | forces quoting. Tried alone — routed around |
| `max_offset_cents` | 50 | how far from mid it can hide. **Pair with `min_size` to close both hatches** |
| `tick_size` | 1 | quote granularity |
| `wakeup_interval` | `"1s"` | **decision frequency — see confound C1 below** |
| `max_inventory` | default (10_000 in `train.py`) | position limit; also the termination trigger |
| `starting_cash` | default | scale of the equity series |

Not currently a knob: the agent cancels all orders and reposts on every wake-up
(`actions.py` docstring calls this out as the cheap choice). Quote persistence is
a real strategic dimension we've hardcoded away.

## D. Reward

`env.py:323` — `reward = (equity − prev_equity) − inventory_penalty * abs(inventory)`

| knob | default | notes |
|---|---|---|
| `inventory_penalty` | `1e-4` | **linear in \|inventory\|, not quadratic — see D1** |
| `termination_penalty` | `-1.0` | penalty on breaching `max_inventory` |
| reward basis | equity delta, cents | the 10⁷-magnitude problem VecNormalize papers over |
| shaping | none | turnover / fill bonus, spread-capture reward, Sharpe-like, drawdown penalty |

Reward shaping is the axis reviewers will ask about ("why didn't you tweak the
reward?"). Worth having one deliberate experiment here even if the spine is
elsewhere.

## E. Observation

| knob | default | notes |
|---|---|---|
| `k` (book levels) | 10 → obs dim 4k+4 = **44** | does depth beyond L1 matter? `describe_obs(k)` names every feature, so feature-group ablation is cheap |
| `norm_obs` | **`False`** | **see E1 — *not* the reward-scale bug; it equalises how much each feature moves** |

## F. Learning

| knob | default | status |
|---|---|---|
| `total_timesteps` | 50_000 | [free] — seed variance may just be undertraining |
| `n_steps` | 512 | [free] |
| `num_vec_envs` | 1 | [free] — each sim spawns 2000+ background agents, so parallelism is expensive |
| `VecNormalize(norm_reward, clip_reward)` | `True`, `10.0` | [free] |
| PPO hyperparams | SB3 defaults (lr 3e-4, γ 0.99, n_epochs 10, net_arch [64,64], ent_coef 0) | [wire] |
| train seed | varies | [free] |
| algorithm | PPO | [code] — SAC/A2C via SB3 |

## G. Evaluation

| knob | default | notes |
|---|---|---|
| eval seeds | `[42]` only | **the biggest hole in current results** — nothing tests generalisation across market draws |
| `max_steps` | 3600 | |
| deterministic policy | yes | stochastic eval would show policy entropy |
| baseline compared against | F at chosen params | F-best vs F-default changes the story |

---

# Layer 2 — the knobs below our wrapper

Everything above is what *our* code exposes. `rmsc03_simple.build_config`
forwards `**kwargs` straight into upstream `rmsc03.build_config`, which takes
**~35 more parameters**. Several are better instruments than anything in layer 1,
because they are the actual physics of adverse selection rather than proxies for
it. All are reachable today via `config_kwargs` — no ABIDES changes needed.

## The informed-flow dials (better than agent counts)

| knob | default | why it matters |
|---|---|---|
| `val_lambda_a` | `7e-11` | **informed-trader arrival rate.** A *continuous* informed-flow dial — a far better x-axis for the phase diagram than integer agent counts |
| `val_kappa` | `1.67e-15` | how fast value agents pull price to their belief — the *strength* of informed pressure |
| `val_vol` | `1e-8` | noise in their private signal — effectively how informed the informed are |
| `fund_vol` | `1e-3` | volatility of the fundamental — how much there *is* to be informed about |

## The stress-test dials

| knob | default | why it matters |
|---|---|---|
| `fund_megashock_lambda_a` | `2.77778e-18` | jump arrival rate — **a built-in flash-crash generator** |
| `fund_megashock_mean` | `1000` | jump size |
| `fund_megashock_var` | `50_000` | jump dispersion |
| `execution_pov` | `0.1` | size of large directional orders (with `execution_agents=True`) |

Does a learned maker withdraw during a shock, the way real liquidity evaporates?
That is a one-parameter experiment, not a research programme.

## The competitor is fully parameterised

rmsc03 already runs an **adaptive market maker** alongside our agents, and it has
nine knobs: `mm_pov`, `mm_window_size`, `mm_min_order_size`, `mm_num_ticks`,
`mm_wake_up_freq`, `mm_skew_beta`, `mm_level_spacing`, `mm_spread_alpha`,
`mm_backstop_quantity`.

This means **the competition axis does not require more learning agents.** Tuning
the background maker from passive to aggressive sweeps competitive pressure
directly, at eval-only cost for the baseline arm.

## Latency — the edge we have never given the agent

`abides-core/abides_core/latency_model.py`, wired in by rmsc03 via
`generate_latency_model(agent_count)` with `default_computation_delay = 50ns`.

Speed is *the* canonical market-maker edge in the real world, and it is sitting
here unused. This is the concrete mechanism for the "give it an edge and see if
it trades" counterfactual: if withdrawal is rational *without* an edge, then a
latency advantage should produce genuine market making. That closes the argument
in a way no amount of reward shaping can.

## Also available

`end_time` (default 16:00 — we simulate 1h of a 6.5h day), `book_log_depth`
(10, matches our `k`), `ticker` / `historical_date`, `book_logging`,
`log_orders`, `fund_r_bar` / `fund_kappa` / `fund_sigma_s`.

## What is still genuinely missing

Not parameters — these need code:
- **Order-flow imbalance in the observation.** The single most-cited predictive
  feature in microstructure, and our 44-dim observation does not include it.
- **Order types beyond passive limits** — market orders, IOC, cancels as actions.
- **Own-order state** — the agent cannot see its own queue position.

---

# Confounds and discrepancies found while cataloguing

These are not knobs, they are things to fix or disclose.

**C1 — the PPO-vs-F comparison is confounded on decision frequency.**
`MarlLobEnv` sets `wakeup_interval="1s"`; the F baseline uses
`wake_up_freq=10s`. PPO gets **10× more decisions per episode** (3600 vs 360) —
which is also why PPO eval trajectories have 3601 snapshots and F's have 360.
Any "PPO vs F" claim needs either matched frequency or an explicit caveat.

**D1 — the reward penalty is linear, the design said quadratic.**
The Phase-1 plan specified "PnL increment minus quadratic inventory penalty
(Avellaneda–Stoikov flavoured)". The implementation is `abs(inventory)`. Either
a deliberate simplification that was never written down, or a slip. It matters:
linear penalties don't discourage large positions the way quadratic ones do.

**E1 — ~~observations are unnormalised raw cents~~ features are bounded but
not equally visible.** *(corrected 2026-08-30 — the original claim was wrong.)*

`extract_obs` already returns prices as distance from mid, sizes over
`max_size`, and clips the whole vector to [-1, 1] (`observation_extractor.py`,
final line). There are no raw cents in the observation, so this is **not** the
reward-scale bug on the input side.

The real asymmetry is *within* [-1, 1].

*(re-measured 2026-08-31 — the original figures came from a smoke rollout in
which the policy made **zero fills**. On a rollout where the agent actually
trades the gap is wider, and no feature is dead.)*

| feature | std, smoke (0 fills) | std, real (867 fills) |
|---|---|---|
| `ask_size_L1_norm` | 0.332 | 0.337 |
| `time_to_close_norm` | 0.289 | 0.289 |
| `ask_price_L1_norm` | 0.017 | 0.017 |
| `inventory_norm` | **0.000** | 0.0020 |
| `spread_norm` | 0.00026 | **0.00010** |

max/min-nonzero: **1,297× on the smoke rollout, 3,341× here**, and 7,910× on a
`both_hatches` run with 1,745 fills. Two caveats on the original measurement:
`inventory_norm` and `cash_norm` read std 0.000 *only* because that policy never
traded — in a trading run all 44 dims move. And the gap scales with how much the
agent trades, so the more it behaves like a market maker, the blinder it gets to
the spread. First-layer weights all initialise at the same scale, so a
feature's influence on the network tracks how much it moves — the policy is
close to blind to the spread, the quantity a market maker most needs. `--norm-obs`
(VecNormalize per-dim running mean/std) equalises them; the risk is that
near-constant features have their jitter amplified to full scale, i.e. noise.

**Prediction, before running:** no material difference — any effect should sit
inside the seed variance we already know is large, making the honest reading at
3 seeds "this experiment can't tell us".

**Outcome (2026-08-31, corrected evals).** Prediction held. norm_obs mean
−1069 vs norm_obs_off −686; effect −383 against pooled seed sd 601, i.e.
**0.64× the noise**. Note these numbers required `fc0e970`: the first run
evaluated norm_obs policies on raw observations, and those numbers were
meaningless. The control is clean — norm_obs_off re-evaluated bit-identical.
Also measured: `--norm-obs` sharply *increases* seed variance (sd ~985 vs ~80),
which is the "amplify near-constant features into noise" risk in this section
showing up in data.

**E2 — ~~`MarlChild` doesn't log fills~~ fixed 2026-08-30.**
The diagnosis was slightly off: fills always reached the `Trajectory` (via the
`traj_row` contract, and `eval.py` even printed the count) — `np.savez` just
dropped them. `eval.py` now writes fills, actions and observations through the
shared `save_trajectory`, so `markout.py` runs on PPO trajectories. Training
also writes `progress.csv` and `vecnormalize.pkl`, and `collect.py` archives
raw artifacts into `results/`, which is the only directory `teardown.sh` keeps.

**E3 — SuperSuit's `dones` are `uint8`, so `VecNormalize` corrupted every
reward.** *(found 2026-08-30 on the runner machine; fixed in `808a6c0`.)*

SB3 runs `self.returns[dones] = 0` (`vec_normalize.py:206`). On an integer
dtype that is fancy indexing, not boolean masking. At `n_agents=1` it raises
`IndexError` and the run dies; at `n_agents>1` it silently zeroes `returns[0]`
on every non-terminal step. With `norm_reward=True` the corrupted `returns`
feed `ret_rms`, which divides every reward — and the corrupted fraction is
`1/n_agents`, so the distortion tracked the grid's own independent variable.
Measured: the grid went 24/36 → 36/36; `v50_n2/seed0` Δequity moved +1179 →
−595; the same 24 cells cost 414 → 128 train-minutes.

**E4 — the grid's training budget was confounded with `n_agents`.**
*(found 2026-08-30; fixed by scaling `timesteps` in `matrix.py`.)*

SB3 counts `total_timesteps` as the sum across the vec env and SuperSuit sets
`num_envs = n_agents`, so a flat 50_000 gave 98 / 49 / 25 rollout iterations —
50,176 / 25,088 / 12,800 env steps per agent — at `n_agents` 1 / 2 / 4. Agents
in the competitive cells were four times less trained than the solo baseline,
along the exact axis the grid varies. Prediction 2 ("more competitors → less
profit each") could not have been separated from "less training".

**E4b — the fix landed in `suite_grid` only; `suite_cheap` still runs at
25k/agent.** *(noticed 2026-09-04, macmini; not yet acted on.)*

`job()` defaults to `timesteps=50_000` and `suite_cheap` takes that default at
`n_agents=2`, so every cheap cell trained on 25,088 env steps per agent while
every grid cell trained on ~50,176. This is **not** a confound inside the cheap
suite — `n_agents` is constant at 2 there, so `norm_obs` vs `norm_obs_off` vs
`both_hatches` vs `invpen_default` are all budget-matched to each other, and
the E1 outcome above stands. What it does mean is that **cheap numbers must not
be compared against grid numbers**, and that the May-2026 runs (also flat
50_000 at `n_agents=2`, so also 25k/agent) happen to be budget-matched to cheap
but not to the grid.

Consequence for the retrain below: the two cells that duplicate cheap
(`exp1` == `norm_obs_off`, `vecnorm_only` == `invpen_default`) are re-run
rather than borrowed, so the whole act-3/4 ablation sits at one budget.

---

# Candidate experiments

Ordered by value per unit effort. Costs assume 3 seeds unless noted.

### Free — no new training
- **Markout on the F sweep.** Done: `scripts/markout.py`.
- **Feature-group ablation planning** via `describe_obs(k)`.

### Cheap — hours
1. ~~**Fix E2, re-eval existing checkpoints.**~~ Done 2026-08-30 — PPO markout
   is unlocked. **Correction 2026-08-31: the "without retraining" half of this
   was wrong and is withdrawn.** E3 (below) shows every checkpoint trained with
   `norm_reward=True` — the default — learned against a corrupted reward scale.
   Re-evaluating them faithfully reproduces a policy trained on a bug. The May
   PPO checkpoints need **retraining**, not re-evaluation. Only re-*evaluation*
   is sufficient for the `norm_obs` arm, whose training was correct and whose
   `vecnormalize.pkl` survives.

   **The retrain is specified as `experiments/run.sh retrain`** (added
   2026-09-04, `suite_retrain` in `matrix.py`): six cells x 3 seeds = 18 runs
   at 50k env steps per agent, matching the grid.

   | cell | flags | what act 3-4 needs it for |
   |---|---|---|
   | `noop` | `--no-vecnormalize --inventory-penalty 1e-4` | the refusal: 0 fills, $0 |
   | `invpen0_only` | `--no-vecnormalize --inventory-penalty 0.0` | penalty removal alone is a noop |
   | `vecnorm_only` | `--inventory-penalty 1e-4` | May's best PPO cell (+$117 / +$113) |
   | `exp1` | `--inventory-penalty 0.0` | fix the scale, it trades and loses |
   | `min_size_10` | `--min-size 10` | routed around by quoting ~50c off mid |
   | `both_hatches` | `--min-size 10 --max-offset-cents 5` | the hatch-closing close |

   `noop` and `invpen0_only` never wrapped `VecNormalize`, so E3 could not have
   reached them and their May numbers are probably still sound. They are in the
   suite anyway: the 2x2 only reads as a 2x2 if all four cells come off the same
   code. `both_hatches` is re-run for the same reason - the
   `min_size_10` -> `both_hatches` contrast is void if the two cells differ in
   training budget as well as in constraints.

   Cost: ~15-18 min/run on macmini (4-core, 5.5 core-hours total, and it shares
   the box with the media stack), far less on the desktop. **Deferred to the
   desktop by Pavel, 2026-09-04** rather than run here.
2. **`norm_obs=True`** (E1, restated above). One flag, 3 seeds ≈ 35 min.
3. **Close both hatches** — `min_size=10` **and** `max_offset_cents` clamped.
   3 seeds ≈ 35 min. Three possible outcomes, all publishable: it makes markets;
   it finds a third hatch (most likely skewing to one side); or it bleeds, which
   *proves* no edge exists.
4. **Matched decision frequency** (C1) — F at 1s, or learners at 10s. Eval-only
   for F. Makes the headline comparison honest.

### The spine — one overnight run
5. **2D grid: informed fraction × number of market makers.**
   `num_value_agents ∈ {10, 50, 150, 400}` × `n_agents ∈ {1, 2, 4}` = 12 cells,
   3 seeds = 36 runs ≈ **7 hours**, i.e. one night.
   Dependent variables: chosen half-spread, fill rate, capture vs adverse
   selection (now measurable), Δequity.
   Pre-registered predictions — commit to these *before* running:
   - more informed flow → wider quotes, fewer fills, eventually withdrawal
   - more competitors → tighter quotes, less profit each
   - **interaction:** competition removes the cushion that lets a maker survive
     adverse selection, so the withdrawal threshold arrives at a *lower* informed
     fraction as `n_agents` rises
   Requires: wiring `--n-agents` and the population kwargs into `train.py`
   (~20 lines) and a smoke test at `n_agents != 2`, which nothing currently covers.

   **Outcome (2026-09-04) — see `notes/grid_results.md` for the full writeup.**
   Run at **65 seeds/cell**, not 3: 780 runs, zero failures, on code with E3
   and E4 fixed. **All three predictions unsupported**, and the reason is
   common to all of them: *the agent never made markets*.

   - Δequity contrast n=4 − n=1 = **−88, SE 238, t = −0.37** (pooled 65 seeds:
     +36, 95% CI [−355, +426]). Bounded near zero, not merely unmeasured.
   - Quoting was tested for the first time and is also flat: quoted spread
     0.50 / 0.54 / 0.54¢ and fills 612 / 663 / 659 for n = 1 / 2 / 4, all null.
   - Spread capture is **under 1% of |P&L|** at every cell (~8¢ against ~−900
     of adverse selection). The P&L is inventory mark-to-market, not liquidity
     provision.
   - The policy posts a genuine two-sided market on **~2% of decision steps**.
     Sub-tick offsets round to zero and cross (~60% of steps); sizes average
     0.3 against `max_size=100` and round to zero (~62%).

   So the grid could not test competition: the treatment was never applied.
   This is a *broken* refusal rather than a correct one — not an agent
   declining to quote because adverse selection makes it unprofitable, but a
   policy in a degenerate optimum that never learned to quote. Fixing it means
   removing the degenerate region from the action space (minimum tick offset
   and minimum size, i.e. the `both_hatches` configuration, which is the one
   setting measured so far that produced real quoting) and re-running the grid
   on top of that.

   Sample size caveat worth carrying into any future design: at 3 seeds the
   cells were unresolvable (signal/noise 0.73), and at 15 seeds the contrast
   read **+448, t = 1.31** — a trend in the *opposite* direction to the
   prediction, which reversed sign by 50 seeds. Anything claimed off 3 seeds
   here would have been an artifact.

### Newly visible, worth considering for the spine
6. **`val_lambda_a` as the x-axis** instead of `num_value_agents` — continuous,
   so the withdrawal threshold can be located properly rather than bracketed
   between four integer settings.
7. **Latency advantage** (layer 2) — the cleanest form of the "give it an edge"
   counterfactual, and the one a microstructure audience will expect.
8. **Background-MM aggression sweep** — competition without extra learning
   agents, at eval-only cost for the baseline arm.

### Deferred — name them on a future-work slide, don't run them
Algorithm comparison (SAC/A2C) · self-play leagues · full observation ablation ·
shock injection / flash-crash robustness · training beyond 50k · quote
persistence · communication or hierarchical MARL · predator/prey (learning
informed trader vs learning maker) · heterogeneous agent populations.

The point of this list is that it is *written down*. The failure mode now is
running six of these badly instead of two well.
