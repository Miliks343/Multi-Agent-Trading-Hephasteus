# Module C — PettingZoo wrapper design

Internal planning doc. Lives in Pavel's notebook, not the team repo. Final code lands in `src/marl_lob/env.py` + friends in the team repo once this design is signed off.

## 1. End goal — what we want to be running

C is not a deliverable on its own. It exists to enable two runnable artifacts:

### 1a. Baseline loop — May 10 floor
```
python scripts/run_baseline.py
  → spins up RMSC03 + N copies of F (constant-spread MM)
  → runs 1 trading hour
  → builds Trajectory from F's holdings/order log
  → metrics.compute_all → Sharpe / drawdown / pnl_curve printed + plotted
```
**Does not need C.** Needs: a `rmsc03_with_baseline.py` config and a `baseline_traj.py` adapter that turns an ABIDES post-sim log into a `Trajectory`. ~1h of work, separate from C.

### 1b. Training loop — May 10 ceiling
```
python scripts/train.py
  → SB3 PPO + SuperSuit + MarlLobEnv (= C wrapping A+B inside RMSC03)
  → tensorboard learning curve, checkpoint saved

python scripts/eval.py runs/<ckpt>
  → rolls trained policy through the same env
  → SB3 callback accumulates info["traj_row"] → Trajectory
  → metrics.compute_all printed next to F's numbers
```
**This is what C unlocks.** Success criterion: `eval.py` prints two Sharpe numbers and the trained one is higher than F's.

## 2. State of the pieces — what's actually missing to combine them

| Module | Status | Gap |
|---|---|---|
| A obs extractor | merged | none |
| B action translator | merged | `MarlAgent` in `agent_adapter.py` is policy-driven — wrong shape for C; need a sibling `MarlChild` class. Neil's tests stay |
| D metrics + Trajectory | merged | none, but **no producer for the baseline path** — F runs natively in ABIDES and never builds a Trajectory |
| F constant-spread MM | merged | needs `rmsc03_with_baseline.py` config (Lollo's hand-off candidate) and the post-sim → Trajectory adapter above |
| C env wrapper | not started | the work below |
| E training pipeline | toy on `feat/training-pipeline-toy` (PR open) | needs swap from `simple_spread_v3` to `MarlLobEnv` |
| `rmsc03_with_marl.py` config | does not exist | RMSC03 + 1 MarlCoordinator + N MarlChild |
| `scripts/run_baseline.py`, `scripts/train.py`, `scripts/eval.py` | none beyond `train_toy.py` | new |
| SB3 traj-buffering callback for eval | does not exist | Neil's hand-off candidate |

So "starting C" means: **C itself + ~5 small connecting pieces**. C is the load-bearing one.

## 3. The kernel-pause mechanism (research output)

From `abides-core/abides_core/kernel.py:363-381` and `abides-gym/abides_gym/envs/core_environment.py`:

- ABIDES `Kernel.runner()` is a single-threaded event loop.
- When an agent's `wakeup(t)` returns a non-None value, the kernel breaks out of the loop and returns `{"done": False, "result": <wakeup_return>}` to whoever called `runner()`.
- The next call `kernel.runner((exp_agent, action_list))` calls `exp_agent.apply_actions(action_list)` first, then re-enters the loop until the next non-None wakeup or `stop_time`.

**This is the cooperative-pause hook. We don't touch the kernel.** All complexity goes into the agent we register.

**Constraint that drives the design:** only one agent can be the "experimental" agent — only one can return non-None from wakeup, because the kernel pauses on the first. Multi-agent has to be multiplexed through a single coordinator.

## 4. Architecture — coordinator + children

Two new ABIDES agents in `src/marl_lob/marl_coordinator.py`:

### `MarlCoordinator` — the experimental agent
- Subclasses `CoreBackgroundAgent` (the same lineage `FinancialGymAgent` uses, so we get L1 market-data subscription + per-step `inter_wakeup_executed_orders` machinery for free if we want it; but the coordinator itself doesn't trade — it's a state-aggregator, the children trade).
- Holds a list of `MarlChild` references (set at construction by the env config).
- Wakes every 1 sim-second (`ConstantTimeGenerator(step_duration="1s")`). On wakeup:
  1. For each child: read `holdings`, cash, mid (from market-data subscription), drained `inter_wakeup_executed_orders` since last tick.
  2. Call A's observation extractor per child (it expects per-agent inventory/cash/mid + the L1 book).
  3. Return a `raw_state` dict: `{"per_agent": [{"obs": np.ndarray, "inventory": int, "cash": int, "mid": int, "fills_qty": int, "fills_vwap": int}, ...], "timestamp_s": float, "stop_time": int}`.
- This non-None return interrupts the kernel. Env reads it.
- After env computes N actions, env calls `kernel.runner((coordinator, [{"agent_idx": i, "action_vec": (...)}, ...]))`. Coordinator's `apply_actions` writes `child[i].pending_action = action_vec` for each child.

### `MarlChild` — the trading agent
- Subclasses `CoreBackgroundAgent` (so `inter_wakeup_executed_orders` and L1 subscription Just Work). Or `TradingAgent` if `CoreBackgroundAgent` brings unwanted background-agent behavior — open question, verify when implementing.
- Wakes ~10 nanoseconds after the coordinator each tick (latency offset; ensures action propagation order).
- On wakeup: reads `self.pending_action`, runs Neil's `translate_action(...)` + `dispatch_intents(...)` from `agent_adapter.py` — exact same code path Neil already wrote and tested. **Reusing B verbatim.**
- Returns None — does not interrupt the kernel.

### Why this design
- **Single experimental agent** satisfies the kernel constraint.
- **Children own the trading state** (inventory, cash, resting orders) — that's what ABIDES expects; we don't fight the framework.
- **Reuses B unchanged** — `translate_action` doesn't care who calls it.
- **Reuses A unchanged** — coordinator builds the per-child obs view A expects.
- **The `MarlAgent` Neil already wrote stays** — useful for non-PettingZoo deployment (e.g., a single-agent sanity-check sim with a hard-coded policy). Two classes, no churn on Neil's side.

## 5. `MarlLobEnv` — the PettingZoo surface

```python
class MarlLobEnv(pettingzoo.ParallelEnv):
    metadata = {"name": "marl_lob_v0", "is_parallelizable": True}

    def __init__(self, n_agents=2, config_kwargs=None, ...):
        self.possible_agents = [f"mm_{i}" for i in range(n_agents)]
        # observation_space = Box(-1, 1, shape=(4*K + 4,), dtype=float32) — from A's constants
        # action_space      = Box(low=[0, 0, 0, 0], high=[max_off, max_off, max_size, max_size]) — from B
```

- `reset(seed=None)`: build config via `rmsc03_with_marl.build_config(...)`, instantiate `MarlCoordinator` + N `MarlChild`, append to config's agent list, instantiate `Kernel`, call `kernel.runner()` (no args) → returns first raw_state from coordinator → unpack into `obs_dict`. Return `(obs_dict, info_dict)`.
- `step(actions_dict)`:
  1. Convert to coordinator's expected list-of-dicts shape.
  2. `raw_state = self.kernel.runner((self.coordinator, action_list))`.
  3. Unpack per-agent obs.
  4. Compute reward per agent: `ΔPnL_t − λ·|inv_t| − fees_t` where `PnL = cash + inv·mid`. Held in env state across steps.
  5. Build `traj_row` per agent (see §6).
  6. Termination: `|inv| > MAX_INVENTORY` → that agent terminated, big negative reward. Truncation: `raw_state["done"]` (kernel hit stop_time).
  7. Return standard PettingZoo 5-tuple of dicts.
- `close()`: nothing to do.

**Reward state**: env tracks `prev_equity[agent]` between steps. ΔPnL = current_equity − prev_equity. Penalty λ default 1e-4. Fees default 0 (until we wire ABIDES exchange fees, which is post-Phase 1).

**SuperSuit adapter for SB3**: `pettingzoo_env_to_vec_env_v1(supersuit.pad_action_space_v0(supersuit.pad_observation_space_v0(env)))` — same chain as `train_toy.py`, just a different env. Should drop in without further edits to E.

## 6. The `traj_row` contract — extending to 6-tuple ourselves

**New contract** (replaces the old 5-tuple in `trajectory.py:8-12`):
```
(timestamp_s, inventory, cash_cents, mid_cents, fill_signed_qty, fill_price_cents)
```
- `fill_signed_qty`: net signed shares filled this step (+N buy, −N sell, 0 none).
- `fill_price_cents`: VWAP of fills this step, or 0 if no fill.

**Why extend**: today `Trajectory.from_tuples` synthesizes `Fill.price = mid_cents`, which is wrong on any step where the mid moved during the wakeup window. We have the real fill prices for free from ABIDES' `inter_wakeup_executed_orders`. Costs one extra int per row, gives D correct `Fill.price`.

**Doing this ourselves, not asking Neil.** Edit is two lines in `trajectory.py:from_tuples` plus the dataclass docstring. Updating Neil's existing tests is a `0` → `(0, 0)` swap on a few hand-built rows. Belongs in chunk 3 (per-step accounting), since that's the chunk that actually produces fill prices. Tag Neil in the PR description so he sees it; no blocker.

## 7. Sequenced work chunks

Estimates assume Pavel solo, focused hours.

| # | Chunk | Where | Effort |
|---|---|---|---|
| 0 | Sign off this doc | — | now |
| 1 | `MarlChild` + `MarlCoordinator` agents | `src/marl_lob/marl_coordinator.py` | 2–3h |
| 2 | `MarlLobEnv` reset/step skeleton, dummy reward, no termination | `src/marl_lob/env.py` | 3h |
| 3 | Per-step accounting; emit 6-tuple `info["traj_row"]`; extend `trajectory.py` to accept it (incl. updating Neil's tests) | env + coordinator + `trajectory.py` | 1.5h |
| 4 | Reward (ΔPnL − λ|inv|) + termination/truncation | env.py | 1h |
| 5 | `rmsc03_with_marl.py` config | `src/marl_lob/configs/` | 30min |
| 6 | Tests: random-policy smoke + `pettingzoo.test.parallel_api_test` | `tests/test_env.py` | 1h |
| 7 | E swap: change `train_toy.py` to use `MarlLobEnv`; verify PPO learns at all (rough — might just train for 50k steps and check ep_rew not NaN) | `scripts/train.py` | 1h |
| 8 | Baseline path: `baseline_traj.py` + `scripts/run_baseline.py` + `rmsc03_with_baseline.py` config | several | 1.5h |
| 9 | `scripts/eval.py` + SB3 trajectory-buffering callback | scripts/ | 1h |

**Total: ~12–14h.** Coding runway is through ~May 13. Comfortable, with slack for surprises.

**Critical path** for May 10 demo: chunks 1-7 (training run kicked off) and 8 (baseline numbers exist for comparison). Chunk 9 (eval script) is needed for a clean demo but the comparison can be hand-done at the REPL if 9 slips.

## 8. Open questions

- **6-tuple vs 5-tuple `traj_row`** — pending Neil reply.
- **`MarlChild` parent class** — `CoreBackgroundAgent` (gives us fills + L1 sub for free) vs `TradingAgent` (lighter, but we'd reimplement bookkeeping). Verify when starting chunk 1; default to `CoreBackgroundAgent`.
- **Wakeup offset between coordinator and children** — coordinator wakes at T, children wake at T + δ. δ = 10 ns or `agent_computation_delay`? Verify by looking at how `FinancialGymAgent` schedules wakeups. Trivial but easy to get wrong silently.
- **Where does the env get the L1 book for A's obs** — through the coordinator's market-data subscription, or by reading from the `ExchangeAgent` directly? Subscription is the abides-gym pattern; default to that.
- **Reward computation timing** — first step has no `prev_equity`. Either set `prev_equity[step 0] = starting_cash` (clean) or zero out the first reward (dishonest). Go with the former.
- **Should termination on `|inv| > cap` actually end the agent, or just penalize and continue?** PettingZoo supports per-agent termination. Default: terminate (matches phase0 sketch); revisit if it turns out PPO can't recover from agent-dropping mid-episode.

## 9. What I will NOT do in this phase

- No fee model — exchange fees stay 0 until post-Phase 1.
- No spread-capture bonus or fancy reward shaping — vanilla `ΔPnL − λ|inv|` only.
- No self-play opponent rotation — N=2 with parameter sharing, single live policy.
- No K calibration or tick-rate sweep — K=10 and 1-sim-second are fixed defaults from A and phase0_env_wrapper.md.
- No `abides-gym` dependency — we crib the pattern, not the package.
