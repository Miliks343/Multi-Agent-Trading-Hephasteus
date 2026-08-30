# Phase 1 plan

What we build between **May 2** and the **May 10 pre-exam event**, then resume after exams.

## Deliverables

Three concrete artifacts, plus a stretch fourth:

1. **`MarlLobEnv`** — PettingZoo `parallel_env` wrapping ABIDES. Critical path; nothing else trains without it.
2. **Constant-spread baseline market maker** — non-learning agent. Doubles as wrapper sanity check and as the benchmark our trained agents have to beat.
3. **Metrics harness** — Sharpe, max drawdown, inventory distribution, per-episode P&L. Used for both baseline and trained agents.
4. **First PPO training run** *(stretch)* — independent PPO via SuperSuit + SB3. Doesn't need to *win*, just needs to *run*.

## The wrapper isn't one task — it's three sub-modules

| Sub-module | What it is | Depends on |
|---|---|---|
| **A. Observation extractor** | ABIDES book state → feature vector. Pure ABIDES-side helper. | ABIDES output format |
| **B. Action translator** | Action vector → ABIDES `Order` messages. Pure ABIDES-side helper. | ABIDES order API |
| **C. PettingZoo glue** | `ParallelEnv` class wiring A + B with reset/step/done. | A + B |

A and B are independent — different people can write them in parallel. C is the integration layer; small once A and B exist.

### Interfaces

These are starting-point signatures — exact shapes finalize once Lollo's RMSC03 notebook tells us what ABIDES emits.

**A. Observation extractor** — pure function, ABIDES-side.

```python
def extract_obs(book_state, agent_state, mid_price) -> np.ndarray:
    # input:  ABIDES order book (top K levels each side),
    #         agent's inventory/cash, current mid-price
    # output: 1D numpy array — observation vector for one agent
```

**B. Action translator** — pure function, ABIDES-side.

```python
def translate_action(action, agent_id, mid_price, resting_orders) -> list[OrderMessage]:
    # input:  4-tuple (bid_offset, ask_offset, bid_size, ask_size),
    #         agent's currently-resting orders
    # output: list of ABIDES messages — cancels for old orders +
    #         new limit orders for the new quote
```

**C. PettingZoo glue** — the actual env class. Calls A and B; doesn't reach into ABIDES internals beyond "advance simulation by Δt" and "submit messages."

```python
class MarlLobEnv(pettingzoo.ParallelEnv):
    def reset(self, seed=None) -> dict[str, np.ndarray]:
        # spin up ABIDES, return {agent_id: extract_obs(...)} dict

    def step(self, actions: dict[str, np.ndarray]):
        # for each agent: translate_action(...) → submit to ABIDES
        # advance ABIDES by Δt (1 sim-second)
        # for each agent: extract_obs(...), compute reward, check done
        # return (obs_dict, reward_dict, term_dict, trunc_dict, info_dict)
```

**D. Metrics harness** — pure functions, no ABIDES or PettingZoo dependency.

```python
def compute_sharpe(trajectory) -> float
def compute_max_drawdown(trajectory) -> float
def compute_inventory_distribution(trajectory) -> np.ndarray
def compute_pnl_curve(trajectory) -> np.ndarray
# trajectory = list of (timestamp, inventory, cash, mid_price, fill) tuples
```

**E. Training pipeline** — adapts PettingZoo's SB3 tutorial.

```python
env = MarlLobEnv()                              # or toy env in week 1
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=4, base_class="stable_baselines3")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=1_000_000)
model.save("checkpoints/ppo_marl_lob")
```

**F. Constant-spread baseline** — native ABIDES agent.

```python
class ConstantSpreadMarketMaker(TradingAgent):
    def __init__(self, spread_ticks: int, quote_size: int, ...):
        ...
    def wakeup(self, current_time):
        # cancel resting orders
        # post bid at mid - spread_ticks for quote_size
        # post ask at mid + spread_ticks for quote_size
```

Plus three streams that don't touch the wrapper at all:

| Stream | Why it parallelizes |
|---|---|
| **D. Metrics harness** | Pure function: trajectory → metrics dict. Test with synthetic trajectories. Zero ABIDES or PettingZoo dependency. |
| **E. Training pipeline** | Adapt PettingZoo's SB3 tutorial. Develop against PettingZoo's toy env (no wrapper needed); swap the import once C lands. |
| **F. Constant-spread baseline** | Built as a *native ABIDES agent*, not a PettingZoo agent. Drops into the RMSC03 config alongside noise/value/momentum agents. |

## Assignment

| Person | Tasks |
|---|---|
| **Pavel** | C (env wrapper) + E (training pipeline) + own the reward design |
| **Lollo** | A (obs extractor) + F (baseline) |
| **Neil** | B (action translator) + D (metrics harness) + repo upkeep |

**Why this split:**
- **Pavel: C + E.** They sit on the same side of the stack (PettingZoo + SB3). One owner means no coordination overhead at the env/trainer seam — that's where MARL bugs live. E gives Pavel something to build in week 1 against PettingZoo's toy env while A and B land; integration happens in week 2.
- **Lollo: A + F.** Both extend his Phase 0 work; both are ABIDES-flavored. He'll be the only person with ABIDES running on his machine in week 1.
- **Neil: B + D + repo.** B is small (action vector → ABIDES messages). D is real work but parallelizable from day 1 with synthetic data. Plus GitHub stewardship as the codebase grows.

## Sequencing

**Week 1 (May 2–8) — parallel sprints, no one blocked:**
- A, B (sub-modules of wrapper) — Lollo and Neil
- D (metrics, on synthetic data) — Neil
- E (training pipeline against toy env) — Pavel
- F (baseline as ABIDES agent) — Lollo

**Week 2 (May 8–10) — integration:**
- C (PettingZoo glue) wires A and B together — Pavel
- Baseline trajectory feeds into metrics harness → first real Sharpe/drawdown numbers
- *Stretch:* swap the toy env in E for `MarlLobEnv`, kick off first PPO run

## May 10 demo target

**Floor (commitment):** ABIDES running, baseline market maker producing a P&L curve, real metrics computed on it. "Here's the benchmark we're going to beat."

**Ceiling (stretch):** First PPO training run kicked off, with logged training curves to show.

Honest line for the event: **first training run is a stretch, not a commitment.** Better to demo a clean baseline than a broken PPO.

## After May 10

- **May 15+** — exams, project paused for ~3 weeks.
- **June onward** — Phase 2: feature engineering, hyperparameter tuning, self-play stability work, evaluation experiments.

## Risks

- **ABIDES install quirks** could eat into Lollo's week 1. If A is blocked on install issues, F (baseline) is still a clean fallback for him to land.
- **SuperSuit + ParallelEnv compatibility.** The `pettingzoo_env_to_vec_env_v1` wrapper has known sharp edges. Pavel building E against the toy env in week 1 catches this early.
- **Reward shaping.** The λ inventory penalty might suppress learning early. If it does, fall back to PnL-only and reintroduce later.

## Open questions for the meeting

- Does Lollo have ABIDES running cleanly enough by May 1 to start A and F next week, or does he need more install runway?
- Does Neil agree B + D + repo is the right load for him, given he was originally also marked for E?
- Is the May 10 demo line — "baseline running, PPO is a stretch" — what we want to commit to publicly to the association?
