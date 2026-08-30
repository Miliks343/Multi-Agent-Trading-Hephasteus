# PettingZoo env wrapper — interface sketch

This is the contract between ABIDES and SB3. Each learning agent sees the LOB through this interface and acts through it.

**Paper sketch only.** Exact dimensions get pinned in Phase 1, once Lollo's RMSC03 notebook shows us what ABIDES actually outputs. Specifically, before we commit to obs/action shapes we need to see:

- **Price levels exposed** — does ABIDES surface 5 levels each side? 10? 20? Sets `K`.
- **Units** — prices in ticks or cents; sizes in shares or lots. Affects scaling and the action discretization buckets.
- **Available fields** — order flow imbalance, recent trades, last-trade price, queue position — which of these come for free vs. need computing on top.
- **Update frequency** — events per simulated second. Tells us whether 1-sim-second tick rate is reasonable, too fast, or too slow.

Everything below is a starting point that survives those four answers being mildly different from what we guess.

## Who's in the env

- **N learning market makers** — these are the PettingZoo agents. Start with N=2. Scale later.
- **ABIDES background agents** (noise, value, momentum, exchange agent, the RMSC03 zoo) — *not* exposed to PettingZoo. They live underneath the wrapper as part of the environment dynamics. Our agents trade against them.

Initial training plan: all N learning agents share one policy network (parameter sharing). Different inventories and observations give different actions, even with shared weights.

## Observation space (per agent, per step)

Type: `Box` (continuous), normalized — SuperSuit's `normalize_obs_v0` handles scaling.

Components:

- **LOB snapshot** — top K price levels each side, `(price, size)` pairs. Start K=10 → 40 features.
- **Mid-price and spread** — 2 features.
- **Recent returns / volatility window** — short rolling stats. Exact features TBD.
- **Own state** — inventory, cash, unrealized P&L, time since last fill.
- **Time-of-day** — markets aren't stationary across the trading day. One scalar (normalized seconds since open).

Open: do we feed prices in absolute units or as offsets from mid? Offsets are probably better (translation-invariant), but we'd want to confirm against the paper Lollo dug up.

## Action space (per agent, per step)

Initial cut: a parameterized quote action. Submitting an action **cancels the agent's resting orders and posts a new quote**.

4 dimensions:

- `bid_offset_ticks` — how far below mid to bid (e.g., 0–9 ticks)
- `ask_offset_ticks` — how far above mid to ask
- `bid_size` — discretized: `{0, 1, 2, 5, 10}`
- `ask_size` — same set

Type: `MultiDiscrete`. Switch to `Box` (continuous) if discretization hurts later.

Convention: **size = 0 means "don't quote on that side."** That's our cancel-only signal. No separate "do nothing" action needed.

## Reward (per agent, per step)

```
r_t  =  ΔPnL_t  −  λ · |inventory_t|  −  fees_t
```

- `ΔPnL_t` — change in mark-to-mid total value: `(cash + inventory · mid)`. Computed every step, not just on fills.
- `λ · |inventory_t|` — inventory penalty. `λ` is small, e.g. 1e-4 per share per step. Tunes risk aversion.
- `fees_t` — nominal cost per fill. Stops the agent churning for free.

No terminal bonus — we want continuous incentive, not a sparse end-of-episode signal.

Open: does the inventory penalty hurt early learning? Possible Phase 2 ablation: PnL-only vs PnL-with-penalty.

## reset / step / done

- **`reset()`** — ABIDES re-seeds and starts a fresh trading window. Inventories zero, cash = baseline. Returns initial obs dict for all agents.
- **`step(actions)`** — wraps one wrapper-tick of ABIDES (which is many ABIDES events; see cadence below). Returns `(obs, rewards, terminations, truncations, infos)`.
- **`done`** — episode ends when:
  - the trading window closes (truncation), or
  - an agent's `|inventory|` exceeds a hard cap → terminate that agent with a big negative penalty (termination).

## Cadence

ABIDES is event-driven and fires at microsecond resolution. Our agents don't need that.

- **Wrapper tick = 1 simulated second.** Agents observe and act once per sim-second.
- Between our ticks, ABIDES background agents keep firing on their own schedules — order book evolves naturally.
- Our resting orders sit on the book between ticks and can get filled by background flow.

This is exactly what PettingZoo's `parallel_env` API expects: all agents observe and act on the same clock.

Open: is 1 sim-second the right rate? Faster (100ms) gives more decisions but more noise. Slower (10s) might miss opportunities. Phase 1 calibration.

## Open questions for Phase 1

These don't block the wrapper skeleton — they get pinned once we can run a training loop and see what behaves.

- **Tick rate** — 1 sim-sec is a guess.
- **K** — 10 LOB levels: enough? too many?
- **Feature set** — LOB snapshot only, or also recent trade flow / order flow imbalance?
- **Reward shaping** — λ value, fee level, whether to add a spread-capture bonus.
- **Self-play scheme** — all agents share one live policy, vs. frozen-opponent rotation, vs. a small population. Start simplest (shared live policy); revisit if instability shows up.

## What this means for Phase 1

Three concrete deliverables fall out of this sketch:

1. **`MarlLobEnv`** — the PettingZoo `parallel_env` class implementing the spaces above against ABIDES.
2. **A constant-spread baseline agent** — non-learning market maker that always quotes at fixed offsets. Sanity check: our env runs end-to-end and produces sensible numbers.
3. **A metrics harness** — Sharpe, max drawdown, inventory distribution, per-episode P&L. Reusable across the baseline and the trained agents.
