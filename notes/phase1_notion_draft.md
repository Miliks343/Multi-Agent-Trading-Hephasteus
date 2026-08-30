# MARL LoB Trading

## ── Phase 1 — Build ──

Goal: get a baseline + metrics running end-to-end through real ABIDES by May 10. First PPO training run is a stretch target, not a commitment.

Next checkpoint: **Wed May 6, 7pm online.**

## Pavel — env glue + training pipeline + reward

- **C: PettingZoo `parallel_env` class** wiring A (observations) + B (actions) into a multi-agent env with `reset` / `step` / `done`. This is the integration piece — depends on A and B landing.
- **E: training pipeline** — SB3 + SuperSuit + independent PPO with parameter sharing. Develop against a PettingZoo toy env in week 1 so it's ready when C lands; swap to `MarlLobEnv` once C is in.
- **Reward design** — initial form: realised P&L per step minus inventory penalty (λ · |inventory|). λ stays a config knob.
- Done = `MarlLobEnv` passes a smoke episode end-to-end (random policy, no NaNs); SB3+SuperSuit script trains on toy env without crashing.

## Lollo — observation extractor + baseline agent

- **A: observation extractor** — ABIDES book → numpy vector. Lives ABIDES-side as a helper. First task: figure out what RMSC03 actually exposes (price levels, units, update frequency) and write that up so we can pin the obs vector shape.
- **F: constant-spread baseline** — native ABIDES `TradingAgent` subclass that quotes a fixed-width spread around mid. Runs alongside RMSC03 background agents. **Does not depend on the PettingZoo wrapper** — you can build and run F entirely inside ABIDES, which is why this is your week-1 unblocked work.
- Done = A produces a documented obs vector from a live RMSC03 run; F runs in an RMSC03 sim and shows up in the trade log.

## Neil — action translator + metrics + repo upkeep

- **B: action translator** — action vector → ABIDES `Order` / `Cancel` messages. ABIDES-side helper, mirror of A.
- **D: metrics harness** — pure functions over trajectories computing Sharpe, max drawdown, inventory distribution, PnL. Pure functions = testable with synthetic trajectory data, no ABIDES dependency. Develop in parallel with everyone else.
- **Repo upkeep** — CI, dep bumps, merging.
- Done = B round-trips an action vector to a placed order in an RMSC03 sim; D produces a metrics report from a synthetic trajectory and from a real baseline run.

## Handoffs — interface contracts

These are the shapes that cross people. Pin them early; renegotiate explicitly if they change.

- **Trajectory tuple (C → D):** list of `(timestamp, inventory, cash, mid_price, fill)` per agent per step. D's input format. Lets D be tested with synthetic data before C lands.
- **Observation vector (A → C):** shape pinned after Lollo's RMSC03 inspection. Until then C uses a placeholder vector matching whatever Lollo documents.
- **Action vector (C → B):** small discrete or low-dim continuous — exact shape decided jointly by Pavel + Neil once B's ABIDES-side surface is clear. Default proposal: `(side, price_offset, size)` per agent per step.

## ── Timeline ──

- ~~Fri May 1 — Phase 0 done, meeting to plan Phase 1~~ ✓
- **Wed May 6, 7pm online** — Phase 1 checkpoint meeting
- **Sun May 10** — pre-exam event: floor = baseline + metrics running through real ABIDES; ceiling = first PPO training run kicked off
- May 15+ — exams, project paused
- June onward — Phase 2, 3 (training scale-up → experiments)

[Meeting Notes](https://www.notion.so/Meeting-Notes-348031bc99a08020a7cedf454eb14afa?pvs=21)

## ── Phase 0 — Bootstrap (done) ──

### Neil — GitHub repo ✓ (repo created; skeleton TBD — see note below)

- Create the repo, add Pavel + Lollo as collaborators
- pyproject.toml with Python 3.11; deps: numpy, pandas, matplotlib, pettingzoo, stable-baselines3, supersuit. Skip ABIDES for now — Lollo handles install separately.
- src/marl_lob/ with empty __init__.py, notebooks/, tests/test_smoke.py with one trivial passing test
- .gitignore (Python template), README.md (1-line description + install + how to run pytest), LICENSE (MIT)
- Done = clone on a fresh machine, run pip install -e . and pytest, both succeed

### Lollo — ABIDES import

- Use abides-jpmc-public (https://github.com/jpmorganchase/abides-jpmc-public). Heads up: archived but still works; if the install fights you, check the repo's issues tab — most quirks are documented.
- Install ABIDES locally and run the RMSC03 config (their reference market simulation — pre-built scenario with noise, value, momentum agents and a market maker; produces a realistic-looking order book out of the box)
- Create notebooks/00_abides_hello.ipynb that runs RMSC03 for ~1 simulated hour and plots (a) mid-price over time, (b) LOB snapshot at one moment (top 10 levels each side)
- Add an 'Installing ABIDES' section to the README documenting any quirks
- Done = pull the repo, follow the README, run the notebook end-to-end without errors

### Pavel — design + onboarding doc ✓

- Write a 'what we're building' page in Notion explaining the stack (ABIDES → PettingZoo → SuperSuit → SB3) so everyone has the same picture before Phase 1
- Sketch the PettingZoo env wrapper interface: observations (LOB features), actions (place/cancel orders), rewards (P&L + inventory penalty). Paper sketch, no code.
- Done = both docs in Notion before the May 1 meeting

[What we're building — the stack](https://www.notion.so/What-we-re-building-the-stack-353031bc99a08080ba64c0b883bb3117?pvs=21)

[**PettingZoo env wrapper — interface sketch**](https://www.notion.so/PettingZoo-env-wrapper-interface-sketch-353031bc99a080d5aabedc7ec082b62f?pvs=21)

## Research Tasks (Done)

- Task 1: Find the Market Simulator → ✓ Decided: ABIDES (jpmc fork; archived but still the one with gym wrappers)
- Task 2: Market Data → ✓ Decided: simulated via ABIDES; optional historical crypto data for validation later
- Task 3: Literature Review → ✓ Done
- Task 4: Metrics & Baselines → ✓ Decided: Sharpe ratio + max drawdown / inventory risk; baseline = naive market-maker (constant spread quoter)
- Task 5: MARL Framework → ✓ Decided: PettingZoo + Stable-Baselines3 + SuperSuit; RLlib as fallback
