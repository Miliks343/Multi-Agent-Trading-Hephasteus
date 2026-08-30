# What we're building — the stack

This is so when someone asks "wait, why are we using SuperSuit?" mid-Phase-1, there's a single answer to point at. The framework decision was made in meeting 2 (see Task 5 pitch); this page just makes the picture explicit.

We're plugging four pieces together:

```
ABIDES  →  PettingZoo wrapper  →  SuperSuit  →  Stable-Baselines3
(market sim)   (multi-agent API)    (adapter)        (PPO trainer)
```

Each piece does one job and hands off to the next.

---

## ABIDES — the simulator

A discrete-event market simulator from JPMorgan (we use the `abides-jpmc-public` fork). It already does the things we don't want to write:

- runs an exchange with a real matching engine
- spawns background traders (noise, value, momentum, market makers) that submit and cancel orders
- emits the order book state and message log at every event

What we get out of it: a realistic-enough LOB that reacts to our orders. What we don't write: the matching engine, the background-agent zoo, the event scheduler.

Reference scenario: **RMSC03** — their canned multi-agent config. Lollo's Phase 0 task is to get this running and plot the mid-price.

## PettingZoo — the multi-agent env API

The de-facto standard interface for multi-agent RL. Same group as Gymnasium (Farama Foundation). Defines five things:

- who the agents are
- what observation each one sees per step
- what action space each one has
- what reward each one gets
- when the episode ends

We write a wrapper that makes ABIDES *look like* a PettingZoo env to anything upstream. That wrapper is the contract — see the env wrapper sketch page for the actual shapes.

Why this layer matters: it's the abstraction boundary. If we swap trainers later (SB3 → RLlib), the env doesn't change. Lock-in is cheap.

## SuperSuit — the adapter

SB3 is natively single-agent. PettingZoo is multi-agent. SuperSuit bridges them — it converts a PettingZoo parallel env into something SB3 can `.learn()` on (parameter-shared parallel envs, observation normalization, frame stacking).

Without it, the two libraries don't speak. It's not optional, it's the glue.

## Stable-Baselines3 — the trainer

PyTorch library with battle-tested implementations of PPO, DQN, SAC, etc. We don't write a training loop — we import PPO, point it at the SuperSuit-wrapped env, call `.learn()`.

Initial algorithm: **independent PPO** with parameter sharing across our learning agents. Standard starting point for competitive multi-agent — every agent runs the same policy network but gets its own observations and rewards.

---

## What a single training step looks like

1. ABIDES advances simulated time by Δt
2. Wrapper extracts LOB features for each learning agent → observation dict
3. SuperSuit reshapes into batched tensors SB3 expects
4. SB3 policy picks actions
5. Wrapper translates each action into ABIDES order messages (place / cancel)
6. ABIDES inserts them, the matching engine fires, time advances again
7. Wrapper computes per-agent rewards (ΔP&L − inventory penalty) and emits them back up

Repeat until the trading window ends.

## Why this stack, not something fancier

The constraint is **onboarding speed**. Most of us are new to RL. We're researching what *strategies* emerge in an LOB, not how PPO works internally. SB3 minimizes the distance between "env wrapper works" and "something is learning" — once the wrapper is done, calling `.learn()` is a one-liner instead of weeks of training-loop debugging.

We're not doing methods research, so we don't need:

- centralized critics (cooperative-MARL feature; we're competitive)
- hand-rolled PyTorch training loops (two-week detour, teaches us nothing about microstructure)
- JAX speed (we're bottlenecked on understanding, not wall-clock)

## Fallback: when we'd switch

If self-play instability hits — agents collapsing into trivial strategies, training oscillating — we'll need **league training** (frozen-policy pools, opponent sampling). SB3 doesn't have those primitives.

The migration is **RLlib**. Cost is low because PettingZoo is the contract: same env, different trainer. The wrapper code we write in Phase 1 doesn't get thrown away.

We don't pre-build for this. We migrate when the symptom shows up.

---

## TL;DR

- ABIDES gives us a market.
- PettingZoo defines how our agents see and act on it.
- SuperSuit makes that compatible with SB3.
- SB3 does the actual learning.
- If we hit a wall, we swap SB3 for RLlib. Nothing else changes.
