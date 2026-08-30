# MARL Library — Speaking Notes (2–3 min)

## Opening (~20s)

The key insight up front: **this isn't one decision, it's three.** Picking "a MARL library" is the wrong framing. We need to pick three things:

1. The **environment API** — how agents talk to the market
2. The **training library** — the actual learning algorithms
3. The **simulator** — the LOB itself (that's [teammate]'s research)

I'll focus on the first two.

---

## The environment API: PettingZoo (~30s)

For the env API, the recommendation is **PettingZoo**. This is the non-negotiable part.

It's the de facto standard for multi-agent RL — maintained by the Farama Foundation, same people behind Gymnasium. Every serious MARL training library speaks PettingZoo. It handles the edge cases that matter for us: agents acting on different cadences, heterogeneous roles, agents entering and leaving the market.

Committing to PettingZoo early is low-regret. Everything downstream plugs into it.

---

## The training library: SB3 + SuperSuit (~60s)

For training, the recommendation is **Stable-Baselines3 with SuperSuit**.

**SB3** is a PyTorch library with clean, tested implementations of PPO, DQN, and the other standard algorithms. You don't write training loops — you import PPO, point it at an environment, call `.learn()`. It has the best documentation in the RL ecosystem.

**SuperSuit** is the adapter that makes SB3 work with PettingZoo, since SB3 is natively single-agent. It's not optional — it's the glue that turns two separate libraries into one pipeline.

**Why this combo:** our main constraint is onboarding. Most of us are new to RL. SB3 maximizes the probability that everyone on the team gets something training in the first couple weeks. We're not doing research on *how PPO works* — we're researching *what strategies emerge in a LOB*. SB3 lets us focus on that.

This is a competitive multi-agent setup with self-play — agents trading against each other, each optimizing their own P&L. For that, **independent PPO** is the right starting algorithm, and that's exactly what SB3 + SuperSuit gives us out of the box.

---

## What we're giving up and when we'd switch (~40s)

To be honest about the tradeoffs: SB3 is fundamentally a single-agent library. It doesn't support fancy MARL techniques like centralized critics, and it doesn't have built-in opponent sampling for league training.

For our setup, we don't need centralized critics — that's a cooperative-MARL thing, we're competitive. But we *might* hit self-play instability, and if we do, we'll want league training.

**The migration path is cheap.** Because we committed to PettingZoo as the env API, switching training libraries later is a wrapper change, not a rewrite. The natural upgrade is **RLlib** — it has the opponent-sampling primitives we'd need.

---

## Closing (~20s)

So the recommendation:
- **PettingZoo** for the env API — commit now, it locks in.
- **SB3 + SuperSuit** for training — start simple, learn the problem.
- **RLlib** as the fallback if we hit a wall.

The vote is really on the training layer, since the env API is the obvious call. Happy to take questions.

---

## Q&A prep — likely questions

**"Why not RLlib from the start?"**
Ray cluster operational overhead + steep learning curve. Onboarding cost outweighs the migration cost later.

**"Why not JAX (Mava/JaxMARL)?"**
Faster training, but requires JAX fluency and a JAX-based env. ABIDES isn't JAX, and our bottleneck right now is understanding the problem, not wall-clock speed.

**"Why not write it ourselves in PyTorch?"**
Two-week detour that teaches us nothing about market microstructure, which is what we're actually researching.

**"What exactly does SuperSuit do?"**
It's the adapter between PettingZoo (multi-agent) and SB3 (single-agent). Also handles preprocessing like frame stacking and observation normalization. Without it, SB3 can't read PettingZoo envs.

**"What about AgileRL?"**
Strong alternative — cleaner MARL algorithms than RLlib, PyTorch-based. Smaller community means more time stuck when we hit bugs. Reasonable Phase 2 option alongside RLlib.
