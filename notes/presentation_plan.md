# Pre-exam event presentation — plan & study notes

**Format:** notebook → exported HTML, full-screen in browser.
**Length:** 10–15 min, target ~12 min talking + ~3 min Q&A.
**Presenters:** Pavel (lead), Neil, Lollo. ~4 min each on average; Pavel takes a longer tail because the PPO/ablation/variance arc is one continuous story.

This file is the **study draft** — fuller prose, talking points, what to emphasise. Once we're aligned at the next meeting we tighten the wording and move it into the notebook. Md cells in the notebook will be sparser than what's written here.

**Major rewrite 2026-05-11** after experiments: the original "PPO beats F by 40%" framing is gone. The new arc is:

1. Built a working multi-agent ABIDES pipeline.
2. First training run collapsed to noop — diagnosed reward-scale issue.
3. Ablation 2×2 cleanly identifies VecNormalize as the load-bearing fix (and overturns our initial attribution).
4. F-default was a strawman; tuned F is profitable. PPO on best seed roughly matches F-best.
5. PPO has large seed variance — this is the next thing to fix.

The new story is harder to tell but more credible. A student team that overturned one of its own claims via ablation and reports honest seed variance reads as serious. Lean into that.

---

## 0. Title slide (10s) — whoever opens

**On screen:**
> # Multi-Agent Reinforcement Learning in Limit Order Books
> Pavel · Neil · Lollo
> Pre-exam progress check, May 2026

**Notes:** quick "hi we're three students working on …" — don't burn time here.

---

## 1. Problem (~90s) — **Lollo**

**Goal:** by the end of this section the audience knows what a limit order book is, what market making is, and why doing it with multiple learning agents is interesting and hard.

**Key points to hit:**
- A **limit order book (LOB)** is the data structure every modern exchange runs on. Two sides — bids and asks — with prices and quantities. Trades happen when a buy and sell cross.
- A **market maker** posts quotes on both sides and earns the spread, but eats inventory risk if the price moves against them.
- Classical market-making strategies are hand-tuned (Avellaneda–Stoikov etc.). RL is a natural fit — the agent learns from interaction.
- The twist: **multiple learning agents**. Real markets have many MMs competing. If you train one agent against scripted opponents, it overfits to a strawman. Multi-agent RL trains them against each other, and that's where it gets interesting and unstable.
- Our research question (informal): can independent PPO with parameter sharing produce a market-making policy that's competitive with a well-tuned constant-spread baseline in a realistic simulator?

**On screen:** one diagram of a LOB (bids/asks/spread), one line of motivation. Don't put the research question on screen — say it.

**Lollo phrasing tips:**
- "Limit order book" said once, then "the book" afterward.
- Avoid jargon for this audience.

---

## 2. Stack & architecture (~60s) — **Lollo**

**Goal:** show we picked tools deliberately and the pieces fit together.

**Key points:**
- **Simulator: ABIDES** (jpmc fork). Agent-based market simulator with realistic background flow — value, momentum, noise traders. We use the RMSC03 config.
- **Multi-agent env API: PettingZoo.** Standard interface; same shape as gym but for N agents.
- **Trainer: Stable-Baselines3 + SuperSuit.** SB3 is single-agent; SuperSuit adapts a PettingZoo parallel env into something SB3 can train on (parameter sharing across agents).
- **Algo: independent PPO** with shared weights.
- **Baseline:** constant-spread market maker, native ABIDES agent — runs alongside the background flow.

**On screen:** module diagram. Boxes for A (obs), B (actions), C (PettingZoo glue), D (metrics), F (baseline), E (training). Arrows showing data flow:
```
ABIDES kernel ──► A (obs) ──► MarlLobEnv ──► SB3 (PPO) ──► B (actions) ──► ABIDES kernel
                                  │
                                  └──► D (metrics) ──► Sharpe / MaxDD / PnL
F (baseline) ──► ABIDES kernel directly (no env wrapper)
```

**Lollo phrasing tips:**
- "We didn't reinvent — we picked the standard pieces and the work was making them talk to each other."
- Keep this short, it sets up the next two sections.

---

## 3. The interfaces (~90s) — **Neil**

**Goal:** show that we agreed on a clean contract between the simulator and the learning code, and that the contract is in code, not on a whiteboard.

**Key points:**
- **Observation (module A, Lollo's work):** `4*K + 4` floats per agent. K = 10 levels of book depth. The four extras are inventory, cash, mid, time-of-day. Cents convention everywhere — no floats-as-prices bugs.
- **Action (module B, Neil's work):** continuous 4-tuple `(bid_offset, ask_offset, bid_size, ask_size)`. Offsets in ticks from mid; sizes in shares. Continuous so PPO has gradients to follow.
- **Trajectory contract (module D, Neil's work):** every step the env emits `info[agent] = {"traj_row": (timestamp_s, inventory, cash_cents, mid_cents, fill_signed_int, fill_price_cents)}`. Metrics consume this — Sharpe, max drawdown, inventory distribution, PnL curve — as pure functions of the trajectory list.
- This decoupling is what made the build parallelisable: A, B, D, F were developed independently against synthetic trajectories.

**On screen:** the `Trajectory` dataclass (~10 lines) and the action 4-tuple type. Code cell, syntax-highlighted. Maybe one line showing `compute_all(traj)` returning a dict.

**Neil phrasing tips:**
- "The interface is the deliverable" — sell this as engineering discipline, not just plumbing.
- Mention dtype enforcement (integer cents) briefly if there's time.

---

## 4. The constant-spread baseline (~45s) — **Neil**

**Goal:** introduce F as our yardstick. Don't show numbers yet — that's §8.

**Key points:**
- Native ABIDES `TradingAgent` subclass — runs in the simulator like any other participant, not through the PettingZoo wrapper.
- Three parameters: spread in ticks, quote size, wakeup frequency.
- It's deliberately dumb — no model of the market. The point: can the learner beat a strategy with no model?
- **Important callback:** we'll come back to F later. Picking the right parameters for F turned out to be a real part of the story.

**On screen:** maybe a 4-line snippet of `LoggingConstantSpreadMM`'s `wakeup` method, or just describe. Lean toward describe.

**Neil phrasing tips:**
- "Naive on purpose. We want a low bar — *but a low bar that's still realistically tuned.*"
- Set up the §8 callback explicitly: "We'll see in a minute that 'reasonable defaults' for F isn't as simple as it sounds."

---

## 5. Integration: the C wrapper (~90s) — **Pavel**

**Goal:** show the one piece that's actually hard — making ABIDES (event-driven, kernel-paced) talk to PettingZoo (step-paced, gym-shaped).

**Key points:**
- ABIDES is a discrete-event simulator. Its kernel runs in its own loop, agents respond to messages.
- PettingZoo expects `step(action) → obs, reward, done, info`. Two completely different control flows.
- Solution: **MarlCoordinator** is a single ABIDES agent that pauses the kernel by returning `None` from its wakeup callback. The PettingZoo env drives the loop: when `step()` is called, it resumes the kernel, the coordinator multiplexes obs from N **MarlChild** trading agents (which hold inventory and place orders via Neil's translator B), then pauses again.
- This means **only one experimental agent talks to the kernel** even though we have N learners. Clean and avoids race conditions.
- Trade-off: tightly coupled to ABIDES' kernel API. Not a general MARL+ABIDES bridge.

**On screen:** a control-flow diagram (more important than code here). Two timelines: ABIDES kernel ticking along, PettingZoo `step()` calls intercepting. The "kernel paused" state shaded.

**Pavel phrasing tips:**
- This is the most "research-engineering" slide of the talk. Pace it slow, the audience needs the picture.
- "The trick is that the kernel can be paused" — that's the headline sentence.

---

## 6. — code section starts here —

From here on, every cell has either a plot or a small code+output pair. Md cells get terse. The HTML export uses `--no-input` for sections 0–5; sections 6 onward keep input visible for 2–3 cells where the code itself is the point.

---

## 7. Live ABIDES sim (~30s) — Pavel narrates

**Goal:** "the simulator works." One slide of evidence.

**On screen:**
- Mid-price + spread plot from a 1h RMSC03 run. Pre-rendered.
- Caption: "1 hour of simulated trading, ~3s wall-clock."

**Talking point:** "This is RMSC03 — the standard ABIDES config — running with our setup. Value, momentum, and noise traders, plus the exchange. Everything from here on runs against this."

---

## 8. Finding a fair F (~90s) — **Neil**

**Goal:** show that the obvious F config (spread=10, size=100) is a strawman, and that tuned F is actually profitable. This sets up the honest comparison in §10.

**On screen — F sweep 2×2 table:**

| spread \ size | 10 | 100 |
|---|---|---|
| **10 ticks** | −$1.0k / −$1.2k, MaxDD 1.9% | **−$16.7k / −$11.7k, MaxDD ~60%** (default) |
| **30 ticks** | **+$590 / −$75, MaxDD 5.4%** ✅ | −$20.1k / −$22.6k, MaxDD ~33% |

Two PnL curves side by side: F-default (bleeding to −$16k) vs F-best (flat-ish, slightly positive).

**Talking points:**
- "We started with the obvious config — spread 10 ticks, quote size 100. It bleeds about $16k per hour. Sharpe deeply negative, max drawdown 60%."
- "Why? Constant spread, fixed size 100, gets adversely selected by ABIDES' value and momentum flow. Whenever price trends, F keeps quoting the same width and gets picked off."
- "We swept the params. Smaller quote size (10 instead of 100) dramatically reduces bleed. Wider spread (30 instead of 10) at small size is actually slightly *profitable* on one agent."
- "**Lesson:** the baseline isn't a constant. Tuning matters. This 2×2 is what an honest yardstick looks like."

**Neil phrasing tips:**
- This is a callback to §4's "we'll come back to F." Land it explicitly.
- The bold cells in the table are the two we'll compare PPO against in §10.

---

## 9. First PPO run + noop (~75s) — **Pavel**

**Goal:** the setup for the diagnosis. Don't deliver the punchline yet.

**On screen:**
- ep_rew_mean training curve (flat near zero).
- A small text block: PPO 50k steps · default config · 0 fills · Δequity $0.

**Talking points:**
- "First training run. PPO 50k steps, default reward (PnL in cents), default config."
- "Result: zero fills. The agent learned to do nothing."
- *(joke beat)* "Honestly, 'don't trade' is the best real-world strategy for a lot of desks. We'd accept that paper as a final result, but unfortunately we wanted ours to actually market-make."
- "Diagnosis: rewards are in cents. PnL per episode is order 10⁷. Value loss blew up to 10⁷–10¹². Approx KL stayed around 10⁻⁵ — meaning the policy barely moved. PPO couldn't learn at that scale, so it converged to the safest possible action: size zero."

**Pavel phrasing tips:**
- Land the joke once, don't milk it.
- The numbers (10⁷, 10⁻⁵) are the credibility move — pronounce them clearly.

---

## 10. The fix — and what we got wrong (~120s) — **Pavel**

**Goal:** the centerpiece of the talk. The ablation 2×2 that overturns our initial attribution. This is the scientific result.

**On screen — ablation 2×2:**

| inv_penalty \ VecNormalize | OFF | ON |
|---|---|---|
| **default (1e-4)** | noop, 0 fills, $0 | **+$117 / +$113, MaxDD 0.66%, 571 fills** ✅ |
| **0.0** | noop, 0 fills, $0 | −$6.7k / −$6.9k, MaxDD 10%, 1186 fills |

**Talking points:**
- "Two candidate fixes. First, **normalize rewards** — wrap the env with VecNormalize so PPO sees rewards in roughly unit scale. Second, **remove the inventory penalty** — maybe the agent was being told to hold zero inventory, which size-zero satisfies trivially."
- "We tried both together. PPO started trading: about 1,200 fills per agent, lost ~$6.7k per agent. Better than baseline default."
- "Then we ablated. *Which fix actually mattered?*"
- Pause on the 2×2.
- "Removing the inventory penalty **alone** does nothing — still noop. Value loss still blows up."
- "Keeping the penalty *and* adding VecNormalize is **better than removing the penalty.** +$115 per agent, max drawdown under 1%, 571 fills. This is our best PPO result."
- "**The lesson is unflattering and we're keeping it on the slide:** we changed two things at once and credited both. The ablation showed only one was load-bearing — and our second 'fix' was actively making things worse. *Change one knob at a time.*"

**Pavel phrasing tips:**
- The "unflattering lesson" line is the most important sentence in the talk. Land it. An audience that hears a student team report a self-correction credibly raises their estimate of everything else in the presentation.
- Keep the 2×2 on screen long enough for the audience to read every cell.

---

## 11. The comparison — and the variance problem (~75s) — **Pavel**

**Goal:** honest side-by-side with the right F config; expose seed variance as the next problem.

**On screen — two things, stacked:**

(a) **Comparison table:**

| Strategy | Δequity / 1h | MaxDD | Fills | Notes |
|---|---|---|---|---|
| Random policy | −$157k | — | many | trades constantly, bleeds |
| F-default (spread=10, size=100) | −$14k | ~60% | ~300 | strawman |
| F-best (spread=30, size=10) | +$258 mean | 5.4% | ~600 | tuned baseline |
| PPO-best (Ablation #4, seed 42) | +$115 mean | 0.7% | 571 | matches F-best on this seed |
| PPO across seeds {0,1,7,42} | range −$6.9k to +$0.3k | 0.2%–10% | 130–1186 | high variance |

(b) **Seed-variance plot:** 4 dots, one per training seed, on Δequity-vs-seed.

**Talking points:**
- "Side by side: PPO-best on its best seed is roughly tied with F-best. PPO isn't beating a well-tuned baseline yet — it's matching it."
- "And **PPO is sensitive to the training seed.** Δequity ranges from −$6.9k to +$0.3k across four seeds with everything else held constant. F is deterministic given the simulator seed; PPO isn't."
- "This is the next thing to fix. The plan: longer training, evaluate each checkpoint on multiple market seeds (not just the training seed), and look at the variance distribution. If PPO is consistently in F's neighborhood with smaller spread, that's a real result. Right now we have a best case, not a robust one."

**Pavel phrasing tips:**
- Don't apologise for the variance. Frame it as "we measured it, now we know what to do about it." Measuring variance honestly is itself a contribution.
- "Best case, not robust case" — short and clear.

---

## 12. What's next (~30s) — **Pavel**

**Goal:** show we know what to do; close the loop.

**On screen — four bullets:**
- Reduce seed variance: longer training, evaluate on multiple market seeds, log per-seed distributions.
- Wider F sweep — include simulator-seed sweep, not just param sweep.
- Self-play with multiple learners; opponent sampling.
- Reward shaping experiments — turnover-aware reward, inventory-aware reward (now that we know inv_penalty=0 is wrong).

**Talking points:**
- "Summer is for completion. Pre-exam target was a working end-to-end pipeline and an honest first result — we have both."
- "The interesting science starts when variance is under control and we put two learners against each other."

---

## 13. Thanks / Q&A (~10s + remainder)

**Anticipated questions:**

- **"Why ABIDES and not a real exchange feed?"** — Real data is replay-only; you can't test counterfactual policies. ABIDES gives us a closed-loop simulator with realistic background flow.
- **"Why PPO?"** — On-policy, stable-ish, well-supported in SB3, common baseline in MARL papers. Not committed to it long-term.
- **"How do you know your sim is realistic?"** — We don't yet, fully. RMSC03 has been validated in the ABIDES literature for stylised facts (return distributions, volatility clustering). We'll do our own sanity checks as we go.
- **"What's the team split?"** — Pavel integration + training, Neil actions + metrics, Lollo obs + baseline + sim setup.
- **"How much of the seed variance is the simulator vs the policy?"** — Honest answer: we don't know yet, eval is on a single market seed. Disentangling that is the next experiment.
- **"Did you re-train F's parameters per simulator seed or are those numbers also single-seed?"** — Single seed (42). F's robustness across simulator seeds is also unverified — same caveat applies.
- **"What's the inventory penalty doing, if removing it hurts?"** — It scales with `|inventory|`. With VecNormalize active, that signal apparently helps the value head learn faster than it would on raw PnL alone. We don't have a clean theoretical story for it yet — empirical observation.

---

## Open questions for the team

Things to settle at the meeting before we tighten this into the notebook:

1. **Joke calibration in §9.** Pavel proposed the "best real-world strategy is don't trade" beat. Team comfortable with it? Probably fine — it sets up the diagnosis well.
2. **The §10 "we got it wrong" framing.** This is the talk's biggest credibility move *if it lands* and an own-goal *if it doesn't.* Comfortable owning it publicly?
3. **Live demo or fully pre-rendered?** Current plan is fully pre-rendered. Alternative: alt-tab to a notebook at the end and run one cell (3-second RMSC03 sim) as a closer. Risk vs payoff.
4. **§8 presenter — Neil or Lollo?** Drafted as Neil. Lollo built F, but Neil is already on stage from §3 and chaining is smoother.
5. **§11 presentation of seed variance.** Dot plot vs bar chart vs just the table. Dot plot reads cleanest at 4 seeds.
6. **Time budget.** ~12 min target; sections sum to ~11.5 min. Slack for ~30s of stumbling. If we need to cut: §4 (-15s by skipping the snippet), §7 (-15s).
7. **What gets cut if we run long.** Pre-decide: drop the "lesson is unflattering" delivery moment to a single sentence, drop the fourth bullet of §12.

---

## Production checklist (post-meeting)

- [ ] Convert this plan to a `.ipynb` in the team repo under `notebooks/presentation.ipynb`.
- [ ] Tighten md cells — aim for ~30–50% of the prose here.
- [ ] Pre-execute all code cells, commit with outputs.
- [ ] Generate plots:
  - ABIDES mid+spread (§7)
  - F-default vs F-best PnL curves (§8)
  - ep_rew_mean flatline (§9)
  - Ablation 2×2 — render as a styled table or heatmap (§10)
  - Seed-variance dot plot (§11)
- [ ] `jupyter nbconvert --to html --no-input notebooks/presentation.ipynb` for sections 0–5; re-export with input visible for 6–11 (or use cell-level tags).
- [ ] Set `plt.rcParams["font.size"] = 14` minimum, white background.
- [ ] Open the HTML on the projector machine ahead of time, full-screen, check legibility from the back of the room.
- [ ] One dry run with all three of us before the event.
