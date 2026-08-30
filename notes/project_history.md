# Project history — team phase (2026-04-20 to 2026-05-11)

Archived 2026-08-19 from the Claude memory file `project_marl_lob.md`, which had grown to 19,930 bytes of append-only status entries and contained superseded claims (a 2026-05-10 "breakthrough" retracted on 05-11, three out-of-order "Status as of 2026-05-06" sections, a \`min_size\` note contradicting its own revert). Kept verbatim below as the record of the team phase. Current state lives in memory; results live in `experiments_results.md`.

---

---
name: MARL LoB project
description: Team-of-4 research project on multi-agent RL in limit order books; Pavel is project lead
type: project
originSessionId: 2c5deaf2-7d54-46c1-bfbe-7171a19a95cf
---
Team project on Multi-Agent RL in Limit Order Books, kicked off 2026-04-20. **Active team of 3**: Pavel (lead), Neil, Lorenzo "Lollo" Mossini (lollomossa@gmail.com). Salome is still in the group chat but skipped meeting 1, submitted Task 1 (simulator) research on 2026-04-24, then dropped before meeting 2 (2026-04-27). Treat as inactive contributor but socially still present — don't write her out of the chat or remove from Notion mentions in a way that would feel cold.

**External deadline (per "the association" message forwarded on 2026-04-22):** there's an event before exams where the team presents progress. Project doesn't need to be finished by then — more advanced = better. Summer is for completion. Pavel's first exam is **2026-05-20** (he has a full week to study beforehand, so coding runway is roughly through ~2026-05-13). Exact event date TBD — Pavel messaged the association heads on 2026-05-06 to ask; original forwarded message only said "before exams" with no specific date.

**Phase 0 status (closed 2026-05-02):**
- **Pavel → design + onboarding doc:** done. Both Notion pages posted as child pages of "MARL LoB Trading" (sources at `/home/pavel/marl_lob/notes/phase0_stack.md` and `/home/pavel/marl_lob/notes/phase0_env_wrapper.md`).
- **Neil → GitHub repo:** repo created (https://github.com/Miliks343/Multi-Agent-Trading-Hephasteus) but skeleton wasn't built. **Pavel pushed the skeleton on 2026-05-02 with Neil's prior agreement over DM** — pyproject (Python ≥3.11; numpy/pandas/matplotlib/pettingzoo/mpe2/sb3/supersuit/tensorboard), src/marl_lob/, tests/test_smoke.py, .gitignore, README. NO LICENSE (Pavel's call — student project, can add later if open-sourced). Verified `pip install -e . && pytest` on Linux + Mac (conda).
- **Lollo → ABIDES import:** ABIDES installed locally + RMSC03 running as of 2026-05-01 meeting; hello-world notebook still to come.

**Stack decided (locked unless something forces a re-eval):**
- Simulator: ABIDES (jpmc fork)
- Multi-agent env API: PettingZoo
- Trainer: Stable-Baselines3 with SuperSuit as the multi→single-agent adapter
- Initial algo: independent PPO with parameter sharing across learning agents
- Fallback if self-play instability hits: RLlib for opponent-sampling / league training
- Eval metrics: Sharpe + max drawdown / inventory risk
- Baseline: naive constant-spread market maker

**Phase 1 plan (proposed, presenting at 3:30pm 2026-05-01 meeting — final assignments may shift post-meeting):**
- Source of truth: `/home/pavel/marl_lob/notes/phase1_plan.md`.
- Wrapper decomposition: **A** = observation extractor (ABIDES book → numpy vector, ABIDES-side helper), **B** = action translator (action vector → ABIDES Order/Cancel messages, ABIDES-side helper), **C** = PettingZoo glue (`parallel_env` class wiring A + B with reset/step/done). A and B are independent and parallelizable; C is the integration.
- Parallel streams: **D** = metrics harness (Sharpe / drawdown / inventory dist / PnL — pure functions over trajectories, testable with synthetic data), **E** = training pipeline (SB3 + SuperSuit + PPO; develop against PettingZoo toy env in week 1, swap to MarlLobEnv when C lands), **F** = constant-spread baseline as a *native ABIDES agent* (subclasses ABIDES TradingAgent; runs alongside RMSC03 background agents — does NOT depend on the PettingZoo wrapper).
- Proposed assignments: **Pavel** = C + E + reward design; **Lollo** = A + F; **Neil** = B + D + repo upkeep.
- May 10 pre-exam event demo target: floor = baseline + metrics running through real ABIDES; ceiling = first PPO training run kicked off. Public-facing line proposed: "first training run is a stretch, not a commitment."
- Trajectory contract (the data shape between C and D, also D's input format for synthetic-data testing): list of (timestamp, inventory, cash, mid_price, fill) tuples per agent.
- Open questions for the meeting: ABIDES install solidity for Lollo (week 1 risk for A and F), Neil's load (he was originally penciled for E too), public commitment level for May 10.

**Local notebook directory layout** (`/home/pavel/marl_lob/`):
- `papers/01_hephaestus_marl_lob.pdf` — Lollo's lit review deliverable
- `notes/neil_task2_4_qa.pdf` — Neil's Task 2/4 Q&A doc (data + metrics + baseline answers)
- `notes/task5_framework_pitch.md` — Pavel's framework speaking notes (the pitch that locked PettingZoo + SB3 + SuperSuit)
- No code yet. The eventual repo lives on Neil's GitHub.

**Local dev environment (Pavel's box, set up 2026-05-06):**
- Miniconda at `$HOME/miniconda3` (installed via official `Miniconda3-latest-Linux-x86_64.sh`).
- Conda env `marl_lob` (Python 3.11, conda-forge only — Anaconda channel ToS not accepted, use `--override-channels -c conda-forge` for any new conda installs).
- ABIDES jpmc fork cloned at `/home/pavel/marl_lob/abides-jpmc-public/`. Installed editable: `abides-core` and `abides-markets`. NOT installed: `abides-gym`. Skipped pinned deps: `gym`, `ray`, `pomegranate` (not needed for our path; full reasoning in team repo README).
- Local-only patch in `abides-jpmc-public/abides-markets/abides_markets/configs/rmsc03.py` — POVExecutionAgent guarded as soft `None` import (the symbol is missing from this fork). Documented in team README under "ABIDES patch"; if Pavel re-clones ABIDES he must reapply.
- Activate the env in any new shell: `source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate marl_lob`.
- Jupyter on tailscale: bind to all interfaces with `--ip=0.0.0.0` (default localhost-only blocks `macmini:8888` access). Stop with `jupyter server stop 8888` when done.

**Status as of 2026-05-06 (post-meeting + integration session):**
- Integration commit `97c888b` pushed to main: vendored `rmsc03_simple` config (in our repo, not upstream), fixed Neil's wrong ABIDES import path in `agent_adapter.py` (`agent.TradingAgent` → `abides_markets.agents.trading_agent`) so the previously-skipped test now passes (25/25 no skips), patched Lollo's notebook to drop None rows before merging L1 snapshots, removed Lollo's accidentally-committed `log/` dumps and added `log/` to `.gitignore`, full install steps written into README.
- ABIDES verified end-to-end: 1h RMSC03 sim runs in ~3s wallclock, mid-price + spread plots render correctly. LOB snapshot cell (`code cell 10` in `00_abides_hello.ipynb`) is broken — known Lollo issue, not blocking.
- Sent message to association heads on 2026-05-06 asking for the exact event date — no reply yet.

**Status as of 2026-05-06 (end of day, all of C built):**
- All 9 chunks of Phase 1 module C done in one push: branch `feat/pettingzoo-env` on team repo (https://github.com/Miliks343/Multi-Agent-Trading-Hephasteus), 10 commits, draft PR #1 open. Final branch tip: `775b1e4`.
- New code (in team repo): `src/marl_lob/marl_agents.py` (MarlCoordinator + MarlChild + pure helpers), `src/marl_lob/env.py` (MarlLobEnv ParallelEnv), `src/marl_lob/baseline_traj.py` (LoggingConstantSpreadMM that wraps F), `scripts/{train,run_baseline,eval}.py`, `tests/{test_marl_agents,test_env,test_baseline_traj}.py`. Trajectory.from_tuples extended to 6-tuple `(ts, inv, cash, mid, fill_qty, fill_price)`.
- Tests: 56 passing (was 36 before today). Includes PettingZoo `parallel_api_test` with 50 cycles. ~17s wallclock.
- Pure-helper TDD pattern was right call — pure functions extracted from agents/env so unit tests don't need the kernel; integration tests under `@pytest.mark.abides` exercise the real kernel.
- Design doc + chunk plan: `/home/pavel/marl_lob/notes/c_wrapper_design.md` (in this notebook, NOT team repo). Architecture: single MarlCoordinator is the kernel's only experimental agent (it can pause kernel.runner via non-None wakeup return); coordinator multiplexes obs/action between PettingZoo env and N MarlChild trading agents that hold inventory and place orders via Neil's translate_action (B reused unchanged).

**Critical ABIDES patch (do NOT lose):** `abides_core/utils.py:str_to_ns` was 1000× off on modern pandas (returned μs not ns via `.to_timedelta64().astype(int)`). Fixed locally with `return pd.to_timedelta(string).value`. Documented in team README "ABIDES patch" section. **Lollo's "1h sim ran in 3s" was actually a 3.6-second sim** — measurements collected before today are off accordingly. Reapply if Pavel re-clones the ABIDES fork.

**Other ABIDES integration discoveries (all fixed today):**
- Real ABIDES `Order` has `.side.is_bid()` / `.side.is_ask()`, NOT `.is_buy_order`. Neil's MagicMock-based unit tests in `test_agent_adapter.py` used the wrong attribute and silently passed; integration would have crashed. Fixed both `agent_adapter.project_resting_orders` and the new `marl_agents.snapshot_fills`.
- `place_limit_order` expects `Side` enum, not bool. Wrapped in `MarlChild.act_on_wakeup`.
- `CoreBackgroundAgent.wakeup` does NOT call `set_wakeup` — agents must do it themselves in `act_on_wakeup`. Without it, kernel exhausts after one tick.
- L1 book lives in `self.parsed_mkt_data["bids"]` / `["asks"]` (filled by L2 subscription), not `self.known_bids` (only populated by explicit exchange queries).
- `TradingAgent.executed_orders` is initialized empty but **never appended to** by `TradingAgent.order_executed` — that's a `CoreBackgroundAgent` extension. F (which subclasses TradingAgent directly, not CoreBackgroundAgent) needs its own fill tracking. `LoggingConstantSpreadMM` overrides `order_executed` for this.

**Demo-relevant numbers as of end-of-day 2026-05-06:**
- Random policy (uniform on action space): avg cumulative reward ≈ −$157k per 1h episode.
- F (constant-spread MM, defaults: spread_ticks=10, quote_size=100, wake_up_freq=10s): Sharpe ≈ −22 / −9, MaxDD ≈ 60%, Δequity ≈ −$14k. Bleeds because constant-spread + size 100 gets adversely selected by ABIDES' value/momentum/noise flow; ends with ~2300-2800 shares long.
- PPO after 50k timesteps from default init: **learned noop**. Zero fills, Sharpe 0, Δequity 0. Network mean stayed near initialization (approx_kl ~10⁻⁵), output clipped to action_space.low which corresponds to "size 0 = don't quote."
- So: **PPO ($0) > F (−$14k) > Random (−$157k)** — but PPO "wins" by sitting out, not by trading skillfully. Hollow comparison.
- **Reward-scale problem confirmed:** value_loss oscillates 10⁷–10¹¹; ep_rew_mean ~−10⁷. Rewards in cents → ΔPnL telescopes to massive magnitudes per episode. PPO can't learn from gradients this scale.

**Required pre-merge tuning (post chunk 9):**
- Raise `action_space.low` for size dims to ≥1 so size=0 (noop) is impossible — forces the agent to actually quote.
- Reward-scale: either express reward in dollars (÷100) or wrap with `VecNormalize`.
- F parameter sweep: a "fair" F config (likely wider spread, smaller size) before claiming PPO can beat F. Pavel agreed to a future matrix sweep over `spread_ticks × quote_size × wake_up_freq`.

**Breakthrough on 2026-05-10 (later corrected — see 2026-05-11 entries):** PPO actually trades when (a) `inventory_penalty=0.0` passed to `MarlLobEnv`, (b) env wrapped in `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)` in `scripts/train.py`. After 50k steps on seed 42, per-agent: PPO 1181/1191 fills, Δeq ≈ −$6.7k/−$6.9k, MaxDD ≈ 10%, final inv ≈ −380 (short). Checkpoint at `runs/marl_lob_ppo/no_inv_penalty_vecnorm/ppo_marl_lob.zip` (NOT committed). With VecNormalize: value_loss 0.006, std 1.00→0.92, approx_kl ~0.01, explained_variance ~0.33. `ep_rew_mean` is misleading — VecMonitor logs raw rewards (still −1e7), rolling 100-episode buffer barely turns over at 50k steps.

**Ablation 2×2 on 2026-05-11 (50k steps, seed 42):** `inv_pen=0 + no VecNorm` → noop (0 fills, $0). `inv_pen=1e-4 + no VecNorm` → original failed run (noop). `inv_pen=0 + VecNorm` → Exp 1 (−$6.7k/agent). **`inv_pen=1e-4 + VecNorm` → +$117/+$113 ✅ — best PPO result.** VecNormalize is the only load-bearing change; removing inventory penalty did nothing on its own AND was actively counterproductive when paired with VecNorm. Right setup = default inv_penalty + VecNormalize. Checkpoints `runs/marl_lob_ppo/abl_{vecnorm_only,invpen0_only}/ppo_marl_lob.zip`.

**Seed robustness on 2026-05-11:** Exp 1 config (inv_pen=0, VecNorm) at train seeds {0,1,7,42}, all eval seed 42. Δequity range −$6,874 → +$271 ($7k spread); fills 130–1191; final inventory flips short↔long across seeds. PPO behavior is highly seed-dependent — sometimes lands on profitable policy (seed 7: +$268/+$271), sometimes on −$6.7k loss (seeds 1, 42), sometimes light trading (seed 0: −$1.6k).

**F parameter sweep on 2026-05-11 (seed 42, no training):** **F at spread=30, size=10 is profitable (+$590/−$75)** — breaks the original "PPO beats F" demo claim. F at spread=10, size=10 loses only ~$1k. Default F (spread=10, size=100) was a strawman: large quote_size gets adversely-selected. Trajectories `runs/baseline_sweep/spread{10,30}_size{10,100}/`.

**Demo headline reframing (2026-05-11):** Original "PPO beats F by 40% with 6× better drawdown" was true only against poorly-tuned F + single seed. Honest claim is "PPO learns market-making when reward scale is normalized (VecNorm); matches well-tuned baseline on best seeds, but high seed variance." Full writeup at `/home/pavel/marl_lob/notes/experiments_results.md`. Lead the presentation with the reward-scale diagnosis + ablation 2×2 — that's the clean scientific result.

**Scripts (`scripts/train.py`) changes on 2026-05-11:** added `--seed`, `--inventory-penalty`, `--no-vecnormalize` CLI flags; `make_env` takes `inventory_penalty` + `vecnormalize` kwargs; uses `set_random_seed` instead of PPO(seed=) because SuperSuit's `ConcatVecEnv` lacks `.seed()` so SB3's seed propagation crashes (AttributeError). Reverted `min_size=10` to default (Exp 2 leftover) in both train.py and eval.py.

**min_size experiment on 2026-05-10:** added `min_size: int = 0` param to `MarlLobEnv.__init__` so `action_space.low` for size dims can be raised. Trained 50k with `min_size=10` (forcing ≥10 shares quoted per side). Result: **PPO routed around the constraint** — instead of trading more, it learned to quote at the max offset (~50 cents off mid) so its mandatory size-10 quotes almost never fill. Per-agent on seed 42: 7/8 fills (vs 1181/1191 unconstrained), Δeq −$36/−$20, MaxDD 0.08%, final inv +30/+26. Checkpoint `runs/marl_lob_ppo/min_size_10/ppo_marl_lob.zip`. **Lesson: PPO's local minimum on this env isn't about size=0; it's about avoiding adverse selection. Single-knob constraints get routed around — to force activity we'd need to clamp `max_offset_cents` simultaneously, add a fill/turnover reward, or penalize wide spreads directly.** Both train.py and eval.py now pass `min_size=10`; reset to 0 for honest comparisons.

**PR review status:** draft. The "also fixes" section in the PR body documents the bool→Side and str_to_ns issues for Neil's awareness — no separate ping, he sees them in the PR.

**Status as of 2026-05-06 (post-meeting):**
- Phase 1 modules A, B, D, F all merged to main:
  - **A** (obs extractor, `src/marl_lob/observation_extractor.py`) — Lollo. 4*K+4 obs vector, default K=10 → 44 dims, cents convention.
  - **B** (action translator, `src/marl_lob/actions.py` + `agent_adapter.py`) — Neil. Continuous 4-tuple `(bid_offset, ask_offset, bid_size, ask_size)`. ABIDES import guarded; one test skipped behind that guard.
  - **D** (metrics, `src/marl_lob/trajectory.py` + `metrics.py`) — Neil. `Trajectory` enforces integer-cents dtypes in `__post_init__`. `compute_all` returns Sharpe / max_drawdown / inventory dist / pnl_curve.
  - **F** (`baseline_market_maker.py`) — Lollo. Native ABIDES `TradingAgent` subclass.
  - Lollo also shipped `notebooks/00_abides_hello.ipynb` (RMSC03 hello-world).
- Lollo accidentally committed `log/1778061202/` and `log/1778061225/` (ABIDES summary logs). `log/` not yet in `.gitignore`.
- README still says "ABIDES install — see Lollo's notes (incoming)"; Lollo has not documented install steps. ABIDES is NOT yet installed on Pavel's machine.
- **Meeting outcome (2026-05-06, 7pm):** Neil and Lollo both flagged exam load and limited time. Agreed Pavel does the integration solo for now; if he hits a wall he'll hand off small chunks. Pre-staged hand-off candidates: `log/` gitignore + ABIDES install docs (Lollo, ~20 min); `rmsc03_with_baseline.py` config (Lollo, ~30 min); E PR review (Neil); trajectory-buffering helper for callback (Neil).
- **Pavel's solo plan:** install ABIDES → document install in README → verify Lollo's notebook runs → run full pytest (check whether the ABIDES-gated test in `agent_adapter.py` passes against real ABIDES) → write C (PettingZoo wrapper) → adapt Neil's B if needed → reward v1 → swap E from `simple_spread_v3` to `MarlLobEnv` → kick off first PPO training run.
- **Integration contract decided:** `env.step()` returns `info[agent] = {"traj_row": (timestamp_s, inventory, cash_cents, mid_cents, fill_signed_int)}`. C must emit this so Neil's eventual callback consumes it without rework.
- E (Pavel's toy training pipeline) still on `feat/training-pipeline-toy`, unmerged. PR was awaiting Neil's review since 2026-05-02. Decision deferred — likely merge alongside C.

**Status as of 2026-05-02:**
- Pavel is project lead — responsible for planning per-person tasks and handing them out.
- Simulator: **ABIDES**.
- Phase 0 closed (status above). Phase 1 plan accepted at 2026-05-01 meeting with no changes. Full plan posted to Notion as a subpage of "MARL LoB Trading"; per-person summary on the main page.
- **Phase 1 in progress:**
  - **Pavel — E (training pipeline):** toy version on branch `feat/training-pipeline-toy` in the team repo, PR open. `scripts/train_toy.py` trains PPO on `mpe2.simple_spread_v3` (continuous actions to match expected MarlLobEnv shape), wired through SuperSuit. Verified learning over 500k steps (-25.5 → -22.7 ep_rew_mean). Awaiting Neil review or May 6 meeting.
  - **Pavel — C (env glue), reward design:** not started.
  - **Lollo — A (obs extractor), F (constant-spread baseline):** not started; A blocked on RMSC03 inspection (his next step).
  - **Neil — B (action translator), D (metrics harness):** not started.
- Local clone of team repo: `/home/pavel/marl_lob/Multi-Agent-Trading-Hephasteus/` (separate git repo, lives inside the notebook directory for convenience).
- Next meeting: **2026-05-06, 7pm online** (Wed).
- Prior research artifacts:
  - Pavel's Task 5 framework research → `/home/pavel/marl_lob/notes/task5_framework_pitch.md`.
  - Neil's Task 2/4 research → `/home/pavel/marl_lob/notes/neil_task2_4_qa.pdf` (data sourcing + metrics/baseline).
  - Lollo's Task 3 lit review → `/home/pavel/marl_lob/papers/01_hephaestus_marl_lob.pdf`.
- Team Q&A settled: ABIDES sim, RL training, eval by Sharpe + max drawdown / inventory risk, baseline = naive constant-spread market maker.

**Why:** Pavel needs to translate the research-phase task list into a build-phase plan. The original Tasks 1–5 are mostly research-completed; new tasks are concrete engineering chunks (repo, ABIDES import, env wrapper, baseline impl, metrics harness, training loop). Salome's simulator decision unblocks everyone.

**How to apply:** When Pavel asks for help, default to the lead's perspective — task decomposition, sequencing, dependency awareness, what to hand to whom — rather than just doing the work. The framework decision is locked unless something forces a re-eval. Watch for the GitHub+ABIDES-import duo as the critical-path bootstrap; nothing else can really start until those two land.
