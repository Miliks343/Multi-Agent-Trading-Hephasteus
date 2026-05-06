"""MarlLobEnv — PettingZoo ParallelEnv wrapping ABIDES via MarlCoordinator.

Skeleton chunk (chunk 2). Reset spins up an RMSC03_simple sim with one
coordinator + N children injected; step calls into the kernel one wrapper
tick at a time and unpacks per-agent state into PettingZoo dicts.

Reward is a stub (zeros) in this chunk — chunk 4 wires the real
ΔPnL − λ·|inv| formula. Termination on |inv| > cap also lands in chunk 4.

Pattern adapted from abides-gym/abides_gym/envs/core_environment.py, with
the multi-agent multiplexing handled by MarlCoordinator.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from abides_core import Kernel
from abides_core.generators import ConstantTimeGenerator
from abides_core.utils import str_to_ns, datetime_str_to_ns, subdict
from abides_markets.utils import config_add_agents

from .configs import rmsc03_simple
from .marl_agents import MarlChild, MarlCoordinator
from .observation_extractor import (
    DEFAULT_K,
    DEFAULT_MAX_INVENTORY,
    DEFAULT_MAX_SIZE,
    DEFAULT_STARTING_CASH,
    obs_vector_size,
)


class MarlLobEnv(ParallelEnv):
    """N-market-maker PettingZoo environment over ABIDES RMSC03.

    Parameters
    ----------
    n_agents : int
        Number of learning market makers. Each gets its own MarlChild.
    starting_cash : int
        Per-agent starting CASH (cents). Also used for obs normalisation.
    k : int
        LOB depth in the observation vector (4*K + 4 features total).
    max_size : int
        Cap on per-side order size; also the action_space upper bound.
    max_offset_cents : int
        Cap on bid/ask offset from mid (cents); the action_space upper bound
        for the two offset dimensions.
    tick_size : int
        ABIDES tick (cents). Defaults to 1.
    historical_date : str
        YYYYMMDD passed through to RMSC03 — fixes the oracle's date.
    start_time, end_time : str
        HH:MM:SS market window.
    config_kwargs : dict | None
        Extra kwargs forwarded to rmsc03_simple.build_config (agent counts,
        etc.). seed is handled via reset(seed=...) and overrides this.
    """

    metadata = {"name": "marl_lob_v0", "is_parallelizable": True}

    def __init__(
        self,
        n_agents: int = 2,
        *,
        starting_cash: int = DEFAULT_STARTING_CASH,
        k: int = DEFAULT_K,
        max_size: int = 100,
        max_offset_cents: int = 50,
        tick_size: int = 1,
        max_inventory: int = DEFAULT_MAX_INVENTORY,
        symbol: str = "ABM",
        historical_date: str = "20200603",
        start_time: str = "09:30:00",
        end_time: str = "10:30:00",
        wakeup_interval: str = "1s",
        inventory_penalty: float = 1e-4,
        termination_penalty: float = -1.0,
        config_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.n_agents = n_agents
        self.starting_cash = starting_cash
        self.k = k
        self.max_size = max_size
        self.max_offset_cents = max_offset_cents
        self.tick_size = tick_size
        self.max_inventory = max_inventory
        self.symbol = symbol
        self.historical_date = historical_date
        self.start_time = start_time
        self.end_time = end_time
        self.wakeup_interval = wakeup_interval
        self.inventory_penalty = float(inventory_penalty)
        self.termination_penalty = float(termination_penalty)
        self.config_kwargs = dict(config_kwargs or {})

        self.possible_agents = [f"mm_{i}" for i in range(n_agents)]
        self.agents: list[str] = []

        obs_dim = obs_vector_size(k)
        self._obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [max_offset_cents, max_offset_cents, max_size, max_size],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.kernel: Optional[Kernel] = None
        self._coord: Optional[MarlCoordinator] = None
        self._children: list[MarlChild] = []
        self._mkt_open_ns: int = 0
        self._mkt_close_ns: int = 0
        # Per-agent equity from the previous step, for ΔPnL reward.
        # Indexed by possible_agents[i] (we keep all N entries even after
        # an agent terminates; lookups for terminated agents stop happening).
        self._prev_equity: dict[str, int] = {}
        self._terminated: dict[str, bool] = {}

    # ── PettingZoo space accessors ──────────────────────────────────────────
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is None:
            seed = np.random.randint(low=0, high=2**32, dtype=np.uint64)

        # Build background config
        cfg_kwargs = dict(self.config_kwargs)
        cfg_kwargs.setdefault("start_time", self.start_time)
        cfg_kwargs.setdefault("end_time", self.end_time)
        cfg_kwargs["seed"] = int(seed) % (2**32 - 1)
        bg_config = rmsc03_simple.build_config(**cfg_kwargs)

        # Compute market window in ns (matches rmsc03's own calc)
        date_ns = datetime_str_to_ns(self.historical_date)
        self._mkt_open_ns = date_ns + str_to_ns(self.start_time)
        self._mkt_close_ns = date_ns + str_to_ns(self.end_time)

        # Children + coordinator. ID space continues from the background agents.
        next_id = len(bg_config["agents"])
        wakeup_ns = str_to_ns(self.wakeup_interval)
        coord_first = wakeup_ns
        # Children wake a hair after the coordinator each tick so action
        # distribution lands before order placement.
        child_first = wakeup_ns + 100  # 100 ns offset

        self._children = [
            MarlChild(
                id=next_id + i,
                symbol=self.symbol,
                starting_cash=self.starting_cash,
                max_size=self.max_size,
                tick_size=self.tick_size,
                wakeup_interval_generator=ConstantTimeGenerator(
                    step_duration=wakeup_ns
                ),
                first_interval=child_first,
                subscribe_num_levels=self.k,
            )
            for i in range(self.n_agents)
        ]
        self._coord = MarlCoordinator(
            id=next_id + self.n_agents,
            symbol=self.symbol,
            starting_cash=self.starting_cash,
            children=self._children,
            mkt_open_ns=self._mkt_open_ns,
            mkt_close_ns=self._mkt_close_ns,
            k=self.k,
            max_inventory=self.max_inventory,
            max_size=self.max_size,
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=wakeup_ns
            ),
            first_interval=coord_first,
            subscribe_num_levels=self.k,
        )
        config = config_add_agents(
            bg_config, list(self._children) + [self._coord]
        )

        self.kernel = Kernel(
            random_state=np.random.RandomState(seed=int(seed) % (2**32 - 1)),
            **subdict(
                config,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
        )
        self.kernel.initialize()
        raw = self.kernel.runner()  # runs until coordinator's first non-None wakeup

        self.agents = list(self.possible_agents)
        self._terminated = {a: False for a in self.possible_agents}
        obs, infos = self._unpack(raw)
        # Seed prev_equity from the first per-agent snapshot so the first
        # step's ΔPnL is computed against a real baseline rather than zero.
        self._prev_equity = {
            a: self._equity_from_traj_row(infos[a]["traj_row"])
            for a in self.possible_agents
        }
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Pass actions only for agents that haven't been terminated. For a
        # terminated agent, send a no-op (size 0) so the coordinator still
        # has a slot per child but no orders go in.
        action_list = []
        for i, agent in enumerate(self.possible_agents):
            if self._terminated[agent]:
                vec = (0.0, 0.0, 0.0, 0.0)
            else:
                vec = tuple(float(x) for x in actions[agent])
            action_list.append({"agent_idx": i, "action_vec": vec})

        raw = self.kernel.runner((self._coord, action_list))
        obs, infos = self._unpack(raw)

        truncated = bool(raw.get("done", False))
        rewards: dict[str, float] = {}
        terms: dict[str, bool] = {}
        truncs: dict[str, bool] = {}
        newly_terminated: list[str] = []

        for agent in list(self.agents):
            row = infos[agent]["traj_row"]
            inv = int(row[1])
            equity = self._equity_from_traj_row(row)
            term_now = abs(inv) > self.max_inventory

            reward = self._compute_reward(
                prev_equity=self._prev_equity[agent],
                equity=equity,
                inventory=inv,
                inventory_penalty=self.inventory_penalty,
                terminated=term_now,
                termination_penalty=self.termination_penalty,
            )
            self._prev_equity[agent] = equity

            if term_now:
                self._terminated[agent] = True
                newly_terminated.append(agent)

            rewards[agent] = reward
            terms[agent] = term_now
            truncs[agent] = truncated

        # Drop terminated/truncated agents from the active set so PettingZoo
        # stops sending us actions for them.
        if truncated:
            self.agents = []
        else:
            for a in newly_terminated:
                if a in self.agents:
                    self.agents.remove(a)

        return obs, rewards, terms, truncs, infos

    def close(self) -> None:
        self.kernel = None
        self._coord = None
        self._children = []
        self.agents = []

    # ── Internals ───────────────────────────────────────────────────────────
    @staticmethod
    def _equity_from_traj_row(row: tuple) -> int:
        """Mark-to-mid equity = cash + inventory * mid_cents (integer cents)."""
        _ts, inv, cash, mid, _fq, _fp = row
        return int(cash) + int(inv) * int(mid)

    @staticmethod
    def _compute_reward(
        prev_equity: int,
        equity: int,
        inventory: int,
        inventory_penalty: float,
        terminated: bool,
        termination_penalty: float,
    ) -> float:
        """ΔPnL minus inventory penalty, plus termination penalty if terminated.

        ΔPnL is mark-to-mid: equity − prev_equity (cents). Inventory penalty
        is symmetric in sign: λ·|inv| applies to long and short positions
        identically.
        """
        reward = float(equity - prev_equity) - inventory_penalty * abs(inventory)
        if terminated:
            reward += termination_penalty
        return reward

    def _unpack(
        self, raw: dict
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        result = deepcopy(raw.get("result") or {})
        per_agent = result.get("per_agent", [])
        timestamp_s = float(result.get("timestamp_s", 0.0))

        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict] = {}
        for i, agent in enumerate(self.possible_agents):
            if i < len(per_agent):
                entry = per_agent[i]
                obs[agent] = entry["obs"]
                infos[agent] = {
                    "traj_row": (
                        timestamp_s,
                        entry["inventory"],
                        entry["cash"],
                        entry["mid_cents"],
                        entry["fill_signed_qty"],
                        entry["fill_price_cents"],
                    )
                }
            else:
                # Defensive fallback (shouldn't happen if coord wired right).
                obs[agent] = np.zeros(self._obs_space.shape, dtype=np.float32)
                infos[agent] = {"traj_row": (timestamp_s, 0, 0, 0, 0, 0)}
        return obs, infos
