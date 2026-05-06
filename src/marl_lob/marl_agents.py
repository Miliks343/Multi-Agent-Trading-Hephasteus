"""Module C, chunk 1 — MARL coordinator + child agents.

The coordinator is the single ABIDES experimental agent that talks to the
PettingZoo env. Children are plain trading agents that receive their next
action from a slot the coordinator writes to.

Pure-Python helpers live at module level so they can be unit-tested without
ABIDES on the import path. The ABIDES classes (`MarlChild`,
`MarlCoordinator`) are thin wrappers that snapshot kernel state into those
helpers and execute the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .actions import translate_action
from .agent_adapter import dispatch_intents, project_resting_orders
from .observation_extractor import (
    DEFAULT_K,
    DEFAULT_MAX_INVENTORY,
    DEFAULT_MAX_SIZE,
    DEFAULT_STARTING_CASH,
    extract_obs,
)

PriceLevels = list[tuple[int, int]]


# ─────────────────────────────────────────────────────────────────────────────
# Plain-data carriers — what the pure helpers operate on
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutedFill:
    """One fill, signed. Positive = agent bought; negative = agent sold."""
    signed_qty: int
    price_cents: int


@dataclass(frozen=True)
class ChildSnapshot:
    inventory: int
    cash: int
    fills: list[ExecutedFill]


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def distribute_actions(
    action_list: list[dict[str, Any]],
    children: list[Any],
) -> None:
    """Write each ``action_vec`` into ``children[agent_idx].pending_action``."""
    for entry in action_list:
        idx = entry["agent_idx"]
        if idx < 0 or idx >= len(children):
            raise IndexError(
                f"agent_idx {idx} out of range for {len(children)} children"
            )
        children[idx].pending_action = entry["action_vec"]


def snapshot_fills(executed_orders: list[Any]) -> list[ExecutedFill]:
    """Convert ABIDES executed-order objects into signed `ExecutedFill` records.

    Caller passes the contents of ``child.inter_wakeup_executed_orders`` (full
    Order objects, since the parsed-tuple form drops the side bit).
    """
    out: list[ExecutedFill] = []
    for order in executed_orders:
        sign = 1 if order.side.is_bid() else -1
        out.append(
            ExecutedFill(
                signed_qty=sign * int(order.quantity),
                price_cents=int(order.fill_price),
            )
        )
    return out


def _aggregate_fills(fills: list[ExecutedFill]) -> tuple[int, int]:
    """Net signed quantity + abs-qty-weighted VWAP across all fills this step.

    Mixed-side steps collapse both sides into one VWAP — a known imprecision,
    documented in the design doc. With 1-second wakeups, MM agents almost
    never both buy and sell in a single step.
    """
    if not fills:
        return 0, 0
    net_signed = sum(f.signed_qty for f in fills)
    abs_total = sum(abs(f.signed_qty) for f in fills)
    if abs_total == 0:
        return 0, 0
    vwap = sum(abs(f.signed_qty) * f.price_cents for f in fills) / abs_total
    return net_signed, int(round(vwap))


def build_per_agent_state(
    snapshots: list[ChildSnapshot],
    known_bids: PriceLevels,
    known_asks: PriceLevels,
    current_time_ns: int,
    mkt_open_ns: int,
    mkt_close_ns: int,
    *,
    k: int = DEFAULT_K,
    starting_cash: int = DEFAULT_STARTING_CASH,
    max_inventory: int = DEFAULT_MAX_INVENTORY,
    max_size: int = DEFAULT_MAX_SIZE,
) -> list[dict[str, Any]]:
    """Build the per-agent payload the env consumes.

    Each entry has the obs vector (from A) plus the raw fields C needs for
    reward computation and the `traj_row` info contract: inventory, cash,
    mid_cents, fill_signed_qty, fill_price_cents.
    """
    best_bid = known_bids[0][0] if known_bids else None
    best_ask = known_asks[0][0] if known_asks else None
    if best_bid is not None and best_ask is not None:
        mid_cents = (best_bid + best_ask) // 2
    elif best_bid is not None:
        mid_cents = best_bid
    elif best_ask is not None:
        mid_cents = best_ask
    else:
        mid_cents = 0

    out: list[dict[str, Any]] = []
    for snap in snapshots:
        obs = extract_obs(
            known_bids=known_bids,
            known_asks=known_asks,
            inventory=snap.inventory,
            cash=snap.cash,
            current_time=current_time_ns,
            mkt_open=mkt_open_ns,
            mkt_close=mkt_close_ns,
            k=k,
            max_inventory=max_inventory,
            max_size=max_size,
            starting_cash=starting_cash,
        )
        fill_qty, fill_px = _aggregate_fills(snap.fills)
        out.append({
            "obs": obs,
            "inventory": int(snap.inventory),
            "cash": int(snap.cash),
            "mid_cents": int(mid_cents),
            "fill_signed_qty": int(fill_qty),
            "fill_price_cents": int(fill_px),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ABIDES wrappers — only resolve when ABIDES is importable
# ─────────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover — environment-dependent
    from abides_markets.agents.background_v2.core_background_agent import (
        CoreBackgroundAgent,
    )
    from abides_markets.orders import Side  # for the bool→Side adapter below
    _ABIDES_AVAILABLE = True
except Exception:  # pragma: no cover
    CoreBackgroundAgent = object  # type: ignore[misc, assignment]
    _ABIDES_AVAILABLE = False


class MarlChild(CoreBackgroundAgent):  # type: ignore[misc, valid-type]
    """ABIDES trading agent driven by an externally-set `pending_action`.

    The coordinator's `apply_actions` writes a 4-tuple into `pending_action`
    each wrapper-tick. The next time this agent's wakeup fires (a few ns
    after the coordinator's), it reads the action, runs `translate_action`,
    dispatches the resulting intents, and clears `pending_action`.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        *,
        max_size: int = 100,
        tick_size: int = 1,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state=None,
        **kwargs: Any,
    ) -> None:
        if not _ABIDES_AVAILABLE:
            raise RuntimeError("MarlChild requires ABIDES on the import path.")
        super().__init__(
            id,
            symbol=symbol,
            starting_cash=starting_cash,
            name=name or f"MarlChild-{id}",
            type=type or "MarlChild",
            random_state=random_state,
            **kwargs,
        )
        self.max_size = max_size
        self.tick_size = tick_size
        self.pending_action: Optional[tuple[float, float, float, float]] = None

    def act_on_wakeup(self) -> None:  # pragma: no cover — ABIDES-driven
        # Schedule the next wakeup unconditionally — even if we have no
        # pending action, we need to keep ticking so the coordinator's
        # apply_actions has somewhere to land.
        self.set_wakeup(
            self.current_time + self.wakeup_interval_generator.next()
        )
        if self.pending_action is None:
            return
        bids = self.parsed_mkt_data.get("bids", []) or []
        asks = self.parsed_mkt_data.get("asks", []) or []
        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) // 2
        elif bids:
            mid = bids[0][0]
        elif asks:
            mid = asks[0][0]
        else:
            self.pending_action = None
            return
        intents = translate_action(
            action=tuple(self.pending_action),
            agent_id=self.id,
            symbol=self.symbol,
            mid_price=int(mid),
            resting_orders=project_resting_orders(self.orders),
            max_size=self.max_size,
            tick_size=self.tick_size,
        )
        # ABIDES' place_limit_order expects a Side enum, not a bool. Adapt
        # here so dispatch_intents stays ABIDES-free for Neil's unit tests.
        def _place(symbol, qty, is_buy, limit_price):
            self.place_limit_order(
                symbol, qty, Side.BID if is_buy else Side.ASK, limit_price
            )

        dispatch_intents(
            intents,
            orders_dict=self.orders,
            place_fn=_place,
            cancel_fn=self.cancel_order,
        )
        self.pending_action = None


class MarlCoordinator(CoreBackgroundAgent):  # type: ignore[misc, valid-type]
    """The single experimental agent. Aggregates state across N children and
    multiplexes one `kernel.runner()` round-trip per wrapper-tick.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        children: list[MarlChild],
        mkt_open_ns: int,
        mkt_close_ns: int,
        *,
        k: int = DEFAULT_K,
        max_inventory: int = DEFAULT_MAX_INVENTORY,
        max_size: int = DEFAULT_MAX_SIZE,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state=None,
        **kwargs: Any,
    ) -> None:
        if not _ABIDES_AVAILABLE:
            raise RuntimeError("MarlCoordinator requires ABIDES on the import path.")
        super().__init__(
            id,
            symbol=symbol,
            starting_cash=starting_cash,
            name=name or "MarlCoordinator",
            type=type or "MarlCoordinator",
            random_state=random_state,
            **kwargs,
        )
        self.children = children
        self.mkt_open_ns = mkt_open_ns
        self.mkt_close_ns = mkt_close_ns
        self.k = k
        self.max_inventory = max_inventory
        self.max_size_norm = max_size
        self._starting_cash_norm = starting_cash

    def apply_actions(self, actions: list[dict[str, Any]]) -> None:  # pragma: no cover
        distribute_actions(actions, self.children)

    def act_on_wakeup(self) -> dict[str, Any]:  # pragma: no cover — ABIDES-driven
        # Schedule next wakeup; without this, the agent fires once and the
        # episode effectively ends.
        self.set_wakeup(
            self.current_time + self.wakeup_interval_generator.next()
        )
        snaps = [
            ChildSnapshot(
                inventory=int(child.get_holdings(child.symbol)),
                cash=int(child.get_holdings("CASH")),
                fills=snapshot_fills(child.inter_wakeup_executed_orders),
            )
            for child in self.children
        ]
        for child in self.children:
            child.inter_wakeup_executed_orders = []
            child.parsed_inter_wakeup_executed_orders = []
        return {
            "per_agent": build_per_agent_state(
                snaps,
                known_bids=self.parsed_mkt_data.get("bids", []) or [],
                known_asks=self.parsed_mkt_data.get("asks", []) or [],
                current_time_ns=int(self.current_time),
                mkt_open_ns=self.mkt_open_ns,
                mkt_close_ns=self.mkt_close_ns,
                k=self.k,
                starting_cash=self._starting_cash_norm,
                max_inventory=self.max_inventory,
                max_size=self.max_size_norm,
            ),
            "timestamp_s": (int(self.current_time) - self.mkt_open_ns) / 1e9,
        }
