"""Thin TradingAgent adapter that executes ``translate_action`` intents.

This is the only module in this package that imports ABIDES. The import is
guarded so the file loads on machines without ABIDES; the class itself only
works when ABIDES is on the path.

Intent dispatch is one method call per intent — kept deliberately small so
swapping the JPMorgan fork (or any other ABIDES variant) is cheap.
"""
from __future__ import annotations

from collections.abc import Callable

from .actions import (
    CancelIntent,
    Intent,
    PlaceIntent,
    RestingOrder,
    translate_action,
)

try:  # pragma: no cover — import is environment-dependent
    from agent.TradingAgent import TradingAgent  # type: ignore
    _ABIDES_AVAILABLE = True
except Exception:  # pragma: no cover
    TradingAgent = object  # type: ignore[misc, assignment]
    _ABIDES_AVAILABLE = False


def project_resting_orders(orders_dict: dict) -> list[RestingOrder]:
    """Convert ABIDES ``self.orders`` to ``RestingOrder`` records."""
    out: list[RestingOrder] = []
    for order_id, order in orders_dict.items():
        out.append(
            RestingOrder(
                order_id=int(order_id),
                is_buy_order=bool(order.is_buy_order),
                limit_price=int(order.limit_price),
                quantity=int(order.quantity),
            )
        )
    return out


def dispatch_intents(
    intents: list[Intent],
    *,
    orders_dict: dict,
    place_fn: Callable[..., None],
    cancel_fn: Callable[..., None],
) -> None:
    """Push intents through the supplied ABIDES helpers.

    Split out from ``MarlAgent.wakeup`` so it is testable without instantiating
    a real ABIDES agent — pass in mocks for ``place_fn`` / ``cancel_fn`` and a
    plain dict for ``orders_dict``.
    """
    for intent in intents:
        if isinstance(intent, CancelIntent):
            order = orders_dict.get(intent.order_id)
            if order is not None:
                cancel_fn(order)
        elif isinstance(intent, PlaceIntent):
            place_fn(
                intent.symbol,
                intent.quantity,
                intent.is_buy_order,
                intent.limit_price,
            )


class MarlAgent(TradingAgent):  # type: ignore[misc, valid-type]
    """ABIDES TradingAgent driven by a (policy, observation_extractor) pair.

    Constructed with:
      - ``policy_fn(obs) -> action_tuple`` (Pavel's E provides this)
      - ``observation_fn(self, current_time) -> obs`` (Lollo's A provides this)
      - ``symbol``, ``max_size``, ``tick_size`` — passed to ``translate_action``.

    The class body only resolves when ABIDES is importable; instantiating
    without ABIDES raises ``RuntimeError``.
    """

    def __init__(
        self,
        id: int,
        name: str,
        type: str,
        symbol: str,
        policy_fn: Callable,
        observation_fn: Callable,
        max_size: int = 100,
        tick_size: int = 1,
        random_state=None,
    ) -> None:
        if not _ABIDES_AVAILABLE:
            raise RuntimeError(
                "MarlAgent requires ABIDES on the import path. "
                "Install ABIDES or run on a machine that has it."
            )
        super().__init__(id, name, type, random_state=random_state)
        self.symbol = symbol
        self.policy_fn = policy_fn
        self.observation_fn = observation_fn
        self.max_size = max_size
        self.tick_size = tick_size

    def wakeup(self, current_time):  # pragma: no cover — exercised in ABIDES-gated test
        super().wakeup(current_time)
        obs = self.observation_fn(self, current_time)
        action = self.policy_fn(obs)
        mid_price = int(obs["mid_price"]) if isinstance(obs, dict) else int(obs[-1])
        intents = translate_action(
            action=tuple(action),
            agent_id=self.id,
            symbol=self.symbol,
            mid_price=mid_price,
            resting_orders=project_resting_orders(self.orders),
            max_size=self.max_size,
            tick_size=self.tick_size,
        )
        dispatch_intents(
            intents,
            orders_dict=self.orders,
            place_fn=self.placeLimitOrder,
            cancel_fn=self.cancelOrder,
        )
