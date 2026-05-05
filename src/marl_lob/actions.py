"""Action translator: continuous action vector -> ABIDES order/cancel intents.

Pure Python; no ABIDES import. The thin TradingAgent adapter in
``agent_adapter`` consumes the intents and dispatches them through ABIDES's
``placeLimitOrder`` / ``cancelOrder`` helpers.

Action vector semantics
-----------------------
The action is a 4-tuple ``(bid_offset, ask_offset, bid_size, ask_size)``:

* ``bid_offset``, ``ask_offset`` — continuous floats in **cents**, clipped to
  ``>= 0``, then quantized to ``tick_size`` cents via ``round()``.
* ``bid_size``, ``ask_size`` — continuous floats in ``[0, max_size]``, then
  ``int(round(...))``.
* Quote prices: ``bid_price = mid_price - bid_offset_ticks``,
  ``ask_price = mid_price + ask_offset_ticks``.
* If ``bid_price >= ask_price`` post-rounding, both sides are zeroed (no place).
* Any side with non-positive price or non-positive size is dropped.
* NaN / inf in any field is treated as 0.

Reconciliation policy: naive cancel-all-then-replace each step. Simple and
fast in simulator time; the smarter "only cancel if price/size differs" is
left as future work.

Pavel (Module C) must import ``TICK_SIZE_CENTS`` from this module rather
than redefining it — keeps action-vector semantics in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TICK_SIZE_CENTS = 1


@dataclass(frozen=True)
class PlaceIntent:
    symbol: str
    quantity: int
    is_buy_order: bool
    limit_price: int          # cents
    tag: str | None = None


@dataclass(frozen=True)
class CancelIntent:
    order_id: int


@dataclass(frozen=True)
class RestingOrder:
    order_id: int
    is_buy_order: bool
    limit_price: int
    quantity: int


Intent = PlaceIntent | CancelIntent


def _safe_float(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def _quantize_offset(raw: float, tick_size: int) -> int:
    raw = max(_safe_float(raw), 0.0)
    return int(round(raw / tick_size)) * tick_size


def _quantize_size(raw: float, max_size: int) -> int:
    raw = max(_safe_float(raw), 0.0)
    raw = min(raw, max_size)
    return int(round(raw))


def translate_action(
    action: tuple[float, float, float, float],
    agent_id: int,
    symbol: str,
    mid_price: int,
    resting_orders: list[RestingOrder],
    max_size: int = 100,
    tick_size: int = TICK_SIZE_CENTS,
) -> list[Intent]:
    """Translate a 4-tuple action into ABIDES order/cancel intents.

    See module docstring for action-vector semantics.

    The ``agent_id`` argument is accepted for forward compatibility (e.g.
    tagging orders per-agent) but is not used by the current intent shape.
    """
    bid_off_raw, ask_off_raw, bid_size_raw, ask_size_raw = action

    bid_off = _quantize_offset(bid_off_raw, tick_size)
    ask_off = _quantize_offset(ask_off_raw, tick_size)
    bid_qty = _quantize_size(bid_size_raw, max_size)
    ask_qty = _quantize_size(ask_size_raw, max_size)

    bid_price = mid_price - bid_off
    ask_price = mid_price + ask_off

    if bid_price <= 0 or bid_qty <= 0:
        bid_price, bid_qty = 0, 0
    if ask_price <= 0 or ask_qty <= 0:
        ask_price, ask_qty = 0, 0
    if bid_qty > 0 and ask_qty > 0 and bid_price >= ask_price:
        bid_qty, ask_qty = 0, 0

    intents: list[Intent] = [CancelIntent(order_id=o.order_id) for o in resting_orders]

    if bid_qty > 0:
        intents.append(
            PlaceIntent(
                symbol=symbol,
                quantity=bid_qty,
                is_buy_order=True,
                limit_price=bid_price,
            )
        )
    if ask_qty > 0:
        intents.append(
            PlaceIntent(
                symbol=symbol,
                quantity=ask_qty,
                is_buy_order=False,
                limit_price=ask_price,
            )
        )
    return intents
