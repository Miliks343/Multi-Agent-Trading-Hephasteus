"""Action translator: continuous action vector -> ABIDES order/cancel intents.

Pure Python; no ABIDES import. The thin TradingAgent adapter in
``agent_adapter`` consumes the intents and dispatches them through ABIDES's
``placeLimitOrder`` / ``cancelOrder`` helpers.

Two action-vector layouts
-------------------------
``gated=False`` (legacy, 4-tuple) is kept so that trajectories and checkpoints
produced before the gated space landed can still be replayed and analysed. New
work should use ``gated=True``.

**Gated (6-tuple, preferred):**
``(bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size)``.

A side quotes when its gate is ``> 0``, and posts nothing otherwise. When a
side is on, its offset is floored at ``min_offset_ticks`` and its size at
``min_quote_size``, so a quote that reaches the book is always a *real* quote.

The point of the gates is that withdrawal has to be a **decision** rather than
a rounding artifact. In the legacy layout the action space floor is 0 on every
field, sub-tick offsets and sub-unit sizes both round to zero, and a policy
initialised near zero therefore starts in a region where no parameter change
alters the posted quote - and so no parameter change alters the reward. That
region has zero gradient and the optimiser cannot feel its way out of it;
measured on the 2026-09 grid, the policy posted a genuine two-sided market on
about 2% of steps. Gating separates "do not quote" (one dimension, with a
gradient) from "quote here, this big" (the rest), so withdrawal stays available
- it is the behaviour Glosten-Milgrom predicts - while ceasing to be the
default the policy is initialised into.

A second property falls out for free: with both gates on and both offsets at
least one tick, ``bid_price <= mid - 1 < mid + 1 <= ask_price``, so the two
sides can no longer cross. The legacy layout self-voided on 54.7% of steps in
the ``min_size_10`` cell; under gating that is structurally impossible.

**Legacy (4-tuple):**
``(bid_offset, ask_offset, bid_size, ask_size)``:

* ``bid_offset``, ``ask_offset`` — continuous floats in **cents**, clipped to
  ``>= 0``, then quantized to ``tick_size`` cents via ``round()``.
* ``bid_size``, ``ask_size`` — continuous floats in ``[0, max_size]``, then
  ``int(round(...))``.
* Quote prices: ``bid_price = mid_price - bid_offset_ticks``,
  ``ask_price = mid_price + ask_offset_ticks``.
* If ``bid_price >= ask_price`` post-rounding, both sides are zeroed (no place).
* Any side with non-positive price or non-positive size is dropped.
* NaN / inf in any field is treated as 0.

Common to both layouts: any side with non-positive price or size is dropped,
NaN/inf is treated as 0, and in the legacy layout both sides are voided if the
rounded bid would cross the rounded ask.

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

# Floors applied to a side that a gate has switched ON. One tick and one share
# are the smallest quantities the exchange can represent, so these are the
# weakest floors that still guarantee a gated-on side reaches the book.
DEFAULT_MIN_OFFSET_TICKS = 1
DEFAULT_MIN_QUOTE_SIZE = 1


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


def _gated_side(
    gate_raw: float,
    off_raw: float,
    size_raw: float,
    *,
    max_size: int,
    tick_size: int,
    min_offset_ticks: int,
    min_quote_size: int,
) -> tuple[int, int]:
    """Resolve one side of a gated action into ``(offset_cents, quantity)``.

    A side is off unless its gate is strictly positive. SB3's Gaussian policy
    initialises with mean ~0, so a threshold at 0 puts a fresh policy at the
    decision boundary - it quotes on roughly half of steps at initialisation
    and therefore sees reward from both choices immediately, which is exactly
    what the legacy layout denied it.

    A side that is on is floored, so it cannot round its way back to silence.
    """
    if not (_safe_float(gate_raw) > 0.0):
        return 0, 0
    off = max(_quantize_offset(off_raw, tick_size), min_offset_ticks * tick_size)
    qty = min(max(_quantize_size(size_raw, max_size), min_quote_size), max_size)
    return off, qty


def translate_action(
    action: tuple[float, ...],
    agent_id: int,
    symbol: str,
    mid_price: int,
    resting_orders: list[RestingOrder],
    max_size: int = 100,
    tick_size: int = TICK_SIZE_CENTS,
    *,
    gated: bool = False,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
) -> list[Intent]:
    """Translate an action vector into ABIDES order/cancel intents.

    ``gated=True`` expects the 6-tuple
    ``(bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size)``;
    ``gated=False`` expects the legacy 4-tuple. See the module docstring for
    why the gated layout exists.

    The ``agent_id`` argument is accepted for forward compatibility (e.g.
    tagging orders per-agent) but is not used by the current intent shape.
    """
    expected = 6 if gated else 4
    if len(action) != expected:
        raise ValueError(
            f"gated={gated} expects a {expected}-tuple action, "
            f"got length {len(action)}"
        )

    if gated:
        bid_gate, ask_gate, bid_off_raw, ask_off_raw, bid_size_raw, ask_size_raw = action
        bid_off, bid_qty = _gated_side(
            bid_gate, bid_off_raw, bid_size_raw, max_size=max_size,
            tick_size=tick_size, min_offset_ticks=min_offset_ticks,
            min_quote_size=min_quote_size)
        ask_off, ask_qty = _gated_side(
            ask_gate, ask_off_raw, ask_size_raw, max_size=max_size,
            tick_size=tick_size, min_offset_ticks=min_offset_ticks,
            min_quote_size=min_quote_size)
    else:
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
    # Under gating with min_offset_ticks >= 1 this can never fire, since
    # bid_price <= mid - 1 < mid + 1 <= ask_price. Kept as a guard so that
    # min_offset_ticks=0 (which reopens the legacy failure) still degrades
    # safely rather than posting a crossed quote.
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
