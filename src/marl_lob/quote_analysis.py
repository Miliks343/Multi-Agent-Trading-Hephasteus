"""Replay saved action vectors through the live quantisation.

Delta-equity says how much money a policy made; it does not say whether the
policy was making markets, or whether two agents were doing anything
different from each other. Both questions are answered by pushing the saved
``actions`` array through the same rounding, flooring and voiding that
``actions.translate_action`` applies before orders reach ABIDES.

An action vector is not a quote. Offsets are continuous cents rounded to whole
ticks and sizes are rounded to integers, so a policy emitting sub-tick offsets
and sub-unit sizes posts nothing at all while still producing a full-looking
action array.

**Two layouts coexist on disk.** Trajectories written before the gated action
space have 4 columns; those written after have 6. Every reader here dispatches
on width, because a 6-wide array read as the legacy layout reports the gates as
offsets - wrong, and silently so.
"""
from __future__ import annotations

import numpy as np

from .actions import (
    DEFAULT_MIN_OFFSET_TICKS,
    DEFAULT_MIN_QUOTE_SIZE,
    TICK_SIZE_CENTS,
    _gated_side,
    _quantize_offset,
    _quantize_size,
)

LEGACY_ACTION_NAMES = ("bid_offset", "ask_offset", "bid_size", "ask_size")
GATED_ACTION_NAMES = ("bid_gate", "ask_gate", "bid_offset", "ask_offset",
                      "bid_size", "ask_size")
QUOTE_NAMES = LEGACY_ACTION_NAMES


def action_names(width: int) -> tuple[str, ...]:
    """Column labels for an action array, chosen by its width."""
    if width == 6:
        return GATED_ACTION_NAMES
    if width == 4:
        return LEGACY_ACTION_NAMES
    raise ValueError(f"unrecognised action width {width}; expected 4 or 6")


def is_gated(actions: np.ndarray) -> bool:
    """True when the array uses the 6-column gated layout."""
    actions = np.asarray(actions)
    width = actions.shape[1] if actions.ndim == 2 else 0
    action_names(width)
    return width == 6


def effective_quotes(
    actions: np.ndarray,
    max_size: int = 100,
    *,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
) -> np.ndarray:
    """Quantise an action array into the quotes the exchange would see.

    Returns an (n, 4) int array of ``(bid_off, ask_off, bid_qty, ask_qty)``.
    Prices are ``mid -/+ offset``, so the cross test ``bid_price >= ask_price``
    reduces to ``-bid_off >= ask_off`` and needs no mid price.
    """
    actions = np.asarray(actions, dtype=float)
    gated = is_gated(actions)

    out = np.zeros((len(actions), 4), dtype=int)
    for i, row in enumerate(actions):
        if gated:
            b_gate, a_gate, b_off, a_off, b_sz, a_sz = row
            bo, bq = _gated_side(
                b_gate, b_off, b_sz, max_size=max_size,
                tick_size=TICK_SIZE_CENTS, min_offset_ticks=min_offset_ticks,
                min_quote_size=min_quote_size)
            ao, aq = _gated_side(
                a_gate, a_off, a_sz, max_size=max_size,
                tick_size=TICK_SIZE_CENTS, min_offset_ticks=min_offset_ticks,
                min_quote_size=min_quote_size)
        else:
            b_off, a_off, b_sz, a_sz = row
            bo = _quantize_offset(b_off, TICK_SIZE_CENTS)
            ao = _quantize_offset(a_off, TICK_SIZE_CENTS)
            bq = _quantize_size(b_sz, max_size)
            aq = _quantize_size(a_sz, max_size)
        if bq > 0 and aq > 0 and (-bo) >= ao:
            bq = aq = 0
        if bq <= 0:
            bo, bq = 0, 0
        if aq <= 0:
            ao, aq = 0, 0
        out[i] = (bo, ao, bq, aq)
    return out


def participation_stats(actions: np.ndarray, max_size: int = 100) -> dict:
    """How often did this policy actually put a market on the book?

    ``two_sided_pct`` is the headline: the share of steps carrying a genuine
    two-sided quote. ``crossed_pct`` is the legacy failure mode - both sides
    voided because the rounded offsets crossed - and is 0 by construction under
    gating with ``min_offset_ticks >= 1``.
    """
    q = effective_quotes(actions, max_size)
    bid_off, ask_off, bid_qty, ask_qty = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    raw = np.asarray(actions, dtype=float)
    gated = is_gated(raw)

    bid_ok, ask_ok = bid_qty > 0, ask_qty > 0
    two_sided = bid_ok & ask_ok
    any_quote = bid_ok | ask_ok

    # Recover the crossings the quantisation voided: a voided pair leaves zero
    # quantity on both sides, so it has to be recomputed from the raw offsets.
    if gated:
        crossed = np.zeros(len(raw), dtype=bool)
        withdrawn = ~((raw[:, 0] > 0) | (raw[:, 1] > 0))
    else:
        rb = np.array([_quantize_offset(x, TICK_SIZE_CENTS) for x in raw[:, 0]])
        ra = np.array([_quantize_offset(x, TICK_SIZE_CENTS) for x in raw[:, 1]])
        qb = np.array([_quantize_size(x, max_size) for x in raw[:, 2]])
        qa = np.array([_quantize_size(x, max_size) for x in raw[:, 3]])
        crossed = (qb > 0) & (qa > 0) & ((-rb) >= ra)
        withdrawn = ~any_quote & ~crossed

    return {
        "two_sided_pct": float(two_sided.mean() * 100),
        "any_quote_pct": float(any_quote.mean() * 100),
        "crossed_pct": float(crossed.mean() * 100),
        "withdrawn_pct": float(withdrawn.mean() * 100),
        "spread_cents": (float((bid_off + ask_off)[two_sided].mean())
                         if two_sided.any() else 0.0),
        "quote_size": (float((bid_qty + ask_qty)[two_sided].mean() / 2.0)
                       if two_sided.any() else 0.0),
        "gated": gated,
    }
