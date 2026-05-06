"""
Task A — Observation Extractor
================================
Converts the ABIDES order book state + agent state into a fixed-size 1D numpy
vector that can be fed directly into a PettingZoo / SB3 policy network.

Observation vector layout (total: 4*K + 4 elements, default K=10 → 44 dims)
─────────────────────────────────────────────────────────────────────────────
Index       Feature
─────────────────────────────────────────────────────────────────────────────
0 .. K-1    Bid prices  (top K levels), normalised by mid-price, descending
K .. 2K-1   Bid sizes   (top K levels), normalised by max_size
2K .. 3K-1  Ask prices  (top K levels), normalised by mid-price, ascending
3K .. 4K-1  Ask sizes   (top K levels), normalised by max_size
4K          Inventory   (signed shares), normalised by max_inventory
4K+1        Cash        (cents),         normalised by starting_cash
4K+2        Spread      (cents / mid),   dimensionless
4K+3        Time-to-close (seconds),     normalised by session_duration_s
─────────────────────────────────────────────────────────────────────────────

All values are clipped to [-1, 1] before returning.

Units / conventions inherited from ABIDES
─────────────────────────────────────────
- Prices are in CENTS  (e.g. $1000.00 → 100_000)
- known_bids / known_asks: list of (price_cents, quantity) tuples, best first
- Holdings cash ('CASH' key) is in cents
- NanosecondTime timestamps are int64 nanoseconds since epoch
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases (mirror ABIDES internals; no import needed here)
# ──────────────────────────────────────────────────────────────────────────────
PriceLevels = List[Tuple[int, int]]   # [(price_cents, qty), ...]
NanosecondTime = int

# ──────────────────────────────────────────────────────────────────────────────
# Constants — change here to resize the obs vector (update PettingZoo spaces too)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_K            = 10        # number of price levels each side
DEFAULT_MAX_INVENTORY = 1_000    # shares  — used for normalisation
DEFAULT_MAX_SIZE      = 1_000    # shares  — used for normalising queue sizes
DEFAULT_STARTING_CASH = 10_000_000  # cents ($100 000)
NS_PER_SECOND         = 1_000_000_000


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def obs_vector_size(k: int = DEFAULT_K) -> int:
    """Returns the length of the observation vector for a given depth K."""
    return 4 * k + 4


def extract_obs(
    known_bids: PriceLevels,
    known_asks: PriceLevels,
    inventory: int,
    cash: int,
    current_time: NanosecondTime,
    mkt_open: NanosecondTime,
    mkt_close: NanosecondTime,
    k: int = DEFAULT_K,
    max_inventory: int = DEFAULT_MAX_INVENTORY,
    max_size: int = DEFAULT_MAX_SIZE,
    starting_cash: int = DEFAULT_STARTING_CASH,
) -> np.ndarray:
    """
    Convert ABIDES book state + agent state → 1D float32 numpy array.

    Parameters
    ----------
    known_bids : list of (price_cents, qty) — best bid first, from agent.known_bids[symbol]
    known_asks : list of (price_cents, qty) — best ask first, from agent.known_asks[symbol]
    inventory  : signed share count (positive = long), from agent.holdings[symbol]
    cash       : cash in cents, from agent.holdings['CASH']
    current_time, mkt_open, mkt_close : NanosecondTime ints from ABIDES kernel
    k          : number of price levels to include each side
    max_inventory, max_size, starting_cash : normalisation constants

    Returns
    -------
    obs : np.ndarray, shape (4*k+4,), dtype float32, values in [-1, 1]
    """
    # ── mid-price (cents) ────────────────────────────────────────────────────
    best_bid = known_bids[0][0] if known_bids else None
    best_ask = known_asks[0][0] if known_asks else None

    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
    elif best_bid is not None:
        mid = float(best_bid)
    elif best_ask is not None:
        mid = float(best_ask)
    else:
        mid = 1.0   # fallback — avoids division by zero at open

    # ── book features ────────────────────────────────────────────────────────
    bid_prices = _pad_prices(known_bids, k, mid, side="bid")
    bid_sizes  = _pad_sizes(known_bids, k, max_size)
    ask_prices = _pad_prices(known_asks, k, mid, side="ask")
    ask_sizes  = _pad_sizes(known_asks, k, max_size)

    # ── scalar features ──────────────────────────────────────────────────────
    inv_norm    = np.clip(inventory / max_inventory, -1.0, 1.0)
    cash_norm   = np.clip(cash / starting_cash - 1.0, -1.0, 1.0)  # 0 at start

    spread_norm = 0.0
    if best_bid is not None and best_ask is not None and mid > 0:
        spread_norm = np.clip((best_ask - best_bid) / mid, 0.0, 1.0)

    session_ns  = max(mkt_close - mkt_open, 1)
    elapsed_ns  = max(current_time - mkt_open, 0)
    ttc_norm    = np.clip(1.0 - elapsed_ns / session_ns, 0.0, 1.0)  # 1=open, 0=close

    # ── assemble ─────────────────────────────────────────────────────────────
    obs = np.concatenate([
        bid_prices, bid_sizes,
        ask_prices, ask_sizes,
        [inv_norm, cash_norm, spread_norm, ttc_norm],
    ]).astype(np.float32)

    return np.clip(obs, -1.0, 1.0)


def describe_obs(k: int = DEFAULT_K) -> List[str]:
    """
    Returns a list of human-readable feature names in the same order as
    extract_obs(). Useful for debugging and documentation.
    """
    names = []
    for i in range(k):
        names.append(f"bid_price_L{i+1}_norm")
    for i in range(k):
        names.append(f"bid_size_L{i+1}_norm")
    for i in range(k):
        names.append(f"ask_price_L{i+1}_norm")
    for i in range(k):
        names.append(f"ask_size_L{i+1}_norm")
    names += ["inventory_norm", "cash_norm", "spread_norm", "time_to_close_norm"]
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pad_prices(
    levels: PriceLevels,
    k: int,
    mid: float,
    side: str,
) -> np.ndarray:
    """
    Extract up to k price levels, normalise relative to mid, pad missing levels.

    For bids: (price - mid) / mid  → negative values (below mid)
    For asks: (price - mid) / mid  → positive values (above mid)
    Missing levels are filled with ±1 (far away from mid).
    """
    out = np.zeros(k, dtype=np.float64)
    fill = -1.0 if side == "bid" else 1.0

    for i in range(k):
        if i < len(levels) and mid > 0:
            price = levels[i][0]
            out[i] = np.clip((price - mid) / mid, -1.0, 1.0)
        else:
            out[i] = fill

    return out


def _pad_sizes(
    levels: PriceLevels,
    k: int,
    max_size: int,
) -> np.ndarray:
    """Extract up to k queue sizes, normalise by max_size, pad missing with 0."""
    out = np.zeros(k, dtype=np.float64)

    for i in range(k):
        if i < len(levels):
            out[i] = np.clip(levels[i][1] / max_size, 0.0, 1.0)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic book: 3 bid levels, 3 ask levels
    mock_bids = [(99_950, 100), (99_900, 200), (99_850, 150)]
    mock_asks = [(100_050, 80),  (100_100, 300), (100_150, 120)]

    mkt_open  = 0
    mkt_close = 3_600 * NS_PER_SECOND     # 1 hour
    current   = 1_800 * NS_PER_SECOND     # halfway through

    obs = extract_obs(
        known_bids=mock_bids,
        known_asks=mock_asks,
        inventory=50,
        cash=9_900_000,
        current_time=current,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
    )

    names = describe_obs()
    print(f"Observation vector — shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"{'Feature':<30} {'Value':>10}")
    print("─" * 42)
    for name, val in zip(names, obs):
        print(f"{name:<30} {val:>10.4f}")

    assert obs.shape == (obs_vector_size(),), "Shape mismatch!"
    assert obs.dtype == np.float32, "Wrong dtype!"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Values out of [-1, 1]!"
    print("\n✓ All assertions passed.")
