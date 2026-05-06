"""Pure-helper tests for MarlLobEnv reward + equity arithmetic.

Full kernel-driven integration tests live in chunk 6 alongside the
PettingZoo conformance check. These tests pin the formulas in isolation
so a sign error or off-by-one in the reward function fails fast,
without spinning up ABIDES.
"""
from __future__ import annotations

from marl_lob.env import MarlLobEnv


# ── _equity_from_traj_row ───────────────────────────────────────────────────

def test_equity_long_inventory():
    # row = (ts, inv, cash, mid, fill_qty, fill_price)
    row = (1.0, 10, 1_000_000, 100_000, 0, 0)
    # cash + inv*mid = 1_000_000 + 10*100_000 = 2_000_000
    assert MarlLobEnv._equity_from_traj_row(row) == 2_000_000


def test_equity_short_inventory():
    row = (1.0, -5, 1_500_000, 100_000, 0, 0)
    # 1_500_000 + (-5)*100_000 = 1_000_000
    assert MarlLobEnv._equity_from_traj_row(row) == 1_000_000


def test_equity_zero_inventory_is_just_cash():
    row = (1.0, 0, 9_999_999, 100_000, 0, 0)
    assert MarlLobEnv._equity_from_traj_row(row) == 9_999_999


# ── _compute_reward ─────────────────────────────────────────────────────────

def test_reward_pure_pnl_no_inventory_no_termination():
    """No inventory, no termination → reward is just ΔPnL."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_500,
        inventory=0, inventory_penalty=1e-4,
        terminated=False, termination_penalty=-1.0,
    )
    assert r == 500.0


def test_reward_inventory_penalty_subtracts():
    """ΔPnL=0, inv=100, λ=1e-4 → reward = -0.01."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_000,
        inventory=100, inventory_penalty=1e-4,
        terminated=False, termination_penalty=-1.0,
    )
    assert r == -0.01


def test_reward_inventory_penalty_symmetric_long_vs_short():
    """λ·|inv| is symmetric: long 50 and short 50 incur the same penalty."""
    common = dict(prev_equity=0, equity=0, inventory_penalty=1e-3,
                  terminated=False, termination_penalty=-1.0)
    long = MarlLobEnv._compute_reward(inventory=50, **common)
    short = MarlLobEnv._compute_reward(inventory=-50, **common)
    assert long == short == -0.05


def test_reward_long_inventory_with_price_up_is_positive_pnl_minus_penalty():
    """Held 10 long, mid moved 50 cents in our favor → +500 ΔPnL minus penalty."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_500,   # +500 cents
        inventory=10, inventory_penalty=1.0,        # large λ for visibility
        terminated=False, termination_penalty=-1.0,
    )
    # 500 - 1.0*10 = 490
    assert r == 490.0


def test_reward_termination_adds_penalty():
    """Termination penalty stacks on top of the ΔPnL/penalty arithmetic."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_000,
        inventory=0, inventory_penalty=1e-4,
        terminated=True, termination_penalty=-1.0,
    )
    assert r == -1.0


def test_reward_termination_with_pnl_and_penalty():
    """All three terms compose additively."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=999_500,    # -500 ΔPnL
        inventory=200, inventory_penalty=1e-3,    # -0.2 penalty
        terminated=True, termination_penalty=-5.0,
    )
    # -500 - 0.2 - 5.0 = -505.2
    assert r == -505.2
