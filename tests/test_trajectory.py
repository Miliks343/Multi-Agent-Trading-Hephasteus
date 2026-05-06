from __future__ import annotations

import numpy as np
import pytest

from marl_lob.trajectory import Fill, Trajectory


def test_from_tuples_records_signed_fills():
    rows = [
        (0.0, 0, 1_000_000, 10_000, 0,  0),
        (1.0, 5,   950_000, 10_010, 5,  10_011),   # buy 5 @ 10011 (mid was 10010)
        (2.0, 3,   970_000, 10_020, -2, 10_019),   # sell 2 @ 10019 (mid was 10020)
        (3.0, 3,   970_000, 10_030, 0,  0),
    ]
    traj = Trajectory.from_tuples(rows)
    assert len(traj) == 4
    assert len(traj.fills) == 2
    assert traj.fills[0] == Fill(timestamp=1.0, side=1, price=10_011, quantity=5)
    assert traj.fills[1] == Fill(timestamp=2.0, side=-1, price=10_019, quantity=2)


def test_from_tuples_fill_price_distinct_from_mid():
    """Pin the contract: Fill.price comes from fill_price, not mid_price."""
    rows = [(0.0, 0, 0, 99_999, 1, 100_005)]   # mid 99999, fill at 100005
    traj = Trajectory.from_tuples(rows)
    assert traj.fills[0].price == 100_005
    assert traj.mid_price[0] == 99_999


def test_from_tuples_empty():
    traj = Trajectory.from_tuples([])
    assert len(traj) == 0
    assert traj.fills == []


def test_equity_marks_to_market():
    traj = Trajectory(
        timestamps=np.array([0.0, 1.0]),
        inventory=np.array([10, 5], dtype=np.int64),
        cash=np.array([100_000, 150_000], dtype=np.int64),
        mid_price=np.array([10_000, 10_000], dtype=np.int64),
    )
    eq = traj.equity()
    # cash + inventory * mid
    assert eq.tolist() == [100_000 + 10 * 10_000, 150_000 + 5 * 10_000]


def test_dtype_guards_reject_floats():
    with pytest.raises(TypeError):
        Trajectory(
            timestamps=np.array([0.0]),
            inventory=np.array([0], dtype=np.int64),
            cash=np.array([1.0]),  # float — should be cents
            mid_price=np.array([10_000], dtype=np.int64),
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Trajectory(
            timestamps=np.array([0.0, 1.0]),
            inventory=np.array([0], dtype=np.int64),
            cash=np.array([0, 0], dtype=np.int64),
            mid_price=np.array([10_000, 10_000], dtype=np.int64),
        )
