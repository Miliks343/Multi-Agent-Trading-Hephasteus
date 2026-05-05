from __future__ import annotations

import numpy as np
import pytest

from marl_lob.trajectory import Fill, Trajectory


def test_from_tuples_records_signed_fills():
    rows = [
        (0.0, 0, 1_000_000, 10_000, 0),
        (1.0, 5, 950_000, 10_010, 5),    # buy 5 @ 10010
        (2.0, 3, 970_000, 10_020, -2),   # sell 2 @ 10020
        (3.0, 3, 970_000, 10_030, 0),
    ]
    traj = Trajectory.from_tuples(rows)
    assert len(traj) == 4
    assert len(traj.fills) == 2
    assert traj.fills[0] == Fill(timestamp=1.0, side=1, price=10_010, quantity=5)
    assert traj.fills[1] == Fill(timestamp=2.0, side=-1, price=10_020, quantity=2)


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
