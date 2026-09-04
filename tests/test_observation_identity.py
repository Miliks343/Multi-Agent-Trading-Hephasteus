"""Tests for the agent-identity one-hot appended to the observation.

Measured across 4,350 agent pairs, the parameter-shared agents posted
identical quotes on 97-99.9% of steps: of the 4*K+4 features, only inventory
and cash could differ between two agents watching the same book, and that was
not enough to make them distinguishable
(`notes/agent_divergence_results.md`). The one-hot is what lets a shared
network condition on which agent it is.
"""
import numpy as np
import pytest

from marl_lob.observation_extractor import (
    describe_obs,
    extract_obs,
    obs_vector_size,
)

BOOK = dict(known_bids=[(9_990, 50), (9_980, 30)],
            known_asks=[(10_010, 40), (10_020, 20)])


def _obs(**kw):
    return extract_obs(inventory=0, cash=10_000_000, current_time=5,
                       mkt_open=0, mkt_close=10, **BOOK, **kw)


# ── sizing ──────────────────────────────────────────────────────────────────

def test_default_layout_is_unchanged():
    """Width 0 must reproduce the pre-2026-09 vector exactly, so legacy
    checkpoints keep evaluating against the contract they were trained on."""
    assert obs_vector_size() == 44
    assert _obs().shape == (44,)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_width_extends_the_vector(width):
    assert obs_vector_size(10, width) == 44 + width
    assert _obs(agent_index=0, agent_id_width=width).shape == (44 + width,)


def test_negative_width_is_treated_as_absent():
    assert obs_vector_size(10, -3) == 44


# ── contents ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_one_hot_marks_the_right_agent(idx):
    tail = _obs(agent_index=idx, agent_id_width=4)[-4:]
    assert tail.tolist() == [1.0 if i == idx else 0.0 for i in range(4)]


def test_agents_differ_even_with_identical_book_and_holdings():
    """The point of the change: two agents seeing the same book and holding
    the same position previously produced bit-identical observations."""
    a = _obs(agent_index=0, agent_id_width=2)
    b = _obs(agent_index=1, agent_id_width=2)
    assert not np.array_equal(a, b)
    assert np.array_equal(a[:-2], b[:-2]), "only the identity tail may differ"


def test_without_the_one_hot_they_are_identical():
    """The control for the test above — this is the measured failure."""
    assert np.array_equal(_obs(agent_index=0), _obs(agent_index=1))


def test_body_is_untouched_by_the_suffix():
    assert np.array_equal(_obs(), _obs(agent_index=1, agent_id_width=3)[:-3])


def test_index_outside_the_pinned_width_is_all_zeros():
    """A pinned width smaller than n_agents leaves the extra agents
    unidentifiable rather than raising or corrupting a neighbour's bit."""
    assert _obs(agent_index=9, agent_id_width=4)[-4:].tolist() == [0.0] * 4


def test_missing_index_is_all_zeros():
    assert _obs(agent_id_width=3)[-3:].tolist() == [0.0] * 3


def test_one_hot_survives_the_final_clip():
    """extract_obs clips to [-1, 1]; a hot bit of 1.0 must come through."""
    assert _obs(agent_index=1, agent_id_width=2)[-1] == 1.0


# ── names ───────────────────────────────────────────────────────────────────

def test_describe_obs_tracks_the_width():
    names = describe_obs(10, 3)
    assert len(names) == obs_vector_size(10, 3)
    assert names[-3:] == ["agent_id_0", "agent_id_1", "agent_id_2"]


def test_describe_obs_default_is_unchanged():
    assert len(describe_obs()) == 44
    assert describe_obs()[-1] == "time_to_close_norm"
