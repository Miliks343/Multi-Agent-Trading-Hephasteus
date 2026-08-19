"""The N-agent path was never exercised: every run used n_agents=2.

These are construction-only checks (MarlLobEnv builds no ABIDES kernel until
reset), so they need no `abides` marker and run in milliseconds.
"""
import numpy as np
import pytest

from marl_lob.env import MarlLobEnv
from marl_lob.observation_extractor import obs_vector_size


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_agent_set_scales_with_n_agents(n):
    env = MarlLobEnv(n_agents=n)
    assert env.n_agents == n
    assert env.possible_agents == [f"mm_{i}" for i in range(n)]
    assert len(set(env.possible_agents)) == n


@pytest.mark.parametrize("n", [1, 3])
def test_spaces_defined_for_every_agent(n):
    env = MarlLobEnv(n_agents=n)
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert obs_space.shape == (obs_vector_size(env.k),)
        assert act_space.shape == (4,)


def test_min_size_raises_action_low_on_size_dims_only():
    env = MarlLobEnv(n_agents=3, min_size=7)
    low = env.action_space("mm_0").low
    assert low[0] == 0.0 and low[1] == 0.0, "offset dims must stay at 0"
    assert low[2] == 7.0 and low[3] == 7.0, "size dims must be raised to min_size"


def test_max_offset_cents_caps_action_high():
    env = MarlLobEnv(n_agents=2, max_offset_cents=12, max_size=40)
    high = env.action_space("mm_1").high
    assert np.allclose(high, [12.0, 12.0, 40.0, 40.0])


def test_config_kwargs_are_retained_for_forwarding():
    env = MarlLobEnv(n_agents=2, config_kwargs={"num_value_agents": 150})
    assert env.config_kwargs["num_value_agents"] == 150
