"""Smaller RMSC03 variant for fast local runs.

Wraps upstream `abides_markets.configs.rmsc03` with reduced agent counts
that still produce meaningful order-book activity:

  1 Exchange | 1 Adaptive MM | 50 Value | 10 Momentum | 2000 Noise

Noise count is the volume driver — drop below ~1000 and the L1 barely
moves. Values were calibrated on a 1h sim (seed=42) producing ~140
post-dropna L1 events, ~3s wallclock.

Note: upstream rmsc03 imports POVExecutionAgent which is missing from the
current JPMC fork. See README install section for the small patch.
"""
from abides_markets.configs import rmsc03


def build_config(start_time="09:30:00", end_time="10:30:00", seed=1, **kwargs):
    return rmsc03.build_config(
        start_time=start_time,
        end_time=end_time,
        seed=seed,
        num_noise_agents=kwargs.pop("num_noise_agents", 2000),
        num_value_agents=kwargs.pop("num_value_agents", 50),
        num_momentum_agents=kwargs.pop("num_momentum_agents", 10),
        execution_agents=kwargs.pop("execution_agents", False),
        **kwargs,
    )
