# Multi-Agent-Trading-Hephasteus

Multi-agent reinforcement learning in limit order books (ABIDES + PettingZoo + SB3).

## Install

Requires Python 3.11. ABIDES's pinned deps don't build on 3.12, so use a 3.11 env (conda recommended).

```bash
# 1) conda env (conda-forge only — avoids the Anaconda-channel ToS)
conda create -n marl_lob -c conda-forge --override-channels python=3.11 -y
conda activate marl_lob
conda install -c conda-forge --override-channels pip -y

# 2) clone ABIDES (jpmc fork) as a sibling of this repo
git clone https://github.com/jpmorganchase/abides-jpmc-public.git ../abides-jpmc-public

# 3) ABIDES needs a small patch — see "ABIDES patch" below — apply it before installing

# 4) install ABIDES + this repo (relaxed deps; we skip gym/ray/pomegranate)
pip install coloredlogs numpy pandas psutil scipy tqdm matplotlib jupyter
pip install -e ../abides-jpmc-public/abides-core
pip install -e ../abides-jpmc-public/abides-markets
pip install -e ".[dev]"
```

### ABIDES patch

Upstream `abides-jpmc-public/abides-markets/abides_markets/configs/rmsc03.py` imports
`POVExecutionAgent` (which doesn't exist in this fork) and instantiates it
unconditionally. Two small edits:

```python
# at the top — replace the POVExecutionAgent import:
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
)
POVExecutionAgent = None  # not in this fork; only used when execution_agents=True
```

```python
# around line 250 — guard the instantiation:
if execution_agents and POVExecutionAgent is not None:
    pov_agent = POVExecutionAgent(
        # ...existing kwargs...
    )
    agents.append(pov_agent)
    agent_types.extend("ExecutionAgent")
    agent_count += 1
```

We don't use execution agents, so this is purely a "make the import work" patch.

### Skipped ABIDES deps

- `gym==0.18.0` — old setup metadata won't build on modern setuptools. Only used in `abides-gym` (which we don't need; we use PettingZoo + SB3).
- `ray==1.7.0` — only used in `abides-gym`.
- `pomegranate==0.14.5` — used by `OrderSizeModel` in rmsc04. We use rmsc03.

## Run tests

```bash
pytest
```

Should report 25 passed, 0 skipped.

## Smoke-test ABIDES

```bash
jupyter notebook notebooks/00_abides_hello.ipynb
```

Run all cells. The mid-price + spread plot should render. ~3s sim wallclock for 1 simulated hour.
