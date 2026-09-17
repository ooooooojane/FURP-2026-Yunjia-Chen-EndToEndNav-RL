# Upstream modifications

The training and simulation code for this project is the open-source project
**[`reiniscimurs/DRL-robot-navigation-IR-SIM`](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM)**
(2D LiDAR navigation with deep RL). We use it as-is rather than vendoring a copy,
so that the upstream provenance stays traceable.

Everything we **changed** in it is contained in a single patch:

```
upstream_modifications.patch
```

**Base commit:** `31e1a4d511bb607e6ea38f4f8fccc842fbc7dd77`
(`Update README with link to MARL implementation link`, the `master` HEAD we forked from)

**Scope:** 4 files, +201 / −150

| File | Change |
|---|---|
| `robot_nav/SIM_ENV/sim.py` | +50 / −78 |
| `robot_nav/rl_train.py` | +127 / −72 |
| `robot_nav/worlds/circle_world.yaml` | +12 / −0 |
| `robot_nav/worlds/cross_world.yaml` | +12 / −0 |

## How to apply

```bash
git clone https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM.git
cd DRL-robot-navigation-IR-SIM
git checkout 31e1a4d511bb607e6ea38f4f8fccc842fbc7dd77
git apply /path/to/upstream_modifications.patch
```

## What the changes actually do

### 1. Deterministic scenario seeding — `SIM_ENV/sim.py`

`__init__` gains `reward_mode` and `seed` parameters. When a seed is given, it
seeds **three** independent RNGs *before* the environment is constructed and
also forwards `seed=` into `irsim.make()`:

```python
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    from irsim.util.random import set_seed
    set_seed(seed)
```

This matters more than it looks. Seeding only IR-SIM's internal RNG is **not**
sufficient: obstacle placement also consumes the Python `random` and NumPy
global RNGs, so different processes produced different layouts. Seeding all
three makes obstacle layouts reproducible **across processes**, which is what
makes the paper's paired-scenario protocol possible — every policy is evaluated
on byte-identical scenarios, and paired statistical tests (McNemar) are valid.

### 2. Configurable reward function — `SIM_ENV/sim.py`

`get_reward` changes from a `@staticmethod` to an instance method taking a
`distance` argument, with four modes: `default` / `dense` / `sparse` / `harsh`.

- `default` is **numerically identical** to upstream
  (`action[0] - abs(action[1])/2 - r_obstacle(min(scan))/2`, goal `+100`,
  collision `−100`), so pre-existing checkpoints remain valid.
- `dense` adds progress shaping: `base + (prev_distance - distance) * 5.0`
- `sparse` returns `0.0` for all non-terminal steps
- `harsh` doubles the collision penalty to `−200`

This backs the reward-sensitivity study (Weekly logs W4–W5).

### 3. Training CLI and model factory — `rl_train.py`

- Adds an `argparse` CLI: `--algo {CNNTD3,TD3,DDPG,SAC,PPO}`, `--epochs`,
  `--save-every`, `--no-plot`, `--reward {default,dense,sparse,harsh}`, `--resume`.
  Hyperparameters are otherwise untouched from upstream (state_dim 185,
  action_dim 2, 70 eps/epoch, batch 64, max_steps 300).
- Adds a `create_model(algo, ...)` factory dispatching to the five algorithms,
  with reward-tagged checkpoint names (`f"{algo}_{reward_mode}"`).
- **PPO correction:** forces `train_every_n = 20`, `training_iterations = 10`.
  Upstream's `train_every_n=2` is catastrophic for an on-policy algorithm —
  it re-chews a ~200-transition rollout 80 times, and PPO collapsed to ~1% goal
  rate. This is a genuine finding of the project, documented in W4.
- Import fix: `from utils import get_buffer` → `from robot_nav.utils import get_buffer`
  (upstream relied on the current working directory being `robot_nav/`).

### 4. Explicit LiDAR sensor — `worlds/{circle,cross}_world.yaml`

Both worlds gain an explicit 180-beam, 7 m, noisy LiDAR block
(`range_max: 7`, `std: 0.08`, `angle_std: 0.1`), matching `robot_world.yaml`.
Previously these two worlds silently inherited defaults, so they were not the
same sensor model as the world used for the main experiments. Pure addition —
nothing was removed.

## Verification

The patch was reverse-apply-checked against the working tree it was generated
from, so it reproduces those exact 4 file states.
