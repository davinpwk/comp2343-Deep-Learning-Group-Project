# Car Racing DQN — COMP3242 Project

This is my group project for COMP3242 Deep Learning at ANU. The plan was
to take what I built in **lab10** (a DQN that solves CartPole) and point
it at a harder problem — a tile-based car racing env — then push it
further by **transfer-learning** from normal grip to a slippery / icy
regime.

The whole point of the slippery toggle is to isolate the transfer-learning
signal. Same track, same physics, same action space, same observation
shape — the **only** thing that changes is a single friction constant
that controls how persistent lateral velocity is. If my pretrained agent
falls over when I flip the switch, I know it's because of slip, not
because of something incidental in the dynamics.

---

## Files

| File              | What I use it for                                                                            |
|-------------------|----------------------------------------------------------------------------------------------|
| `car_env.py`      | The Gymnasium env (`CarRacingEnv`). Started as a friend's pygame game, I refactored it.     |
| `track.py`        | Track generator and tile codes (`DIRT`, `ROAD`, `WALL`, `START`, `FINISH`).                  |
| `train_util.py`   | Lab10-style `set_seed` / `TrainConfig` / `EpisodeLogger`, adapted for this env.              |
| `dqn.ipynb`       | Main notebook. Same task layout as lab10 (Q-Net → Replay → DQN agent → train).               |
| `test_env.py`     | pytest-style smoke tests. `python test_env.py` to run them as a script.                      |
| `archive/legacy.py` | The original keyboard game I started from. Kept for reference, nothing imports it.         |

---

## Install

```bash
pip install gymnasium numpy pygame torch
```

Python 3.10+ (I'm on 3.13).

---

## Sanity-check the env without training

Before plugging it into DQN, I always make sure the env actually drives:

```bash
# Drive the car myself with WASD or arrow keys
python car_env.py

# Same, but with the icy regime
python car_env.py --slippery

# Random-action rollout — does the env run end-to-end without crashing?
python car_env.py --random
```

I should see a top-down view of the car, yellow LIDAR rays projecting
outward, and a HUD in the top-left (mode, step, action, reward, progress
%, speed). The keyboard-play mode uses `action_set="full"` so "no keys
pressed" maps to a valid NOOP.

---

## Using the env

```python
from car_env import CarRacingEnv

env = CarRacingEnv(
    slippery=False,        # False = grippy (pretrain), True = icy (transfer)
    max_steps=3000,        # episode is truncated after this many env steps
    render_mode=None,      # 'human' for a live window, None for headless training
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Standard Gymnasium 5-tuple. Headless mode doesn't import pygame, so it's
fast for training.

---

## Observation space

`spaces.Box(low=-1, high=1, shape=(13,), dtype=float32)` by default. I
deliberately kept it small and dense so a tiny MLP can handle it — no
need for a CNN.

| idx   | feature              | meaning                                                                                                                                                                                          |
|-------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0..6  | `ray_0..6`           | distance to nearest non-road tile along each of 7 LIDAR rays, / `ray_max_dist`. `1.0` = nothing within range. Rays at angles `[-90, -45, -20, 0, +20, +45, +90]` degrees from the car's heading. |
| 7     | `speed`              | signed speed / `CAR_SPEED`. Approx. range [-0.5, 1].                                                                                                                                             |
| 8     | `fwd_vel`            | actual velocity projected onto the car's heading axis / `CAR_SPEED`.                                                                                                                             |
| 9     | `lat_vel`            | actual velocity projected onto the car's right axis / `CAR_SPEED`. **Non-zero when sliding** — this is the feature that lets the agent "see" slip.                                               |
| 10    | `is_off_road`        | 1.0 if the car is on dirt, else 0.0.                                                                                                                                                             |
| 11    | `angle_to_center`    | signed angle from heading toward a centerline point `centerline_lookahead` indices ahead, in `[-1, 1]` (radians / π). Negative = steer left to recenter, positive = steer right.                |
| 12    | `distance_to_center` | unsigned euclidean distance from the car to the current-row centerline point, / `ray_max_dist`. Clamped to `[0, 1]`. Reads ~0 on-track, grows as the car drifts off the racing line.            |

LIDAR is pure-math raycasts against the tile array — no image, no CNN,
no pygame surfaces. Vectorised in NumPy, costs a few hundred microseconds
per step.

The `lat_vel` channel is the most important thing in this whole obs for
the transfer experiment. If I ever strip it, the agent is blind to slip
and the transfer study is meaningless. `test_env.py::test_slip_observability`
is my canary for that.

---

## Action space

I expose two presets via `CarRacingEnv(action_set=...)`. Default is
**`no_noop`** (6 actions). I dropped NOOP after watching my first DQN
runs sit at spawn for 200 episodes — NOOP becomes a "do nothing safe"
local optimum and random exploration can't get past it.

**`action_set="no_noop"` (default, `Discrete(6)`):**

| idx | name      | throttle | steer | effect                       |
|-----|-----------|----------|-------|------------------------------|
| 0   | Acc       | +1       |  0    | speed up, straight           |
| 1   | Dec       | -1       |  0    | active brake, straight       |
| 2   | LEFT      |  0       | -1    | coast + turn left            |
| 3   | RIGHT     |  0       | +1    | coast + turn right           |
| 4   | Acc+LEFT  | +1       | -1    | speed up, turn left          |
| 5   | Acc+RIGHT | +1       | +1    | speed up, turn right         |

`LEFT` and `RIGHT` carry throttle=0, which falls into the `coast_factor`
branch in `step()` — so the agent gets to slow into a corner without me
having to add an explicit "brake while turning" action.

**`action_set="full"` (`Discrete(7)`):** original set, NOOP at index 0.
I use this only for keyboard play.

```python
env = CarRacingEnv()                          # default: no_noop, 6 actions
env.action_index(1, 0)                        # 0  (Acc)
env.action_names[env.action_index(1, -1)]     # 'Acc+LEFT'
```

I learned the hard way that hardcoding action ints (`step(1)` to mean
"Acc") silently breaks across action-set changes. So I added
`env.action_index(throttle, steer)` and use that in all my tests.

I also dropped `Dec+LEFT` and `Dec+RIGHT` from both sets — never useful
for racing.

---

## Reward function

```python
DEFAULT_REWARDS = {
    "progress":   3.0,   # × delta arc-length along the centerline this step
    "time":      -0.05,  # flat per-step penalty
    "off_road":  -3.0,   # added each step the car is on dirt
    "wall_hit": -10.0,   # one-shot, episode ends (terminated)
    "finish":  100.0,    # one-shot, episode ends (terminated)
}
```

Five terms, each one is there for a specific reason:

- **`progress`** — the one I care about most. It's the delta in
  arc-length along a precomputed centerline this step. Going forward =
  positive, going backward or sideways = ~0 or negative. The centerline
  is a curve, so this correctly rewards "follow the track" on a winding
  shape without me having to hand-tune a direction vector.

- **`time`** — a flat per-step penalty. Without this, "sit still
  forever" is a tie with "drive carefully", because both give zero
  progress. The penalty makes inactivity strictly worse than doing
  something. -0.05 is small enough not to drown out the progress signal
  but big enough to be felt.

- **`off_road`** — added every step the car is on dirt. The env already
  caps off-road speed at `off_road_speed` (much lower than `car_speed`),
  but I want a direct signal so the agent learns "stay on the asphalt"
  faster than it would from the speed cap alone.

- **`wall_hit`** — one-shot, fires when the car drives into a wall, and
  ends the episode (`terminated=True`). I keep it modest at -10 so the
  agent doesn't become so wall-averse that it refuses to take corners,
  but it's enough that random crashing isn't free.

- **`finish`** — one-shot +100 for crossing the finish line, also
  terminates the episode. This is the big sparse signal at the end of
  the lap. With `progress` already shaping per-step reward, `finish`
  doesn't need to be huge — but I want it to clearly dominate any
  partial-lap return.

I deliberately don't reward "centering" anymore. The agent already gets
`angle_to_center` and `distance_to_center` in the observation, so it
can learn to drive on the racing line *if that's actually optimal*
under the other rewards — I don't want to pre-bake that assumption
into the reward shape and limit cornering strategies.

I keep these defaults active in `dqn.ipynb`. In my first DQN run I
zeroed `wall_hit` and `off_road` to "make it more like CartPole" — that
was a mistake. Standing still became strictly dominant (zero progress
beats negative crash penalty when the crash penalty is also zero) and
my agent learned a NOOP policy. With the defaults back, time pressure
+ crash cost force the agent to actually try.

Two ways to override:

```python
# At construction
env = CarRacingEnv(reward_config={"progress": 2.0, "wall_hit": -100})

# Mid-training
env.rewards["off_road"] = -1.0
env.set_reward(off_road=-1.0, finish=200.0)
```

Each `step()` returns the individual reward components in `info`:

```python
info["r_progress"], info["r_time"], info["r_off_road"],
info["r_wall_hit"], info["r_finish"]
```

When training stalls I plot these separately — usually one term is
dominating everything else.

---

## Slippery toggle

```python
pretrain_env = CarRacingEnv(slippery=False)   # friction = 0.1 (grippy)
transfer_env = CarRacingEnv(slippery=True)    # friction = 0.95 (icy)
```

Or flip an existing env without rebuilding:

```python
env.set_slippery(True)
```

The math: `vel = vel * friction + target_vel * (1 - friction)`. With
`friction = 0.1` the actual velocity follows the heading within ~3 ticks.
With `friction = 0.95` the velocity persists for ~20+ ticks — turning
the wheel rotates the chassis but the car keeps sliding in the previous
direction. That's the icy feel.

The naming is admittedly inverted — higher `friction` = more slippery,
because it's the *retention coefficient* of the previous velocity. Both
values are constructor params (`friction_normal`, `friction_slippery`)
if I want to tune them.

---

## Episode termination

| condition                                                     | terminated | truncated | typical reward |
|---------------------------------------------------------------|------------|-----------|----------------|
| Car drives into a wall tile                                   | True       | False     | -10            |
| Car crosses the finish line                                   | True       | False     | +100           |
| `max_steps` reached without ending                            | False      | True      | 0              |
| Lost ≥ `early_terminate_backward_pct` of arc (off by default) | False      | True      | 0              |
| `early_terminate_stagnation_steps` without progress (off)     | False      | True      | 0              |

Important detail I learned from this project: the early-termination
conditions are **`truncated`, not `terminated`**. Since DQN bootstraps
with `terminated` to zero the next-state value, calling them terminal
would bias Q-values downward at near-stagnation states. I keep
`info["backward_truncated"]` and `info["stagnation_truncated"]` to
flag which truncation fired.

---

## Rendering during training

Three ways I render. Trade-offs are pretty obvious:

**A. Eval-only rendering (cheapest).** Train headless, periodically
spin up a second env in `render_mode="human"` to watch a greedy episode.
This is what `dqn.ipynb`'s final cell does.

**B. Mixed training (what I usually use).** `train_mixed(cfg, render_every=10)`
in the notebook keeps a headless env for 9 episodes out of 10 and
renders the 10th. Both envs share the same agent and replay buffer, so
it's one training run, not two.

**C. Render every step (slowest, only for debugging).**
`render_mode="human"` directly on the training env. Useful when I just
want to confirm the agent isn't doing something absurd.

```python
env = CarRacingEnv(slippery=False, render_mode="human", render_fps=120)
```

The HUD shows current action name, instantaneous reward, % progress,
speed, and slip mode. Yellow lines are the LIDAR rays — visually
confirms what the agent is "seeing".

---

## My experiment plan

1. **Pretrain on `slippery=False`** until the agent finishes the lap
   consistently. Save weights + replay buffer.
2. **Zero-shot eval on `slippery=True`** with the pretrained weights —
   this is my "no transfer" baseline.
3. **Transfer-train on `slippery=True`**, starting from the pretrained
   weights. Variants I want to compare:
   - same replay buffer (warm start)
   - fresh replay buffer, same weights
   - freeze early layers, fine-tune the head
   - and training-from-scratch on slippery as a control
4. **Plot:** reward components over training, lap-completion rate vs.
   env steps, and "steps to recover" — how many transfer steps to match
   pretrained performance.

Right now I'm in step 1.
