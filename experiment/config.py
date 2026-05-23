"""Step-based experiment configuration.

Replaces ../train_util.py's TrainConfig (which is episode-based) with a
budget unit of *environment steps*, per EXPERIMENT_PLAN §2/§0.3.
Episodes still exist (they're what the env terminates on) but the outer
training loop runs until the env-step budget is exhausted.

The config object holds three logically-distinct bundles:
  - env knobs                 (action_set, max_episode_steps, ...)
  - algorithm knobs           (lr, gamma, ε schedule, batch size, ...)
  - run/budget knobs          (max_env_steps, eval_every, seed, ...)

All as a single flat dataclass to keep call-sites short and
serialisation trivial (asdict -> JSON).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# Surface tile-friction presets here so consumers don't need to import
# car_env just to know what "normal" or "slippery" means.
FRICTION_NORMAL: float = 0.1
FRICTION_SLIPPERY: float = 0.95
FRICTION_DR_RANGE: tuple = (FRICTION_NORMAL, FRICTION_SLIPPERY)


@dataclass
class ExperimentConfig:
    """All knobs for one training run.

    Most fields mirror ../train_util.py's TrainConfig so existing call
    sites don't have to relearn names; the budget unit, however, is
    `max_env_steps` rather than `episodes`.
    """

    # === friction regime ===============================================
    # One of:
    #   "fixed"      -> use `friction` for every episode
    #   "dr"         -> sample friction ~ Uniform(friction_dr_low, friction_dr_high)
    #                   at every env.reset()
    #   "curriculum" -> linear ramp from friction_curr_start -> friction_curr_end
    #                   over `max_env_steps` (advance by env-step count, not
    #                   episode count, so episode length doesn't bias the ramp)
    friction_mode: str = "fixed"
    friction: float = FRICTION_NORMAL
    friction_dr_low: float = FRICTION_DR_RANGE[0]
    friction_dr_high: float = FRICTION_DR_RANGE[1]
    friction_curr_start: float = FRICTION_NORMAL
    friction_curr_end: float = FRICTION_SLIPPERY

    # === maps ==========================================================
    # Paths relative to project root. Episodes pick one map uniformly at
    # random; if a single path is given, that map is used every episode.
    map_paths: tuple = (
        "maps/straight_turn.txt",
        "maps/narrow_straight_wide_turn.txt",
        "maps/wide_straight_narrow_turn.txt",
    )

    # === per-episode env knobs ========================================
    max_episode_steps: int = 3000
    reward_overrides: dict = field(default_factory=dict)
    early_terminate_backward_pct: float = 0.05
    early_terminate_stagnation_steps: int = 300
    align_spawn_to_tangent: bool = True
    action_set: str = "no_noop"

    # === DQN / optimiser ==============================================
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_capacity: int = 100_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden_sizes: tuple = (128, 128)
    grad_clip: float = 10.0

    # === budget / scheduling ==========================================
    max_env_steps: int = 500_000        # the only "how long do we train?" knob
    warmup_env_steps: int = 1_000       # collect random transitions before updates
    eval_every_env_steps: int = 10_000  # cadence of the eval harness
    eval_n_episodes: int = 10           # episodes per (map, friction) cell
    # Friction values to evaluate at every checkpoint. Slippery and normal
    # by default so catastrophic forgetting is measured for free.
    eval_frictions: tuple = (FRICTION_NORMAL, FRICTION_SLIPPERY)

    # === misc ==========================================================
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "runs"
    run_name: str = "unnamed"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for run-metadata JSON. Tuples become lists, which is
        fine -- they round-trip through json.dump / json.load cleanly."""
        return asdict(self)
