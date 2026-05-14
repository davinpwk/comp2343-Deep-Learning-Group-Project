"""
Car racing environment for Deep Q-Network and transfer learning experiments.

Two surface modes, controlled by the `slippery` flag at construction:
  - slippery=False : grippy road  (friction = 0.1)  -- pretraining mode
  - slippery=True  : icy road     (friction = 0.95) -- transfer-target mode

Everything else (track layout, physics constants, action space, observation
shape) stays identical across the two modes so the *only* difference your
DQN sees is the slip behavior.
"""

import math
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import (
    DIRT, ROAD, WALL, START, FINISH, TILE_SIZE,
    DEFAULT_TRACK, generate_winding_map,
)


# ---------------------------------------------------------------------------
# Discrete action sets, encoded as (throttle, steer) in {-1, 0, +1}.
#
# Two presets, selectable via `CarRacingEnv(action_set=...)`:
#   - "no_noop"  (default): 6 actions. NOOP dropped to remove the
#     "sit still at spawn" trap that wrecks early DQN exploration. The
#     pure-steer actions (LEFT, RIGHT) coast as a side-effect of the
#     throttle=0 branch in step(), giving the agent a slow-down option.
#   - "full":               7 actions, includes NOOP. Kept for backward
#     compatibility and for human keyboard play.
#
# Reverse-while-turning combos (Dec+LEFT, Dec+RIGHT) are dropped from both
# sets — they're never useful for racing and only dilute exploration.
# ---------------------------------------------------------------------------
ACTIONS_FULL = [
    (0, 0),     # 0  NOOP
    (1, 0),     # 1  Acc
    (-1, 0),    # 2  Dec
    (0, -1),    # 3  LEFT
    (0, 1),     # 4  RIGHT
    (1, -1),    # 5  Acc+LEFT
    (1, 1),     # 6  Acc+RIGHT
]
ACTION_NAMES_FULL = [
    "NOOP", "Acc", "Dec", "LEFT", "RIGHT",
    "Acc+LEFT", "Acc+RIGHT",
]

ACTIONS_NO_NOOP = [
    (1, 0),     # 0  Acc
    (-1, 0),    # 1  Dec
    (0, -1),    # 2  LEFT       (coasts while turning, since throttle=0)
    (0, 1),     # 3  RIGHT      (coasts while turning, since throttle=0)
    (1, -1),    # 4  Acc+LEFT
    (1, 1),     # 5  Acc+RIGHT
]
ACTION_NAMES_NO_NOOP = [
    "Acc", "Dec", "LEFT", "RIGHT", "Acc+LEFT", "Acc+RIGHT",
]

ACTION_SETS = {
    "full":    (ACTIONS_FULL,    ACTION_NAMES_FULL),
    "no_noop": (ACTIONS_NO_NOOP, ACTION_NAMES_NO_NOOP),
}

# Module-level aliases reflect the new default (no_noop). Import these for
# convenience; the per-env truth lives on the instance (self.actions /
# self.action_names) once an env is constructed.
ACTIONS = ACTIONS_NO_NOOP
ACTION_NAMES = ACTION_NAMES_NO_NOOP


# ---------------------------------------------------------------------------
# LIDAR: 7 rays, forward-biased (degrees from car heading)
# ---------------------------------------------------------------------------
DEFAULT_RAY_ANGLES = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)

# Non-ray observation count:
#   (speed, fwd_vel, lat_vel, is_off_road, angle_to_center, distance_to_center)
N_EXTRA_OBS = 6


# ---------------------------------------------------------------------------
# Default reward weights. Mutable at runtime: env.rewards["wall_hit"] = -100
# ---------------------------------------------------------------------------
DEFAULT_REWARDS = dict(
    progress=3.0,    # multiplied by Δ arc-length per step
    time=-0.05,      # flat per-step time penalty
    off_road=-3.0,   # added each step the car is on dirt
    wall_hit=-10.0,  # one-shot, ends the episode
    finish=100.0,    # one-shot, ends the episode
)


class CarRacingEnv(gym.Env):
    """Gymnasium-compatible tile racing environment.

    Observation (default 13 floats, all in roughly [-1, 1]):
        [ray_0, ..., ray_{N-1}, speed, fwd_vel, lat_vel,
         is_off_road, angle_to_center, distance_to_center]

        ray_i              distance to nearest non-road tile / ray_max_dist (in [0, 1])
        speed              signed speed scalar / CAR_SPEED                  (in ~[-0.5, 1])
        fwd_vel            actual velocity · heading / CAR_SPEED            (forward slip)
        lat_vel            actual velocity · right-axis / CAR_SPEED         (lateral slip)
        is_off_road        1.0 if on dirt, else 0.0
        angle_to_center    signed angle from heading toward nearest centerline point,
                           in [-1, 1] (radians/pi). Negative = steer left to recenter.
        distance_to_center unsigned euclidean distance from car to current-row
                           centerline point, / ray_max_dist (clamped to [0, 1]).

    Action: discrete. Default is `action_set="no_noop"` (6 actions); pass
    `action_set="full"` for the original 7-action set including NOOP.
    See ACTIONS_FULL / ACTIONS_NO_NOOP at module level, or query the live
    env via `env.actions` / `env.action_names`.

    Reward (configurable, see DEFAULT_REWARDS):
        r = progress*Δarc + time + (off_road if on dirt)
                                 + (wall_hit if crashed)
                                 + (finish   if crossed line)

    info dict from step() contains each component split out as
    r_progress / r_time / r_off_road / r_wall_hit / r_finish, plus
    is_off_road, wall_hit, finish, progress_pct, speed, car_x, car_y,
    car_angle. Useful for plotting "where the reward is coming from".

    Friction convention: `friction` is the *retention* coefficient of the
    previous velocity (so higher = MORE slippery, since less of the
    target-velocity is blended in each step). Indexing names "slip" might
    be more intuitive, but `friction` is preserved for backward compat.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ----- render constants -------------------------------------------------
    SCREEN_W = 1000
    SCREEN_H = 750
    ASPHALT     = (50, 50, 60)
    GRASS       = (55, 110, 55)
    WALL_COL    = (180, 180, 180)
    START_COL   = (240, 240, 240)
    FINISH_COL  = (220, 190, 0)
    RED         = (220, 40, 40)
    WHITE       = (255, 255, 255)
    RAY_COL     = (255, 255, 0)

    def __init__(
        self,
        slippery: bool = False,
        reward_config: dict = None,
        max_steps: int = 3000,
        ray_angles=DEFAULT_RAY_ANGLES,
        ray_max_dist: float = 200.0,
        friction_normal: float = 0.1,
        friction_slippery: float = 0.95,
        car_speed: float = 4.5,
        off_road_speed: float = 1.8,
        turn_speed: float = 4.0,
        accel: float = 0.15,
        accel_off_road: float = 0.08,
        coast_factor: float = 0.92,
        reverse_throttle_cap: float = 0.5,
        off_road_overspeed_decay: float = 0.85,
        centerline_lookahead: int = 10,
        render_mode: str = None,
        render_fps: int = 60,
        track_config: dict = None,
        show_rays_on_render: bool = True,
        screen_size: tuple = None,
        early_terminate_backward_pct: float = 0.0,
        early_terminate_stagnation_steps: int = 0,
        align_spawn_to_tangent: bool = False,
        action_set: str = "no_noop",
    ):
        super().__init__()
        # --- action set ---------------------------------------------------
        if action_set not in ACTION_SETS:
            raise ValueError(
                f"action_set must be one of {list(ACTION_SETS)}, got {action_set!r}"
            )
        self.action_set = action_set
        self.actions, self.action_names = ACTION_SETS[action_set]
        # --- track --------------------------------------------------------
        cfg = {**DEFAULT_TRACK, **(track_config or {})}
        self.track_cfg = cfg
        self.track = generate_winding_map(**cfg)

        # Spawn position. If multiple START tiles ever appear, the first
        # (row-major) wins -- single START is the documented assumption.
        loc = np.argwhere(self.track == START)
        if len(loc) == 0:
            raise ValueError("Track has no START tile (value 3).")
        if len(loc) > 1:
            # Not an error -- there's no reason to forbid multiple starts --
            # but make the silent pick explicit so future debugging is easier.
            print(
                f"[CarRacingEnv] {len(loc)} START tiles found; using first at "
                f"row={loc[0, 0]}, col={loc[0, 1]}.",
                file=sys.stderr,
            )
        sr, sc = int(loc[0, 0]), int(loc[0, 1])
        self.spawn_x = sc * TILE_SIZE + TILE_SIZE / 2
        self.spawn_y = sr * TILE_SIZE + TILE_SIZE / 2

        # Centerline & cumulative arc length
        self._build_centerline(cfg)

        # --- spaces -------------------------------------------------------
        self.action_space = spaces.Discrete(len(self.actions))
        self.ray_angles = np.array(ray_angles, dtype=np.float32)
        self.n_rays = len(self.ray_angles)
        self.ray_max_dist = float(ray_max_dist)
        obs_dim = self.n_rays + N_EXTRA_OBS
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Named indices into the obs vector (useful for tests / debug)
        self.obs_idx = {
            "rays":               slice(0, self.n_rays),
            "speed":              self.n_rays + 0,
            "fwd_vel":            self.n_rays + 1,
            "lat_vel":            self.n_rays + 2,
            "is_off_road":        self.n_rays + 3,
            "angle_to_center":    self.n_rays + 4,
            "distance_to_center": self.n_rays + 5,
        }

        # --- reward config (mutable at runtime) ---------------------------
        self.rewards = {**DEFAULT_REWARDS, **(reward_config or {})}

        # --- physics ------------------------------------------------------
        self.car_speed = car_speed
        self.off_road_speed = off_road_speed
        self.turn_speed = turn_speed
        self.accel = accel
        self.accel_off_road = accel_off_road
        self.coast_factor = coast_factor
        self.reverse_throttle_cap = reverse_throttle_cap
        self.off_road_overspeed_decay = off_road_overspeed_decay
        self._CENTERLINE_LOOKAHEAD = int(centerline_lookahead)
        self.friction_normal = friction_normal
        self.friction_slippery = friction_slippery
        self.slippery = bool(slippery)
        self.friction = friction_slippery if slippery else friction_normal
        self.max_steps = int(max_steps)

        # --- early-termination thresholds (0 = disabled) ------------------
        self.early_terminate_backward_pct = float(early_terminate_backward_pct)
        self.early_terminate_stagnation_steps = int(early_terminate_stagnation_steps)
        self.align_spawn_to_tangent = bool(align_spawn_to_tangent)
        # Precompute spawn-tangent angle (degrees, env convention) once.
        if self._cl_n >= 2:
            dx = float(self.centerline[1, 0] - self.centerline[0, 0])
            dy = float(self.centerline[1, 1] - self.centerline[0, 1])
            self._spawn_tangent_deg = math.degrees(math.atan2(dx, -dy))
        else:
            self._spawn_tangent_deg = 0.0

        # --- ray sampling cache ------------------------------------------
        self._ray_step = TILE_SIZE / 2
        self._max_ray_steps = int(self.ray_max_dist / self._ray_step)
        self._ray_step_distances = (
            np.arange(1, self._max_ray_steps + 1, dtype=np.float32) * self._ray_step
        )

        # --- render state (lazy pygame init) ------------------------------
        if screen_size is not None:
            self.SCREEN_W, self.SCREEN_H = int(screen_size[0]), int(screen_size[1])
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.show_rays_on_render = show_rays_on_render
        self._pygame = None         # cached module handle after first import
        self._screen = None
        self._font = None
        self._clock = None
        self._track_surf = None     # pre-rendered full track as a pygame.Surface
        self._track_rgb = None      # (H, W, 3) uint8 -- kept for rgb_array path

        # --- episode state (filled by reset) ------------------------------
        self.car_x = self.car_y = self.car_angle = 0.0
        self.speed = self.vel_x = self.vel_y = 0.0
        self.steps = 0
        self.prev_arc = 0.0
        self._max_arc_reached = 0.0
        self._steps_since_progress = 0
        self._last_action = 0
        self._last_reward = 0.0

    # ======================================================================
    # Convenience API
    # ======================================================================

    def set_slippery(self, slippery: bool):
        """Toggle slip mode without rebuilding the env.

        NOTE: this mutates `friction` but leaves the current `vel_x`/`vel_y`
        from the previous regime untouched. The slip-blend will catch up
        within a few steps, but if you're toggling mid-episode for transfer
        experiments, prefer calling reset() right after for a clean handoff.
        """
        self.slippery = bool(slippery)
        self.friction = self.friction_slippery if slippery else self.friction_normal

    def set_reward(self, **kwargs):
        """Update one or more reward weights. Same as env.rewards.update(...)."""
        for k, v in kwargs.items():
            if k not in self.rewards:
                raise KeyError(f"Unknown reward key: {k}. "
                               f"Known: {list(self.rewards)}")
            self.rewards[k] = v

    # ======================================================================
    # Internals
    # ======================================================================

    def _build_centerline(self, cfg):
        path_start = cfg["path_start"]
        path_end = cfg["path_end"]
        start_row = cfg["start_row"]
        finish_row = cfg["finish_row"]

        def center_col(row):
            t = (row - path_start) / (path_end - path_start)
            return cfg["base_center"] + cfg["amplitude"] * math.sin(2 * math.pi * t)

        step = -1 if finish_row < start_row else 1
        rows = list(range(start_row, finish_row + step, step))

        cl = np.array([
            [center_col(r) * TILE_SIZE + TILE_SIZE / 2,
             r * TILE_SIZE + TILE_SIZE / 2]
            for r in rows
        ], dtype=np.float32)
        self.centerline = cl

        diffs = np.diff(cl, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.centerline_arc = np.concatenate(
            [[0.0], np.cumsum(seg_lens)]
        ).astype(np.float32)
        self.total_arc = float(self.centerline_arc[-1])

        if self.total_arc <= 0.0:
            raise ValueError(
                "Degenerate centerline: total_arc <= 0. "
                "Check start_row vs finish_row in track_config."
            )

        self._cl_start_row = start_row
        self._cl_direction = step
        self._cl_n = len(cl)

    def _arc_at(self, x: float, y: float) -> float:
        # Centerline is monotonic in row, so use y as a direct index.
        row = y / TILE_SIZE
        idx_f = (row - self._cl_start_row) / self._cl_direction
        idx = int(np.clip(idx_f, 0, self._cl_n - 1))
        return float(self.centerline_arc[idx])

    def _tile_at(self, x: float, y: float) -> int:
        c = int(x // TILE_SIZE)
        r = int(y // TILE_SIZE)
        H, W = self.track.shape
        if 0 <= r < H and 0 <= c < W:
            return int(self.track[r, c])
        return WALL

    def _cast_rays(self) -> np.ndarray:
        """Vectorized LIDAR — pure-math raycast against the tile grid."""
        angles_rad = np.radians(self.car_angle + self.ray_angles)
        dx = np.sin(angles_rad)
        dy = -np.cos(angles_rad)

        # Sample points along each ray
        xs = self.car_x + dx[:, None] * self._ray_step_distances[None, :]
        ys = self.car_y + dy[:, None] * self._ray_step_distances[None, :]
        cs = (xs // TILE_SIZE).astype(np.int32)
        rs = (ys // TILE_SIZE).astype(np.int32)

        H, W = self.track.shape
        in_bounds = (cs >= 0) & (cs < W) & (rs >= 0) & (rs < H)
        cs_clip = np.clip(cs, 0, W - 1)
        rs_clip = np.clip(rs, 0, H - 1)
        tiles = self.track[rs_clip, cs_clip]
        hit = (tiles != ROAD) | (~in_bounds)

        any_hit = hit.any(axis=1)
        first_idx = np.argmax(hit, axis=1)
        distances = np.where(
            any_hit,
            self._ray_step_distances[first_idx],
            self.ray_max_dist,
        )
        return distances.astype(np.float32)

    def _car_frame_velocity(self):
        a = math.radians(self.car_angle)
        s, c = math.sin(a), math.cos(a)
        fwd = self.vel_x * s + self.vel_y * (-c)
        lat = self.vel_x * c + self.vel_y * s
        return fwd, lat

    def _get_obs(self, angle_to_center: float = None) -> np.ndarray:
        if angle_to_center is None:
            angle_to_center = self._angle_to_centerline()
        rays = self._cast_rays() / self.ray_max_dist
        fwd, lat = self._car_frame_velocity()
        is_off_road = float(self._tile_at(self.car_x, self.car_y) == DIRT)
        extras = np.array([
            self.speed / self.car_speed,
            fwd / self.car_speed,
            lat / self.car_speed,
            is_off_road,
            angle_to_center,
            self._distance_to_centerline(),
        ], dtype=np.float32)
        return np.concatenate([rays, extras]).astype(np.float32)

    def _distance_to_centerline(self) -> float:
        """Unsigned euclidean distance from car to current-row centerline point,
        normalised by ray_max_dist. Clamped to [0, 1].

        On-track this typically reads ~0.0-0.1; gets larger when the car drifts
        off into dirt, giving the agent a continuous "how far off-line am I"
        signal alongside the binary `is_off_road` flag.
        """
        row = self.car_y / TILE_SIZE
        idx_f = (row - self._cl_start_row) / self._cl_direction
        idx = int(np.clip(idx_f, 0, self._cl_n - 1))
        cx, cy = self.centerline[idx]
        dx = self.car_x - float(cx)
        dy = self.car_y - float(cy)
        d = math.hypot(dx, dy)
        return float(min(d / self.ray_max_dist, 1.0))

    def _angle_to_centerline(self) -> float:
        """Signed angle from the car's heading to a centerline point
        `_CENTERLINE_LOOKAHEAD` indices ahead, in [-1, 1] (radians / pi).
        Negative = steer left to recenter, positive = steer right.
        """
        row = self.car_y / TILE_SIZE
        idx_f = (row - self._cl_start_row) / self._cl_direction
        idx = int(np.clip(idx_f + self._CENTERLINE_LOOKAHEAD, 0, self._cl_n - 1))
        cx, cy = self.centerline[idx]
        dx = float(cx) - self.car_x
        dy = float(cy) - self.car_y
        if dx == 0.0 and dy == 0.0:
            return 0.0
        # Same convention as car_angle: 0 = pointing -y, positive = clockwise.
        target_deg = math.degrees(math.atan2(dx, -dy))
        diff = target_deg - self.car_angle
        diff = (diff + 180.0) % 360.0 - 180.0
        return float(diff / 180.0)

    # ======================================================================
    # Gymnasium API
    # ======================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x = self.spawn_x
        self.car_y = self.spawn_y
        self.car_angle = self._spawn_tangent_deg if self.align_spawn_to_tangent else 0.0
        self.speed = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.steps = 0
        self.prev_arc = self._arc_at(self.car_x, self.car_y)
        self._max_arc_reached = self.prev_arc
        self._steps_since_progress = 0
        self._last_action = 0
        self._last_reward = 0.0
        return self._get_obs(), {}

    def action_index(self, throttle: int, steer: int) -> int:
        """Look up an action index by its (throttle, steer) pair.

        Lets callers stay action-set-agnostic: instead of hardcoding
        `step(1)` and hoping that means "Acc", use
        `step(env.action_index(1, 0))`. Raises ValueError if the pair
        isn't in the active action set.
        """
        try:
            return self.actions.index((int(throttle), int(steer)))
        except ValueError as e:
            raise ValueError(
                f"({throttle}, {steer}) not in action_set={self.action_set!r}; "
                f"valid pairs: {self.actions}"
            ) from e

    def step(self, action):
        throttle, steer = self.actions[int(action)]

        cur_tile = self._tile_at(self.car_x, self.car_y)
        is_off_road = (cur_tile == DIRT)
        max_v = self.off_road_speed if is_off_road else self.car_speed
        accel = self.accel_off_road if is_off_road else self.accel

        # Throttle
        if throttle > 0:
            self.speed = min(self.speed + accel, max_v)
        elif throttle < 0:
            self.speed = max(self.speed - accel, -max_v * self.reverse_throttle_cap)
        else:
            self.speed *= self.coast_factor

        if is_off_road and self.speed > self.off_road_speed:
            self.speed *= self.off_road_overspeed_decay

        # Steer (only effective when moving)
        if abs(self.speed) > 0.1:
            turn = (self.speed / self.car_speed) * self.turn_speed
            if steer < 0:
                self.car_angle -= turn
            elif steer > 0:
                self.car_angle += turn

        # Wrap car_angle into [-180, 180] so info["car_angle"] stays bounded
        # over long episodes.
        self.car_angle = (self.car_angle + 180.0) % 360.0 - 180.0

        # Target velocity (where the car is pointed * speed)
        rad = math.radians(self.car_angle)
        target_vx = math.sin(rad) * self.speed
        target_vy = -math.cos(rad) * self.speed

        # Slippery blend: velocity is pulled toward target, more slowly when icy
        self.vel_x = self.vel_x * self.friction + target_vx * (1 - self.friction)
        self.vel_y = self.vel_y * self.friction + target_vy * (1 - self.friction)

        nx = self.car_x + self.vel_x
        ny = self.car_y + self.vel_y

        wall_hit = False
        finish = False
        terminated = False

        next_tile = self._tile_at(nx, ny)
        if next_tile == WALL:
            wall_hit = True
            terminated = True
            # Don't actually move into the wall (keeps last-good pose for the
            # final render of the crash)
        else:
            self.car_x = nx
            self.car_y = ny
            if next_tile == FINISH:
                finish = True
                terminated = True

        # Reward
        cur_arc = self._arc_at(self.car_x, self.car_y)
        d_arc = cur_arc - self.prev_arc
        self.prev_arc = cur_arc

        # Track peak arc + stagnation for early-termination conditions
        if cur_arc > self._max_arc_reached:
            self._max_arc_reached = cur_arc
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        # Early-stop conditions are time-limits, NOT MDP terminal states, so
        # they go through `truncated` (Gymnasium convention). Using
        # `terminated` here would zero the Q-bootstrap target and bias values
        # downward at near-stagnation states.
        backward_truncated = (
            self.early_terminate_backward_pct > 0.0
            and (self._max_arc_reached - cur_arc) / self.total_arc
                >= self.early_terminate_backward_pct
        )
        stagnation_truncated = (
            self.early_terminate_stagnation_steps > 0
            and self._steps_since_progress
                >= self.early_terminate_stagnation_steps
        )

        angle_to_center = self._angle_to_centerline()

        r_progress  = self.rewards["progress"] * d_arc
        r_time      = self.rewards["time"]
        r_off_road  = self.rewards["off_road"] if is_off_road else 0.0
        r_wall      = self.rewards["wall_hit"] if wall_hit else 0.0
        r_finish    = self.rewards["finish"]   if finish else 0.0
        reward = r_progress + r_time + r_off_road + r_wall + r_finish

        self.steps += 1
        max_steps_truncated = self.steps >= self.max_steps
        truncated = (
            (max_steps_truncated or backward_truncated or stagnation_truncated)
            and not terminated
        )

        info = {
            "r_progress": float(r_progress),
            "r_time":     float(r_time),
            "r_off_road": float(r_off_road),
            "r_wall_hit": float(r_wall),
            "r_finish":    float(r_finish),
            "is_off_road": bool(is_off_road),
            "wall_hit":    bool(wall_hit),
            "finish":      bool(finish),
            "backward_truncated":   bool(backward_truncated),
            "stagnation_truncated": bool(stagnation_truncated),
            "progress_pct": min(float(cur_arc / self.total_arc), 1.0),
            "speed": float(self.speed),
            "car_x": float(self.car_x),
            "car_y": float(self.car_y),
            "car_angle": float(self.car_angle),
        }

        obs = self._get_obs(angle_to_center=angle_to_center)
        self._last_action = int(action)
        self._last_reward = float(reward)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Rendering
    # ======================================================================

    def _ensure_pygame(self):
        """Lazy-import pygame on first use; cache the module handle."""
        if self._pygame is None:
            import pygame
            self._pygame = pygame
        return self._pygame

    def render(self):
        """Render current state.

        - render_mode='human'     -> draws to a pygame window, returns None
        - render_mode='rgb_array' -> returns an (H, W, 3) uint8 numpy array
        - render_mode=None        -> returns None (no work)
        """
        if self.render_mode is None:
            return None
        pygame = self._ensure_pygame()
        if self._screen is None:
            self._init_render()
        self._draw_frame()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            pygame.display.flip()
            self._clock.tick(self.render_fps)
            return None

        if self.render_mode == "rgb_array":
            # pygame surfaces are (W, H, 3); transpose to (H, W, 3) for gym convention.
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2)).copy()

        return None

    def _init_render(self):
        pygame = self._ensure_pygame()
        if self.render_mode == "human":
            pygame.init()
            self._screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
            pygame.display.set_caption(
                f"CarRacingEnv ({'SLIPPERY' if self.slippery else 'NORMAL'})"
            )
        else:
            # rgb_array: headless surface, no display required
            pygame.font.init()
            self._screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self._font = pygame.font.SysFont("monospace", 18, bold=True)
        self._clock = pygame.time.Clock()

        color_lut = np.array([
            self.GRASS,    # 0 DIRT
            self.ASPHALT,  # 1 ROAD
            self.WALL_COL, # 2 WALL
            self.START_COL,# 3 START
            self.FINISH_COL# 4 FINISH
        ], dtype=np.uint8)
        self._track_rgb = color_lut[self.track]  # (H, W, 3) uint8

        # Build the full track surface once (TILE_SIZE px per tile).
        # Per-frame work is then just a sub-rect blit with a camera offset.
        H, W = self.track.shape
        big = np.repeat(np.repeat(self._track_rgb, TILE_SIZE, axis=0),
                        TILE_SIZE, axis=1)
        # pygame.surfarray expects (W, H, 3)
        self._track_surf = pygame.surfarray.make_surface(big.transpose(1, 0, 2))

    def _draw_frame(self):
        pygame = self._ensure_pygame()
        screen = self._screen
        screen.fill(self.WALL_COL)

        cam_x = self.car_x - self.SCREEN_W / 2
        cam_y = self.car_y - self.SCREEN_H / 2
        # Blit the pre-rendered track at camera offset. SDL clips for us, but
        # we still place it relative to (0,0) on the camera frame.
        screen.blit(self._track_surf, (int(-cam_x), int(-cam_y)))

        # LIDAR rays
        if self.show_rays_on_render:
            distances = self._cast_rays()
            cx_screen, cy_screen = self.SCREEN_W // 2, self.SCREEN_H // 2
            for theta, d in zip(self.ray_angles, distances):
                rad = math.radians(self.car_angle + theta)
                ex = cx_screen + math.sin(rad) * d
                ey = cy_screen - math.cos(rad) * d
                pygame.draw.line(
                    screen, self.RAY_COL,
                    (cx_screen, cy_screen),
                    (int(ex), int(ey)),
                    1,
                )

        # Car
        self._draw_car(screen, self.SCREEN_W // 2, self.SCREEN_H // 2,
                       self.car_angle)

        # HUD
        hud = pygame.Surface((320, 130), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 170))
        screen.blit(hud, (10, 10))
        mode_str = "SLIPPERY" if self.slippery else "NORMAL"
        progress_pct = (self.prev_arc / self.total_arc * 100) if self.total_arc > 0 else 0
        progress_pct = min(progress_pct, 100.0)
        lines = [
            (f"mode:     {mode_str}",       self.WHITE),
            (f"step:     {self.steps}",      self.WHITE),
            (f"action:   {self.action_names[self._last_action]}", self.WHITE),
            (f"reward:   {self._last_reward:+.2f}",          self.FINISH_COL),
            (f"progress: {progress_pct:.1f}%",               self.WHITE),
            (f"speed:    {self.speed:.2f}",                  self.WHITE),
        ]
        for i, (txt, col) in enumerate(lines):
            screen.blit(self._font.render(txt, 1, col), (20, 15 + i * 18))

    def _draw_car(self, surface, x, y, angle):
        pygame = self._ensure_pygame()
        CAR_W, CAR_H = 18, 30
        s = pygame.Surface((CAR_W + 4, CAR_H + 4), pygame.SRCALPHA)
        pygame.draw.rect(s, self.RED, (2, 2, CAR_W, CAR_H), border_radius=2)
        pygame.draw.rect(s, (160, 20, 20), (2, 2, CAR_W, 8))
        rot = pygame.transform.rotate(s, -angle)
        surface.blit(rot, rot.get_rect(center=(int(x), int(y))).topleft)

    def close(self):
        if self._screen is not None and self._pygame is not None:
            if self.render_mode == "human":
                self._pygame.quit()
            self._screen = None
            self._track_surf = None


# ===========================================================================
# Demo / sanity check
# ===========================================================================

def keyboard_play(slippery: bool = False):
    """Drive the env manually with W/A/S/D (or arrows). For sanity-checking.

    Uses action_set="full" so NOOP (no keys pressed) is a valid action;
    no_noop would crash here when the keys map to (0, 0).
    """
    import pygame
    env = CarRacingEnv(
        slippery=slippery, render_mode="human", max_steps=10_000,
        action_set="full",
    )
    env.reset()
    action_map = {ts: i for i, ts in enumerate(env.actions)}
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        keys = pygame.key.get_pressed()
        up    = keys[pygame.K_w] or keys[pygame.K_UP]
        down  = keys[pygame.K_s] or keys[pygame.K_DOWN]
        left  = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        throttle = 1 if up else (-1 if down else 0)
        steer    = -1 if left else (1 if right else 0)
        action = action_map[(throttle, steer)]
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()


def random_rollout(slippery: bool = False, render: bool = True, n_steps: int = 2000):
    """Random-action rollout — useful for verifying the env runs end-to-end."""
    env = CarRacingEnv(
        slippery=slippery,
        render_mode="human" if render else None,
        max_steps=n_steps,
    )
    obs, _ = env.reset()
    total = 0.0
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            print(f"episode ended: total_reward={total:.2f} "
                  f"progress={info['progress_pct']*100:.1f}% wall={info['wall_hit']}")
            obs, _ = env.reset()
            total = 0.0
    env.close()


if __name__ == "__main__":
    slip = "--slippery" in sys.argv
    if "--random" in sys.argv:
        random_rollout(slippery=slip)
    else:
        keyboard_play(slippery=slip)
