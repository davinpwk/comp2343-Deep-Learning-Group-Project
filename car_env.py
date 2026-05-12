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
# 9 discrete actions, encoded as (throttle, steer) in {-1, 0, +1}
# ---------------------------------------------------------------------------
ACTIONS = [
    (0, 0),     # 0  NOOP
    (1, 0),     # 1  UP
    (-1, 0),    # 2  DOWN
    (0, -1),    # 3  LEFT
    (0, 1),     # 4  RIGHT
    (1, -1),    # 5  UP+LEFT
    (1, 1),     # 6  UP+RIGHT
    (-1, -1),   # 7  DOWN+LEFT
    (-1, 1),    # 8  DOWN+RIGHT
]
ACTION_NAMES = [
    "NOOP", "UP", "DOWN", "LEFT", "RIGHT",
    "UP+LEFT", "UP+RIGHT", "DOWN+LEFT", "DOWN+RIGHT",
]


# ---------------------------------------------------------------------------
# LIDAR: 7 rays, forward-biased (degrees from car heading)
# ---------------------------------------------------------------------------
DEFAULT_RAY_ANGLES = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)


# ---------------------------------------------------------------------------
# Default reward weights. Mutable at runtime: env.rewards["wall_hit"] = -100
# ---------------------------------------------------------------------------
DEFAULT_REWARDS = dict(
    progress=1.0,    # multiplied by Delta arc-length per step
    time=-0.01,      # flat per-step time penalty
    off_road=-0.3,   # added each step the car is on dirt
    wall_hit=-50.0,  # one-shot, ends the episode
    finish=100.0,    # one-shot, ends the episode
)


class CarRacingEnv(gym.Env):
    """Gymnasium-compatible tile racing environment.

    Observation (default 11 floats, all in roughly [-1, 1]):
        [ray_0, ..., ray_{N-1}, speed, fwd_vel, lat_vel, is_off_road]

        ray_i        distance to nearest non-road tile / ray_max_dist  (in [0, 1])
        speed        signed speed scalar / CAR_SPEED                    (in ~[-0.5, 1])
        fwd_vel      actual velocity . heading / CAR_SPEED              (forward slip)
        lat_vel      actual velocity . right-axis / CAR_SPEED           (lateral slip)
        is_off_road  1.0 if on dirt, else 0.0

    Action (Discrete(9)): see ACTIONS / ACTION_NAMES at module level.

    Reward (configurable, see DEFAULT_REWARDS):
        r = progress*delta_arc + time + (off_road if on dirt)
                                      + (wall_hit if crashed)
                                      + (finish   if crossed line)

    info dict from step() contains each component split out as
    r_progress / r_time / r_off_road / r_wall_hit / r_finish, plus
    is_off_road, wall_hit, finish, progress_pct, speed, car_x, car_y,
    car_angle. Useful for plotting "where the reward is coming from".
    """

    metadata = {"render_modes": ["human", None], "render_fps": 60}

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
        render_mode: str = None,
        render_fps: int = 60,
        track_config: dict = None,
        show_rays_on_render: bool = True,
    ):
        super().__init__()
        # --- track --------------------------------------------------------
        cfg = {**DEFAULT_TRACK, **(track_config or {})}
        self.track_cfg = cfg
        self.track = generate_winding_map(**cfg)

        # Spawn position (single START tile)
        loc = np.argwhere(self.track == START)
        if len(loc) == 0:
            raise ValueError("Track has no START tile (value 3).")
        sr, sc = int(loc[0, 0]), int(loc[0, 1])
        self.spawn_x = sc * TILE_SIZE + TILE_SIZE / 2
        self.spawn_y = sr * TILE_SIZE + TILE_SIZE / 2

        # Centerline & cumulative arc length
        self._build_centerline(cfg)

        # --- spaces -------------------------------------------------------
        self.action_space = spaces.Discrete(9)
        self.ray_angles = np.array(ray_angles, dtype=np.float32)
        self.n_rays = len(self.ray_angles)
        self.ray_max_dist = float(ray_max_dist)
        obs_dim = self.n_rays + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # --- reward config (mutable at runtime) ---------------------------
        self.rewards = {**DEFAULT_REWARDS, **(reward_config or {})}

        # --- physics ------------------------------------------------------
        self.car_speed = car_speed
        self.off_road_speed = off_road_speed
        self.turn_speed = turn_speed
        self.accel = accel
        self.accel_off_road = accel_off_road
        self.coast_factor = coast_factor
        self.friction_normal = friction_normal
        self.friction_slippery = friction_slippery
        self.slippery = bool(slippery)
        self.friction = friction_slippery if slippery else friction_normal
        self.max_steps = int(max_steps)

        # --- ray sampling cache ------------------------------------------
        self._ray_step = TILE_SIZE / 2
        self._max_ray_steps = int(self.ray_max_dist / self._ray_step)
        self._ray_step_distances = (
            np.arange(1, self._max_ray_steps + 1, dtype=np.float32) * self._ray_step
        )

        # --- render state (lazy pygame init) ------------------------------
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.show_rays_on_render = show_rays_on_render
        self._screen = None
        self._font = None
        self._clock = None
        self._track_rgb = None  # (H, W, 3) uint8 color lookup of the tile grid

        # --- episode state (filled by reset) ------------------------------
        self.car_x = self.car_y = self.car_angle = 0.0
        self.speed = self.vel_x = self.vel_y = 0.0
        self.steps = 0
        self.prev_arc = 0.0
        self._last_action = 0
        self._last_reward = 0.0

    # ======================================================================
    # Convenience API
    # ======================================================================

    def set_slippery(self, slippery: bool):
        """Toggle slip mode without rebuilding the env. Useful for transfer eval."""
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
        # Quirk: exit_row is where START sits, spawn_row is where FINISH sits
        start_row = cfg["exit_row"]
        finish_row = cfg["spawn_row"]

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

    def _get_obs(self) -> np.ndarray:
        rays = self._cast_rays() / self.ray_max_dist
        fwd, lat = self._car_frame_velocity()
        is_off_road = float(self._tile_at(self.car_x, self.car_y) == DIRT)
        extras = np.array([
            self.speed / self.car_speed,
            fwd / self.car_speed,
            lat / self.car_speed,
            is_off_road,
        ], dtype=np.float32)
        return np.concatenate([rays, extras]).astype(np.float32)

    # ======================================================================
    # Gymnasium API
    # ======================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x = self.spawn_x
        self.car_y = self.spawn_y
        self.car_angle = 0.0
        self.speed = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.steps = 0
        self.prev_arc = self._arc_at(self.car_x, self.car_y)
        self._last_action = 0
        self._last_reward = 0.0
        return self._get_obs(), {}

    def step(self, action):
        throttle, steer = ACTIONS[int(action)]

        cur_tile = self._tile_at(self.car_x, self.car_y)
        is_off_road = (cur_tile == DIRT)
        max_v = self.off_road_speed if is_off_road else self.car_speed
        accel = self.accel_off_road if is_off_road else self.accel

        # Throttle
        if throttle > 0:
            self.speed = min(self.speed + accel, max_v)
        elif throttle < 0:
            self.speed = max(self.speed - accel, -max_v * 0.5)
        else:
            self.speed *= self.coast_factor

        if is_off_road and self.speed > self.off_road_speed:
            self.speed *= 0.85

        # Steer (only effective when moving)
        if abs(self.speed) > 0.1:
            turn = (self.speed / self.car_speed) * self.turn_speed
            if steer < 0:
                self.car_angle -= turn
            elif steer > 0:
                self.car_angle += turn

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

        r_progress = self.rewards["progress"] * d_arc
        r_time     = self.rewards["time"]
        r_off_road = self.rewards["off_road"] if is_off_road else 0.0
        r_wall     = self.rewards["wall_hit"] if wall_hit else 0.0
        r_finish   = self.rewards["finish"]   if finish else 0.0
        reward = r_progress + r_time + r_off_road + r_wall + r_finish

        self.steps += 1
        truncated = (self.steps >= self.max_steps) and not terminated

        info = {
            "r_progress": float(r_progress),
            "r_time":     float(r_time),
            "r_off_road": float(r_off_road),
            "r_wall_hit": float(r_wall),
            "r_finish":   float(r_finish),
            "is_off_road": bool(is_off_road),
            "wall_hit":    bool(wall_hit),
            "finish":      bool(finish),
            "progress_pct": float(cur_arc / self.total_arc) if self.total_arc > 0 else 0.0,
            "speed": float(self.speed),
            "car_x": float(self.car_x),
            "car_y": float(self.car_y),
            "car_angle": float(self.car_angle),
        }

        obs = self._get_obs()
        self._last_action = int(action)
        self._last_reward = float(reward)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Rendering
    # ======================================================================

    def render(self):
        """Render current state to a pygame window (render_mode='human' only)."""
        if self.render_mode != "human":
            return None
        if self._screen is None:
            self._init_render()
        self._draw_frame()

        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        pygame.display.flip()
        self._clock.tick(self.render_fps)
        return None

    def _init_render(self):
        import pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        pygame.display.set_caption(
            f"CarRacingEnv ({'SLIPPERY' if self.slippery else 'NORMAL'})"
        )
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

    def _draw_frame(self):
        import pygame
        screen = self._screen
        screen.fill(self.WALL_COL)

        # Visible tile range
        cam_x = self.car_x - self.SCREEN_W / 2
        cam_y = self.car_y - self.SCREEN_H / 2
        H, W = self.track.shape
        tx0 = max(0, int(cam_x // TILE_SIZE))
        ty0 = max(0, int(cam_y // TILE_SIZE))
        tx1 = min(W, int((cam_x + self.SCREEN_W) // TILE_SIZE) + 1)
        ty1 = min(H, int((cam_y + self.SCREEN_H) // TILE_SIZE) + 1)

        if tx1 > tx0 and ty1 > ty0:
            sub = self._track_rgb[ty0:ty1, tx0:tx1]
            sub_big = np.repeat(np.repeat(sub, TILE_SIZE, axis=0),
                                TILE_SIZE, axis=1)
            tile_surf = pygame.surfarray.make_surface(sub_big.transpose(1, 0, 2))
            blit_x = int(tx0 * TILE_SIZE - cam_x)
            blit_y = int(ty0 * TILE_SIZE - cam_y)
            screen.blit(tile_surf, (blit_x, blit_y))

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
        lines = [
            (f"mode:     {mode_str}",       self.WHITE),
            (f"step:     {self.steps}",      self.WHITE),
            (f"action:   {ACTION_NAMES[self._last_action]}", self.WHITE),
            (f"reward:   {self._last_reward:+.2f}",          self.FINISH_COL),
            (f"progress: {(self.prev_arc / self.total_arc * 100) if self.total_arc > 0 else 0:.1f}%",
             self.WHITE),
            (f"speed:    {self.speed:.2f}",  self.WHITE),
        ]
        for i, (txt, col) in enumerate(lines):
            screen.blit(self._font.render(txt, 1, col), (20, 15 + i * 18))

    def _draw_car(self, surface, x, y, angle):
        import pygame
        CAR_W, CAR_H = 18, 30
        s = pygame.Surface((CAR_W + 4, CAR_H + 4), pygame.SRCALPHA)
        pygame.draw.rect(s, self.RED, (2, 2, CAR_W, CAR_H), border_radius=2)
        pygame.draw.rect(s, (160, 20, 20), (2, 2, CAR_W, 8))
        rot = pygame.transform.rotate(s, -angle)
        surface.blit(rot, rot.get_rect(center=(int(x), int(y))).topleft)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


# ===========================================================================
# Demo / sanity check
# ===========================================================================

def keyboard_play(slippery: bool = False):
    """Drive the env manually with W/A/S/D (or arrows). For sanity-checking."""
    import pygame
    env = CarRacingEnv(slippery=slippery, render_mode="human", max_steps=10_000)
    env.reset()
    # Map (throttle, steer) -> action index
    action_map = {ts: i for i, ts in enumerate(ACTIONS)}
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
