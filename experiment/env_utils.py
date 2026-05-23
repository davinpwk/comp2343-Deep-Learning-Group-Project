"""Env wrappers for the transfer experiment.

  - `build_env(map_path, **kwargs)`: thin factory around `CarRacingEnv`
    that hides the env-kwarg dance and applies an initial friction.
  - `MultiMapEnv`: holds N pre-built CarRacingEnv instances (one per
    map), and on `reset()` randomly picks one. The active env's
    `step()` is forwarded directly. Friction can be set either via a
    fixed value, sampled per-episode (DR), or driven by a curriculum
    scheduler. Episode-level RNG is seeded from the training seed for
    reproducibility.

The reason for *pre-building* one env per map (instead of constructing
on-the-fly inside reset) is that env construction loads the map file
from disk, builds the centerline, and pre-renders the track surface
(if rendering) -- all relatively expensive. Pre-building once and then
just rotating between instances keeps per-episode reset fast.
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Optional

import numpy as np

from .car_env import CarRacingEnv


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def build_env(
    map_path: str,
    *,
    friction: float = 0.1,
    max_episode_steps: int = 3000,
    reward_overrides: dict = None,
    early_terminate_backward_pct: float = 0.05,
    early_terminate_stagnation_steps: int = 300,
    align_spawn_to_tangent: bool = True,
    action_set: str = "no_noop",
    render_mode: Optional[str] = None,
) -> CarRacingEnv:
    """Build a single CarRacingEnv from a map file with the right kwargs.

    `friction` is applied via `env.set_friction(...)` after construction
    so we don't go through the `slippery` boolean.
    """
    env = CarRacingEnv(
        map_path=map_path,
        max_steps=max_episode_steps,
        reward_config=reward_overrides,
        early_terminate_backward_pct=early_terminate_backward_pct,
        early_terminate_stagnation_steps=early_terminate_stagnation_steps,
        align_spawn_to_tangent=align_spawn_to_tangent,
        action_set=action_set,
        render_mode=render_mode,
    )
    env.set_friction(friction)
    return env


# ---------------------------------------------------------------------------
# Pre-built eval env cache
# ---------------------------------------------------------------------------

def build_eval_env_set(
    map_paths: Iterable[str],
    env_kwargs: dict | None = None,
) -> dict[str, "CarRacingEnv"]:
    """Build one env per map for use by evaluate().

    Returns a `{map_path: CarRacingEnv}` dict. Each env is constructed
    once (map loaded + centerline built + track surface implicitly
    cached) and then reused across every eval call -- friction is the
    only thing that varies per call, and it's set via env.set_friction.

    Caller is responsible for `env.close()` (use `close_eval_env_set`).
    """
    env_kwargs = dict(env_kwargs or {})
    return {p: build_env(p, friction=0.1, **env_kwargs) for p in map_paths}


def close_eval_env_set(envs: dict[str, "CarRacingEnv"]) -> None:
    for e in envs.values():
        e.close()


# ---------------------------------------------------------------------------
# Multi-map wrapper
# ---------------------------------------------------------------------------

class MultiMapEnv:
    """Random-sample-a-map-on-reset wrapper around N CarRacingEnv instances.

    Not a Gymnasium subclass on purpose -- the action/observation spaces
    are identical across the underlying envs (same action set, same obs
    dim), but Gymnasium's vector envs would over-complicate the
    single-agent training loop. We just forward `reset` / `step`.

    Friction handling:
      - `friction_provider` is a callable `() -> float` invoked at every
        `reset()` to determine the friction of the upcoming episode.
        For fixed friction, pass `lambda: 0.1`. For domain randomisation,
        pass `lambda: rng.uniform(lo, hi)`. For curriculum, pass a
        closure over a step counter (see CurriculumScheduler below).
    """

    def __init__(
        self,
        map_paths: list[str],
        friction_provider: Callable[[], float],
        env_kwargs: dict = None,
        seed: int = 0,
    ) -> None:
        if not map_paths:
            raise ValueError("MultiMapEnv needs at least one map_path.")
        env_kwargs = dict(env_kwargs or {})
        # Use a fixed neutral friction at construction; the real friction
        # is set per-episode by friction_provider via reset().
        self.envs: list[CarRacingEnv] = [
            build_env(p, friction=0.1, **env_kwargs) for p in map_paths
        ]
        self.map_paths = list(map_paths)
        self.friction_provider = friction_provider

        # Both envs must agree on observation/action shape -- they do as
        # long as the action_set and ray config match, but be defensive.
        first = self.envs[0]
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        for e, p in zip(self.envs[1:], self.map_paths[1:]):
            if e.action_space.n != first.action_space.n:
                raise ValueError(f"Action space mismatch on {p}")
            if e.observation_space.shape != first.observation_space.shape:
                raise ValueError(f"Observation space mismatch on {p}")

        # The map-choice RNG is independent of the per-env Gymnasium RNGs
        # so that we can reproduce the same map sequence regardless of
        # what each env does internally.
        self._map_rng = random.Random(seed)
        self._active_idx: int = 0
        self._active_friction: float = 0.1

    # ---- gymnasium-style API ------------------------------------------

    def reset(self, seed: Optional[int] = None):
        self._active_idx = self._map_rng.randrange(len(self.envs))
        self._active_friction = float(self.friction_provider())
        env = self.envs[self._active_idx]
        env.set_friction(self._active_friction)
        return env.reset(seed=seed)

    def step(self, action):
        return self.envs[self._active_idx].step(action)

    def close(self):
        for e in self.envs:
            e.close()

    # ---- introspection ------------------------------------------------

    @property
    def active_map(self) -> str:
        return self.map_paths[self._active_idx]

    @property
    def active_friction(self) -> float:
        return self._active_friction


# ---------------------------------------------------------------------------
# Friction providers
# ---------------------------------------------------------------------------

def fixed_friction(value: float) -> Callable[[], float]:
    """Friction provider that returns `value` every reset."""
    v = float(value)
    return lambda: v


def domain_randomized_friction(
    low: float, high: float, rng: Optional[random.Random] = None
) -> Callable[[], float]:
    """Friction provider that samples U[low, high] every reset.

    Pass an explicit `rng` (seeded) so the sampled sequence is
    reproducible across runs.
    """
    if not (0.0 <= low <= high < 1.0):
        raise ValueError(f"need 0 <= low <= high < 1; got {low}, {high}")
    r = rng if rng is not None else random.Random()
    return lambda: r.uniform(low, high)


class CurriculumScheduler:
    """Linearly ramp friction from `start` to `end` over `total_env_steps`.

    Usage: pass `scheduler.provider` to MultiMapEnv as `friction_provider`,
    and call `scheduler.tick(n_steps)` from the training loop each step
    (or in bulk per episode). The provider reads the current step count
    when reset() is invoked, so the friction is sampled at episode start
    and held fixed for the episode (matching how DR/fixed work).
    """

    def __init__(self, start: float, end: float, total_env_steps: int) -> None:
        if total_env_steps <= 0:
            raise ValueError("total_env_steps must be > 0")
        self.start = float(start)
        self.end = float(end)
        self.total = int(total_env_steps)
        self.env_steps = 0

    def tick(self, n: int = 1) -> None:
        self.env_steps = min(self.env_steps + int(n), self.total)

    def current_friction(self) -> float:
        if self.total == 0:
            return self.end
        t = min(self.env_steps / self.total, 1.0)
        return self.start + (self.end - self.start) * t

    @property
    def provider(self) -> Callable[[], float]:
        # Captures self so the provider reads live values.
        return self.current_friction
