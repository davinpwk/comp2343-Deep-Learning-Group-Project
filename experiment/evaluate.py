"""Periodic evaluation harness.

`evaluate(agent, eval_envs, frictions, n_episodes)` rolls out `n_episodes`
greedy (ε=0) episodes for every (map, friction) cell and returns a dict
of summary statistics. Cell key is the tuple `(map_path, friction)`.

The eval envs are pre-built once by the caller (via
`env_utils.build_eval_env_set(...)`) and reused across every call -- map
files are touched exactly once over the lifetime of a run, regardless of
how many eval checkpoints fire.

Per cell we return:
  - return_mean / return_std         (per-episode total reward)
  - progress_mean                    (final progress_pct, in [0, 1])
  - finish_rate / crash_rate         (in [0, 1])
  - returns                          (raw list, for downstream plots)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .dqn import DQNAgent


def _rollout_one(agent: DQNAgent, env, seed: int) -> tuple[float, dict]:
    """One greedy episode. Returns (total_return, last_info)."""
    obs, _ = env.reset(seed=seed)
    total = 0.0
    info: dict = {}
    done = False
    while not done:
        a = agent.select_action(obs, greedy=True)
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        done = term or trunc
    return total, info


def evaluate(
    agent: DQNAgent,
    eval_envs: dict,
    frictions: Iterable[float],
    n_episodes: int = 10,
    base_seed: int = 1000,
) -> dict:
    """Greedy eval across the (map, friction) cross-product.

    `eval_envs` is a `{map_path: CarRacingEnv}` dict (see
    `env_utils.build_eval_env_set`). The envs are held externally so map
    files load once per run, not once per eval call.

    `base_seed` controls the reset seed sequence (`base_seed + i` for the
    i-th episode within a cell). Picking the same base_seed across calls
    means the eval is deterministic, so checkpoint-to-checkpoint deltas
    aren't polluted by reset noise.
    """
    results: dict[tuple[str, float], dict] = {}
    for map_path, env in eval_envs.items():
        for friction in frictions:
            env.set_friction(float(friction))
            returns, progresses, finishes, crashes = [], [], [], []
            for i in range(n_episodes):
                total, info = _rollout_one(agent, env, seed=base_seed + i)
                returns.append(total)
                progresses.append(float(info.get("progress_pct", 0.0)))
                finishes.append(int(info.get("finish", False)))
                crashes.append(int(info.get("wall_hit", False)))
            results[(map_path, float(friction))] = {
                "return_mean":   float(np.mean(returns)),
                "return_std":    float(np.std(returns)),
                "progress_mean": float(np.mean(progresses)),
                "finish_rate":   float(np.mean(finishes)),
                "crash_rate":    float(np.mean(crashes)),
                "returns":       [float(x) for x in returns],
            }
    return results


# ---------------------------------------------------------------------------
# Convenience: roll up cell-level results into the per-friction means
# the report's main plots want.
# ---------------------------------------------------------------------------

def aggregate_by_friction(cells: dict) -> dict:
    """Average cell return_mean across maps, grouped by friction.

    Returns `{friction: {"return_mean": ..., "return_std_across_maps": ...,
    "progress_mean": ..., "finish_rate": ..., "crash_rate": ...}}`.

    The `return_std_across_maps` is the spread of *map-level means*, not
    of individual episodes -- it's a quick "does this policy generalise
    across maps?" signal.
    """
    by_f: dict[float, list[dict]] = {}
    for (_, friction), stats in cells.items():
        by_f.setdefault(friction, []).append(stats)

    out: dict = {}
    for friction, stats_list in by_f.items():
        rets = np.array([s["return_mean"] for s in stats_list], dtype=np.float64)
        out[friction] = {
            "return_mean":            float(rets.mean()),
            "return_std_across_maps": float(rets.std()),
            "progress_mean":          float(np.mean([s["progress_mean"] for s in stats_list])),
            "finish_rate":            float(np.mean([s["finish_rate"]   for s in stats_list])),
            "crash_rate":             float(np.mean([s["crash_rate"]    for s in stats_list])),
        }
    return out
