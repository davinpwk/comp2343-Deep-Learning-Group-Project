"""Step-based DQN training loop for the transfer-learning experiment.

Differences from ../train_util.py's loop:
  - Outer budget is env steps (`cfg.max_env_steps`), not episodes.
  - Periodic eval (`evaluate(...)`) every `cfg.eval_every_env_steps`,
    on all maps and both friction conditions, logged to disk.
  - Friction mode is picked by `cfg.friction_mode` ("fixed" | "dr" |
    "curriculum") and applied via env_utils' friction providers.
  - Loads from a pretrained checkpoint (`pretrained_ckpt`) optionally,
    and supports first-layer freezing for fine-tune experiments.
  - Writes config.json, train_log.jsonl, eval_log.jsonl, checkpoint.pt
    under <save_dir>/<run_name>/seed_<seed>/.
"""
from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
import torch

from .config import ExperimentConfig
from .dqn import DQNAgent
from .env_utils import (
    MultiMapEnv,
    fixed_friction,
    domain_randomized_friction,
    CurriculumScheduler,
    build_eval_env_set,
    close_eval_env_set,
)
from .evaluate import evaluate
from .run_io import (
    run_dir,
    save_config,
    save_checkpoint,
    load_checkpoint_into,
    append_jsonl,
    serialise_eval_cells,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_friction_provider(cfg: ExperimentConfig, seed: int):
    """Build the friction provider and (if curriculum) its scheduler."""
    if cfg.friction_mode == "fixed":
        return fixed_friction(cfg.friction), None
    if cfg.friction_mode == "dr":
        rng = random.Random(seed + 31)
        return domain_randomized_friction(
            cfg.friction_dr_low, cfg.friction_dr_high, rng=rng
        ), None
    if cfg.friction_mode == "curriculum":
        sched = CurriculumScheduler(
            start=cfg.friction_curr_start,
            end=cfg.friction_curr_end,
            total_env_steps=cfg.max_env_steps,
        )
        return sched.provider, sched
    raise ValueError(
        f"Unknown friction_mode {cfg.friction_mode!r}; "
        f"expected 'fixed', 'dr', or 'curriculum'."
    )


def train(
    cfg: ExperimentConfig,
    pretrained_ckpt: Optional[str] = None,
    freeze_first: bool = False,
    verbose: bool = True,
) -> dict:
    """Train one DQN run. Returns the in-memory log dict for caller use.

    Side effects: writes config.json, train_log.jsonl, eval_log.jsonl,
    final_checkpoint.pt under <cfg.save_dir>/<cfg.run_name>/seed_<seed>/.
    """
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    save_config(cfg)
    train_log_path = rd / "train_log.jsonl"
    eval_log_path = rd / "eval_log.jsonl"
    # Wipe previous logs for this run+seed so re-runs don't append to old data.
    for p in (train_log_path, eval_log_path):
        if p.exists():
            p.unlink()

    # ---- env -----------------------------------------------------------
    friction_provider, scheduler = _make_friction_provider(cfg, cfg.seed)
    env_kwargs = dict(
        max_episode_steps=cfg.max_episode_steps,
        reward_overrides=dict(cfg.reward_overrides),
        early_terminate_backward_pct=cfg.early_terminate_backward_pct,
        early_terminate_stagnation_steps=cfg.early_terminate_stagnation_steps,
        align_spawn_to_tangent=cfg.align_spawn_to_tangent,
        action_set=cfg.action_set,
    )
    multi_env = MultiMapEnv(
        map_paths=list(cfg.map_paths),
        friction_provider=friction_provider,
        env_kwargs=env_kwargs,
        seed=cfg.seed,
    )

    # Pre-build one env per map for the eval harness. Map files load once
    # here and these envs are reused across every eval checkpoint -- if
    # we built them inside evaluate() instead, we'd pay the load cost
    # ~max_env_steps / eval_every_env_steps times per run.
    eval_envs = build_eval_env_set(cfg.map_paths, env_kwargs=env_kwargs)

    # ---- agent ---------------------------------------------------------
    obs_dim = int(np.prod(multi_env.observation_space.shape))
    n_actions = multi_env.action_space.n
    agent = DQNAgent(
        obs_dim             = obs_dim,
        n_actions           = n_actions,
        lr                  = cfg.lr,
        gamma               = cfg.gamma,
        epsilon_start       = cfg.epsilon_start,
        epsilon_end         = cfg.epsilon_end,
        epsilon_decay_steps = cfg.epsilon_decay_steps,
        batch_size          = cfg.batch_size,
        buffer_capacity     = cfg.buffer_capacity,
        target_update_freq  = cfg.target_update_freq,
        hidden_sizes        = list(cfg.hidden_sizes),
        grad_clip           = cfg.grad_clip,
        device              = cfg.device,
    )

    if pretrained_ckpt:
        load_checkpoint_into(pretrained_ckpt, agent, load_target=True)
        # After loading a pretrained policy, re-arm ε for fine-tuning. The
        # config knobs are interpreted as the *fine-tune* schedule when a
        # checkpoint is loaded.
        agent.set_epsilon_schedule(
            cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps
        )

    if freeze_first:
        agent.freeze_first_layer()

    # ---- training loop -------------------------------------------------
    if verbose:
        print()
        print("=" * 60)
        print(f"  Run     : {cfg.run_name}  seed={cfg.seed}")
        print(f"  Budget  : {cfg.max_env_steps:,} env steps")
        print(f"  Friction: {cfg.friction_mode} "
              f"(value={cfg.friction if cfg.friction_mode == 'fixed' else 'dynamic'})")
        print(f"  Maps    : {len(cfg.map_paths)}")
        print(f"  Pretrained: {pretrained_ckpt or '-'}")
        print(f"  Freeze first layer: {freeze_first}")
        print("=" * 60)

    env_steps = 0
    episodes = 0
    next_eval_at = cfg.eval_every_env_steps  # eval at this many env steps
    t0 = time.time()

    obs, _ = multi_env.reset(seed=cfg.seed)
    ep_return = 0.0
    ep_steps = 0
    ep_start_step = 0
    last_update_info: dict = {}

    try:
        while env_steps < cfg.max_env_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = multi_env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, float(reward),
                                   next_obs, terminated)

            if env_steps >= cfg.warmup_env_steps:
                u_info = agent.update()
                if u_info:
                    last_update_info = u_info

            env_steps += 1
            if scheduler is not None:
                scheduler.tick(1)

            ep_return += float(reward)
            ep_steps += 1
            obs = next_obs

            # Periodic eval -- before episode boundary so the cadence is
            # tight regardless of how long episodes happen to be.
            if env_steps >= next_eval_at or env_steps >= cfg.max_env_steps:
                cells = evaluate(
                    agent,
                    eval_envs=eval_envs,
                    frictions=cfg.eval_frictions,
                    n_episodes=cfg.eval_n_episodes,
                )
                rec = {
                    "env_steps":     env_steps,
                    "episodes":      episodes,
                    "wall_seconds":  time.time() - t0,
                    "cells":         serialise_eval_cells(cells),
                }
                append_jsonl(eval_log_path, rec)
                if verbose:
                    msg = "  ".join(
                        f"f={f:.2f}:{cells.get((cfg.map_paths[0], f), {}).get('return_mean', float('nan')):+.1f}"
                        for f in cfg.eval_frictions
                    )
                    print(f"  [eval @ {env_steps:>8,}] {msg}")
                next_eval_at = env_steps + cfg.eval_every_env_steps

            if done:
                episodes += 1
                append_jsonl(train_log_path, {
                    "env_steps":   env_steps,
                    "episode":     episodes,
                    "ep_return":   ep_return,
                    "ep_steps":    ep_steps,
                    "map":         multi_env.active_map,
                    "friction":    multi_env.active_friction,
                    "progress":    float(info.get("progress_pct", 0.0)),
                    "finish":      bool(info.get("finish", False)),
                    "crash":       bool(info.get("wall_hit", False)),
                    "epsilon":     last_update_info.get("epsilon", agent.epsilon),
                })
                if verbose and episodes % 20 == 0:
                    elapsed = time.time() - t0
                    sps = env_steps / max(elapsed, 1e-9)
                    print(f"  ep {episodes:>5} "
                          f"step {env_steps:>8,}/{cfg.max_env_steps:,}  "
                          f"ret={ep_return:+8.1f}  "
                          f"prog={info.get('progress_pct', 0)*100:>5.1f}%  "
                          f"ε={agent.epsilon:.3f}  "
                          f"sps={sps:>5.0f}")
                obs, _ = multi_env.reset()
                ep_return = 0.0
                ep_steps = 0
                ep_start_step = env_steps

    finally:
        save_checkpoint(cfg, agent, name="final_checkpoint.pt",
                        extra={"env_steps": env_steps, "episodes": episodes})
        multi_env.close()
        close_eval_env_set(eval_envs)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Done. env_steps={env_steps:,} episodes={episodes} "
              f"wall={elapsed:.1f}s")

    return {
        "env_steps": env_steps,
        "episodes":  episodes,
        "run_dir":   str(rd),
    }
