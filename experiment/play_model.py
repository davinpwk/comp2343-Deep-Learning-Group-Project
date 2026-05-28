"""Greedy playback of a trained DQN checkpoint, rendered live.

Mirrors the keyboard_play CLI in car_env.py but drives with a learned policy
instead of WASD. Episodes auto-reset; close the pygame window to quit.

Usage (from project root):
    # Run by run directory (loads final_checkpoint.pt + config.json).
    python -m experiment.play_model --run runs/phase1_pilot_normal/seed_0

    # Override the map and/or friction.
    python -m experiment.play_model --run runs/phase1_pilot_slippery/seed_0 \
        --map maps/winding_frequent.txt --friction 0.95

    # Or point at a bare checkpoint (config.json in the same dir is optional).
    python -m experiment.play_model --ckpt checkpoints/pretrained_normal_seed_0.pt \
        --map maps/straight_turn.txt
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .car_env import CarRacingEnv
from .dqn import DQNAgent
from .run_io import load_checkpoint_into


def _resolve_ckpt_and_config(run_dir: pathlib.Path | None,
                             ckpt_path: pathlib.Path | None
                             ) -> tuple[pathlib.Path, dict]:
    """Resolve the checkpoint path and (best-effort) load the sibling config."""
    if run_dir is not None:
        ck = run_dir / "final_checkpoint.pt"
        cf = run_dir / "config.json"
    else:
        ck = ckpt_path
        cf = ckpt_path.parent / "config.json"
    if not ck.exists():
        raise SystemExit(f"checkpoint not found: {ck}")
    config = json.loads(cf.read_text()) if cf.exists() else {}
    return ck, config


def _pick_friction(config: dict, override: float | None) -> float:
    """Decide what friction to demo at.

    Priority: --friction CLI > config's `friction` (if fixed mode) >
    `friction_curr_end` (curriculum) > midpoint of DR range > 0.1.
    """
    if override is not None:
        return float(override)
    mode = config.get("friction_mode", "fixed")
    if mode == "fixed":
        return float(config.get("friction", 0.1))
    if mode == "curriculum":
        return float(config.get("friction_curr_end", 0.95))
    if mode == "dr":
        lo = float(config.get("friction_dr_low", 0.1))
        hi = float(config.get("friction_dr_high", 0.95))
        return 0.5 * (lo + hi)
    return 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=pathlib.Path,
                     help="Run directory: loads final_checkpoint.pt + config.json.")
    src.add_argument("--ckpt", type=pathlib.Path,
                     help="Explicit checkpoint .pt path (config.json in same dir is optional).")
    ap.add_argument("--map", dest="map_path", default=None,
                    help="Map file path (defaults to the first map in the run's config).")
    ap.add_argument("--friction", type=float, default=None,
                    help="Friction override (defaults to the training friction in config).")
    ap.add_argument("--max-steps", type=int, default=10_000,
                    help="Per-episode step cap (default 10000).")
    args = ap.parse_args()

    ck, cfg = _resolve_ckpt_and_config(args.run, args.ckpt)

    map_path     = args.map_path or (cfg.get("map_paths") or [None])[0]
    friction     = _pick_friction(cfg, args.friction)
    hidden_sizes = list(cfg.get("hidden_sizes", [128, 128]))
    action_set   = cfg.get("action_set", "no_noop")

    print(f"checkpoint : {ck}")
    print(f"map        : {map_path or '(procedural)'}")
    print(f"friction   : {friction}")
    print(f"action set : {action_set}")
    print(f"hidden     : {hidden_sizes}")
    print("close the pygame window to quit; episodes auto-reset.")

    env = CarRacingEnv(
        map_path=map_path,
        render_mode="human",
        max_steps=args.max_steps,
        action_set=action_set,
    )
    env.set_friction(float(friction))

    obs, _ = env.reset()
    agent = DQNAgent(
        obs_dim      = int(np.prod(env.observation_space.shape)),
        n_actions    = env.action_space.n,
        hidden_sizes = hidden_sizes,
        device       = "cpu",
    )
    load_checkpoint_into(str(ck), agent, load_target=False)

    import pygame
    env.render()  # spawn the window before we start polling events

    total = 0.0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                env.close()
                return
        action = agent.select_action(obs, greedy=True)
        obs, r, term, trunc, info = env.step(action)
        total += float(r)
        if term or trunc:
            print(f"  episode end: return={total:+.1f}  "
                  f"progress={info.get('progress_pct', 0):.1%}  "
                  f"finish={info.get('finish', False)}  "
                  f"crash={info.get('wall_hit', False)}")
            obs, _ = env.reset()
            total = 0.0


if __name__ == "__main__":
    main()
