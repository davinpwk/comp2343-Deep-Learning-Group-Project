"""
Training utilities for DQN on CarRacingEnv.

Mirrors the structure of lab10's `train_util.py` (set_seed, TrainConfig,
EpisodeLogger) but adapted to the car project:

  - No `env_id`: we instantiate `CarRacingEnv` directly so the
    slippery/reward/early-term knobs are first-class config.
  - `EpisodeLogger` tracks the extra car fields (progress, finish, crash)
    that aren't in CartPole.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All DQN + env knobs in one place. Same shape as lab10's TrainConfig,
    but with the car-env-specific fields surfaced at the top level."""

    # === environment ===
    slippery: bool = False                 # False = pretrain, True = transfer target
    max_steps: int = 3000                  # per-episode hard cap
    # Mutable overrides on top of env defaults. Empty dict = use env defaults
    # (progress=3, time=-0.05, off_road=-3, wall_hit=-10, finish=100, centering=-0.5).
    reward_overrides: dict = field(default_factory=dict)
    early_terminate_backward_pct: float = 0.05     # truncate if car loses >5% of peak arc
    early_terminate_stagnation_steps: int = 300    # truncate after 300 stale steps
    align_spawn_to_tangent: bool = True            # spawn pointing along the track
    action_set: str = "no_noop"                    # "full" (7 actions) | "no_noop" (6)

    # === training ===
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 5e-4
    seed: int = 42
    device: str = "cpu"
    render: bool = False                    # render every step during training (slow)
    save_dir: str = "checkpoints"

    # === DQN ===
    batch_size: int = 64
    buffer_capacity: int = 100_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden_sizes: tuple = (128, 128)
    grad_clip: float = 10.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """Per-episode console + in-memory logger.

    Same API as lab10's logger: `record(ep, ret, steps, extra)` and
    `summary()`. Plot includes a finish/crash overlay if those keys appear
    in the per-episode `extra` dict.
    """

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every
        self.history: list[dict[str, Any]] = []

    def record(self, ep: int, ret: float, steps: int, extra: dict) -> None:
        record = {"episode": ep, "return": ret, "steps": steps, **extra}
        self.history.append(record)

        if ep % self.log_every == 0 or ep == 1:
            recent = [h["return"] for h in self.history[-20:]]
            avg = float(np.mean(recent))
            # progress is car-specific; format if present, else fall through.
            progress = extra.get("progress", None)
            prog_str = f"  progress={progress * 100:>5.1f}%" if progress is not None else ""
            extra_str = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in extra.items() if k != "progress"
            )
            print(
                f"[Ep {ep:>5}] return={ret:>7.1f}  "
                f"avg20={avg:>7.1f}  steps={steps:>4}{prog_str}  "
                + extra_str
            )

    def summary(self) -> None:
        if not self.history:
            return
        returns = [h["return"] for h in self.history]
        episodes = [h["episode"] for h in self.history]
        print("\n" + "=" * 60)
        print(f"  Episodes   : {len(returns)}")
        print(f"  Mean return: {np.mean(returns):.2f}")
        print(f"  Max return : {np.max(returns):.2f}")
        print(f"  Last 50 avg: {np.mean(returns[-50:]):.2f}")
        if "progress" in self.history[-1]:
            last50_progress = [h.get("progress", 0.0) for h in self.history[-50:]]
            print(f"  Last 50 progress avg: {np.mean(last50_progress) * 100:.1f}%")
        if "finish" in self.history[-1]:
            last50_finish = [int(h.get("finish", False)) for h in self.history[-50:]]
            print(f"  Last 50 finish rate : {np.mean(last50_finish) * 100:.1f}%")
        print("=" * 60)

        # Always plot return; if car-specific keys exist, overlay progress.
        has_progress = "progress" in self.history[-1]
        if has_progress:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            ax_ret, ax_prog = axes
        else:
            fig, ax_ret = plt.subplots(figsize=(10, 4))
            ax_prog = None

        ax_ret.plot(episodes, returns, linewidth=0.8, alpha=0.5, label="Return")
        window = min(20, len(returns))
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax_ret.plot(episodes[window - 1:], smoothed, linewidth=2, label=f"Avg{window}")
        ax_ret.set_xlabel("Episode")
        ax_ret.set_ylabel("Return")
        ax_ret.set_title("Training Return")
        ax_ret.legend()
        ax_ret.grid(True, alpha=0.3)

        if ax_prog is not None:
            progs = [h.get("progress", 0.0) * 100 for h in self.history]
            ax_prog.plot(episodes, progs, linewidth=0.8, alpha=0.5)
            smoothed_p = np.convolve(progs, np.ones(window) / window, mode="valid")
            ax_prog.plot(episodes[window - 1:], smoothed_p, linewidth=2,
                         label=f"Avg{window}")
            ax_prog.set_xlabel("Episode")
            ax_prog.set_ylabel("Progress (%)")
            ax_prog.set_title("Final-step Progress")
            ax_prog.legend()
            ax_prog.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
