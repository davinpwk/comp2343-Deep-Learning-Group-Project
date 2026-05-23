"""Filesystem layout + serialisation helpers for experiment runs.

Layout: `<save_dir>/<run_name>/seed_<seed>/`
  ├── config.json       # frozen ExperimentConfig
  ├── checkpoint.pt     # final agent weights (and intermediate, if asked)
  ├── train_log.json    # list of {env_steps, episodes, return, ...} records
  └── eval_log.json     # list of {env_steps, cells, by_friction} records

JSON is used for the logs (small, diffable, no torch dependency to read).
Torch's pickle format is used for the agent weights.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import torch

from .config import ExperimentConfig
from .dqn import DQNAgent


def run_dir(cfg: ExperimentConfig) -> pathlib.Path:
    """Return (and create) the run directory for a config."""
    p = pathlib.Path(cfg.save_dir) / cfg.run_name / f"seed_{cfg.seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(cfg: ExperimentConfig) -> pathlib.Path:
    """Freeze the config to <run_dir>/config.json. Idempotent."""
    path = run_dir(cfg) / "config.json"
    path.write_text(json.dumps(cfg.to_dict(), indent=2, default=_json_default))
    return path


def save_checkpoint(
    cfg: ExperimentConfig,
    agent: DQNAgent,
    name: str = "checkpoint.pt",
    extra: dict | None = None,
) -> pathlib.Path:
    """Save agent weights (online + target + ε) to <run_dir>/<name>."""
    path = run_dir(cfg) / name
    payload: dict[str, Any] = {"agent": agent.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint_into(path: str | os.PathLike, agent: DQNAgent,
                        load_target: bool = True) -> dict:
    """Load weights from `path` into an already-constructed agent.

    The agent must have matching obs_dim/n_actions/hidden_sizes -- we
    don't try to reconstruct the architecture from the checkpoint here,
    because the experiment always builds the agent from a known config
    before loading.
    """
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(payload["agent"], load_target=load_target)
    return payload.get("extra", {})


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    """Append a single JSON line to `path`. Creates the file if missing."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    """Read a .jsonl file written by `append_jsonl` back into a list."""
    if not pathlib.Path(path).exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Fallback for tuples-of-tuples, numpy scalars, and (map_path, friction)
    tuple keys that show up in eval cells."""
    if isinstance(obj, tuple):
        return list(obj)
    # numpy scalars / arrays
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(obj).__name__}")


def serialise_eval_cells(cells: dict) -> list[dict]:
    """Convert evaluate()'s `{(map, friction): stats}` to JSON-safe list-of-dicts."""
    out = []
    for (map_path, friction), stats in cells.items():
        row = {"map": map_path, "friction": float(friction)}
        row.update(stats)
        out.append(row)
    return out
