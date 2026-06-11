#!/usr/bin/env python3
"""Render one stitched multi-map GIF for a single evaluation run directory.

Given a run directory (a `seed_*` folder holding periodic `ckpt_step_*.pt`
checkpoints, a `final_checkpoint.pt`, and a `config.json`), this renders the
multi-map rollout at every 20,000-step checkpoint and stitches them -- in step
order -- into a *single* GIF. Each segment carries a banner naming the
checkpoint it came from. If the run's total training length isn't a multiple of
the interval, `final_checkpoint.pt` is appended too, so the last policy is
always captured.

The run is demoed at its *evaluation* friction (the same rule
`experiment.play_model` uses): fixed -> the configured friction, curriculum ->
its end friction, domain-randomisation -> the midpoint of its range.

Run as a script (from the repo root or anywhere):

    python scripts/render_eval_gifs.py runs/phase3_final/E1/seed_101
    python scripts/render_eval_gifs.py runs/phase3_pretrain/seed_101 --out demo.gif
    python scripts/render_eval_gifs.py runs/phase3_final/E1/seed_101 --dry-run

By default the GIF is written under `gifs/eval/`, mirroring the run's path
relative to `runs/`, e.g. `gifs/eval/phase3_final/E1/seed_101.gif`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make sibling scripts (render_gifs) and the `experiment` package importable.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_gifs import DEFAULT_MAPS, rollout_frames, save_gif
from experiment.play_model import _pick_friction

_CKPT_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def _step_of(ckpt: Path) -> int | None:
    m = _CKPT_RE.search(ckpt.name)
    return int(m.group(1)) if m else None


def _resolve_run_dir(arg: str) -> Path:
    """Resolve the run dir, falling back to one relative to the repo root."""
    d = Path(arg)
    if not d.is_absolute() and not d.exists():
        d = REPO_ROOT / arg
    return d.resolve()


def _rel_to_runs(run_dir: Path) -> Path:
    """Path of the run relative to `runs/`, or just its name if outside it."""
    try:
        return run_dir.relative_to((REPO_ROOT / "runs").resolve())
    except ValueError:
        return Path(run_dir.name)


def plan_checkpoints(run_dir: Path, interval: int) -> tuple[list[Path], int]:
    """Pick the checkpoints to render for one run.

    Returns `(checkpoints, total_steps)` where `checkpoints` are the existing
    `ckpt_step_*` files at each `interval`-step multiple up to the run's length,
    with `final_checkpoint.pt` appended when `total_steps` isn't a multiple of
    `interval`. `total_steps` is taken as the largest periodic checkpoint step.
    """
    by_step = {
        s: p for p in run_dir.glob("ckpt_step_*.pt")
        if (s := _step_of(p)) is not None
    }
    total = max(by_step) if by_step else 0

    chosen: list[Path] = []
    if total and interval > 0:
        for step in range(interval, total + 1, interval):
            ckpt = by_step.get(step)
            if ckpt is not None:
                chosen.append(ckpt)
            else:
                print(f"  ! missing expected checkpoint at step {step} in {run_dir}")

    final = run_dir / "final_checkpoint.pt"
    if final.exists() and (total == 0 or total % interval != 0):
        chosen.append(final)

    return chosen, total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir",
                    help="A single run directory, e.g. runs/phase3_final/E1/seed_101.")
    ap.add_argument("--out", default=None,
                    help="Output GIF path. Default: gifs/eval/<run path relative to runs>.gif")
    ap.add_argument("--interval", type=int, default=20_000,
                    help="Stitch a segment every N training steps (default 20000).")
    ap.add_argument("--friction", type=float, default=None,
                    help="Override the run's evaluation friction.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the checkpoints that would be stitched, without rendering.")
    # Rollout/encoding knobs forwarded to rollout_frames / save_gif.
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir)
    if not (run_dir / "config.json").exists():
        print(f"Not a run directory (no config.json): {run_dir}", file=sys.stderr)
        return 1

    rel = _rel_to_runs(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    friction = args.friction if args.friction is not None else _pick_friction(cfg, None)
    maps = list(cfg.get("map_paths") or DEFAULT_MAPS)
    ckpts, total = plan_checkpoints(run_dir, args.interval)
    if not ckpts:
        print(f"No checkpoints found in {run_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else Path("gifs/eval") / f"{rel}.gif"
    note = "" if total % args.interval == 0 else " (+final: total not divisible)"
    print(f"{rel}  total={total} steps  friction={friction:.3g}  "
          f"-> 1 stitched gif from {len(ckpts)} checkpoint(s){note}")

    if args.dry_run:
        print(f"  would write {out_path}")
        for ckpt in ckpts:
            print(f"      {ckpt.name}")
        return 0

    frames: list = []
    for ckpt in ckpts:
        step = _step_of(ckpt)
        tag = f"step {step:,}" if step is not None else "final"
        banner = f"{rel}   {tag}   friction {friction:.3g}"
        print(f"  rollout {ckpt.name} ...")
        frames += rollout_frames(
            ckpt, maps,
            friction=friction, stride=args.stride, scale=args.scale,
            max_steps=args.max_steps, seed=args.seed, device=args.device,
            banner=banner,
        )
    save_gif(frames, out_path, fps=args.fps, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
