#!/usr/bin/env python3
"""Render trained DQN checkpoints driving all three maps at once, as GIFs.

For each checkpoint you pass, this rolls the greedy policy out on the three
training maps *simultaneously* -- one panel per map, stepped in lockstep -- and
records the frames until every episode has ended (finish, crash, stagnation, or
the step cap). Panels that finish early freeze on their last frame while the
others keep going. The three panels are composited side-by-side into a single
GIF per model, then the next checkpoint is rendered.

Run as a script (from the repo root or anywhere):

    python scripts/render_gifs.py checkpoints/dqn_normal_best.pt
    python scripts/render_gifs.py checkpoints/*.pt --out-dir gifs --scale 0.4
    python scripts/render_gifs.py ckpt.pt --friction 0.05   # demo on ice

Or import it from another script:

    from render_gifs import generate_gifs, make_multimap_gif, DEFAULT_MAPS
    paths = generate_gifs(["checkpoints/dqn_normal_best.pt"], out_dir="gifs")

The public entry points are `generate_gifs` (loop over checkpoints) and
`make_multimap_gif` (one checkpoint). Importing has no side effects beyond
putting the repo root on sys.path so `experiment` resolves, and asking SDL for
a headless (dummy) video driver so rendering needs no display.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Headless rendering: no display needed. setdefault so an explicit driver wins.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Make `experiment` importable no matter what the current working directory is.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.dqn import DQNAgent
from experiment.env_utils import build_env

# The three maps the agents are trained and evaluated on (see config.py).
DEFAULT_MAPS = (
    "maps/winding.txt",
    "maps/winding_frequent.txt",
    "maps/winding_varying_width.txt",
)


# ---------------------------------------------------------------------------
# Path + checkpoint helpers
# ---------------------------------------------------------------------------
def _resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a path, falling back to one relative to the repo root."""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.exists() else path


def _extract_online_state_dict(payload) -> dict:
    """Pull the online QNet weights out of either checkpoint format.

    Two shapes show up in checkpoints/:
      - run_io format:  {"agent": {"online": <qnet sd>, ...}, "extra": ...}
      - try2/best format: {"q_state_dict": <qnet sd>, "cfg": ...}
    A bare QNet state_dict (net.0.weight, ...) is also accepted.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Unrecognised checkpoint object: {type(payload).__name__}")
    if "agent" in payload and isinstance(payload["agent"], dict):
        return payload["agent"]["online"]
    if "q_state_dict" in payload:
        return payload["q_state_dict"]
    if any(k.startswith("net.") for k in payload):
        return payload
    raise ValueError(
        f"Can't find QNet weights in checkpoint; top keys: {list(payload)[:6]}"
    )


def _infer_arch(qsd: dict) -> tuple[int, list[int], int]:
    """Infer (obs_dim, hidden_sizes, n_actions) from QNet Linear weight shapes.

    The QNet is `nn.Sequential(Linear, ReLU, Linear, ReLU, ..., Linear)`, so the
    Linear layers live at even indices `net.{0,2,4,...}.weight`. Each weight is
    (out_features, in_features); reading them in order recovers the MLP shape
    without needing the training config.
    """
    linears = sorted(
        (int(k.split(".")[1]), v)
        for k, v in qsd.items()
        if k.startswith("net.") and k.endswith(".weight")
    )
    if not linears:
        raise ValueError("No `net.*.weight` Linear layers found in state dict.")
    outs = [w.shape[0] for _, w in linears]
    obs_dim = int(linears[0][1].shape[1])
    n_actions = int(outs[-1])
    hidden = [int(o) for o in outs[:-1]]
    return obs_dim, hidden, n_actions


def _resolve_action_set(ckpt_path: Path, payload) -> str:
    """Best-effort action_set: sibling config.json > embedded cfg > 'no_noop'."""
    cfg_file = ckpt_path.parent / "config.json"
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text()).get("action_set", "no_noop")
        except (OSError, json.JSONDecodeError):
            pass
    if isinstance(payload, dict) and isinstance(payload.get("cfg"), dict):
        return payload["cfg"].get("action_set", "no_noop")
    return "no_noop"


def load_policy(ckpt_path: str | os.PathLike, device: str = "cpu") -> tuple[DQNAgent, dict]:
    """Load a checkpoint into a greedy-ready DQNAgent, inferring its architecture.

    Returns `(agent, meta)` where `meta` carries the inferred `obs_dim`,
    `hidden_sizes`, `n_actions`, and resolved `action_set` -- enough to build a
    matching env. The agent's online net holds the loaded weights; the target
    net is irrelevant for greedy playback.
    """
    import torch

    ckpt_path = _resolve_path(ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    online_sd = _extract_online_state_dict(payload)
    obs_dim, hidden, n_actions = _infer_arch(online_sd)
    action_set = _resolve_action_set(ckpt_path, payload)

    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_sizes=hidden, device=device,
    )
    agent.online_net.load_state_dict(online_sd)
    agent.online_net.eval()
    meta = {
        "obs_dim": obs_dim, "hidden_sizes": hidden,
        "n_actions": n_actions, "action_set": action_set,
    }
    return agent, meta


# ---------------------------------------------------------------------------
# Frame compositing
# ---------------------------------------------------------------------------
def _font(size: int = 16) -> ImageFont.ImageFont:
    """A legible font, falling back to PIL's bitmap default if no TTF is found."""
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_HEADER_H = 26
_BANNER_H = 26


def _panel(frame: np.ndarray, label: str, scale: float, font) -> Image.Image:
    """Downscale one env frame and stamp a header strip with its status label."""
    img = Image.fromarray(frame)
    if scale != 1.0:
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.BILINEAR,
        )
    out = Image.new("RGB", (img.width, img.height + _HEADER_H), (20, 20, 24))
    out.paste(img, (0, _HEADER_H))
    ImageDraw.Draw(out).text((6, 5), label, fill=(255, 255, 255), font=font)
    return out


def _composite(frames, labels, scale: float, font, banner: str | None = None) -> Image.Image:
    """Lay the per-map panels out left-to-right into one frame.

    If `banner` is given, a full-width caption strip is drawn across the top --
    used when stitching several checkpoints into one GIF so each segment is
    labelled with the checkpoint it came from.
    """
    panels = [_panel(f, lab, scale, font) for f, lab in zip(frames, labels)]
    width = sum(p.width for p in panels)
    body_h = max(p.height for p in panels)
    banner_h = _BANNER_H if banner else 0
    canvas = Image.new("RGB", (width, body_h + banner_h), (20, 20, 24))
    if banner:
        ImageDraw.Draw(canvas).text((6, 5), banner, fill=(255, 230, 120), font=font)
    x = 0
    for p in panels:
        canvas.paste(p, (x, banner_h))
        x += p.width
    return canvas


def _status(map_path: str, step: int, info: dict, done: bool) -> str:
    """One-line per-panel caption: map name, step, progress, and outcome."""
    name = Path(map_path).stem
    pct = float(info.get("progress_pct", 0.0)) * 100.0
    tag = ""
    if done:
        if info.get("finish"):
            tag = "  FINISH"
        elif info.get("wall_hit"):
            tag = "  CRASH"
        else:
            tag = "  END"
    return f"{name}  step {step}  {pct:4.1f}%{tag}"


# ---------------------------------------------------------------------------
# Rollout + GIF writing
# ---------------------------------------------------------------------------
def rollout_frames(
    ckpt_path: str | os.PathLike,
    map_paths: Sequence[str] = DEFAULT_MAPS,
    *,
    friction: float = 0.9,
    stride: int = 3,
    scale: float = 0.5,
    max_steps: int = 3000,
    seed: int = 0,
    device: str = "cpu",
    banner: str | None = None,
) -> list[Image.Image]:
    """Roll one checkpoint out across every map at once, returning the frames.

    All maps are stepped in lockstep; a map that ends freezes on its last frame
    while the rest continue, and the rollout stops once every map has ended (or
    `max_steps` is hit). Only every `stride`-th step is rendered and kept
    (dropped steps are still simulated but not rendered, so a larger stride cuts
    render time as well as frame count; each map's terminal frame is always
    rendered). `scale` shrinks each panel. If `banner` is given it is drawn as a
    full-width caption on every frame -- handy when stitching several rollouts
    into one GIF.

    Returns the list of composited PIL frames (does not write anything).
    """
    agent, meta = load_policy(ckpt_path, device=device)

    envs = [
        build_env(
            str(_resolve_path(m)),
            friction=friction,
            action_set=meta["action_set"],
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )
        for m in map_paths
    ]
    font = _font()

    try:
        obs, infos = [], []
        for env in envs:
            o, _ = env.reset(seed=seed)
            obs.append(o)
            infos.append({})
        done = [False] * len(envs)
        panels = [env.render() for env in envs]

        labels = [_status(m, 0, {}, False) for m in map_paths]
        frames = [_composite(panels, labels, scale, font, banner)]

        step = 0
        while not all(done) and step < max_steps:
            step += 1
            capture = (step % stride == 0)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                action = agent.select_action(obs[i], greedy=True)
                obs[i], _, term, trunc, infos[i] = env.step(action)
                done[i] = bool(term or trunc)
                # Only render when this frame will be kept, or to freeze the
                # terminal frame the instant this map ends (so a crash/finish is
                # never missed). Dropped steps are stepped but not rendered --
                # that's what makes a larger stride cut render time, not just
                # shrink the GIF.
                if capture or done[i]:
                    panels[i] = env.render()
            if capture or all(done):
                labels = [
                    _status(m, step, infos[i], done[i])
                    for i, m in enumerate(map_paths)
                ]
                frames.append(_composite(panels, labels, scale, font, banner))
    finally:
        for env in envs:
            env.close()

    return frames


def save_gif(
    frames: Sequence[Image.Image],
    out_path: str | os.PathLike,
    *,
    fps: int = 20,
    speed: float = 1.0,
) -> Path:
    """Write a list of PIL frames out as a single looping GIF.

    `fps` sets the base playback rate; `speed` is a multiplier on top of it
    (2.0 plays twice as fast by halving each frame's on-screen duration).
    """
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    if not frames:
        raise ValueError("No frames to write.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / (fps * speed)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Wrote {out_path}  ({len(frames)} frames, "
        f"{frames[0].width}x{frames[0].height}px)"
    )
    return out_path


def make_multimap_gif(
    ckpt_path: str | os.PathLike,
    out_path: str | os.PathLike,
    map_paths: Sequence[str] = DEFAULT_MAPS,
    *,
    friction: float = 0.9,
    fps: int = 20,
    speed: float = 1.0,
    stride: int = 3,
    scale: float = 0.5,
    max_steps: int = 3000,
    seed: int = 0,
    device: str = "cpu",
    banner: str | None = None,
) -> Path:
    """Render one checkpoint driving every map simultaneously into a single GIF.

    Thin wrapper: `rollout_frames(...)` then `save_gif(...)`. See those for the
    `stride`/`scale`/`fps`/`speed`/`banner` semantics. Returns the written path.
    """
    frames = rollout_frames(
        ckpt_path, map_paths,
        friction=friction, stride=stride, scale=scale,
        max_steps=max_steps, seed=seed, device=device, banner=banner,
    )
    return save_gif(frames, out_path, fps=fps, speed=speed)


def generate_gifs(
    ckpt_paths: Sequence[str | os.PathLike],
    out_dir: str | os.PathLike = "gifs",
    map_paths: Sequence[str] = DEFAULT_MAPS,
    **kwargs,
) -> list[Path]:
    """Render one multi-map GIF per checkpoint into `out_dir`.

    Each GIF is named `<checkpoint-stem>.gif`. Extra keyword arguments
    (`friction`, `fps`, `stride`, `scale`, `max_steps`, `seed`, `device`) are
    forwarded to `make_multimap_gif`. Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []
    for i, ckpt in enumerate(ckpt_paths, 1):
        ckpt = _resolve_path(ckpt)
        out_path = out_dir / f"{ckpt.stem}.gif"
        print(f"[{i}/{len(ckpt_paths)}] {ckpt.name} -> {out_path}")
        written.append(
            make_multimap_gif(ckpt, out_path, map_paths=map_paths, **kwargs)
        )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+", help="One or more .pt checkpoint files.")
    ap.add_argument("--out-dir", default="gifs", help="Directory for the GIFs (default: gifs).")
    ap.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS),
                    help="Map files to drive simultaneously (default: the three training maps).")
    ap.add_argument("--friction", type=float, default=0.9,
                    help="Surface friction: 0.9 grippy (default), 0.05 icy.")
    ap.add_argument("--fps", type=int, default=20, help="Base GIF playback rate (default 20).")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier on top of --fps (e.g. 2 = twice as fast).")
    ap.add_argument("--stride", type=int, default=3,
                    help="Keep every Nth simulation step as a GIF frame (default 3).")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Per-panel downscale factor (default 0.5).")
    ap.add_argument("--max-steps", type=int, default=3000, help="Per-episode step cap.")
    ap.add_argument("--seed", type=int, default=0, help="Reset seed for every map.")
    ap.add_argument("--device", default="cpu", help="Torch device (default cpu).")
    args = ap.parse_args(argv)

    generate_gifs(
        args.checkpoints,
        out_dir=args.out_dir,
        map_paths=args.maps,
        friction=args.friction,
        fps=args.fps,
        speed=args.speed,
        stride=args.stride,
        scale=args.scale,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
