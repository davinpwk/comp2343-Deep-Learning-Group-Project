#!/usr/bin/env python3
"""Render an entire track map to a single image.

Loads a packed-digit map file (maps/*.txt) and paints the whole grid with the
exact same tile palette CarRacingEnv uses on screen, then writes it out as one
PNG. Unlike the env's per-frame view (a 1000x750 camera crop), this draws the
*whole* track at once, so you can eyeball a map without driving it.

The centerline that drives the progress/angle/distance observations is baked in
by default (vivid yellow, matching the env), so what you see here is what the
agent's reward signal is shaped around.

Usage (run from the repo root or anywhere):

    python scripts/render_map.py maps/winding.txt
    python scripts/render_map.py maps/winding.txt -o /tmp/winding.png
    python scripts/render_map.py maps/winding_frequent.txt --tile-px 2
    python scripts/render_map.py maps/zigzag.txt --no-centerline

By default the longest side is capped (--max-dim) so the full 3000-tile-wide
maps don't produce a 15000px monster; pass --max-dim 0 to disable the cap and
render at full --tile-px resolution.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Make `experiment` importable no matter what the current working directory is.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.car_env import CarRacingEnv
from experiment.track import TILE_SIZE


def build_image(env: CarRacingEnv, tile_px: int, draw_centerline: bool) -> Image.Image:
    """Paint the full track grid into a PIL image at `tile_px` pixels per tile."""
    # Same lookup table the env bakes into its track surface (track.py tile codes).
    color_lut = np.array(
        [
            env.GRASS,      # 0 DIRT
            env.ASPHALT,    # 1 ROAD
            env.WALL_COL,   # 2 WALL
            env.START_COL,  # 3 START
            env.FINISH_COL, # 4 FINISH
        ],
        dtype=np.uint8,
    )
    rgb = color_lut[env.track]                       # (H, W, 3)
    rgb = np.repeat(np.repeat(rgb, tile_px, axis=0), tile_px, axis=1)
    img = Image.fromarray(rgb)

    # Centerline points live in env pixel space (TILE_SIZE px per tile); rescale
    # to our tile_px so the polyline lands on the road exactly as in the env.
    if draw_centerline and env.centerline is not None and len(env.centerline) >= 2:
        CL_COL = (255, 220, 0)  # vivid yellow, same as the env overlay
        scale = tile_px / TILE_SIZE
        pts = [(float(p[0]) * scale, float(p[1]) * scale) for p in env.centerline]
        ImageDraw.Draw(img).line(pts, fill=CL_COL, width=max(1, tile_px // 2))

    return img


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an entire track map to a single PNG.",
    )
    parser.add_argument(
        "map_path",
        help="Path to a packed-digit map file (e.g. maps/winding.txt).",
    )
    parser.add_argument(
        "-o", "--out",
        help="Output image path. Defaults to <map-stem>.png in the current dir.",
    )
    parser.add_argument(
        "--tile-px", type=int, default=TILE_SIZE,
        help=f"Pixels per tile before any downscale (default {TILE_SIZE}, matching the env).",
    )
    parser.add_argument(
        "--max-dim", type=int, default=6000,
        help="Cap on the longest image side; the image is downscaled to fit. "
             "Pass 0 to disable and render at full --tile-px resolution.",
    )
    parser.add_argument(
        "--no-centerline", dest="centerline", action="store_false",
        help="Don't draw the centerline overlay.",
    )
    args = parser.parse_args(argv)

    map_path = Path(args.map_path)
    if not map_path.exists():
        parser.error(f"map file not found: {map_path}")
    if args.tile_px < 1:
        parser.error("--tile-px must be >= 1")

    out_path = Path(args.out) if args.out else Path.cwd() / f"{map_path.stem}.png"

    # Constructing the env loads the map, validates it, and derives the
    # centerline -- all without touching pygame (rendering is lazy).
    env = CarRacingEnv(map_path=str(map_path))
    rows, cols = env.track.shape
    print(f"Loaded {map_path}  ({rows} x {cols} tiles)")

    img = build_image(env, args.tile_px, args.centerline)

    if args.max_dim and max(img.size) > args.max_dim:
        factor = args.max_dim / max(img.size)
        new_size = (max(1, round(img.width * factor)), max(1, round(img.height * factor)))
        print(f"Downscaling {img.size} -> {new_size} (--max-dim {args.max_dim})")
        img = img.resize(new_size, Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Wrote {out_path}  ({img.width} x {img.height} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
