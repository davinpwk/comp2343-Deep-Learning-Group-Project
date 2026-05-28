"""Generate the three vertical-progressing maps used for training.

Produces three .txt tile grids in ../maps/, in the same packed-digit format
load_map_from_file() reads.  All tracks progress along the y-axis (start
near the bottom row, finish near the top), so _build_centerline_from_array
in car_env.py walks them row-by-row and progress_pct stays meaningful.

Run from the repo root:
    python -m experiment.make_maps
"""

import math
import os
from pathlib import Path

import numpy as np

from .track import (
    DIRT, ROAD, WALL, START, FINISH,
    DEFAULT_TRACK, generate_winding_map,
)


MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def write_map(grid: np.ndarray, path: Path) -> None:
    """Serialize a (rows, cols) tile grid as one digit per cell, one row per line."""
    rows, cols = grid.shape
    lines = [
        "".join(chr(ord("0") + int(v)) for v in grid[r])
        for r in range(rows)
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(MAPS_DIR.parent)}  ({rows} x {cols})")


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the structure of generate_winding_map in track.py)
# ---------------------------------------------------------------------------
def _paint_road_row(grid, row, inner_left, inner_right, dirt_border, cols):
    """Paint one row of road tiles + a dirt fringe fanning out into wall."""
    for i in range(dirt_border + 1):
        lc = inner_left - 1 - i
        rc = inner_right + 1 + i
        if 0 <= lc < cols:
            grid[row, lc] = DIRT
        if 0 <= rc < cols:
            grid[row, rc] = DIRT
    for c in range(inner_left, inner_right + 1):
        if 0 <= c < cols:
            grid[row, c] = ROAD


def _place_cap(grid, row_idx, center, inner_w, cols):
    """Dirt cap across the centerline at the row just outside the road region."""
    half = (inner_w - 1) // 2
    left = center - half - 1
    right = center + half + 1
    for c in range(max(0, left), min(cols, right + 1)):
        grid[row_idx, c] = DIRT


def _stamp_start_finish(grid, start_row, finish_row, center_col_fn):
    """Drop a single START tile and convert every ROAD tile on `finish_row`
    into FINISH — a one-row line spanning the road width, perpendicular to
    the car's direction of travel."""
    grid[start_row, center_col_fn(start_row)] = START
    road_mask = grid[finish_row] == ROAD
    grid[finish_row, road_mask] = FINISH


def _convert_finish_block_to_line(grid):
    """Replace any existing FINISH tiles (e.g. the square block stamped by
    track.generate_winding_map) with a one-row finish line, keeping the row
    where they currently live."""
    finish_rows = np.flatnonzero((grid == FINISH).any(axis=1))
    if len(finish_rows) == 0:
        raise ValueError("Grid has no FINISH tiles to convert.")
    # Use the row nearest the start as the line row -- matches the convention
    # that the car crosses the finish row in one step rather than driving
    # through a block. For maps with start at the bottom, that's the max row
    # of the block (closest to the car); we pick the median to be agnostic.
    finish_row = int(np.median(finish_rows))
    grid[grid == FINISH] = ROAD
    road_mask = grid[finish_row] == ROAD
    grid[finish_row, road_mask] = FINISH


# ---------------------------------------------------------------------------
# Map 2: frequent smooth winding (sine centerline, constant width)
# ---------------------------------------------------------------------------
def generate_frequent_winding_map(
        rows=1100, cols=3000,
        path_start=100, path_end=1000,
        base_center=1500, amplitude=55.0,
        n_cycles=7, road_width=56,
        start_row=980, finish_row=120,
        dirt_border=5,
) -> np.ndarray:
    """Smooth sine centerline that winds `n_cycles` times across the path.

    Replaces the old triangle-wave zigzag, which was too hard to learn: a
    triangle wave forces the car to reverse its steering almost instantly at
    each sharp apex.  A sine bends continuously (no apex), and winding more
    often with a smaller amplitude keeps every bend gentle, so the track is
    markedly easier while still demanding constant steering.

    Peak lateral slope is ~2*pi*n_cycles*amplitude / path_len; the defaults
    (7 winds, amplitude 55) match the retired zigzag's peak steepness, so the
    only thing that changed is sharp->smooth and fewer->more frequent bends.
    """
    grid = np.full((rows, cols), WALL, dtype=np.uint8)
    path_len = path_end - path_start

    def offset_at(t):
        return amplitude * math.sin(2.0 * math.pi * n_cycles * t)

    def center_col(row):
        t = (row - path_start) / path_len
        return int(round(base_center + offset_at(t)))

    half = (road_width - 1) / 2.0
    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = offset_at(t)
        inner_left = int(round(base_center + offset - half))
        inner_right = int(round(base_center + offset + half))
        _paint_road_row(grid, row, inner_left, inner_right, dirt_border, cols)

    if path_start > 0:
        _place_cap(grid, path_start - 1, center_col(path_start),
                  road_width, cols)
    if path_end < rows - 1:
        _place_cap(grid, path_end + 1, center_col(path_end),
                  road_width, cols)

    _stamp_start_finish(grid, start_row, finish_row, center_col)
    return grid


# ---------------------------------------------------------------------------
# Map 3: winding centerline with width that breathes along the track
# ---------------------------------------------------------------------------
def generate_winding_varying_width_map(
        rows=1100, cols=3000,
        path_start=100, path_end=1000,
        base_center=1500, amplitude=200.0,
        bend_t=(0.18, 0.40, 0.55, 0.82),
        bend_signs=(+1, -1, +1, -1),
        width_base=44, width_amp=16, width_cycles=3.0, width_phase=0.5,
        min_width=15,
        start_row=980, finish_row=120,
        dirt_border=5,
) -> np.ndarray:
    """Piecewise winding centerline with 4 user-placed bends + width that
    breathes along t.

    The centerline interpolates with a half-cosine between anchor points
    `[(0, 0)] + zip(bend_t, bend_signs*amplitude) + [(1, 0)]`. Spacings
    between consecutive bend_t entries control how close two corners sit —
    the default (0.18, 0.40, 0.55, 0.82) gives short / long / short / long
    inter-bend gaps.

    Half-cosine interp gives zero derivative at every anchor, so the track
    is briefly vertical at each bend extremum — i.e. the car drives "into"
    the corner straight, then out — instead of catching the bend at speed.

    width(t) = max(min_width, width_base + width_amp * sin(2π·width_cycles·t + width_phase))
    """
    if len(bend_t) != len(bend_signs):
        raise ValueError("bend_t and bend_signs must be the same length")
    if any(bend_t[i] >= bend_t[i + 1] for i in range(len(bend_t) - 1)):
        raise ValueError(f"bend_t must be strictly increasing; got {bend_t}")
    if not (0.0 < bend_t[0] and bend_t[-1] < 1.0):
        raise ValueError(f"bend_t must lie strictly inside (0, 1); got {bend_t}")

    anchors_t = (0.0,) + tuple(bend_t) + (1.0,)
    anchors_o = (0.0,) + tuple(s * amplitude for s in bend_signs) + (0.0,)

    def offset_at(t):
        for i in range(len(anchors_t) - 1):
            if anchors_t[i] <= t <= anchors_t[i + 1]:
                t0, t1 = anchors_t[i], anchors_t[i + 1]
                o0, o1 = anchors_o[i], anchors_o[i + 1]
                phase = (t - t0) / (t1 - t0)
                return o0 + (o1 - o0) * (1.0 - math.cos(math.pi * phase)) / 2.0
        return 0.0

    grid = np.full((rows, cols), WALL, dtype=np.uint8)
    path_len = path_end - path_start

    def center_col(row):
        t = (row - path_start) / path_len
        return int(round(base_center + offset_at(t)))

    def width_at(t):
        w = width_base + width_amp * math.sin(
            2.0 * math.pi * width_cycles * t + width_phase
        )
        return max(min_width, w)

    max_width = int(math.ceil(width_base + width_amp))
    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = offset_at(t)
        w = width_at(t)
        half = (w - 1) / 2.0
        inner_left = int(round(base_center + offset - half))
        inner_right = int(round(base_center + offset + half))
        _paint_road_row(grid, row, inner_left, inner_right, dirt_border, cols)

    if path_start > 0:
        _place_cap(grid, path_start - 1, center_col(path_start),
                  max_width, cols)
    if path_end < rows - 1:
        _place_cap(grid, path_end + 1, center_col(path_end),
                  max_width, cols)

    _stamp_start_finish(grid, start_row, finish_row, center_col)
    return grid


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing maps to {MAPS_DIR}")

    # Map 1: default winding track from track.py, with the road widened from
    # 40 to 56 tiles. DEFAULT_TRACK itself stays unchanged so the procedural-
    # generation fallback in CarRacingEnv keeps its original shape.
    print("[1/3] winding.txt")
    g1 = generate_winding_map(**{**DEFAULT_TRACK, "width_swing": 56})
    _convert_finish_block_to_line(g1)
    write_map(g1, MAPS_DIR / "winding.txt")

    # Map 2: frequent smooth winding (replaces the retired triangle-wave zigzag).
    print("[2/3] winding_frequent.txt")
    g2 = generate_frequent_winding_map()
    write_map(g2, MAPS_DIR / "winding_frequent.txt")

    # Map 3: winding with width that oscillates along the track.
    print("[3/3] winding_varying_width.txt")
    g3 = generate_winding_varying_width_map()
    write_map(g3, MAPS_DIR / "winding_varying_width.txt")


if __name__ == "__main__":
    main()
