"""
Track generation for the car racing environment.

Tile codes and the procedural map generator live here so they can be reused
by visualization scripts without pulling in the full Gym env / pygame stack.
"""

import math
from types import MappingProxyType
import numpy as np


# ---------------------------------------------------------------------------
# Tile codes (kept compatible with the original main.py)
# ---------------------------------------------------------------------------
DIRT, ROAD, WALL, START, FINISH = 0, 1, 2, 3, 4
TILE_SIZE = 5


# ---------------------------------------------------------------------------
# Default track (the F1 Circuit from the original main.py).
# Wrapped in MappingProxyType so accidental mutation from another module
# fails loudly instead of silently corrupting later envs.
# ---------------------------------------------------------------------------
DEFAULT_TRACK = MappingProxyType(dict(
    rows=1100, cols=3000,
    path_start=100, path_end=1000,
    base_center=1500, amplitude=400.0,
    width_center=20, width_swing=40,
    start_row=980, finish_row=120,
    dirt_border=5,
    finish_size=5,
))


def _validate_config(rows, cols, path_start, path_end,
                     base_center, amplitude, width_center, width_swing,
                     start_row, finish_row, dirt_border, finish_size):
    if not (0 <= path_start < path_end < rows):
        raise ValueError(
            f"need 0 <= path_start < path_end < rows; "
            f"got path_start={path_start}, path_end={path_end}, rows={rows}"
        )
    if not (0 <= base_center < cols):
        raise ValueError(f"base_center={base_center} not in [0, cols={cols})")
    if width_center < 1 or width_swing < 1:
        raise ValueError("width_center and width_swing must be >= 1")
    if not (path_start <= start_row <= path_end):
        raise ValueError(
            f"start_row={start_row} must lie within "
            f"[path_start={path_start}, path_end={path_end}]"
        )
    if not (path_start <= finish_row <= path_end):
        raise ValueError(
            f"finish_row={finish_row} must lie within "
            f"[path_start={path_start}, path_end={path_end}]"
        )
    if dirt_border < 0 or finish_size < 0:
        raise ValueError("dirt_border and finish_size must be >= 0")


# ---------------------------------------------------------------------------
# Track generator (lifted from the original main.py, returns a numpy array)
# ---------------------------------------------------------------------------
def generate_winding_map(
        rows=26, cols=44,
        path_start=1, path_end=24,
        base_center=21, amplitude=4.5,
        width_center=3, width_swing=4,
        start_row=23, finish_row=3,
        dirt_border=5,
        finish_size=5,
) -> np.ndarray:
    """Generate a winding race track as a (rows, cols) uint8 tile grid.

    Tile codes: 0=DIRT, 1=ROAD, 2=WALL, 3=START, 4=FINISH.

    The track is a sinusoidal road from `path_start` to `path_end`, capped
    by dirt at either end. A single START tile is placed on `start_row`
    (where the car spawns) and a square of FINISH tiles of half-width
    `finish_size` is placed at `finish_row`.
    """
    _validate_config(rows, cols, path_start, path_end,
                     base_center, amplitude, width_center, width_swing,
                     start_row, finish_row, dirt_border, finish_size)

    grid = np.full((rows, cols), WALL, dtype=np.uint8)
    path_len = path_end - path_start

    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)

        half = (width_swing - 1) / 2.0
        inner_left = round(base_center + offset - half)
        inner_right = round(base_center + offset + half)

        # Dirt border fanning out from the road
        for i in range(dirt_border + 1):
            lc = inner_left - 1 - i
            rc = inner_right + 1 + i
            if 0 <= lc < cols: grid[row, lc] = DIRT
            if 0 <= rc < cols: grid[row, rc] = DIRT

        # Road
        for c in range(inner_left, inner_right + 1):
            if 0 <= c < cols:
                grid[row, c] = ROAD

    def place_cap(row_idx, center, inner_w):
        half = (inner_w - 1) // 2
        left = center - half - 1
        right = center + half + 1
        for c in range(max(0, left), min(cols, right + 1)):
            grid[row_idx, c] = DIRT

    if path_start > 0:
        place_cap(path_start - 1, base_center, width_center)
    if path_end < rows - 1:
        place_cap(path_end + 1, base_center, width_center)

    def center_col(row):
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)
        return round(base_center + offset)

    grid[start_row, center_col(start_row)] = START

    fx = center_col(finish_row)
    for dr in range(-finish_size, finish_size + 1):
        for dc in range(-finish_size, finish_size + 1):
            r, c = finish_row + dr, fx + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == ROAD:
                grid[r, c] = FINISH

    return grid
