"""
Track generation for the car racing environment.

Tile codes and the procedural map generator live here so they can be reused
by visualization scripts without pulling in the full Gym env / pygame stack.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Tile codes (kept compatible with the original main.py)
# ---------------------------------------------------------------------------
DIRT, ROAD, WALL, START, FINISH = 0, 1, 2, 3, 4
TILE_SIZE = 5


# ---------------------------------------------------------------------------
# Default track (the F1 Circuit from the original main.py)
# ---------------------------------------------------------------------------
DEFAULT_TRACK = dict(
    rows=1100, cols=3000,
    path_start=100, path_end=1000,
    base_center=1500, amplitude=400.0,
    width_center=20, width_swing=40,
    spawn_row=120, exit_row=980,
    grass_border=5,
)


# ---------------------------------------------------------------------------
# Track generator (lifted from the original main.py, returns a numpy array)
# ---------------------------------------------------------------------------
def generate_winding_map(
        rows=26, cols=44,
        path_start=1, path_end=24,
        base_center=21, amplitude=4.5,
        width_center=3, width_swing=4,
        spawn_row=23, exit_row=3,
        grass_border=5,
) -> np.ndarray:
    """Generate a winding race track as a (rows, cols) uint8 tile grid.

    Tile codes: 0=DIRT, 1=ROAD, 2=WALL, 3=START, 4=FINISH.

    Quirk preserved from the original: the parameters `spawn_row` and
    `exit_row` are reversed from what their names suggest. `exit_row` is the
    row that gets the START tile; `spawn_row` is the row that gets FINISH.
    """
    grid = np.full((rows, cols), WALL, dtype=np.uint8)
    path_len = path_end - path_start

    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)

        half = (width_swing - 1) / 2.0
        inner_left = round(base_center + offset - half)
        inner_right = round(base_center + offset + half)

        # Dirt border fanning out from the road
        for i in range(grass_border + 1):
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

    if path_start <= exit_row <= path_end:
        grid[exit_row, center_col(exit_row)] = START
    if path_start <= spawn_row <= path_end:
        cx = center_col(spawn_row)
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                r, c = spawn_row + dr, cx + dc
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == ROAD:
                    grid[r, c] = FINISH

    return grid
