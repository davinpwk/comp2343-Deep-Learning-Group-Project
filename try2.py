DIRT, ROAD, WALL, START, FINISH = 0, 1, 2, 3, 4

import math
import pygame
import numpy as np


def generate_winding_map(
        rows=1100, cols=3000,
        path_start=100, path_end=1000,
        base_center=1500, amplitude=400.0,
        width_center=10, width_swing=100,
        start_row=980, finish_row=120,
        dirt_border=5,
        finish_size=5,
) -> np.ndarray:
    grid = np.full((rows, cols), WALL, dtype=np.uint8)
    path_len = path_end - path_start
    def road_half_width(t: float) -> float:
        curve_val = abs(math.sin(2 * math.pi * t))
        width = width_swing * curve_val + width_center * (1 - curve_val)
        return (width - 1) / 2.0
    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)
        half = road_half_width(t)
        inner_left  = round(base_center + offset - half)
        inner_right = round(base_center + offset + half)
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

def generate_inside_map(
        rows=1100, cols=340,
        road_width=30,
        amplitude=400.0,
        grass_border=5,
        num_legs=6,
        start_row=980,
        finish_row=120,
        finish_size=5,
) -> np.ndarray:

    grid = [[WALL for _ in range(cols)] for _ in range(rows)]

    center_c = cols // 2
    center_r = (start_row + finish_row) // 2
    half = road_width // 2

    path = []

    # Square spiral: each iteration adds 4 legs (right, up, left, down)
    # amplitude shrinks each full loop
    amp = amplitude
    r, c = start_row, center_c
    path.append((r, c))

    directions = [
        ( 0,  1),   # right
        (-1,  0),   # up
        ( 0, -1),   # left
        ( 1,  0),   # down
    ]

    steps = int(amp)
    dir_idx = 0
    legs_done = 0

    while legs_done < num_legs and steps > 1:
        dr, dc = directions[dir_idx % 4]

        # For horizontal moves use amplitude as col steps,
        # for vertical scale by row/col ratio so the square looks square
        if dc != 0:
            leg_len = steps
        else:
            leg_len = max(1, int(steps * (rows / cols)))

        for _ in range(leg_len):
            r = max(finish_row, min(start_row, r + dr))
            c = max(half, min(cols - 1 - half, c + dc))
            path.append((r, c))

        dir_idx += 1
        legs_done += 1

        # Shrink amplitude every 2 legs (after each half-loop)
        if legs_done % 2 == 0:
            steps = max(1, steps - int(amplitude * 0.4))

    # Steps 3-6 unchanged from before
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                for dr in range(-half, half + 1):
                    rr = r1 + dr
                    if 0 <= rr < rows:
                        grid[rr][c] = ROAD

        elif c1 == c2:
            r_start = min(r1, r2) - half
            r_end   = max(r1, r2) + half
            for rr in range(r_start, r_end + 1):
                if 0 <= rr < rows:
                    for dc in range(-half, half + 1):
                        cc = c1 + dc
                        if 0 <= cc < cols:
                            grid[rr][cc] = ROAD

    from collections import deque
    queue = deque()
    dist = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == ROAD:
                queue.append((r, c, 0))
                dist[(r, c)] = 0
    while queue:
        r, c, d = queue.popleft()
        if d >= grass_border:
            continue
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in dist:
                    dist[(rr, cc)] = d + 1
                    if grid[rr][cc] == WALL:
                        grid[rr][cc] = DIRT
                    queue.append((rr, cc, d + 1))

    start_c = path[0][1]
    for dr in range(-finish_size, finish_size + 1):
        for dc in range(-finish_size, finish_size + 1):
            rr, cc = start_row + dr, start_c + dc
            if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == ROAD:
                grid[rr][cc] = START

    finish_c = path[-1][1]
    for dr in range(-finish_size, finish_size + 1):
        for dc in range(-finish_size, finish_size + 1):
            rr, cc = finish_row + dr, finish_c + dc
            if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == ROAD:
                grid[rr][cc] = FINISH

    return np.array(grid, dtype=np.int8)

CELL_SIZE = 1

def draw_grid(screen, grid):
    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            color = COLOR_MAP.get(val, (255, 0, 255))  # magenta = unknown
            pygame.draw.rect(
                screen,
                color,
                (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

COLOR_MAP = {
    0: (30, 30, 30),      # WALL -> dark gray
    1: (200, 200, 200),   # ROAD -> light gray
    2: (139, 69, 19),     # DIRT -> brown
    3: (0, 255, 0),       # START -> green
    4: (255, 0, 0),       # FINISH -> red
}

def view_grid(grid: np.ndarray):

    pygame.init()

    rows, cols = grid.shape
    screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
    pygame.display.set_caption("Winding Map Viewer")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        draw_grid(screen, grid)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ---- USAGE ----
if __name__ == "__main__":
    grid = generate_inside_map()  # your function
    view_grid(grid)