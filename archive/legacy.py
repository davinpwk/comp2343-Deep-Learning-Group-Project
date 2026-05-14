import pygame
import math
import sys
import time
# import random

# ===========================================================================
#  CHOOSE YOUR TRACK HERE
# ===========================================================================
ACTIVE_TRACK = "F1 Circuit"
# ===========================================================================

# 0 = Dirt
# 1 = Road
# 2 = Wall
# 3 = Start
# 4 = Finish

WIDTH = 40
HEIGHT = 20

# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_winding_map(
        rows: int = 26,
        cols: int = 44,
        path_start: int = 1,
        path_end: int = 24,
        base_center: int = 21,
        amplitude: float = 4.5,
        width_center: int = 3,
        width_swing: int = 4,
        spawn_row: int = 23,
        exit_row: int = 3,
        grass_border: int = 5,
        EMPTY: int = 2,   # Wall (background fill)
        WALL: int = 0,    # Dirt
        FLOOR: int = 1,   # Road
        SPAWN: int = 3,   # Start
        EXIT: int = 4,    # Finish
) -> list[list[int]]:
    exit_size = 5
    grid = [[EMPTY] * cols for _ in range(rows)]

    path_len = path_end - path_start

    for row in range(path_start, path_end + 1):
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)

        half = (width_swing - 1) / 2.0
        inner_left = round(base_center + offset - half)
        inner_right = round(base_center + offset + half)

        # Wall tile + grass_border dirt tiles fanning out on each side
        for i in range(grass_border + 1):
            lc = inner_left  - 1 - i
            rc = inner_right + 1 + i
            if 0 <= lc < cols: grid[row][lc] = WALL
            if 0 <= rc < cols: grid[row][rc] = WALL

        # Road floor
        for c in range(inner_left, inner_right + 1):
            if 0 <= c < cols:
                grid[row][c] = FLOOR

    def place_cap(row_idx: int, center: int, inner_w: int):
        half = (inner_w - 1) // 2
        left  = center - half - 1
        right = center + half + 1
        for c in range(max(0, left), min(cols, right + 1)):
            grid[row_idx][c] = WALL

    if path_start > 0:
        place_cap(path_start - 1, base_center, width_center)
    if path_end < rows - 1:
        place_cap(path_end + 1, base_center, width_center)

    def floor_center_col(row: int) -> int:
        t = (row - path_start) / path_len
        offset = amplitude * math.sin(2 * math.pi * t)
        return round(base_center + offset)

    if path_start <= exit_row <= path_end:
        grid[exit_row][floor_center_col(exit_row)] = SPAWN
    if path_start <= spawn_row <= path_end:
        cx = floor_center_col(spawn_row)
        for dr in range(-exit_size, exit_size + 1):
            for dc in range(-exit_size, exit_size + 1):
                r, c = spawn_row + dr, cx + dc
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == FLOOR:
                    grid[r][c] = EXIT

    return grid


TILE_SIZE = 5

TRACKS = {
    "F1 Circuit": generate_winding_map(
        rows=1100,
        cols=3000,
        path_start=100,
        path_end=1000,
        base_center=1500,
        amplitude=400.0,
        width_center=20,
        width_swing=40,
        spawn_row=120,
        exit_row=980,
        grass_border=5,
    )
}

FPS = 60
CAR_SPEED = 4.5
OFF_ROAD_SPEED = 1.8
TURN_SPEED = 4.0
FRICTION = 0.95   # 0.0 = full grip  →  ~0.95 = icy

# Colors
ASPHALT, GRASS, FINISH_COL = (50, 50, 60), (55, 110, 55), (220, 190, 0)
START_COL, WALL_COL, WHITE, RED = (240, 240, 240), (180, 180, 180), (255, 255, 255), (220, 40, 40)


def tile_at(track, x, y):
    rows = len(track)
    c, r = int(x // TILE_SIZE), int(y // TILE_SIZE)
    if 0 <= r < rows and 0 <= c < len(track[r]):
        return track[r][c]
    return 2  # Out-of-bounds → Wall


def find_tile(track, value):
    for r, row in enumerate(track):
        for c, v in enumerate(row):
            if v == value: return r, c
    return 0, 0


def build_track_surface(track):
    rows = len(track)
    max_cols = max(len(row) for row in track)
    surf = pygame.Surface((max_cols * TILE_SIZE, rows * TILE_SIZE))
    # 0=Dirt, 1=Road, 2=Wall, 3=Start, 4=Finish
    color_map = {0: GRASS, 1: ASPHALT, 2: WALL_COL, 3: START_COL, 4: FINISH_COL}
    for r in range(rows):
        for c in range(max_cols):
            val = track[r][c] if c < len(track[r]) else 2
            pygame.draw.rect(surf, color_map.get(val, GRASS), (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            if val == 2:  # Wall border outline
                pygame.draw.rect(surf, (150, 150, 150), (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)
    return surf


def draw_car(surface, x, y, angle):
    CAR_W, CAR_H = 18, 30
    s = pygame.Surface((CAR_W + 4, CAR_H + 4), pygame.SRCALPHA)
    pygame.draw.rect(s, RED, (2, 2, CAR_W, CAR_H), border_radius=2)
    pygame.draw.rect(s, (160, 20, 20), (2, 2, CAR_W, 8))
    rot = pygame.transform.rotate(s, -angle)
    surface.blit(rot, rot.get_rect(center=(int(x), int(y))).topleft)


def main():
    pygame.init()
    track_data = TRACKS[ACTIVE_TRACK]
    track_surf = build_track_surface(track_data)

    # Camera Viewport Size
    SCREEN_W, SCREEN_H = 1000, 750
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Race - {ACTIVE_TRACK}")
    clock, font = pygame.time.Clock(), pygame.font.SysFont("monospace", 18, bold=True)

    sr, sc = find_tile(track_data, 3)
    spawn_x, spawn_y = sc * TILE_SIZE + TILE_SIZE // 2, sr * TILE_SIZE + TILE_SIZE // 2
    car_x, car_y, car_angle, speed = spawn_x, spawn_y, 0.0, 0.0
    vel_x, vel_y = 0.0, 0.0

    last_lap, start_time = None, time.time()

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

        current_tile = tile_at(track_data, car_x, car_y)
        is_off_road = (current_tile == 0)  # Dirt

        max_v = OFF_ROAD_SPEED if is_off_road else CAR_SPEED
        accel = 0.08 if is_off_road else 0.15

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            speed = min(speed + accel, max_v)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            speed = max(speed - accel, -max_v * 0.5)
        else:
            speed *= 0.92

        if is_off_road and speed > OFF_ROAD_SPEED: speed *= 0.85

        if abs(speed) > 0.1:
            turn = (speed / CAR_SPEED) * TURN_SPEED
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: car_angle -= turn
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: car_angle += turn

        # Where the car wants to go based on its heading
        target_vx = math.sin(math.radians(car_angle)) * speed
        target_vy = -math.cos(math.radians(car_angle)) * speed

        # Blend actual velocity toward target — FRICTION controls how slowly
        vel_x = vel_x * FRICTION + target_vx * (1 - FRICTION)
        vel_y = vel_y * FRICTION + target_vy * (1 - FRICTION)

        nx = car_x + vel_x
        ny = car_y + vel_y

        if tile_at(track_data, nx, ny) == 2:  # Wall collision
            car_x, car_y, car_angle, speed = spawn_x, spawn_y, 0.0, 0.0
            vel_x, vel_y = 0.0, 0.0
        else:
            car_x, car_y = nx, ny

        # Finish Line Logic
        if tile_at(track_data, car_x, car_y) == 4:  # Finish
            last_lap, start_time = time.time() - start_time, time.time()
            car_x, car_y, car_angle, speed = spawn_x, spawn_y, 0.0, 0
            vel_x = 0
            vel_y = 0

        # --- CAMERA OFFSET CALCULATION ---
        offset_x = (SCREEN_W // 2) - car_x
        offset_y = (SCREEN_H // 2) - car_y

        screen.fill(GRASS)
        screen.blit(track_surf, (offset_x, offset_y))
        draw_car(screen, SCREEN_W // 2, SCREEN_H // 2, car_angle)

        # UI
        hud = pygame.Surface((240, 60), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 160))
        screen.blit(hud, (10, 10))
        screen.blit(font.render(f"TIME: {time.time() - start_time:.2f}s", 1, WHITE), (20, 15))
        screen.blit(font.render(f"LAST: {f'{last_lap:.2f}s' if last_lap else '--'}", 1, FINISH_COL), (20, 35))

        pygame.display.flip()


if __name__ == "__main__":
    main()