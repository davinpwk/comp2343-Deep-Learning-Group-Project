"""
Pytest suite for CarRacingEnv.

Run:
    pytest test_env.py -v

These are intentionally cheap smoke tests, not a deep verification suite —
the goal is to catch regressions in observation shape, reward sign, slip
observability, and termination semantics.
"""
import numpy as np

from car_env import CarRacingEnv, ACTION_NAMES


# === 1. Shape and space sanity ============================================
def test_shapes():
    env = CarRacingEnv(slippery=False)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs), \
            "initial obs out of declared range"
        # obs_idx covers every slot in the obs vector
        assert env.obs_idx["rays"] == slice(0, env.n_rays)
        for k in ("speed", "fwd_vel", "lat_vel", "is_off_road",
                  "angle_to_center", "distance_to_center"):
            assert env.n_rays <= env.obs_idx[k] < obs.shape[0]
    finally:
        env.close()


# === 2. Reward sign: pushing UP yields positive progress ==================
def test_progress_sign_positive():
    env = CarRacingEnv(slippery=False)
    try:
        env.reset(seed=0)
        acc = env.action_index(1, 0)
        total_progress = 0.0
        for _ in range(60):
            _, _, term, trunc, info = env.step(acc)
            total_progress += info["r_progress"]
            if term or trunc:
                break
        assert total_progress > 0, \
            "Driving UP should accumulate positive progress reward"
    finally:
        env.close()


# === 3. Slippery vs Normal: lateral velocity is observably larger ========
# The 3x multiplier below is empirical: with friction_normal=0.1 vs
# friction_slippery=0.95, after 20 steps of UP+RIGHT at speed, slippery
# typically shows ~4-7x more lateral drift. 3x is a comfortable floor that
# survives small physics tuning without losing test sensitivity.
def test_slip_observability():
    n_env = CarRacingEnv(slippery=False)
    s_env = CarRacingEnv(slippery=True)
    try:
        n_env.reset(seed=0)
        s_env.reset(seed=0)
        acc       = n_env.action_index(1, 0)
        acc_right = n_env.action_index(1, 1)
        # Spin up to speed
        for _ in range(30):
            n_env.step(acc)
            s_env.step(acc)
        # Hard right turn
        n_obs = s_obs = None
        for _ in range(20):
            n_obs, *_ = n_env.step(acc_right)
            s_obs, *_ = s_env.step(acc_right)
        lat_idx = n_env.obs_idx["lat_vel"]
        n_lat, s_lat = n_obs[lat_idx], s_obs[lat_idx]
        assert abs(s_lat) > abs(n_lat) * 3, (
            f"Slippery should show much larger lateral velocity "
            f"(normal={n_lat:+.3f}, slippery={s_lat:+.3f})"
        )
    finally:
        n_env.close()
        s_env.close()


# === 4a. Termination via wall hit =========================================
def test_wall_hit_terminates():
    # Drive straight forward at max speed; the first turn will smash into
    # the wall since DOWN-track has a sinusoidal centerline. Stop on first
    # terminal step and assert it is a wall hit (terminated, not truncated).
    env = CarRacingEnv(slippery=False, max_steps=2000)
    try:
        env.reset(seed=0)
        acc = env.action_index(1, 0)
        terminated = truncated = False
        info = {}
        for _ in range(2000):
            _, _, terminated, truncated, info = env.step(acc)
            if terminated or truncated:
                break
        assert terminated and not truncated, \
            f"expected wall-hit termination, got term={terminated} trunc={truncated}"
        assert info["wall_hit"], "info['wall_hit'] should be True"
    finally:
        env.close()


# === 4b. Truncation via max_steps =========================================
def test_max_steps_truncates():
    # NOOP isn't in the default no_noop action set, but it's the only
    # action that sits perfectly still — needed to isolate max_steps as
    # the truncation cause. Force action_set="full" for this test only.
    env = CarRacingEnv(slippery=False, max_steps=10, action_set="full")
    try:
        env.reset(seed=0)
        noop = env.action_index(0, 0)
        terminated = truncated = False
        for _ in range(15):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated or truncated:
                break
        assert truncated and not terminated, \
            "max_steps should set truncated=True, terminated=False"
    finally:
        env.close()


# === 5. car_angle stays bounded across long episodes ======================
def test_car_angle_wraps():
    env = CarRacingEnv(slippery=False, max_steps=5000)
    try:
        env.reset(seed=0)
        acc_left = env.action_index(1, -1)
        for _ in range(500):
            _, _, term, trunc, info = env.step(acc_left)
            assert -180.0 <= info["car_angle"] <= 180.0, \
                f"car_angle out of [-180,180]: {info['car_angle']}"
            if term or trunc:
                break
    finally:
        env.close()


# === 6. progress_pct is clamped to [0, 1] =================================
def test_progress_pct_bounded():
    env = CarRacingEnv(slippery=False, max_steps=5000)
    try:
        env.reset(seed=0)
        acc = env.action_index(1, 0)
        for _ in range(2000):
            _, _, term, trunc, info = env.step(acc)
            assert 0.0 <= info["progress_pct"] <= 1.0, \
                f"progress_pct out of [0,1]: {info['progress_pct']}"
            if term or trunc:
                break
    finally:
        env.close()


# === 7. Default action set is the 6-action "no_noop" preset ==============
def test_default_action_set():
    env = CarRacingEnv(slippery=False)
    try:
        assert env.action_set == "no_noop"
        assert env.action_space.n == 6
        # NOOP should NOT be in the default set
        try:
            env.action_index(0, 0)
            raise AssertionError("NOOP unexpectedly present in no_noop set")
        except ValueError:
            pass
        # Acc / Acc+RIGHT should be present
        assert env.action_index(1, 0) >= 0
        assert env.action_index(1, 1) >= 0
    finally:
        env.close()


# === 8. action_set="full" exposes 7 actions including NOOP ===============
def test_full_action_set():
    env = CarRacingEnv(slippery=False, action_set="full")
    try:
        assert env.action_set == "full"
        assert env.action_space.n == 7
        assert env.action_index(0, 0) == 0  # NOOP first
    finally:
        env.close()


# ==========================================================================
# Standalone smoke runner -- handy when you don't want to install pytest.
# ==========================================================================
if __name__ == "__main__":
    import inspect
    import sys
    fns = [(name, fn) for name, fn in globals().items()
           if name.startswith("test_") and inspect.isfunction(fn)]
    failures = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[OK]   {name}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(0 if failures == 0 else 1)
