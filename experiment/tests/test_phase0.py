"""Phase 0 sanity tests.

Run with `python -m experiment.test_phase0` from the project root.
Each test prints a short OK/FAIL line; the script exits 0 if all pass.

Covers (matching EXPERIMENT_PLAN §0.x and the pre-flight checklist):
  - 0.1 map loader works on all three files
  - 0.1 env constructs and rolls out on each map; spawn + centerline OK
  - 0.2 MultiMapEnv samples maps uniformly and forwards step/reset
  - 0.2 reproducibility: same seed -> same map sequence
  - 0.3 step-based budget terminates at max_env_steps
  - 0.4 eval harness returns the right cells with expected shape
  - 0.5 set_friction + CurriculumScheduler mutates env friction live
  - 0.6 freeze_first_layer flips requires_grad and rebuilds optimizer
  - 0.7 save/load round-trip preserves weights
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from collections import Counter

import numpy as np
import torch

from experiment.car_env import CarRacingEnv
from experiment.track import load_map_from_file, START, FINISH, ROAD
from experiment.config import ExperimentConfig, FRICTION_NORMAL, FRICTION_SLIPPERY
from experiment.dqn import DQNAgent
from experiment.env_utils import (
    MultiMapEnv,
    fixed_friction,
    domain_randomized_friction,
    CurriculumScheduler,
    build_env,
    build_eval_env_set,
)
from experiment.evaluate import evaluate, aggregate_by_friction
from experiment.run_io import save_checkpoint, load_checkpoint_into
from experiment.train import train


MAPS = [
    "maps/winding.txt",
    "maps/winding_frequent.txt",
    "maps/winding_varying_width.txt",
]


# ---------------------------------------------------------------------------
# Tiny pytest-flavoured test harness (no external deps)
# ---------------------------------------------------------------------------
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _run(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except AssertionError as e:
        print(f"  {FAIL}  {name}: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  {FAIL}  {name}: unexpected {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_map_basics():
    for p in MAPS:
        arr = load_map_from_file(p)
        assert arr.ndim == 2, f"{p} not 2D"
        assert arr.dtype == np.uint8, f"{p} wrong dtype {arr.dtype}"
        # Must have at least one START and one FINISH tile
        assert (arr == START).sum() >= 1, f"{p} has no START"
        assert (arr == FINISH).sum() >= 1, f"{p} has no FINISH"
        # Must have road
        assert (arr == ROAD).sum() > 0, f"{p} has no ROAD"
        # All values must be in 0..4
        u = np.unique(arr)
        assert set(int(x) for x in u).issubset({0, 1, 2, 3, 4}), f"bad tiles in {p}: {u}"


def test_env_constructs_on_each_map():
    for p in MAPS:
        env = CarRacingEnv(map_path=p, max_steps=200)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape, f"obs shape mismatch on {p}"
        assert env.total_arc > 0, f"degenerate centerline on {p}"
        assert env._cl_n >= 2, f"centerline too short on {p}"
        # Step a few times; nothing should throw.
        for _ in range(20):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            assert obs.shape == env.observation_space.shape
            assert np.isfinite(r)
            if term or trunc:
                break
        env.close()


def test_set_friction_changes_friction():
    env = CarRacingEnv(map_path=MAPS[0], max_steps=200)
    env.set_friction(0.1)
    assert env.friction == 0.1
    assert env.slippery is False
    env.set_friction(0.95)
    assert env.friction == 0.95
    assert env.slippery is True
    # Bounds enforcement
    try:
        env.set_friction(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("set_friction(1.0) should have raised")
    env.close()


def test_multi_map_samples_uniformly_and_is_reproducible():
    provider = fixed_friction(0.1)
    e1 = MultiMapEnv(MAPS, provider, env_kwargs=dict(max_episode_steps=100), seed=0)
    e2 = MultiMapEnv(MAPS, provider, env_kwargs=dict(max_episode_steps=100), seed=0)

    seq1, seq2 = [], []
    for _ in range(60):
        e1.reset()
        e2.reset()
        seq1.append(e1.active_map)
        seq2.append(e2.active_map)
    assert seq1 == seq2, "Same seed should produce same map sequence"

    counts = Counter(seq1)
    # 60 resets / 3 maps = 20 each. Allow generous slack but must hit every map.
    assert set(counts) == set(MAPS), f"Some maps never sampled: {counts}"
    for k, v in counts.items():
        assert v >= 5, f"Map {k} only sampled {v}/60 times -- distribution too skewed"

    e1.close()
    e2.close()


def test_dr_provider_stays_in_range():
    import random as pyrnd
    rng = pyrnd.Random(7)
    prov = domain_randomized_friction(0.1, 0.95, rng=rng)
    vals = [prov() for _ in range(500)]
    assert min(vals) >= 0.1 and max(vals) <= 0.95, f"DR out of range: {min(vals)} {max(vals)}"
    # Some spread expected
    assert max(vals) - min(vals) > 0.3, "DR not exploring enough range"


def test_curriculum_scheduler_linear():
    sched = CurriculumScheduler(start=0.1, end=0.95, total_env_steps=1000)
    assert sched.current_friction() == 0.1
    sched.tick(500)
    midpoint = sched.current_friction()
    assert abs(midpoint - (0.1 + 0.95) / 2) < 1e-6, f"midpoint wrong: {midpoint}"
    sched.tick(500)
    assert sched.current_friction() == 0.95
    sched.tick(1)
    # Saturates at the endpoint
    assert sched.current_friction() == 0.95


def test_curriculum_provider_applied_in_multi_env():
    sched = CurriculumScheduler(start=0.1, end=0.95, total_env_steps=10)
    env = MultiMapEnv([MAPS[0]], sched.provider,
                      env_kwargs=dict(max_episode_steps=10), seed=0)
    env.reset()
    assert env.active_friction == 0.1
    sched.tick(10)
    env.reset()
    assert env.active_friction == 0.95
    env.close()


def test_evaluate_returns_expected_shape():
    agent = DQNAgent(obs_dim=13, n_actions=6,
                     hidden_sizes=(64, 64), device="cpu")
    eval_envs = build_eval_env_set(
        MAPS,
        env_kwargs=dict(max_episode_steps=100,
                        early_terminate_stagnation_steps=50),
    )
    try:
        cells = evaluate(
            agent,
            eval_envs=eval_envs,
            frictions=(FRICTION_NORMAL, FRICTION_SLIPPERY),
            n_episodes=2,
        )
    finally:
        for e in eval_envs.values():
            e.close()
    assert len(cells) == len(MAPS) * 2, f"wrong cell count: {len(cells)}"
    for k, v in cells.items():
        assert isinstance(k, tuple) and len(k) == 2
        for key in ("return_mean", "return_std", "progress_mean",
                    "finish_rate", "crash_rate", "returns"):
            assert key in v, f"missing {key} in {k}"
        assert len(v["returns"]) == 2

    agg = aggregate_by_friction(cells)
    assert set(agg.keys()) == {FRICTION_NORMAL, FRICTION_SLIPPERY}


def test_eval_envs_load_maps_only_once():
    """build_eval_env_set should load each map once, then reuse for repeated evals."""
    from experiment import track as track_mod
    call_count = {"n": 0}
    real_loader = track_mod.load_map_from_file

    def counting_loader(p):
        call_count["n"] += 1
        return real_loader(p)

    track_mod.load_map_from_file = counting_loader
    # Also need to patch the symbol imported into car_env at module load time
    from experiment import car_env as ce_mod
    ce_mod.load_map_from_file = counting_loader
    try:
        eval_envs = build_eval_env_set(
            MAPS,
            env_kwargs=dict(max_episode_steps=50,
                            early_terminate_stagnation_steps=25),
        )
        assert call_count["n"] == len(MAPS), (
            f"Expected {len(MAPS)} loads, got {call_count['n']}"
        )

        agent = DQNAgent(obs_dim=13, n_actions=6,
                         hidden_sizes=(16, 16), device="cpu")
        for _ in range(3):
            evaluate(agent, eval_envs=eval_envs,
                     frictions=(FRICTION_NORMAL, FRICTION_SLIPPERY),
                     n_episodes=1)
        # After 3 more eval calls, no extra file reads should have happened.
        assert call_count["n"] == len(MAPS), (
            f"Map files re-read during eval: {call_count['n']} > {len(MAPS)}"
        )
        for e in eval_envs.values():
            e.close()
    finally:
        track_mod.load_map_from_file = real_loader
        ce_mod.load_map_from_file = real_loader


def test_freeze_first_layer_rebuilds_optimizer():
    agent = DQNAgent(obs_dim=13, n_actions=6, hidden_sizes=(32, 32))
    total_before = sum(p.numel() for p in agent.online_net.parameters())
    trainable_before = sum(
        p.numel() for p in agent.online_net.parameters() if p.requires_grad
    )
    assert total_before == trainable_before

    agent.freeze_first_layer()

    first = agent.online_net.linear_layers()[0]
    for p in first.parameters():
        assert not p.requires_grad, "First layer params not frozen"

    trainable_after = sum(
        p.numel() for p in agent.online_net.parameters() if p.requires_grad
    )
    assert trainable_after < total_before, "Nothing was frozen"

    # Optimizer must not contain any frozen tensors. Identity-check
    # against the actual first-layer parameter tensors.
    frozen_ids = {id(p) for p in first.parameters()}
    for group in agent.optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in frozen_ids, "Optimizer still holds frozen tensors"

    # An update with a tiny buffer fill should run without error and not
    # change the frozen first-layer weights.
    snap = first.weight.detach().clone()
    for _ in range(agent.batch_size + 4):
        agent.store_transition(
            np.zeros(13, dtype=np.float32), 0, 1.0,
            np.zeros(13, dtype=np.float32), False,
        )
    agent.update()
    assert torch.equal(snap, first.weight), "Frozen layer changed during update"


def test_checkpoint_round_trip():
    cfg = ExperimentConfig(
        run_name="phase0_ckpt_test",
        seed=0,
        save_dir=tempfile.mkdtemp(prefix="phase0_"),
        hidden_sizes=(32, 32),
    )
    a = DQNAgent(obs_dim=13, n_actions=6, hidden_sizes=cfg.hidden_sizes)
    # Twiddle weights so the all-zero init isn't a false positive.
    with torch.no_grad():
        for p in a.online_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    a.target_net.load_state_dict(a.online_net.state_dict())

    ckpt = save_checkpoint(cfg, a)
    b = DQNAgent(obs_dim=13, n_actions=6, hidden_sizes=cfg.hidden_sizes)
    load_checkpoint_into(ckpt, b)
    for (k, pa), (_, pb) in zip(a.online_net.state_dict().items(),
                                b.online_net.state_dict().items()):
        assert torch.equal(pa, pb), f"checkpoint mismatch at {k}"
    shutil.rmtree(cfg.save_dir, ignore_errors=True)


def test_step_based_budget_terminates_exactly():
    """End-to-end smoke test of the training loop with a tiny budget."""
    save_dir = tempfile.mkdtemp(prefix="phase0_train_")
    cfg = ExperimentConfig(
        friction_mode="fixed",
        friction=0.1,
        map_paths=(MAPS[0],),                  # one map -> fast
        max_episode_steps=200,
        max_env_steps=2_000,                   # tiny budget
        warmup_env_steps=64,
        eval_every_env_steps=1_000,
        eval_n_episodes=2,
        eval_frictions=(FRICTION_NORMAL, FRICTION_SLIPPERY),
        epsilon_decay_steps=1_000,
        target_update_freq=200,
        batch_size=32,
        buffer_capacity=5_000,
        hidden_sizes=(32, 32),
        run_name="phase0_smoke",
        seed=0,
        save_dir=save_dir,
        # Speed: heavy stagnation cutoff so episodes don't drag
        early_terminate_stagnation_steps=80,
    )
    result = train(cfg, verbose=False)
    # Budget terminates at or just past the cap (the loop checks at the
    # *start* of each step, so the final step may push us 1 over).
    assert cfg.max_env_steps <= result["env_steps"] <= cfg.max_env_steps + 1, (
        f"budget overrun: {result['env_steps']} vs cap {cfg.max_env_steps}"
    )
    # Logs and checkpoint must exist
    rd = result["run_dir"]
    assert os.path.exists(os.path.join(rd, "config.json"))
    assert os.path.exists(os.path.join(rd, "final_checkpoint.pt"))
    assert os.path.exists(os.path.join(rd, "eval_log.jsonl"))
    assert os.path.exists(os.path.join(rd, "train_log.jsonl"))
    shutil.rmtree(save_dir, ignore_errors=True)


def test_train_loads_pretrained_checkpoint():
    """Fine-tune path: save a checkpoint, then start a 2nd run from it."""
    save_dir = tempfile.mkdtemp(prefix="phase0_finetune_")
    cfg_pre = ExperimentConfig(
        friction_mode="fixed",
        friction=0.1,
        map_paths=(MAPS[0],),
        max_episode_steps=150,
        max_env_steps=800,
        warmup_env_steps=32,
        eval_every_env_steps=500,
        eval_n_episodes=1,
        epsilon_decay_steps=400,
        target_update_freq=100,
        batch_size=16,
        buffer_capacity=2_000,
        hidden_sizes=(32, 32),
        run_name="pretrain",
        seed=0,
        save_dir=save_dir,
        early_terminate_stagnation_steps=60,
    )
    r1 = train(cfg_pre, verbose=False)
    ckpt = os.path.join(r1["run_dir"], "final_checkpoint.pt")

    cfg_ft = ExperimentConfig(
        friction_mode="curriculum",
        friction_curr_start=0.1,
        friction_curr_end=0.95,
        map_paths=(MAPS[0],),
        max_episode_steps=150,
        max_env_steps=400,
        warmup_env_steps=0,
        eval_every_env_steps=400,
        eval_n_episodes=1,
        epsilon_start=0.2,                # fine-tune ε schedule
        epsilon_end=0.05,
        epsilon_decay_steps=200,
        target_update_freq=100,
        batch_size=16,
        buffer_capacity=2_000,
        hidden_sizes=(32, 32),
        run_name="finetune",
        seed=0,
        save_dir=save_dir,
        early_terminate_stagnation_steps=60,
    )
    r2 = train(cfg_ft, pretrained_ckpt=ckpt, freeze_first=True, verbose=False)
    assert os.path.exists(os.path.join(r2["run_dir"], "final_checkpoint.pt"))
    shutil.rmtree(save_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("map loader",                          test_load_map_basics),
        ("env constructs on each map",          test_env_constructs_on_each_map),
        ("set_friction",                        test_set_friction_changes_friction),
        ("MultiMapEnv sampling+reproducibility", test_multi_map_samples_uniformly_and_is_reproducible),
        ("DR provider range",                   test_dr_provider_stays_in_range),
        ("curriculum scheduler linear",         test_curriculum_scheduler_linear),
        ("curriculum applied via MultiMapEnv",  test_curriculum_provider_applied_in_multi_env),
        ("eval harness shape",                  test_evaluate_returns_expected_shape),
        ("eval envs load maps once",            test_eval_envs_load_maps_only_once),
        ("freeze first layer + optimizer",      test_freeze_first_layer_rebuilds_optimizer),
        ("checkpoint round-trip",               test_checkpoint_round_trip),
        ("step-based budget terminates",        test_step_based_budget_terminates_exactly),
        ("fine-tune loads pretrained ckpt",     test_train_loads_pretrained_checkpoint),
    ]
    print()
    print("=" * 60)
    print(f"  Phase 0 sanity tests ({len(tests)})")
    print("=" * 60)
    all_ok = True
    for name, fn in tests:
        ok = _run(name, fn)
        all_ok = all_ok and ok
    print("=" * 60)
    print(f"  {'OK' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
