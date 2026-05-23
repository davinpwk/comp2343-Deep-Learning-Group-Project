# Experiment Plan: Transfer Learning for Car Racing DQN

## 1. Environments

Three tracks already exist as text files in `maps/`:
- `straight_turn.txt` (T1 — reference layout)
- `narrow_straight_wide_turn.txt` (T2)
- `wide_straight_narrow_turn.txt` (T3)

During training, sample a map uniformly at random per episode. Spawn position and heading remain at env defaults (no randomization).

**Friction settings used across the plan** (higher value = more slippery, per `car_env.py:130-135`):
- Normal: `friction = 0.1`
- Slippery: `friction = 0.95`
- Domain randomization: `friction ~ Uniform(0.1, 0.95)` per episode
- Curriculum: `friction` schedule from 0.1 → 0.95 over fine-tune budget

## 2. Budget anchoring (do this first)

All budgets are measured in **environment steps**, not episodes.

**Preliminary run (single seed, generous budget):**
1. Train from-scratch on slippery for ~3× expected convergence. Identify slippery convergence step `N_slip`.
2. Train from-scratch on normal for the same budget. Identify normal convergence step `N_norm`.

Set the budget unit:
- `N := N_norm`
- Verify `2N ≥ N_slip`. If not, bump `N` so that `2N` comfortably covers slippery convergence; report this in the methodology.

## 3. Baselines (from-scratch)

| Name | Training data | Budget |
|---|---|---|
| **B-normal** | friction = 0.1 | N |
| **B-slippery** | friction = 0.95 | 2N (matched-compute) |
| **B-DR** | friction ~ U(0.1, 0.95) | 2N (matched-compute) |

B-normal serves dual duty: baseline *and* source of pretrained checkpoints for the transfer experiments.

## 4. Transfer experiments (pretrain on normal → fine-tune)

All start from a B-normal checkpoint. Four conditions form a 2×2 ablation:

| | Full fine-tune | Freeze first layer |
|---|---|---|
| **Direct transfer** (friction jumps 0.1 → 0.95) | E1 | E2 |
| **Curriculum transfer** (friction 0.1 → 0.95 linear) | E3 | E4 |

Each fine-tune uses budget `N`. Total compute per transfer condition: `N (pretrain) + N (fine-tune) = 2N`, matching the baseline budgets.

**Curriculum schedule:** linear ramp of `env.friction` from 0.1 to 0.95 over the `N` fine-tune steps. One fixed schedule — do not search over schedules; that's a separate research question.

## 5. Hyperparameter search

**Fixed across all conditions** (architecture choices, not tuning knobs):
- Network architecture (hidden_sizes), γ=0.99, optimizer=Adam, buffer_capacity, batch_size, grad_clip, reward weights

**Searched per condition:**

*Pretrain & from-scratch baseline grid* (~6 configs):
- LR ∈ {1e-3, 3e-4, 1e-4}
- ε-decay end-step ∈ {30% of budget, 60% of budget}

*Fine-tune grid* (~6 configs), constrained for warm-start:
- LR ∈ {3e-4, 1e-4, 3e-5} — strictly ≤ pretrain LR range
- ε-start ∈ {0.5, 0.2, 0.05} — not 1.0
- ε-decay short (~30% of fine-tune budget)

**Search protocol:**
- 3 validation seeds per config
- Metric: **AUC of the eval-return curve** on the *target* task (slippery)
- Tie-break by lower standard deviation across seeds

**Validation/test split:** the 3 seeds used in HP search are validation seeds. Final reported numbers use 5 fresh held-out seeds.

## 6. Final evaluation

For each of the 7 conditions (3 baselines + 4 transfer variants):
1. Take the winning HP config from §5.
2. Train 5 fresh seeds with that config.
3. Every `X` env steps (pick `X` so you get ~30–50 eval points across the run), pause and evaluate.

**Evaluation protocol per checkpoint:**
- 10 greedy (ε=0) episodes per (map × friction) cell
- Run on **all 3 maps × both friction conditions** (normal + slippery):
  - Slippery eval = primary target performance
  - Normal eval = catastrophic-forgetting probe (mandatory, not optional)
- Record mean return + std per cell

## 7. Reported metrics

For each condition, report:
- **Final mean return** on slippery (averaged across 3 maps, last 20% of training, 5 seeds)
- **AUC of slippery eval curve** (sample efficiency)
- **Catastrophic forgetting**: drop in normal-friction return from start-of-fine-tune to end-of-fine-tune (transfer conditions only)
- **Per-map breakdown** (do conditions generalize unevenly across T1/T2/T3?)
- **Mean ± std across seeds**; note that 5 seeds is too few for strong significance claims

## 8. Compute estimate

Sharing the B-normal pretrained checkpoint across the 4 transfer conditions:

- HP search: 7 conditions × 6 configs × 3 seeds = **126 search runs**
- Final eval: 7 conditions × 5 seeds = **35 final runs**
- **Total: ~161 runs**

Each search run can use a shorter budget (e.g., 50% of full) since AUC is informative early; final runs use the full budget. Estimate single-run wall-clock before committing.

---

# Step-by-step execution plan

## Phase 0 — Infrastructure (do before any training)

### 0.1 Map loader
The env currently builds tracks procedurally via `generate_winding_map`. Add a file loader since the maps already exist as text grids.

- In `track.py`, add `load_map_from_file(path) -> np.ndarray` that:
  - Reads the file as a 2D array of single-digit tile codes (0=DIRT, 1=ROAD, 2=WALL, 3=START, 4=FINISH).
  - Returns a `(rows, cols) uint8` array.
- In `car_env.py:151-211` (the `__init__`), add a `map_path: str | None = None` argument. When set, skip `generate_winding_map(**cfg)` and load from file instead. Keep the centerline construction working: either derive the centerline from the loaded map (scan road tiles per row, take their midpoint) or read it from a sidecar file. Simplest: derive from road tiles row-by-row.
- Add a tiny test in `test_env.py` confirming each of the three maps loads, the env constructs, `reset()` returns a valid obs, and a few random steps don't crash.

### 0.2 Multi-map episode sampling
- Add a wrapper or a `MultiMapEnv` helper that holds N pre-built `CarRacingEnv` instances (one per map) and forwards `reset()`/`step()` to a randomly-selected one per episode.
- Seed the map-choice RNG with the training seed so runs are reproducible.

### 0.3 Step-based budget refactor
The current `TrainConfig.episodes` field assumes episode budgets. Switch to env steps:
- Add `max_env_steps: int` to `TrainConfig`.
- In your DQN training loop, terminate the outer loop on env steps, not episode count.
- Keep episode logging for the plots but make the headline progress bar/budget driven by env steps.

### 0.4 Periodic evaluation harness
Write `evaluate(agent, maps, frictions, n_episodes=10) -> dict`:
- For each `(map, friction)` cell, run `n_episodes` greedy (ε=0) episodes, collect returns.
- Returns a dict keyed by `(map_name, friction_value)` → `{mean, std, returns}`.
- Call this every `X` env steps from the training loop and append to a results list.

### 0.5 Curriculum hook
Add `env.set_friction(value: float)` to `CarRacingEnv` (a 2-line method writing `self.friction`). In the training loop, when `curriculum=True`, call this each episode with the scheduled value.

### 0.6 Layer-freezing hook
Add a helper `freeze_first_layer(net)` that sets `requires_grad=False` on the first `nn.Linear`'s parameters. Verify the optimizer is constructed *after* freezing (or filter `parameters()` by `requires_grad`).

### 0.7 Run-management plumbing
- Decide on a results directory layout, e.g. `runs/{condition}/{hp_config_id}/seed_{seed}/`.
- Save per run: training curve (env_steps, train_return), eval results dict (per checkpoint), final agent weights.
- Save the HP config JSON alongside so reruns are reproducible.

## Phase 1 — Calibration

### 1.1 Pilot from-scratch slippery
- Run a single from-scratch slippery training with one seed, generous budget (estimate: 1M–2M env steps; bump if it's still climbing).
- Plot the smoothed eval curve. Identify `N_slip` = step where the curve plateaus (no improvement over last 20% of training).

### 1.2 Pilot from-scratch normal
- Same procedure on normal friction. Identify `N_norm`.

### 1.3 Lock in `N`
- Set `N = N_norm`.
- Check `2N ≥ N_slip`. If not, set `N = ceil(N_slip / 2)` and document this choice.
- All subsequent runs use `N` (or `2N` for matched-compute baselines).

### 1.4 Lock in `X` (eval cadence)
- Pick `X` so eval fires 30–50 times across a budget-`N` run.
- Time one eval pass (all maps × both frictions × 10 episodes) end-to-end. Ensure `X` is large enough that eval is < ~10% of wall-clock.

## Phase 2 — Hyperparameter search

### 2.1 Define grids
- Write the two grids (pretrain/baseline grid, fine-tune grid) as Python dicts.
- Generate the cartesian product as a list of config IDs.

### 2.2 Search the baseline conditions
For each of B-normal, B-slippery, B-DR:
- Run all ~6 configs × 3 seeds at the appropriate budget. Use 50% budget if compute is tight (AUC is well-defined on partial curves).
- Compute AUC of the slippery-eval curve (for B-normal, AUC on normal-eval since slippery isn't its target; document this asymmetry).
- Pick the winning config per condition.

### 2.3 Pretrain on normal with the winning B-normal HPs
- Train 3 seeds on normal at budget `N` with the winning HPs.
- Save the final agent weights as `checkpoints/pretrained_normal_seed_{seed}.pt`.
- These checkpoints feed every transfer experiment downstream.

### 2.4 Search the fine-tune conditions
For each of E1 (direct, full), E2 (direct, freeze-1st), E3 (curriculum, full), E4 (curriculum, freeze-1st):
- For each fine-tune HP config × 3 validation seeds:
  - Load the pretrained checkpoint matching the seed (one of the three from §2.3).
  - Apply freeze logic if applicable.
  - Run the fine-tune for `N` env steps under the appropriate friction regime (constant 0.95 for direct, linear ramp for curriculum).
- Pick the winning config per condition by AUC on slippery eval.

## Phase 3 — Final evaluation

### 3.1 Re-pretrain with 5 fresh seeds
- Using the winning B-normal HPs, train 5 new seeds on normal at budget `N`.
- Save these 5 checkpoints.

### 3.2 Run the 7 final conditions with 5 fresh seeds each
- B-normal (5 seeds, already done in §3.1)
- B-slippery (5 fresh seeds, budget `2N`, winning HPs)
- B-DR (5 fresh seeds, budget `2N`, winning HPs)
- E1–E4 (5 fresh seeds each, load pretrained checkpoints, apply respective fine-tune protocol with winning HPs)

For each run, eval every `X` env steps on all 3 maps × both friction conditions.

### 3.3 Aggregate results
Compute the §7 reported metrics per condition: final mean return on slippery, AUC, catastrophic-forgetting delta, per-map breakdown, mean±std across 5 seeds.

## Phase 4 — Analysis & write-up

### 4.1 Figures to produce
- Main learning curves: eval return vs. env steps, one line per condition, shaded band = ±1 std across 5 seeds.
- Per-map heatmap: final return per (condition × map × friction) cell.
- Forgetting plot: normal-friction return over fine-tune steps for E1–E4.
- Optional: AUC bar chart with seed-level error bars.

### 4.2 Methodology paragraph
> "We compare three from-scratch baselines (normal, slippery, domain-randomized friction ~ U(0.1, 0.95)) against four transfer variants pretrained on normal friction: direct vs. curriculum fine-tuning, each with full retraining or first-layer freezing. Budgets are anchored to the from-scratch normal convergence point N (in environment steps); transfer conditions use N for pretraining and N for fine-tuning, while from-scratch slippery/DR baselines use 2N to match total compute. Hyperparameters were tuned per condition by AUC of the slippery eval curve over 3 validation seeds, with a separate constrained grid for fine-tuning. Final results use 5 held-out seeds. Evaluation runs every X env steps on all 3 maps under both friction conditions; the normal-friction evaluation of fine-tuned models is reported as a catastrophic-forgetting measurement."

### 4.3 Limitations to state explicitly
- 5 seeds is the minimum for error bars; we do not make significance claims.
- Curriculum schedule is fixed at linear; ablating the schedule is out of scope.
- Fine-tuning conditions see more total environment interaction than the from-scratch normal baseline; we anchor comparisons to target-task sample efficiency, not total compute, and provide 2N from-scratch baselines for slippery and DR as the matched-compute control.
- Single curriculum shape, single freeze granularity; richer ablations are future work.

## Pre-flight checklist (run all of these before launching Phase 2)

- [ ] Map loader works for all three files; env constructs and rolls out cleanly on each.
- [ ] Random-map sampler produces well-formed episodes on all three maps over a 100-episode dry-run.
- [ ] Step-based budget terminates training at the configured `max_env_steps`.
- [ ] Eval harness runs end-to-end on a random policy; output dict shape is as expected.
- [ ] Curriculum hook: friction value actually mutates per episode (print it on a small dry-run).
- [ ] Freeze-first-layer hook: first layer's `requires_grad` is False after the call; optimizer only updates the remaining params (check `optim.param_groups`).
- [ ] Checkpoint save/load round-trips weights correctly.
- [ ] `N` and `X` are committed in a config file shared across all runs.
