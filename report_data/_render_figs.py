"""Render up-to-date Phase 1/2/3 figures straight from each notebook's own
plotting cells, reading the existing runs/ logs. No training is performed:
`train` is monkeypatched to raise, so a missing log surfaces as an error
instead of silently kicking off a run.

Run from the project root:  python report_data/_render_figs.py
"""
import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "report_data"
sys.path.insert(0, str(ROOT))


def run_notebook_cells(nb_path, cell_indices, fig_paths):
    """exec the given code cells (by index) in one shared namespace, saving
    each plt.show() call to the next path in fig_paths."""
    nb = json.loads((ROOT / nb_path).read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}

    saved = []
    pending = list(fig_paths)

    def _save_show(*args, **kwargs):
        fig = plt.gcf()
        if pending:
            p = pending.pop(0)
            fig.savefig(OUT / p, dpi=130, bbox_inches="tight")
            saved.append(p)
        plt.close(fig)

    def _no_train(*a, **k):
        raise RuntimeError("train() called -- an eval log is missing; "
                           "refusing to retrain. Investigate before re-running.")

    for idx in cell_indices:
        cell = nb["cells"][idx]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        exec(compile(src, f"{nb_path}#cell{idx}", "exec"), ns)
        # patch as soon as the imports/knobs cell has run
        plt.show = _save_show
        if "train" in ns:
            ns["train"] = _no_train

    print(f"{nb_path}: saved {len(saved)} figures -> {saved}")
    if pending:
        print(f"  WARNING: {len(pending)} expected figures were not produced: {pending}")


if __name__ == "__main__":
    # ---- Phase 1: pilot curves, convergence detection, on/off-task ----
    run_notebook_cells(
        "phase1_calibration.ipynb",
        cell_indices=[2, 8, 10, 20],
        fig_paths=[
            "phase1_pilot_curves.png",
            "phase1_convergence.png",
            "phase1_on_off_task.png",
        ],
    )

    # ---- Phase 2: AUC bar charts + per-config learning curves ----
    # cells 6 & 12 mix definitions with skip-guarded training loops; train is
    # patched out so they only (re)define BASELINE_CONDITIONS / FT_CONDITIONS /
    # helpers, then cells 8 & 14 rebuild the winners from the logs.
    run_notebook_cells(
        "phase2_hp_search.ipynb",
        cell_indices=[2, 4, 6, 8, 10, 12, 14, 16, 18],
        fig_paths=[
            "phase2_area_under_the_curve_baseline.png",
            "phase2_area_under_the_curve_transfer.png",
            "phase2_baseline_curve.png",
            "phase2_transfer_curve.png",
        ],
    )

    # ---- Phase 3: learning curves, per-map heatmap, forgetting, AUC bars ----
    run_notebook_cells(
        "phase3_final_eval.ipynb",
        cell_indices=[2, 4, 12, 16, 17, 18, 19],
        fig_paths=[
            "phase3_learning_curves_slippery.png",
            "phase3_per_map_heatmap.png",
            "phase3_forgetting.png",
            "phase3_auc_bars.png",
        ],
    )
