#!/usr/bin/env bash
#
# Render a stitched multi-map GIF for every phase-3 run of a given seed.
#
# Walks both phase-3 trees -- runs/phase3_pretrain/seed_<SEED> and
# runs/phase3_final/<setup>/seed_<SEED> -- and calls render_eval_gifs.py on each,
# producing one stitched GIF per run under gifs/eval/.
#
# Usage:
#   scripts/render_phase3_seed.sh <seed> [extra args forwarded to render_eval_gifs.py]
#
# Examples:
#   scripts/render_phase3_seed.sh 101
#   scripts/render_phase3_seed.sh 103 --stride 5 --scale 0.4
#   PYTHON=python3 scripts/render_phase3_seed.sh 101 --dry-run
#
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <seed> [extra args for render_eval_gifs.py]" >&2
  exit 2
fi

SEED="$1"; shift

# Resolve repo paths relative to this script so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RENDER="$SCRIPT_DIR/render_eval_gifs.py"
PYTHON="${PYTHON:-python}"

# Collect every seed_<SEED> run dir under the phase-3 trees (sorted, stable).
mapfile -t RUN_DIRS < <(
  find "$REPO_ROOT"/runs/phase3_* -type d -name "seed_${SEED}" 2>/dev/null | sort
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No phase-3 run dirs found for seed ${SEED} under $REPO_ROOT/runs/phase3_*" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} phase-3 run(s) for seed ${SEED}:"
printf '  %s\n' "${RUN_DIRS[@]}"
echo

failed=0
for run_dir in "${RUN_DIRS[@]}"; do
  echo "=== $run_dir ==="
  if ! "$PYTHON" "$RENDER" "$run_dir" "$@"; then
    echo "!! render failed for $run_dir" >&2
    failed=$((failed + 1))
  fi
  echo
done

if [[ $failed -gt 0 ]]; then
  echo "Done with $failed failure(s) out of ${#RUN_DIRS[@]} run(s)." >&2
  exit 1
fi
echo "Done: rendered ${#RUN_DIRS[@]} phase-3 run(s) for seed ${SEED}."
