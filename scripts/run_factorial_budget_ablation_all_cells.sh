#!/bin/bash
# Sequential orchestration for the factorial budget ablation: one cell at
# a time, workers=2 (the concurrency validated clean in every prior GLM
# campaign in this repo; also clean in this run's own preflight). Never
# runs two GLM cells concurrently.
set -e
cd "$(dirname "$0")/.."
source /home/ycao95/Agent_repo/.venv_qwen_server/bin/activate

for cell in A B C D; do
  echo "=== cell $cell ==="
  python scripts/run_factorial_budget_ablation.py run --cell "$cell" \
    --seeds 0 1 2 --workers 2
done
echo "ALL_FACTORIAL_CELLS_DONE"
