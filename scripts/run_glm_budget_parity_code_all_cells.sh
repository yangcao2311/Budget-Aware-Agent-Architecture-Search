#!/bin/bash
# Sequential orchestration for the GLM code-domain budget-parity
# follow-up: one cohort x arm at a time, workers=2 (the concurrency
# validated clean for the math version and confirmed clean again here in
# preflight -- 0 errors, 0 over_budget). Never runs two GLM cells
# concurrently.
set -e
cd "$(dirname "$0")/.."
source /home/ycao95/Agent_repo/.venv_qwen_server/bin/activate

for cohort in tight loose; do
  for arm in assign same_policy; do
    echo "=== $cohort $arm ==="
    python scripts/run_glm_budget_parity_code.py run --cohort "$cohort" --arm "$arm" \
      --seeds 0 1 2 --workers 2
  done
done
echo "ALL_CODE_CELLS_DONE"
