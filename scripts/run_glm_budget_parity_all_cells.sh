#!/bin/bash
# Sequential orchestration for the GLM budget-parity follow-up: one cell
# (tier x arm) at a time, workers=2 (the concurrency validated clean --
# 10/10 success, 0 rate-limit retries -- in the tight/assign preflight;
# workers=3 hit the account's rate limit, code 1302). Never runs two GLM
# cells concurrently.
set -e
cd "$(dirname "$0")/.."
source /home/ycao95/Agent_repo/.venv_qwen_server/bin/activate

for tier in tight loose; do
  for arm in assign same_policy; do
    echo "=== $tier $arm ==="
    python scripts/run_glm_budget_parity.py run --tier "$tier" --arm "$arm" \
      --seeds 0 1 2 --workers 2
  done
done
echo "ALL_CELLS_DONE"
