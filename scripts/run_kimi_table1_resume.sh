#!/bin/bash
# Resume run_kimi_table1.sh after an interruption: seed 0 already completed;
# seed 1 completed code (loose+tight) before dying mid-math. Skips redone
# work rather than re-running seed 1's code portion.
cd "$(dirname "$0")/.."
export LLM_PROVIDER=kimi

PY=.venv/bin/python
RUN="$PY scripts/run_envelope.py --workers 7 --tag-prefix kimi_"

run_seed() {
  local seed=$1
  local from=$2  # 1=start at math, 0=start at the top
  if [ "$from" = "0" ]; then
    $RUN --split test --families code --tiers loose tight \
         --structures incumbent_refine direct --n 150 --seed $seed || return 1
  fi
  $RUN --split test --families math --tiers loose tight \
       --structures incumbent_refine_cot cot --n 150 --seed $seed || return 1
  $RUN --split logic_prospective --families logic --tiers loose \
       --structures incumbent_refine direct --n 120 --seed $seed || return 1
  $RUN --split ood --families code --tiers loose \
       --structures incumbent_refine direct --n 100 --seed $seed || return 1
  $RUN --split ood --families math --tiers loose \
       --structures incumbent_refine_cot cot --n 100 --seed $seed || return 1
  $RUN --split ood_visible --families code --tiers loose \
       --structures incumbent_refine direct --n 100 --seed $seed || return 1
  $RUN --split test --families code --tiers loose --mask-tests 0.5 \
       --structures verify_refine_3 direct --n 150 --seed $seed || return 1
  $RUN --split test --families code --tiers loose --mask-tests 0.0 \
       --structures verify_refine_3 direct --n 150 --seed $seed || return 1
  echo "=== seed $seed done ==="
}

# Retry each seed up to 3 times if it dies partway (resumes from math each
# retry within a seed -- code loose/tight for that seed would be redone once
# if a retry is needed after code but before math; acceptable given no cost).
for attempt in 1 2 3; do
  run_seed 1 1 && break
  echo "seed 1 attempt $attempt failed, retrying..."
done

for attempt in 1 2 3; do
  run_seed 2 0 && break
  echo "seed 2 attempt $attempt failed, retrying..."
done

echo "=== ALL DONE ==="
