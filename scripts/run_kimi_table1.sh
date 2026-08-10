#!/bin/bash
# Replicate Table 1's 10 conditions on Kimi K3.
# Mirrors scripts/run_envelope.py's original GPT-4o invocations (see
# false_rejection.py's CONDS for the exact structure/family/tag mapping),
# just run against a different model via LLM_PROVIDER=kimi.
set -e
cd "$(dirname "$0")/.."
export LLM_PROVIDER=kimi

PY=.venv/bin/python
RUN="$PY scripts/run_envelope.py --workers 7 --tag-prefix kimi_"

for seed in 0 1 2; do
  # code, oracle tests, loose + tight (rows 1-2; also the direct_code_loose
  # baseline reused by rows 3-4's mask conditions)
  $RUN --split test --families code --tiers loose tight \
       --structures incumbent_refine direct --n 150 --seed $seed

  # math, self-check, loose + tight (rows 5-6)
  $RUN --split test --families math --tiers loose tight \
       --structures incumbent_refine_cot cot --n 150 --seed $seed

  # BBH, self-check, loose (row 10)
  $RUN --split logic_prospective --families logic --tiers loose \
       --structures incumbent_refine direct --n 120 --seed $seed

  # code OOD, no tests (row 8) / math OOD, self-check (row 7)
  $RUN --split ood --families code --tiers loose \
       --structures incumbent_refine direct --n 100 --seed $seed
  $RUN --split ood --families math --tiers loose \
       --structures incumbent_refine_cot cot --n 100 --seed $seed

  # code OOD, tests restored (row 9)
  $RUN --split ood_visible --families code --tiers loose \
       --structures incumbent_refine direct --n 100 --seed $seed

  # code, 50% tests / no tests, loose (rows 3-4)
  $RUN --split test --families code --tiers loose --mask-tests 0.5 \
       --structures verify_refine_3 direct --n 150 --seed $seed
  $RUN --split test --families code --tiers loose --mask-tests 0.0 \
       --structures verify_refine_3 direct --n 150 --seed $seed

  echo "=== seed $seed done ==="
done
echo "=== ALL DONE ==="
