#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

# Wait for the final seed-2 arm file before issuing another provider call.
# The earlier long process exited after writing seed 1 and before appending its
# buffered summary, so raw result completeness—not summary length—is the
# durable completion marker.
while true; do
  final_rows=0
  final_file=experiments/glm4flash_repaired_code_20260909_causal/arm3_diffpolicy_math_loose/results_seed2.jsonl
  if [[ -f "$final_file" ]]; then
    final_rows=$(wc -l < "$final_file")
  fi
  if (( final_rows == 150 )); then
    break
  fi
  sleep 30
done

.venv_shared/bin/python scripts/run_glm4flash_replication.py --stage repaired_matrix_analyze
.venv_shared/bin/python scripts/recover_repaired_matrix_causal_failures.py freeze

baseline_done=0
for pass_no in {1..20}; do
  echo "baseline recovery pass $pass_no"
  .venv_shared/bin/python scripts/recover_repaired_matrix_failures.py run --workers 1
  if .venv_shared/bin/python scripts/recover_repaired_matrix_failures.py analyze; then
    baseline_done=1
    break
  fi
done
(( baseline_done == 1 ))

causal_done=0
for pass_no in {1..20}; do
  echo "causal-only recovery pass $pass_no"
  .venv_shared/bin/python scripts/recover_repaired_matrix_causal_failures.py run --workers 1
  if .venv_shared/bin/python scripts/recover_repaired_matrix_causal_failures.py analyze; then
    causal_done=1
    break
  fi
done
(( causal_done == 1 ))

refiner_done=0
for pass_no in {1..20}; do
  echo "five-refiner pass $pass_no"
  if .venv_shared/bin/python scripts/shared_refiner5_temporal_study.py run --workers 1; then
    refiner_done=1
    break
  fi
done
(( refiner_done == 1 ))
.venv_shared/bin/python scripts/shared_refiner5_temporal_study.py analyze
.venv_shared/bin/python scripts/audit_shared_refiner5_math.py

.venv_shared/bin/python scripts/assign_search_intervention.py reference --workers 1
.venv_shared/bin/python scripts/audit_assign_search_integrity.py --reference-only
.venv_shared/bin/python scripts/assign_search_intervention.py search --workers 1
.venv_shared/bin/python scripts/assign_search_intervention.py analyze
.venv_shared/bin/python scripts/audit_assign_search_integrity.py

.venv_shared/bin/python scripts/render_main_evidence.py
.venv_shared/bin/python scripts/make_clean_matrix_figure.py
(cd paper && LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
python3 -m pytest -q
.venv_shared/bin/python scripts/audit_claims.py
.venv_shared/bin/python scripts/audit_submission_ready.py
echo "ALL_API_GENERATION_AND_MACHINE_AUDIT_STAGES_COMPLETE"
