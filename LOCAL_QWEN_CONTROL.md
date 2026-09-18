# Local Qwen2.5-Coder-7B-Instruct serving control

Deterministic-serving control for the frozen math/`tight` cell: a `cot`
reference baseline plus three provenance arms, 150 frozen tasks x 3 seeds
per arm, served by a local vLLM OpenAI-compatible endpoint. No paid or
remote model API was called at any point. All four cells completed
150/150 unique task rows for all three seeds with zero execution-error
rows and zero missing task/seed pairs.

## Hardware and model

- Host: shared multi-user Linux GPU server (4x NVIDIA RTX PRO 6000
  Blackwell Server Edition, 96 GB each, compute capability 12.0).
  A single GPU was used for the whole campaign.
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, HF revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`, served BF16 (fit comfortably;
  the AWQ fallback was never needed). Per-shard SHA-256 recorded in
  `experiments/qwen25coder7b_server_math_tight_20260914_environment.json`.
- Serving: `vllm==0.29.0`, `torch==2.13.0+cu130`, Python 3.12.3.

## Commands

```bash
# vLLM server (started once, left running for the whole campaign)
source .venv_qwen_server/bin/activate
export TMPDIR="$TMPDIR"   # JIT kernel compiles need real
                                        # scratch space (flashinfer)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --dtype bfloat16 --max-num-seqs 1 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.35 --port 8000 --seed 0

# Provider env for every benchmark call (LLM_PROVIDER=vllm; see hbws/llm.py)
export LLM_PROVIDER=vllm VLLM_API_KEY=local \
       VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
       VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LLM_DISABLE_SEED=0 \
       LLM_PRICE_IN_PER_M=0 LLM_PRICE_OUT_PER_M=0

# Freeze (manifest + environment record; run once, before any benchmark task)
python scripts/run_qwen_vllm_math_tight_nonbinding.py freeze
python scripts/write_qwen_vllm_environment_record.py

# Reference baseline: 150 tasks x 3 seeds, cot structure, ordinary tight tier
python scripts/run_qwen_vllm_math_tight.py baseline --seeds 0 1 2

# Three provenance arms: 150 tasks x 3 seeds x 3 arms, nonbinding caps
python scripts/run_qwen_vllm_math_tight.py causal --seeds 0 1 2

# Analysis
python scripts/analyze_cell_control.py \
  --baseline-dir experiments/qwen25coder7b_server_math_tight_20260914_envelope_test/cot_math_tight \
  --causal-dir experiments/qwen25coder7b_server_math_tight_20260914_causal \
  --arm-suffix _math_tight_nonbinding \
  --condition qwen25coder7b_server_math_tight_nonbinding \
  --output experiments/qwen25coder7b_server_math_tight_20260914_analysis.json

python scripts/qwen_vllm_determinism_check.py \
  --baseline-dir experiments/qwen25coder7b_server_math_tight_20260914_envelope_test/cot_math_tight \
  --arm2-dir experiments/qwen25coder7b_server_math_tight_20260914_causal/arm2_samepolicy_math_tight_nonbinding \
  --output experiments/qwen25coder7b_server_math_tight_20260914_determinism.json

python -m pytest tests/ -v
```

New files, all Linux/vLLM-specific; none of the frozen Mac/MLX or GLM
drivers were touched:

- `scripts/run_qwen_vllm_math_tight.py` — top-level driver (freeze / smoke /
  baseline / causal stages).
- `scripts/run_qwen_vllm_math_tight_nonbinding.py` — the three-arm
  nonbinding-caps causal runner (adapted from the repo's own frozen GLM
  driver, `scripts/run_math_tight_nonbinding.py`, unchanged prompts /
  verifier / suffix graph / per-call limits / seeds / caps); adds a
  `--recover` path that reruns and overlays only rows whose prior status
  was an execution error, never resampling rows that already completed.
- `scripts/write_qwen_vllm_environment_record.py` — freezes hardware/runtime
  metadata (git commit, model shard hashes, data hash, CUDA/torch/vllm
  versions, sampling params, caps) before the first benchmark task.
- `scripts/qwen_vllm_determinism_check.py` — byte-identical-rate check
  described below.

## Determinism

`scripts/qwen_vllm_determinism_check.py` compares, for every task where
`arm2_samepolicy`'s independently-regenerated first draft (same prompt,
temperature 0, same forwarded seed, cache disabled, a genuinely separate
HTTP call) was accepted at the first verify (no refinement triggered),
that first draft against the stored reference baseline's own first-draft
solution for the same task/seed. Result:
**306/306 byte-identical (rate = 1.0)** — see
`experiments/qwen25coder7b_server_math_tight_20260914_determinism.json`.
A second manual check (two back-to-back completions at temperature 0 with
the same seed against the running server, outside the benchmark) was also
byte-identical.

## Validation performed

- `git rev-parse HEAD` == `398d1df2f0516f7205b4b9774f37511861dbfb08` (frozen
  branch tip, verified before any code was read).
- `sha256sum data/math_test.jsonl` ==
  `f520b14219177c3fb5123bd80fcd0f61019b767dbb3f0ce8fbf1e6b1449a01c4`.
- `python -m pytest tests/` — 10/10 passed (offline; no LLM calls).
- Determinism: 306/306 byte-identical (see above) plus a manual
  double-completion check.
- Completeness: every one of the 12 result files (3 seeds x [reference + 3
  arms]) has exactly 150 unique `task_id`s; 1800/1800 rows total; 0 rows
  with an `error:`-prefixed status; 0 missing task/seed/arm keys.
- Full SHA-256 inventory of every manifest, result, summary, log, and
  analysis file: `experiments/qwen25coder7b_server_math_tight_20260914_sha256_inventory.txt`.

## What was not done / out of scope

- No paper text, Figure 2, or any existing (non-Qwen) experiment result was
  modified.
- No API keys, model weights, the response cache, or the `.venv_qwen_server`
  virtual environment are included in the results package.
- The 14 `budget_exceeded` rows above were left as-is rather than patched;
  if the estimator's cross-tokenizer safety margin should be widened for
  future non-OpenAI local-serving campaigns, that is a `hbws/llm.py`
  infrastructure change to make *before* a future run, not a retroactive
  fix to this one.
