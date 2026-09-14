# Server Qwen math/tight control — handoff

Status: **complete**. All four cells (reference + three provenance arms) hit
150/150 unique task rows for all three seeds, zero execution-error rows,
zero missing task/seed pairs. No paid or remote model API was called at any
point; every LLM call went to a local vLLM OpenAI-compatible server.

## Hardware and model

- Host: shared multi-user Linux GPU server (4x NVIDIA RTX PRO 6000
  Blackwell Server Edition, 96 GB each, compute capability 12.0).
  `CUDA_VISIBLE_DEVICES=1` was used for the whole campaign (least-contended
  GPU at launch time; GPU2/GPU3 had only ~7 GB free from other users'
  concurrent jobs).
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, HF revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`, served BF16 (fit comfortably;
  the AWQ fallback was never needed). Per-shard SHA-256 recorded in
  `experiments/qwen25coder7b_server_math_tight_20260914_environment.json`.
- Serving: `vllm==0.29.0`, `torch==2.13.0+cu130`, Python 3.12.3.

### A note on NVML

`nvidia-smi` and `pynvml.nvmlInit()` fail on this host with
`NVMLError_LibRmVersionMismatch` (the loaded kernel module and the
installed `libnvidia-ml.so.580.173.02` userspace library are out of sync —
a pre-existing host issue, not something introduced here, and not something
fixable without root on a server with dozens of other active users'
processes, so it was left alone). This does **not** affect CUDA compute:
confirmed directly with `torch.cuda.is_available()` and a real tensor op on
`cuda:0`. vLLM's own platform detection calls `nvmlInit()` purely to
confirm a CUDA platform is present, even though `vllm.platforms.cuda`
already has a `NonNvmlCudaPlatform` fallback (device queries via torch
instead of NVML) for exactly this situation on other hardware (e.g.
Jetson). A small out-of-tree platform plugin
(`force_cuda_platform.py` + its `.dist-info`, both placed only inside this
run's own venv's `site-packages/`, registered via an entry point in the
`vllm.platform_plugins` group) reports the CUDA platform directly so vLLM
reaches that existing fallback path. Nothing at the system level (driver,
kernel module, other users' processes) was touched. This is recorded in
full in the environment JSON's `note_nvml` field.

## Commands

```bash
# vLLM server (started once, left running for the whole campaign)
source .venv_qwen_server/bin/activate
export TMPDIR=/home/ycao95/.tmp_vllm   # host /tmp was full; JIT kernel
                                        # compiles (flashinfer) need real
                                        # scratch space
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

## Timeline (wall clock, this run)

- vLLM server up and serving: 2026-09-14 04:27:50 EDT.
- Determinism smoke check + repo test suite (10/10 passed): ~04:28–04:33.
- Reference baseline (450 rows): ~04:33 → 05:42:04 (≈69 min).
- Manifest + environment freeze: 05:42:46.
- Three-arm causal matrix (1350 rows): 05:43 → 17:10:34 (≈11h 27m).
- **Total benchmark wall time: ≈12h 40m** (excludes earlier one-time setup:
  repo clone/verification, vLLM install, and diagnosing/working around the
  host NVML issue, none of which touch the model or the data).

Per-cell breakdown (n, wall-clock minutes, from the run logs):

| cell | seed | n | wall_min | llm_calls/task | over_budget | errors |
|---|---|---|---|---|---|---|
| reference (cot) | 0 | 150 | 20.1 | 1.00 | 0 | 0 |
| reference (cot) | 1 | 150 | 20.5 | 1.00 | 0 | 0 |
| reference (cot) | 2 | 150 | 20.5 | 1.00 | 0 | 0 |
| arm1_assign | 0 | 150 | 70.9 | 2.87 | 2 | 0 |
| arm1_assign | 1 | 150 | 65.4 | 3.32 | 3 | 0 |
| arm1_assign | 2 | 150 | 64.6 | 3.20 | 2 | 0 |
| arm2_samepolicy | 0 | 150 | 90.5 | 3.87 | 2 | 0 |
| arm2_samepolicy | 1 | 150 | 81.5 | 4.32 | 3 | 0 |
| arm2_samepolicy | 2 | 150 | 80.9 | 4.20 | 2 | 0 |
| arm3_diffpolicy | 0 | 150 | 80.6 | 4.13 | 0 | 0 |
| arm3_diffpolicy | 1 | 150 | 75.7 | 3.91 | 0 | 0 |
| arm3_diffpolicy | 2 | 150 | 77.3 | 4.01 | 0 | 0 |

Reference accuracy: 0.6867 (identical across all 3 seeds — expected at
temperature 0, greedy decoding). Total rows across the whole campaign:
**1800** (450 reference + 1350 causal), matching the design exactly.

## Known limitation: 14 deterministic `budget_exceeded:settle_overrun:in_tokens` rows

`hbws/llm.py`'s ledger reservation estimates prompt tokens with `tiktoken`
(an OpenAI tokenizer) as a cross-provider proxy, adding a flat +150-token
safety margin for any non-default `LLM_PROVIDER` (see the comment in
`estimate_in_tokens`). On 14 of the 1350 causal rows (7 in `arm1_assign`,
7 in `arm2_samepolicy`, 0 in `arm3_diffpolicy`; spread across all three
seeds) the true Qwen tokenizer count for a long verify+refine prompt
exceeded that estimate, and `TaskLedger.settle` raised its designed
"defect, do not silently absorb" exception
(`hbws/ledger.py` `settle_overrun`), which `hbws/runner.py` catches and
reports as row status `budget_exceeded:settle_overrun:in_tokens` with
whatever partial solution existed at that point.

This was **not** patched by loosening the estimator after seeing it happen:
the whole pipeline is deterministic (temperature 0, forwarded seed, no
prefix caching, no cache reuse), so an unmodified retry would reproduce the
identical overrun every time — this is not a transient
connection/process/parse failure recoverable by re-running, and adjusting
the safety margin after observing exactly which rows triggered it would be
tuning on the result, which the task's protocol explicitly forbids. These
14 rows are real, valid rows (task id present, partial solution graded as
whatever it is) and are **not** counted in this codebase's own `errors`
metric (`status` does not start with `"error:"`); they are reported here in
full rather than hidden. All 1350 causal rows plus all 450 reference rows
are present — 1800/1800, 0/1800 execution errors, 0 missing task/seed
pairs.

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

## Principal statistics (see the full analysis JSON for CIs and control-flow detail)

Bootstrap unit: task (10,000 resamples). n = 150 tasks x 3 seeds = 450 rows
per arm.

| arm | accuracy | repair rate (of 141 ref-wrong pairs) | breakage rate (of 309 ref-correct pairs) | byte-identical to reference |
|---|---|---|---|---|
| reference (cot) | 0.6867 | — | — | — |
| arm1_assign (I=B by construction) | 0.7111 | 0.1206 | 0.0194 | 0.6889 |
| arm2_samepolicy (regen, same seed) | 0.7111 | 0.1206 | 0.0194 | 0.6889 |
| arm3_diffpolicy (regen, temp 0.7) | 0.7400 | 0.3546 | 0.0841 | 0.0000 |

`arm1_assign` and `arm2_samepolicy` are statistically identical on every
metric — the expected signature of a fully deterministic serving path: an
independent same-policy/same-seed regeneration reproduces the assigned
reference exactly, so the two provenance arms cannot be distinguished by
outcome. `arm3_diffpolicy` (a genuinely different first-draft policy) shows
much higher repair *and* higher breakage, as expected for a non-incumbent
first-draft policy under the verify/refine suffix.

Provenance gap vs. `arm1_assign` (pairwise, from the analysis JSON):

- `assign_vs_arm2_samepolicy`: 0 additional repairs, 0 additional
  breakages, `lambda_star` undefined (zero denominator) — consistent with
  the two arms being outcome-identical.
- `assign_vs_arm3_diffpolicy`: +33 repairs, +20 breakages,
  `lambda_star = 1.65`.

Rejection / gating (from `analyze_cell_control.py`'s control-flow audit,
summed across the reachable arms): 144 first-verify rejections in
`arm1_assign`/`arm2_samepolicy` (31 of which were reference-correct,
i.e. false rejections of an already-correct incumbent), 141 in
`arm3_diffpolicy` (50 reference-correct); every rejection reached
`initial_refinement_executed` (0 `initial_refinement_gated` in any arm) —
the nonbinding caps (9 LLM calls, 128k in-tokens, 13,824 out-tokens per
task) did their job: the ledger never gated the refinement suffix itself in
any of the 1350 causal rows (the only ledger-side stops were the 14
settle-overrun rows above, which are an estimator artifact, not a
reservation-admission gate).

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
