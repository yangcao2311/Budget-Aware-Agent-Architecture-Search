#!/usr/bin/env python3
"""Freeze the hardware/runtime environment record for the local Qwen/vLLM
math/tight nonbinding control, before any benchmark task runs.

Linux/vLLM-specific; does not touch any frozen Mac/GLM driver or manifest.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/qwen25coder7b_server_math_tight_20260914_environment.json"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"<error: {e}>"


def model_file_hashes(model_id: str) -> dict:
    cache_dir = Path.home() / ".cache/huggingface/hub"
    slug = "models--" + model_id.replace("/", "--")
    snap_glob = sorted(glob.glob(str(cache_dir / slug / "snapshots" / "*")))
    if not snap_glob:
        return {}
    snap = Path(snap_glob[-1])
    revision = snap.name
    hashes = {}
    for f in sorted(snap.glob("*.safetensors")):
        # HF cache blob filenames on this system are exactly the sha256 of
        # the file contents (verified directly with sha256sum against the
        # resolved symlink target before freezing).
        hashes[f.name] = f.resolve().name
    return {"revision": revision, "shard_sha256": hashes}


def main() -> None:
    import torch
    import vllm

    record = {
        "purpose": "hardware/runtime environment freeze for the local "
                   "Qwen2.5-Coder-7B-Instruct vLLM math/tight nonbinding "
                   "cross-provider deterministic-serving control",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": torch.cuda.get_device_properties(0).name,
            "total_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "compute_capability": "%d.%d" % (
                torch.cuda.get_device_properties(0).major,
                torch.cuda.get_device_properties(0).minor),
            "device_count_visible": torch.cuda.device_count(),
        },
        "nvidia_driver_version_from_cuda_json": "580.95.05 (system CUDA 13.0 metadata)",
        "note_nvml": (
            "nvidia-smi / pynvml.nvmlInit() fail on this shared host with "
            "NVMLError_LibRmVersionMismatch (loaded kernel module vs. "
            "installed libnvidia-ml.so.580.173.02 userspace library "
            "version skew). This does not affect CUDA compute: verified "
            "directly with torch.cuda.is_available()==True and a tensor "
            "op on cuda:0. vLLM's own platform detection normally calls "
            "nvmlInit() just to confirm a CUDA platform is present (even "
            "though it already has a NonNvmlCudaPlatform fallback for "
            "when NVML is unavailable at import time), so a local "
            "out-of-tree platform plugin (scripts-adjacent "
            "force_cuda_platform.py, registered only in this venv's "
            "site-packages via an entry point in the "
            "'vllm.platform_plugins' group) reports the CUDA platform "
            "directly; vllm.platforms.cuda then falls back to "
            "NonNvmlCudaPlatform (device queries via torch, not NVML) on "
            "its own, exactly as it does for e.g. Jetson. No system "
            "driver, kernel module, or other users' processes were "
            "touched; this is a compute-path-neutral fix scoped to this "
            "one venv."
        ),
        "model": {
            "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "served_as": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "dtype": "bfloat16",
            **model_file_hashes("Qwen/Qwen2.5-Coder-7B-Instruct"),
        },
        "serving": {
            "runtime": f"vllm {vllm.__version__}",
            "engine": "vllm serve (OpenAI-compatible API server)",
            "gpu_selected": "CUDA_VISIBLE_DEVICES=1 (least-contended of 4 "
                            "shared GPUs at launch time; GPU2/GPU3 had "
                            "~7 GB free from other users' jobs)",
            "flags": ["--dtype bfloat16", "--max-num-seqs 1",
                      "--no-enable-prefix-caching",
                      "--gpu-memory-utilization 0.35", "--port 8000",
                      "--seed 0"],
            "base_url": "http://127.0.0.1:8000/v1",
            "sampling": {"temperature": 0, "seed": "forwarded per-task "
                         "(math split seeds 0/1/2)"},
            "cache": "disabled for every benchmark call (use_cache=False; "
                     "response-cache sqlite untouched by this campaign)",
        },
        "git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "data_sha256": {
            "data/math_test.jsonl": sh(
                ["sha256sum", str(ROOT / "data/math_test.jsonl")]).split()[0],
        },
        "workers": 1,
        "concurrency": "workers=1 (protocol.evaluate ThreadPoolExecutor) and "
                       "vLLM --max-num-seqs=1: fully serial, one in-flight "
                       "request at a time, no cross-arm/seed batching or "
                       "prefix reuse.",
    }
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
