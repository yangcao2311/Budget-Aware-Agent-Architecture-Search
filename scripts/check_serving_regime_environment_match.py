#!/usr/bin/env python3
"""Compare the current host/model/software environment against the archived
qwen25coder7b_server_math_tight_20260914 environment record, before any
serving-regime benchmark call.

Per the task: if hardware/software cannot be kept identical, this run may
not be called a single-factor serving-configuration intervention -- report
the difference instead of proceeding silently. Only the vLLM serving flags
(concurrency, prefix caching) are meant to vary between S/B/P/S2; every
other field checked here should match exactly.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OLD_ENV = ROOT / "experiments/qwen25coder7b_server_math_tight_20260914_environment.json"
OUT = ROOT / "experiments/qwen_serving_regime_20260915_environment_diff.json"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"<error: {e}>"


def current_model_revision(model_id: str) -> str | None:
    cache_dir = Path.home() / ".cache/huggingface/hub"
    slug = "models--" + model_id.replace("/", "--")
    snaps = sorted(glob.glob(str(cache_dir / slug / "snapshots" / "*")))
    return Path(snaps[-1]).name if snaps else None


def current(gpu_index: int) -> dict:
    import torch
    import vllm
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_properties(gpu_index).name,
        "gpu_total_memory_gb": round(
            torch.cuda.get_device_properties(gpu_index).total_memory / 1e9, 1),
        "gpu_compute_capability": "%d.%d" % (
            torch.cuda.get_device_properties(gpu_index).major,
            torch.cuda.get_device_properties(gpu_index).minor),
        "hostname": platform.node(),
    }


def main() -> None:
    gpu_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    old = json.loads(OLD_ENV.read_text())
    cur = current(gpu_index)
    checks = {
        "python_version": (old["python_version"], cur["python_version"]),
        "torch_version": (old["torch_version"], cur["torch_version"]),
        "torch_cuda_version": (old["torch_cuda_version"], cur["torch_cuda_version"]),
        "vllm_version": (old["vllm_version"], cur["vllm_version"]),
        "gpu_name": (old["gpu"]["name"], cur["gpu_name"]),
        "gpu_total_memory_gb": (old["gpu"]["total_memory_gb"], cur["gpu_total_memory_gb"]),
        "gpu_compute_capability": (old["gpu"]["compute_capability"], cur["gpu_compute_capability"]),
        "hostname": (old["hostname"], cur["hostname"]),
        "model_revision": (old["model"]["revision"],
                          current_model_revision("Qwen/Qwen2.5-Coder-7B-Instruct")),
    }
    mismatches = {k: v for k, v in checks.items() if str(v[0]) != str(v[1])}
    report = {
        "purpose": "hardware/software identity check before the serving-regime "
                   "intervention: same model/hardware/software is a precondition "
                   "for calling this a single-factor serving-configuration test",
        "gpu_index_used_this_campaign": gpu_index,
        "old_gpu_selected": old["serving"]["gpu_selected"],
        "checks": {k: {"archived": v[0], "current": v[1]} for k, v in checks.items()},
        "all_match": len(mismatches) == 0,
        "mismatches": mismatches,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        print("\nWARNING: environment mismatch detected; this run cannot be "
              "described as a pure single-factor serving-configuration "
              "intervention without qualification.", file=sys.stderr)


if __name__ == "__main__":
    main()
