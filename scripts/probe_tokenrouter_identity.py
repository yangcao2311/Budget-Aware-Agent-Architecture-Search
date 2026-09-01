#!/usr/bin/env python
"""Diagnostic model-ID/behavior probe; never used as paper evidence."""
import json
from pathlib import Path

from openai import OpenAI

from run_glm53f_replication import load_key

ROOT = Path(__file__).resolve().parent.parent
PROMPT = (
    "我有一个容量为 5 升的杯子和 3 升的杯子，水不限量。我如何用这两个杯子量出准确的 4 升水？"
    "请注意，我的 5 升杯子底部有个小洞，每秒会漏掉 100 毫升水，我的水龙头每秒出水 200 毫升。"
    "请给出详细步骤，并明确说明每一步花费的时间及漏水量。"
)
MODELS = ("z-ai/glm-5.3-free", "z-ai/glm-5.3-flash")


def main():
    client = OpenAI(api_key=load_key(), base_url="https://api.tokenrouter.com/v1", timeout=180)
    rows = []
    for model in MODELS:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0,
                max_tokens=2048,
                seed=42,
                extra_body={"reasoning_effort": "low"},
            )
            rows.append({
                "requested_model": model,
                "response_model": r.model,
                "usage": r.usage.model_dump() if r.usage else None,
                "content": r.choices[0].message.content or "",
            })
        except Exception as e:
            rows.append({
                "requested_model": model,
                "error_type": type(e).__name__,
                "error": str(e),
            })
    out = ROOT / "experiments" / "tokenrouter_identity_probe.json"
    out.write_text(json.dumps({"prompt": PROMPT, "rows": rows}, ensure_ascii=False, indent=2))
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    print("written to", out)


if __name__ == "__main__":
    main()
