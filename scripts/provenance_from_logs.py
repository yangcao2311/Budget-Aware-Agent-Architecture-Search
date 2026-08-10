#!/usr/bin/env python
"""What existing logs already say about provenance, at zero inference cost.

The three-arm provenance experiment we say we did not run needs a
"regenerate under the same policy" arm to contrast with "reuse the reference's
stored output". One such arm is already on disk. The response-cache key
includes the execution seed, so the SAME baseline prompt at temperature 0 was
issued three times per task -- once per seed -- and each is an independent
regeneration under an identical policy.

That lets us estimate, without new inference:

  (1) how often temperature-0 regeneration reproduces the reference's text, and
  (2) Pr(regenerated draft wrong | reference draft correct), which is the
      exposure of the accepting path in

          b = Pr(accept, I wrong | B correct) + Pr(reject, J wrong | B correct)

      i.e. exactly the term reference preservation sets to zero. It bounds what
      the guarantee buys, and it is what a regenerating arm would have to pay
      before its verifier ever speaks.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]

BASELINES = [
    ("code, loose",  "envelope_test", "direct_code_loose"),
    ("code, tight",  "envelope_test", "direct_code_tight"),
    ("math, loose",  "envelope_test", "cot_math_loose"),
    ("math, tight",  "envelope_test", "cot_math_tight"),
    ("BBH, loose",   "envelope_logic_prospective", "direct_logic_loose"),
    ("BBH, tight",   "envelope_logic_prospective", "direct_logic_tight"),
    ("code OOD",     "envelope_ood", "direct_code_loose"),
    ("math OOD",     "envelope_ood", "cot_math_loose"),
]


def load(tag, name):
    """task_id -> {seed: (solution_text, correct)}"""
    out = defaultdict(dict)
    for s in SEEDS:
        p = EXP / tag / name / f"results_seed{s}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            ok = bool(r.get("success_symbolic", r["success"]))
            out[r["task_id"]][s] = (r.get("solution") or "", ok)
    return out


def analyse(runs):
    """Over ordered seed pairs (reference s, regeneration s'), s != s'."""
    same_text = n_pairs = 0
    ref_correct = regen_wrong = 0
    for _, by_seed in runs.items():
        seeds = sorted(by_seed)
        for s in seeds:
            for t in seeds:
                if s == t:
                    continue
                (txt_s, ok_s), (txt_t, ok_t) = by_seed[s], by_seed[t]
                n_pairs += 1
                same_text += (txt_s == txt_t)
                if ok_s:
                    ref_correct += 1
                    regen_wrong += (not ok_t)
    if not n_pairs:
        return None
    return {
        "tasks": len(runs),
        "pairs": n_pairs,
        "identical": same_text / n_pairs,
        "n_ref_correct": ref_correct,
        "leak": regen_wrong / ref_correct if ref_correct else float("nan"),
    }


def main():
    print("=" * 84)
    print("Regeneration under an identical policy, recovered from stored seeds")
    print("Each ordered seed pair = (reference output, independent T=0 regeneration).")
    print("=" * 84)
    print(f"{'condition':16s}{'tasks':>7s}{'pairs':>7s}{'identical text':>16s}"
          f"{'n ref-correct':>15s}{'Pr(regen wrong)':>17s}")
    rows = []
    for label, tag, name in BASELINES:
        runs = load(tag, name)
        m = analyse(runs)
        if not m:
            print(f"{label:16s}  (no data)")
            continue
        rows.append((label, m))
        print(f"{label:16s}{m['tasks']:>7d}{m['pairs']:>7d}{m['identical']:>15.3f} "
              f"{m['n_ref_correct']:>14d}{m['leak']:>17.3f}")
    print("=" * 84)
    if rows:
        ident = [m["identical"] for _, m in rows]
        leak = [m["leak"] for _, m in rows]
        print(f"Identical-text rate  : {min(ident):.3f} to {max(ident):.3f}")
        print(f"Accepting-path leak  : {min(leak):.3f} to {max(leak):.3f}")
        print()
        print("Reading: temperature 0 does not make regeneration reproducible, so a")
        print("workflow that regenerates its own draft instead of reusing the stored")
        print("reference output holds a WRONG draft on the reference-correct tasks at")
        print("the last rate above -- before its verifier is consulted. Reference")
        print("preservation removes exactly that exposure; it is not a nominal term.")
    out = ROOT / "experiments" / "provenance_from_logs.json"
    out.write_text(json.dumps({k: v for k, v in rows}, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
