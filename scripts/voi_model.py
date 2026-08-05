#!/usr/bin/env python
"""Decision-theoretic model of when refinement pays (paper §Theory).

The repair/breakage decomposition is not just a measurement device: it
implies a closed-form condition for whether a refine step has positive
expected value, and that condition PREDICTS the sign and rough magnitude of
every Part-I result from three measurable quantities.

Setup. An incumbent answer is correct with probability p. A verifier flags
it for refinement; write
    q1 = P(flagged | incumbent wrong)     (sensitivity)
    q0 = P(flagged | incumbent correct)   (1 - specificity)
Refinement applied to a flagged item repairs a wrong answer with prob. r and
breaks a correct one with prob. b. The expected change in success from
adding the refine step is

    Delta = (1-p) * q1 * r  -  p * q0 * b                              (1)

Two corollaries the experiments test:
  * Oracle verifier (q0 = 0)  =>  Delta = (1-p) q1 r >= 0: refinement can
    never hurt. Incumbent protection is the constructive way to force b -> 0
    when q0 > 0, giving the same guarantee without an oracle (C4).
  * No-signal verifier (q0 = q1 = 1, everything flagged) =>
    Delta = (1-p) r - p b, which is negative whenever b/r > (1-p)/p (C3).

This script estimates p, r, b from the frozen test runs, plugs them into (1),
and compares the predicted Delta against the measured Delta in conditions the
parameters were NOT fitted on.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = [0, 1, 2]


def per_task(dirname: str) -> dict[str, float]:
    acc = defaultdict(list)
    for s in SEEDS:
        p = EXP / dirname / f"results_seed{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                acc[r["task_id"]].append(bool(r["success"]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def decompose(struct_dir: str, base_dir: str):
    """Return p, repair rate r_eff = q1*r, breakage rate b_eff = q0*b.

    The observable rates already fold in the flagging probabilities, which
    is what equation (1) needs: Delta = (1-p)*r_eff - p*b_eff.
    """
    S, B = per_task(struct_dir), per_task(base_dir)
    ids = sorted(set(S) & set(B))
    if not ids:
        return None
    p = sum(B[t] for t in ids) / len(ids)
    hard = [S[t] for t in ids if B[t] == 0.0]
    easy = [1 - S[t] for t in ids if B[t] == 1.0]
    return {
        "p": p,
        "r_eff": sum(hard) / len(hard) if hard else 0.0,
        "b_eff": sum(easy) / len(easy) if easy else 0.0,
        "delta_measured": sum(S[t] - B[t] for t in ids) / len(ids),
        "n_hard": len(hard), "n_easy": len(easy),
    }


def predict(p, r_eff, b_eff):
    return (1 - p) * r_eff - p * b_eff


def main():
    print("=" * 78)
    print("DECISION-THEORETIC MODEL:  Delta = (1-p)*r_eff - p*b_eff")
    print("Parameters are read off each condition; the test is whether the")
    print("equation reproduces the measured Delta (it is not fitted).")
    print("=" * 78)

    conditions = [
        ("code vanilla @loose (oracle tests)",
         "envelope_test/verify_refine_3_code_loose", "envelope_test/direct_code_loose"),
        ("code vanilla @loose (50% tests)",
         "envelope_test_mask0.5_k1/verify_refine_3_code_loose",
         "envelope_test_mask0.5_k1/direct_code_loose"),
        ("code vanilla @loose (NO signal)",
         "envelope_test_mask0.0_k1/verify_refine_3_code_loose",
         "envelope_test_mask0.0_k1/direct_code_loose"),
        ("code vanilla @tight (budget floor)",
         "envelope_test/verify_refine_3_code_tight", "envelope_test/direct_code_tight"),
        ("code INCUMBENT-PROTECTED @loose",
         "envelope_test/incumbent_refine_code_loose", "envelope_test/direct_code_loose"),
        ("code INCUMBENT-PROTECTED @tight",
         "envelope_test/incumbent_refine_code_tight", "envelope_test/direct_code_tight"),
        ("math vanilla @loose (non-oracle critic)",
         "envelope_test/verify_refine_3_math_loose", "envelope_test/cot_math_loose"),
        ("math INCUMBENT-PROTECTED @loose",
         "envelope_test/incumbent_refine_cot_math_loose", "envelope_test/cot_math_loose"),
    ]

    print(f"\n{'condition':42s}{'p':>6s}{'r_eff':>7s}{'b_eff':>7s}"
          f"{'pred':>8s}{'obs':>8s}{'err':>7s}")
    rows = []
    for name, s_dir, b_dir in conditions:
        d = decompose(s_dir, b_dir)
        if not d:
            print(f"{name:42s}  (missing)")
            continue
        pred = predict(d["p"], d["r_eff"], d["b_eff"])
        err = pred - d["delta_measured"]
        rows.append((name, d, pred, err))
        print(f"{name:42s}{d['p']:>6.3f}{d['r_eff']:>7.3f}{d['b_eff']:>7.3f}"
              f"{pred:>+8.3f}{d['delta_measured']:>+8.3f}{err:>+7.3f}")

    if rows:
        mae = sum(abs(e) for *_, e in rows) / len(rows)
        signs = sum(1 for _, d, pr, _ in rows
                    if (pr > 0) == (d["delta_measured"] > 0) or
                    abs(d["delta_measured"]) < 0.005)
        print(f"\nmean absolute error = {mae:.4f}   sign agreement = {signs}/{len(rows)}")

    # -- the break-even condition, stated as a falsifiable threshold -------
    print("\n" + "=" * 78)
    print("BREAK-EVEN CONDITION:  refinement pays iff  b_eff/r_eff < (1-p)/p")
    print("=" * 78)
    for name, d, _, _ in rows:
        if d["r_eff"] > 0:
            ratio = d["b_eff"] / d["r_eff"]
            thresh = (1 - d["p"]) / d["p"] if d["p"] > 0 else float("inf")
            verdict = "pays" if ratio < thresh else "harmful"
            agree = (verdict == "pays") == (d["delta_measured"] > -0.005)
            print(f"  {name:42s} b/r={ratio:6.2f}  vs  (1-p)/p={thresh:5.2f}"
                  f"  -> {verdict:8s} {'OK' if agree else 'MISMATCH'}")
        else:
            print(f"  {name:42s} r_eff=0: no repair possible, any b>0 is harmful")

    # -- honesty check: is (1) an identity rather than a prediction? -------
    print("\n" + "=" * 78)
    print("IDENTITY CHECK.  For binary per-task outcomes, (1) is EXACT:")
    print("  mean(S-B) = P(B=0)*E[S|B=0] - P(B=1)*E[1-S|B=1].")
    print("So the table above verifies that the decomposition is EXHAUSTIVE")
    print("(residual = tasks whose seed-averaged baseline is strictly between")
    print("0 and 1), not that the model forecasts anything. The forecast test")
    print("is below: fit r_eff/b_eff in ONE condition, predict ANOTHER.")
    print("=" * 78)

    transfers = [
        ("fit @loose(oracle) -> predict @tight",
         "envelope_test/verify_refine_3_code_loose", "envelope_test/direct_code_loose",
         "envelope_test/verify_refine_3_code_tight", "envelope_test/direct_code_tight"),
        ("fit @tight -> predict @loose(oracle)",
         "envelope_test/verify_refine_3_code_tight", "envelope_test/direct_code_tight",
         "envelope_test/verify_refine_3_code_loose", "envelope_test/direct_code_loose"),
        ("fit code-incumbent@loose -> predict @tight",
         "envelope_test/incumbent_refine_code_loose", "envelope_test/direct_code_loose",
         "envelope_test/incumbent_refine_code_tight", "envelope_test/direct_code_tight"),
        ("fit vanilla@loose -> predict NO-signal@loose",
         "envelope_test/verify_refine_3_code_loose", "envelope_test/direct_code_loose",
         "envelope_test_mask0.0_k1/verify_refine_3_code_loose",
         "envelope_test_mask0.0_k1/direct_code_loose"),
    ]
    print(f"\n{'transfer':44s}{'pred':>8s}{'obs':>8s}{'err':>8s}")
    for name, fs, fb, ts, tb in transfers:
        src, tgt = decompose(fs, fb), decompose(ts, tb)
        if not src or not tgt:
            continue
        pred = predict(tgt["p"], src["r_eff"], src["b_eff"])
        print(f"{name:44s}{pred:>+8.3f}{tgt['delta_measured']:>+8.3f}"
              f"{pred - tgt['delta_measured']:>+8.3f}")
    print("\nA transfer that fails is informative: it localises which parameter"
          "\nthe intervention actually moves (budget moves r_eff by capping"
          "\niterations; verifier noise moves b_eff).")

    out = EXP / "voi_model_table.json"
    json.dump([{"condition": n, **d, "predicted": pr} for n, d, pr, _ in rows],
              open(out, "w"), indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
