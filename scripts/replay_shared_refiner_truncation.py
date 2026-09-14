#!/usr/bin/env python3
"""Post-hoc finish-reason fallback replay; never replaces the frozen arms.

Defined after inspecting the completed study's truncation diagnostics.
No new model calls: if the final provider response ended for reason `length`,
return the stored reference. The guard reads no correctness label at runtime.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/shared_refiner_20260909'


def main():
    from shared_refiner_study import inputs
    tasks,baseline,old,rejected=inputs()
    stages={(r['task_id'],r['arm'],r['stage']):r for r in
        map(json.loads,(OUT/'stages.jsonl').read_text().splitlines())}
    report=dict(status='post_hoc_deterministic_replay_after_outcome_inspection',
        rule='Return stored reference iff final API response finish_reason is length; otherwise use original final output.',
        additional_model_calls=0,original_outputs_unchanged=True,
        limitation='Not a prospectively tested or independently selected workflow. A fallback can discard useful truncated repairs on other data.',
        created_at_utc=datetime.now(timezone.utc).isoformat(),arms={})
    for arm in ['standard','targeted','two_stage']:
        raw=list(map(json.loads,(OUT/f'{arm}_results.jsonl').read_text().splitlines()))
        result=[];corrected=lost=triggered=0
        for r in raw:
            row=dict(r)
            trigger=r['gate_rejected'] and stages[r['task_id'],arm,1 if arm=='two_stage' else 0]['finish_reason']=='length'
            if trigger:
                triggered+=1
                corrected+=int(r['baseline_success'] and not r['success'])
                lost+=int(not r['baseline_success'] and r['success'])
                row.update(solution=baseline[r['task_id']]['solution'],success=r['baseline_success'])
            row['posthoc_length_fallback']=bool(trigger)
            result.append(row)
        counts=Counter(str(int(r['baseline_success']))+str(int(r['success'])) for r in result)
        k=counts['10'];rep=counts['01'];n=len(result)
        draws=np.random.default_rng(20260909).multinomial(n,[k/n,rep/n,1-(k+rep)/n],size=10000)
        deltas=(draws[:,1]-draws[:,0])/n
        report['arms'][arm]=dict(transitions=dict(counts),repairs=rep,breaks=k,
            breakage=k/835,delta=(rep-k)/n,task_bootstrap_ci95=np.quantile(deltas,[.025,.975]).tolist(),
            guarded_executions=triggered,observed_breakages_removed=corrected,observed_repairs_lost=lost)
        (OUT/f'{arm}_posthoc_length_fallback.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    (OUT/'posthoc_length_fallback.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
