#!/usr/bin/env python3
"""Independent raw-trace checks for the completed shared-refiner study.

Uses SciPy beta quantiles as an independent implementation of the CP limits.
Does not call any model or modify the frozen experiment.
"""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/shared_refiner_20260909'


def read(name):
    return [json.loads(line) for line in (OUT/name).read_text().splitlines() if line]


def main():
    from shared_refiner_study import inputs, manifest, prompt_text
    from hbws.llm import estimate_in_tokens
    tasks, base, historical, rejected = inputs()
    frozen = manifest()
    report = json.loads((OUT/'analysis.json').read_text())
    stages = read('stages.jsonl')
    trajectories = read('trajectories.jsonl')
    attempts = read('attempts.jsonl')
    assert len(stages) == 400 and len(trajectories) == 300
    keyed = {(s['task_id'],s['arm'],s['stage']):s for s in stages}
    assert len(keyed)==400
    assert set(s['model'] for s in stages)=={frozen['model']}
    assert sum(a['status']=='completed' for a in attempts)==400
    results = {}
    checks = []
    reservation_margins=[]
    # The estimator used the GLM provider's documented fixed overhead buffer.
    import os
    os.environ['LLM_PROVIDER']='glm'
    for arm,specs in frozen['arms'].items():
        rows=read(f'{arm}_results.jsonl')
        assert len(rows)==len(tasks)
        assert set(r['task_id'] for r in rows)==set(tasks)
        results[arm]={r['task_id']:r for r in rows}
        for r in rows:
            tid=r['task_id']
            if tid not in rejected:
                assert r['solution']==base[tid]['solution']
                assert r['success']==base[tid]['success']
                assert r['calls']==r['tokens']==r['seconds']==0
                continue
            draft=''
            totals=Counter(calls=0,tokens=0,seconds=0)
            gate=next(n for n in historical[tid]['trace'] if n['type']=='verify')['budget']
            in_used=gate['in_tokens'];out_used=gate['out_tokens'];calls_used=gate['llm_calls']
            for idx,spec in enumerate(specs):
                s=keyed[tid,arm,idx]
                assert s['prompt']==prompt_text(spec,tasks[tid]['prompt'],base[tid]['solution'],draft)
                reservation=estimate_in_tokens([dict(role='user',content=s['prompt'])])
                assert s['in_tokens']<=reservation
                assert s['out_tokens']<=spec['max_tokens']
                assert in_used+reservation<=16000 and out_used+spec['max_tokens']<=4000 and calls_used+1<=8
                reservation_margins.append(reservation-s['in_tokens'])
                in_used+=s['in_tokens'];out_used+=s['out_tokens'];calls_used+=1
                totals.update(calls=1,tokens=s['in_tokens']+s['out_tokens'],seconds=s['seconds'])
                draft=s['content']
            assert draft==r['solution']
            for k,v in totals.items():assert abs(r[k]-v)<1e-7
        counts=Counter((bool(base[r['task_id']]['success']),bool(r['success'])) for r in rows)
        k=counts[True,False];rep=counts[False,True]
        expected=report['refiners'][arm]
        assert k==expected['counts'].get('10',0) and rep==expected['counts'].get('01',0)
        for a,label in [(.05,'direct_cp_iid_sensitivity'),(.05/3,'direct_cp_iid_simultaneous_sensitivity')]:
            cp=float(beta.ppf(1-a,k+1,835-k))
            assert abs(cp-expected[label])<1e-10
        for a,label in [(.05,'direct_hoeffding'),(.05/3,'direct_hoeffding_simultaneous')]:
            h=(k+math.sqrt(1194*math.log(1/a)/2))/835
            assert abs(h-expected[label])<1e-10
        for key in ['calls','tokens','seconds']:
            assert abs(sum(r[key] for r in rows)-expected['cost'][key])<1e-7
        checks.append(arm+': outputs, shared acceptance contract, prompts, reservations, costs and four confidence bounds verified')
    for key in ['calls','tokens','seconds']:
        total=sum(report['refiners'][a]['cost'][key] for a in results)
        upstream=report['reference_plus_gate'][key]
        assert abs(total/(upstream+total)-report['multi_candidate_audit_only_savings'][key])<1e-12
    # Optional, explicitly post-hoc robustness: simultaneous utility intervals.
    utility={}
    for arm,rows in results.items():
        diff=np.array([int(rows[t]['success'])-int(base[t]['success']) for t in sorted(tasks)])
        probs=[np.mean(diff==-1),np.mean(diff==1),np.mean(diff==0)]
        draws=np.random.default_rng(20260909).multinomial(len(diff),probs,size=100000)
        d=(draws[:,1]-draws[:,0])/len(diff)
        utility[arm]=dict(status='post_hoc_utility_multiplicity_sensitivity',
            familywise95_bonferroni_task_bootstrap_interval=np.quantile(d,[.05/6,1-.05/6]).tolist())
    # Additional post-hoc target alignment, recorded before grading new arms.
    # On stored B and gate outcomes only the 22 correct/rejected tasks are
    # random for repeated refinement. The common ceiling is deterministic.
    eligible=sum(bool(base[t]['success']) for t in rejected)
    denominator=sum(bool(r['success']) for r in base.values())
    ceiling=eligible/denominator
    radius=math.sqrt(eligible*math.log(3/.05)/2)/denominator
    conditional={}
    for arm,rows in results.items():
        k=sum(bool(base[t]['success']) and not bool(r['success']) for t,r in rows.items())
        conditional[arm]=dict(breaks=k,observed_breakage=k/denominator,
            direct_simultaneous_upper=min(ceiling,k/denominator+radius))
    conditional_report=dict(status='additional_post_hoc_conditional_prefix_analysis',
        target='Repeat refinement on the fixed task bank with stored B and gate, not a new draw of the full pipeline.',
        assumption='Refiner executions independent across correct/rejected tasks; candidate definitions fixed before new refinement randomness.',
        reference_correct=denominator,reference_correct_rejected=eligible,
        deterministic_shared_ceiling=ceiling,direct_simultaneous_radius=radius,
        refiners=conditional,
        thresholds=[dict(epsilon=e,shared_passes=3*int(ceiling<=e),
            direct_passes=sum(r['direct_simultaneous_upper']<=e for r in conditional.values()))
            for e in frozen['secondary_thresholds']])
    (OUT/'conditional_prefix_analysis.json').write_text(json.dumps(conditional_report,indent=2)+'\n')
    summary=dict(status='PASS',checks=checks,stages=400,trajectories=300,full_task_rows=3582,
        minimum_input_reservation_margin=min(reservation_margins),
        acceptance_identity_checks=1094*3,scipy_cp_cross_checks=6,
        posthoc_utility_sensitivity=utility,
        source_manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
    (OUT/'independent_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
