#!/usr/bin/env python3
"""Frozen, post-hoc fixed-pool shared-gate study. No candidate tuning on results.

freeze: lock prompts, input hashes, gate mask and analysis before new calls.
run: replay the stored gate; interleave three genuinely executed refiners.
analyze: grade only after generation, preserving failures and all candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/shared_refiner_20260909'
DATA = ROOT / 'data/math_dcert_glm4flash.jsonl'
BASE = ROOT / 'experiments/glm4flash_dcert/baseline/results_seed0.jsonl'
OLD = ROOT / 'experiments/glm4flash_dcert/candidate/results_seed0.jsonl'
MODEL = 'glm-4-flash-250414'
FEEDBACK = 'an independent solution reached a different answer; re-examine your reasoning step by step'
FORMAT = 'End your response with the final answer in \\boxed{...}.'
ARMS = {
    'standard': [dict(max_tokens=1536, temperature=.7, prompt=
        'Problem:\n{task}\n\nPrevious attempt:\n{reference}\n\nFeedback:\n' + FEEDBACK +
        '\n\nRework the solution carefully. ' + FORMAT)],
    'targeted': [dict(max_tokens=512, temperature=.3, prompt=
        'Problem:\n{task}\n\nPrevious solution:\n{reference}\n\nAn initial check rejected this solution. '
        'Find the earliest concrete mathematical error and repair only the steps depending on it. '
        'Preserve valid reasoning. If you cannot substantiate an error, keep the original final answer. '
        'Be concise and reserve space for the final answer. ' + FORMAT)],
    'two_stage': [dict(max_tokens=1024, temperature=.7, prompt=
        'Problem:\n{task}\n\nSolve this problem independently from first principles. '
        'Check the essential calculations. Be concise. ' + FORMAT),
        dict(max_tokens=1024, temperature=.3, prompt=
        'Problem:\n{task}\n\nOriginal solution:\n{reference}\n\nIndependent solution:\n{draft}\n\n'
        'Compare both solutions against the problem. Resolve disagreements by checking the '
        'mathematical steps or substituting candidate answers. Return your best complete solution; '
        'do not average the answers. Be concise. ' + FORMAT)],
}
LOCK = threading.Lock()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line] if Path(path).exists() else []


def append(path, row):
    with LOCK, Path(path).open('a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()


def inputs():
    tasks = {r['id']: r for r in load(DATA)}
    baseline = {r['task_id']: r for r in load(BASE)}
    old = {r['task_id']: r for r in load(OLD)}
    assert tasks.keys() == baseline.keys() == old.keys()
    rejected = []
    for tid, row in old.items():
        assert row['status'] == baseline[tid]['status'] == 'completed'
        assert sum(n['type'] == 'verify' for n in row['trace']) == 1
        if any(n['type'] == 'refine' for n in row['trace']):
            rejected.append(tid)
        else:
            assert row['solution'] == baseline[tid]['solution']
    assert len(tasks) == 1194 and len(rejected) == 100
    return tasks, baseline, old, sorted(rejected)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, baseline, old, rejected = inputs()
    payload = dict(
        status='post_hoc_fixed_pool; frozen before new refiner calls',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        model=MODEL, endpoint='https://open.bigmodel.cn/api/paas/v4',
        arms=ARMS, source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
            [DATA, BASE, OLD, Path(__file__), ROOT/'hbws/runner.py', ROOT/'hbws/verify.py',
             ROOT/'hbws/ledger.py', ROOT/'hbws/prompts.py']},
        rejected_task_ids=rejected, task_count=len(tasks), workers=2,
        planned_refinement_calls=400, max_attempts_per_call=4,
        max_campaign_api_attempts=800, max_campaign_billed_tokens=2000000,
        primary_epsilon=.05, secondary_thresholds=[.005,.01,.02,.03,.04,.05,.06,.08,.10],
        alpha=.05, direct_simultaneous_alpha=.05/3,
        analysis='All three arms and historical standard; pointwise and familywise Hoeffding; CP only under additional iid mixture model. Task paired bootstrap 10000 draws. No tuning or outcome-based exclusions.',
        estimand='Stored reference and gate on fixed 1194-task pool. Exploratory outcomes; inherited FRR confidence retains original assumptions, not new independent deployment validation.',
        randomization='Seed 20260909 shuffles rejected tasks; rotate arm order by task index; no correctness used.',
        fallback='Gate accept returns stored reference without a call. API failures resumable; no answer-quality retries. Incomplete arms preclude complete comparison.',
        costs='Reference and gate reusable by both comparators. Every returned API response logged before settlement, all attempts recorded. Shared audit-only savings exclude utility-evaluation costs.',
        runtime={p: importlib.metadata.version(p) for p in ['openai','tiktoken','sympy','antlr4-python3-runtime','numpy','scipy']},
    )
    path = OUT/'manifest.json'
    if path.exists():
        current = json.loads(path.read_text())
        payload['frozen_at_utc'] = current['frozen_at_utc']
        if payload != current:
            raise RuntimeError('Refusing to alter frozen study')
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    print('Frozen study:', len(rejected), 'rejected tasks, three arms, 400 planned calls', flush=True)


def manifest():
    p = json.loads((OUT/'manifest.json').read_text())
    assert p['arms'] == ARMS
    for name, sha in p['source_hashes'].items():
        assert digest(ROOT/name) == sha, f'Frozen source changed: {name}'
    return p


def prompt_text(stage, task, reference, draft):
    # Replace placeholders before insertion so source text is never templated.
    import re
    values = dict(task=task, reference=reference, draft=draft)
    return re.sub(r'\{(task|reference|draft)\}', lambda m: values[m[1]], stage['prompt'])


def run(limit=None):
    cfg = manifest()
    from run_glm4flash_replication import load_key
    os.environ.update(LLM_PROVIDER='glm', GLM_API_KEY=load_key(), GLM_MODEL=MODEL,
        GLM_BASE_URL=cfg['endpoint'], LLM_PRICE_IN_PER_M='0', LLM_PRICE_OUT_PER_M='0')
    from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
    from hbws.llm import estimate_in_tokens
    from hbws.ledger import TaskLedger, BUDGET_TIERS, BudgetCaps, llm_call_vec
    tasks, baseline, old, rejected = inputs()
    existing = {(r['task_id'],r['arm']):r for r in load(OUT/'trajectories.jsonl')}
    completed_stages = {(r['task_id'],r['arm'],r['stage']):r for r in load(OUT/'stages.jsonl')}
    prior_attempts = load(OUT/'attempts.jsonl')
    counters = dict(attempts=len(prior_attempts), tokens=sum(r.get('in_tokens',0)+r.get('out_tokens',0) for r in completed_stages.values()))
    random.Random(20260909).shuffle(rejected)
    jobs=[]
    names=list(ARMS)
    for i,tid in enumerate(rejected):
        for j in range(3):
            arm=names[(i+j)%3]
            if (tid,arm) not in existing:
                jobs.append((tid,arm))
    if limit:
        jobs=jobs[:limit]
    start=time.monotonic()
    stamp=datetime.now(timezone.utc).isoformat()
    print(f'Start {stamp}: {len(jobs)} trajectories remaining in this invocation',flush=True)

    def one(tid,arm):
        client=OpenAI(api_key=os.environ['GLM_API_KEY'],base_url=cfg['endpoint'],timeout=75,max_retries=0)
        ledger=TaskLedger(BudgetCaps(**vars(BUDGET_TIERS['loose'])))
        gate=next(n for n in old[tid]['trace'] if n['type']=='verify')
        ledger.used.update({k:v for k,v in gate['budget'].items() if k!='wall_sec'})
        draft=''
        records=[]
        for idx,stage in enumerate(ARMS[arm]):
            saved=completed_stages.get((tid,arm,idx))
            if saved:
                # Completed calls are not reissued on resume.
                for k in ['llm_calls','in_tokens','out_tokens']:
                    ledger.used[k]+=1 if k=='llm_calls' else saved[k]
                draft=saved['content'];records.append(saved);continue
            prompt=prompt_text(stage,tasks[tid]['prompt'],baseline[tid]['solution'],draft)
            messages=[dict(role='user',content=prompt)]
            lease=ledger.reserve(llm_call_vec(estimate_in_tokens(messages),stage['max_tokens']))
            if lease is None:
                raise RuntimeError('Unexpected reservation failure; keep incomplete and inspect')
            for attempt in range(4):
                with LOCK:
                    if counters['attempts']>=cfg['max_campaign_api_attempts'] or counters['tokens']>=cfg['max_campaign_billed_tokens']:
                        raise RuntimeError('Campaign cap reached')
                    counters['attempts']+=1
                t0=time.monotonic()
                try:
                    response=client.chat.completions.create(model=MODEL,messages=messages,
                        temperature=stage['temperature'],max_tokens=stage['max_tokens'])
                except (RateLimitError,APIConnectionError,APITimeoutError,InternalServerError) as e:
                    append(OUT/'attempts.jsonl',dict(task_id=tid,arm=arm,stage=idx,status='retryable_error',error_type=type(e).__name__,seconds=time.monotonic()-t0,time=time.time()))
                    if attempt==3: raise RuntimeError('Provider retries exhausted') from None
                    time.sleep([3,10,30][attempt]);continue
                except Exception as e:
                    append(OUT/'attempts.jsonl',dict(task_id=tid,arm=arm,stage=idx,status='fatal_error',error_type=type(e).__name__,seconds=time.monotonic()-t0,time=time.time()))
                    raise RuntimeError('Provider fatal error: '+type(e).__name__) from None
                seconds=time.monotonic()-t0
                if response.usage is None: raise RuntimeError('Missing usage')
                record=dict(task_id=tid,arm=arm,stage=idx,model=response.model,
                    in_tokens=response.usage.prompt_tokens,out_tokens=response.usage.completion_tokens,
                    content=response.choices[0].message.content or '',finish_reason=response.choices[0].finish_reason,
                    prompt=prompt,seconds=seconds,time=time.time())
                append(OUT/'stages.jsonl',record)
                append(OUT/'attempts.jsonl',dict(task_id=tid,arm=arm,stage=idx,status='completed',seconds=seconds,time=time.time()))
                with LOCK: counters['tokens']+=record['in_tokens']+record['out_tokens']
                if response.model != MODEL: raise RuntimeError('Unexpected provider model identity')
                ledger.settle(lease,dict(llm_calls=1,in_tokens=record['in_tokens'],out_tokens=record['out_tokens']))
                draft=record['content'];records.append(record);break
        row=dict(task_id=tid,arm=arm,solution=draft,status='completed',gate_rejected=True,
            calls=len(records),tokens=sum(r['in_tokens']+r['out_tokens'] for r in records),
            seconds=sum(r['seconds'] for r in records),
            finish_reasons=[r['finish_reason'] for r in records],time=time.time())
        append(OUT/'trajectories.jsonl',row)
        return row

    errors=0
    # Small batches bound the work that remains in flight after any failure.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0,len(jobs),2):
            futures=[pool.submit(one,*job) for job in jobs[offset:offset+2]]
            for f in as_completed(futures):
                try: f.result()
                except Exception as e:
                    errors+=1;print('Execution stopped:',type(e).__name__,str(e),flush=True)
            if (offset+2)%10==0 or offset+2>=len(jobs):
                print(f'progress trajectories={min(offset+2,len(jobs))}/{len(jobs)} errors={errors} attempts={counters["attempts"]} billed_tokens={counters["tokens"]}',flush=True)
            if errors: break
    append(OUT/'invocations.jsonl',dict(started_at=stamp,elapsed_seconds=time.monotonic()-start,errors=errors,scheduled=len(jobs)))
    if errors: raise SystemExit(2)


def analyze():
    cfg=manifest()
    import numpy as np
    from hbws.verify import grade_math
    from hbws.stats import binomial_exact_upper
    tasks, baseline, old, rejected=inputs()
    trajectories={(r['task_id'],r['arm']):r for r in load(OUT/'trajectories.jsonl')}
    assert len(trajectories)==300, f'Incomplete: {len(trajectories)}/300'
    rows={}
    grading_mismatches=[]
    # Verify grading consistency on the gate-rejected stratum before comparison.
    for tid in rejected:
        for label, archive in [('baseline',baseline),('historical',old)]:
            now=bool(grade_math(archive[tid]['solution'],tasks[tid]['gold_answer']))
            if now != bool(archive[tid]['success']):
                grading_mismatches.append(dict(task_id=tid,source=label,old=bool(archive[tid]['success']),now=now))
    (OUT/'grading_consistency.json').write_text(json.dumps(grading_mismatches,indent=2)+'\n')
    if grading_mismatches: raise RuntimeError('Grader mismatch; inspect before mixing labels')
    for arm in ARMS:
        result=[]
        for tid in tasks:
            if tid in rejected:
                row=dict(trajectories[tid,arm]);row['success']=bool(grade_math(row['solution'],tasks[tid]['gold_answer']))
            else:
                row=dict(task_id=tid,arm=arm,solution=baseline[tid]['solution'],success=bool(baseline[tid]['success']),
                    status='reference_preserved',gate_rejected=False,calls=0,tokens=0,seconds=0)
            row['baseline_success']=bool(baseline[tid]['success'])
            assert row['gate_rejected'] or row['solution']==baseline[tid]['solution']
            result.append(row)
        (OUT/f'{arm}_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
        rows[arm]=result
    n=len(tasks);den=sum(bool(r['success']) for r in baseline.values());fr=sum(bool(baseline[t]['success']) for t in rejected)
    def bound(k,alpha=.05):return min(1,(k+math.sqrt(n*math.log(1/alpha)/2))/den)
    shared=dict(events=fr,hoeffding=bound(fr),cp_iid_sensitivity=binomial_exact_upper(fr,den))
    gate_cost=Counter(calls=0,tokens=0,seconds=0)
    for tid,b in baseline.items():
        v=next(z for z in old[tid]['trace'] if z['type']=='verify')
        gate_cost.update(calls=b['budget']['llm_calls']+v['budget']['llm_calls'],
            tokens=b['budget']['in_tokens']+b['budget']['out_tokens']+v['budget']['in_tokens']+v['budget']['out_tokens'],
            seconds=b['budget']['wall_sec']+v['sec'])
    reports={}
    for arm,result in rows.items():
        counts=Counter(str(int(r['baseline_success']))+str(int(r['success'])) for r in result)
        k=counts['10'];repair=counts['01'];rng=np.random.default_rng(20260909)
        bootstrap=rng.multinomial(n,[k/n,repair/n,1-(k+repair)/n],size=10000)
        delta=(bootstrap[:,1]-bootstrap[:,0])/n
        cost={key:sum(r[key] for r in result) for key in ['calls','tokens','seconds']}
        reports[arm]=dict(counts=dict(counts),breakage=k/den,repair=repair/(n-den),delta=(repair-k)/n,
            delta_ci95=np.quantile(delta,[.025,.975]).tolist(),cost=cost,
            direct_hoeffding=bound(k),direct_hoeffding_simultaneous=bound(k,.05/3),
            direct_cp_iid_sensitivity=binomial_exact_upper(k,den),
            direct_cp_iid_simultaneous_sensitivity=binomial_exact_upper(k,den,alpha=.05/3),
            fr_harm_fraction=k/fr,
            truncation_stage_count=sum(r['finish_reason']=='length' for r in load(OUT/'stages.jsonl') if r['arm']==arm),
            single_candidate_savings={key:cost[key]/(gate_cost[key]+cost[key]) for key in cost})
    extra={key:sum(r['cost'][key] for r in reports.values()) for key in ['calls','tokens','seconds']}
    comparisons=[]
    for eps in cfg['secondary_thresholds']:
        comparisons.append(dict(epsilon=eps,shared_hoeffding_passes=3*int(shared['hoeffding']<=eps),
            direct_hoeffding_simultaneous_passes=sum(r['direct_hoeffding_simultaneous']<=eps for r in reports.values()),
            shared_cp_iid_passes=3*int(shared['cp_iid_sensitivity']<=eps),
            direct_cp_iid_simultaneous_passes=sum(r['direct_cp_iid_simultaneous_sensitivity']<=eps for r in reports.values())))
    attempts=load(OUT/'attempts.jsonl')
    report=dict(status='completed_post_hoc_fixed_pool_study',n=n,baseline_correct=den,gate_rejections=len(rejected),shared=shared,
        refiners=reports,reference_plus_gate=dict(gate_cost),total_measured_refinement=extra,
        multi_candidate_audit_only_savings={key:extra[key]/(gate_cost[key]+extra[key]) for key in extra},
        threshold_comparison=comparisons,attempt_status_counts=dict(Counter(r['status'] for r in attempts)),
        new_execution_wall_seconds=sum(r['elapsed_seconds'] for r in load(OUT/'invocations.jsonl')),
        model_identities=sorted(set(r['model'] for r in load(OUT/'stages.jsonl'))),
        statistical_scope=cfg['estimand'],manifest_sha256=digest(OUT/'manifest.json'))
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze']);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='run':run(args.limit)
    else:analyze()
