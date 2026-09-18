#!/usr/bin/env python3
"""Existing-log content diagnostics and paired widened-cap contrasts; no API."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_final_auxiliary_evidence import ARMS, EXP, keyed
from scripts.analyze_clean_matrix_byte_identity import CELLS, trace_types
from hbws.data import load_split
from hbws.verify import extract_boxed


def first_accepted(row):
    types = trace_types(row)
    return (row['status'] == 'completed' and types[:2] == ['generate', 'verify']
            and 'refine' not in types)


def signed(v):
    return f'{v:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')


def ci_tex(point, interval):
    return f'${signed(point)}$ [{signed(interval[0])}, {signed(interval[1])}]'


def main():
    report = {'selection': 'first-accepted same-policy rows; byte equality is not semantic equality',
              'bootstrap_draws': 10000, 'bootstrap_seed': 20260916,
              'bootstrap_unit': 'task, retaining available eligible seeds',
              'content': {}, 'nonbinding_pairwise': {}, 'acceptance_cases': []}
    content_rows = []
    for cell, (base_name, _, same_name) in CELLS.items():
        base = keyed(EXP / 'glm4flash_repaired_matrix_clean_20260910_envelope_test' / base_name)
        same = keyed(EXP / 'glm4flash_repaired_matrix_final_20260910_causal' / same_name)
        counts = Counter()
        for key in base:
            b, w = base[key], same[key]
            if not first_accepted(w):
                continue
            counts['accepted'] += 1
            if b['solution'] == w['solution']:
                assert b['success'] == w['success']
                counts['identical'] += 1
            else:
                status = ('both_correct' if b['success'] and w['success'] else
                          'both_wrong' if not b['success'] and not w['success'] else
                          'repair' if w['success'] else 'breakage')
                counts[status] += 1
        categories = ('identical', 'both_correct', 'both_wrong', 'repair', 'breakage')
        assert sum(counts[k] for k in categories) == counts['accepted']
        report['content'][cell] = {k: counts[k] for k in ('accepted', *categories)}
        family, profile = cell.split('/')
        content_rows.append(f'{family}/\\texttt{{{profile}}} & ' +
                            ' & '.join(str(counts[k]) for k in ('accepted', *categories)) + r' \\')

    base = keyed(EXP / 'glm4flash_repaired_code_20260909_envelope_test/cot_math_tight')
    arms = {arm: keyed(EXP / 'glm4flash_math_tight_nonbinding_20260911' /
                       f'{arm}_math_tight_nonbinding') for arm in ARMS}
    eligible = {k for k, r in base.items() if r['status'] == 'completed'}
    assert len(eligible) == 383
    assert sum(base[k]['success'] for k in eligible) == 294
    tasks = {t['id']: t for t in load_split('math', 'test')[:150]}
    for key in sorted(eligible):
        b, w = base[key], arms[ARMS[1]][key]
        if b['success'] and not w['success'] and first_accepted(w):
            verifier = next(s for s in w['trace'] if s['type'] == 'verify')['verifier']
            report['acceptance_cases'].append(dict(
                task_id=key[0], seed=key[1], problem=tasks[key[0]]['prompt'],
                gold=tasks[key[0]]['gold_answer'],
                reference_answer=extract_boxed(b['solution']),
                accepted_answer=extract_boxed(w['solution']),
                checker_answers=[c['checker_answer'] for c in verifier['checks']],
                agreements=[c['agreement'] for c in verifier['checks']],
                reference_text=b['solution'], draft_text=w['solution'],
                checker_texts=[c['checker_text'] for c in verifier['checks']]))
    assert len(report['acceptance_cases']) == 3
    ids = sorted({k[0] for k in eligible})
    by_task = np.zeros((len(ids), 6), dtype=float)
    # Columns: N, baseline correct, baseline wrong, repair gap, break gap,
    # accuracy gap. Pool ratios over eligible pairs, not equal-weight task ratios.
    for i, task in enumerate(ids):
        for seed in range(3):
            key = task, seed
            if key not in eligible:
                continue
            b = bool(base[key]['success'])
            a = bool(arms[ARMS[0]][key]['success'])
            s = bool(arms[ARMS[1]][key]['success'])
            by_task[i] += (1, b, not b, (not b) * (int(a) - int(s)),
                           b * (int(not a) - int(not s)), int(a) - int(s))
    rng = np.random.default_rng(20260916)
    samples = rng.integers(0, len(ids), size=(10000, len(ids)))
    totals = by_task.sum(axis=0)
    boot = by_task[samples].sum(axis=1)
    assert np.all(boot[:, :3] > 0)
    metrics = [('repair', 3, 2), ('breakage', 4, 1), ('accuracy', 5, 0)]
    metric_columns = []
    for name, numerator, denominator in metrics:
        point = float(totals[numerator] / totals[denominator])
        interval = np.quantile(boot[:, numerator] / boot[:, denominator], [.025, .975]).tolist()
        report['nonbinding_pairwise'][name] = {'point': point, 'ci95': interval}
        metric_columns.append(ci_tex(point, interval))
    utility_columns = []
    for loss in (1, 3, 5):
        point = float((totals[3] - loss * totals[4]) / totals[0])
        interval = np.quantile((boot[:, 3] - loss * boot[:, 4]) / boot[:, 0], [.025, .975]).tolist()
        report['nonbinding_pairwise'][f'utility_lambda_{loss}'] = {'point': point, 'ci95': interval}
        utility_columns.append(ci_tex(point, interval))
    report['nonbinding_pairwise']['eligible_pairs'] = int(totals[0])
    (ROOT / 'paper/generated_nonbinding_pairwise_row.tex').write_text(' & '.join(metric_columns) + '\n')
    (ROOT / 'paper/generated_nonbinding_utility_row.tex').write_text(' & '.join(utility_columns) + '\n')
    (ROOT / 'paper/generated_accepted_content_rows.tex').write_text('\n'.join(content_rows + [r'\bottomrule']) + '\n')
    (EXP / 'submission_provenance_content_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'acceptance_cases'}, indent=2))
    print('cases', [(c['task_id'], c['seed'], c['gold'], c['accepted_answer'], c['checker_answers'])
                    for c in report['acceptance_cases']])


if __name__ == '__main__':
    main()
