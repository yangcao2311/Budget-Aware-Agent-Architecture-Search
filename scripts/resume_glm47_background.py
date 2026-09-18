#!/usr/bin/env python3
"""Serial, resumable operational scheduler for the frozen free GLM follow-up."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = ROOT / 'experiments/glm47_background_20260916_status.json'
LOG = ROOT / 'experiments/glm47_background_20260916.log'


def write_state(**fields):
    payload = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   model='glm-4.7-flash', paid_api=False, **fields)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(STATE)
    print(json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        if STATE.exists():
            old = json.loads(STATE.read_text())
            if old.get('pid'):
                try:
                    os.kill(old['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Existing scheduler is still alive')
        with LOG.open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        print('started scheduler', child.pid, 'log', LOG, flush=True)
        return

    import scripts.recover_glm47_causal_cell_failures as recovery
    import scripts.run_glm47_causal_one_cell as driver
    import scripts.run_glm47_within_family as frozen
    if frozen.MODEL != 'glm-4.7-flash':
        raise RuntimeError('Refusing a different endpoint/model')
    workers = 2
    original_evaluate = recovery.evaluate

    def scheduled_evaluate(*a, **kw):
        kw['workers'] = workers
        return original_evaluate(*a, **kw)

    recovery.evaluate = scheduled_evaluate
    driver.evaluate = scheduled_evaluate
    # Only concurrency/cooldown change. Requests, seeds, references, caps,
    # scoring and execution-status-only recovery remain unchanged.
    amendment = ROOT / 'experiments/glm47_operational_resume_20260916.json'
    if not amendment.exists():
        amendment.write_text(json.dumps(dict(
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            initial_workers=2, fallback_workers=1,
            fallback_trigger='any newly observed RateLimitError',
            cooldown_seconds=600, maximum_cooldown_seconds=1800,
            unchanged=['model', 'endpoint', 'tasks', 'seeds', 'references',
                       'workflows', 'caps', 'grader', 'no-cache'],
            selection='execution-status failures only',
        ), indent=2) + '\n')
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    completed = []
    try:
        for seed in (0, 1, 2):
            for arm in driver.ARMS:
                key = f'{arm}/seed{seed}'
                failures = 0
                while True:
                    write_state(status='running', pid=os.getpid(), cell=key,
                                workers=workers, completed=completed)
                    rp = recovery.source(arm, seed)
                    if not rp.exists():
                        driver.run(arm, seed)
                    rows = recovery.read_rows(rp)
                    if len(rows) != 150:
                        raise RuntimeError(f'Unexpected incomplete result: {key}')
                    errors = sum(str(r.get('status', '')).startswith('error') for r in rows)
                    if not errors:
                        completed.append(key)
                        break
                    if not recovery.manifest(arm, seed).exists():
                        recovery.freeze(arm, seed)
                    cfg = recovery.config(arm, seed)
                    remaining = set(cfg['targets']) - set(recovery.recovered(arm, seed))
                    if not remaining:
                        recovery.apply(arm, seed)
                        continue
                    attempt_log = recovery.cell_dir(arm, seed) / 'attempts.jsonl'
                    offset = attempt_log.stat().st_size if attempt_log.exists() else 0
                    recovery.run(arm, seed)
                    if attempt_log.exists():
                        with attempt_log.open() as stream:
                            stream.seek(offset)
                            rate_limited = any(json.loads(line).get('error_type') ==
                                               'RateLimitError' for line in stream if line.strip())
                        if rate_limited:
                            workers = 1
                    unresolved = set(cfg['targets']) - set(recovery.recovered(arm, seed))
                    if not unresolved:
                        recovery.apply(arm, seed)
                        continue
                    failures = failures + 1 if len(unresolved) >= len(remaining) else 0
                    delay = min(1800, 600 * (1 + failures))
                    write_state(status='cooldown', pid=os.getpid(), cell=key,
                                workers=workers, remaining=len(unresolved),
                                retry_after_seconds=delay, completed=completed)
                    time.sleep(delay)
        write_state(status='complete', pid=os.getpid(), workers=workers,
                    completed=completed)
    except Exception as exc:
        write_state(status='stopped_error', pid=os.getpid(), workers=workers,
                    completed=completed, error=str(exc))
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    main()
