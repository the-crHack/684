"""Quota-aware primary experiment with three spaced, independent API workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import httpx
from openai import OpenAI
from dataset import HERE, load_rows, cases
from evaluate import load_config, plan, core_priority, execute_request

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[1,2,3], default=3)
    parser.add_argument('--rounds', type=int, choices=[1,2,3], default=2)
    args = parser.parse_args()
    config = load_config(HERE/'config.openrouter.json')
    key = os.getenv('OPENROUTER_API_KEY', '').strip()
    if not key:
        raise SystemExit('Set OPENROUTER_API_KEY in .env first.')
    path = HERE/'results/openrouter/predictions.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = sorted([j for j in plan(config, cases(load_rows()[0]), ['baseline','improved'])
                   if core_priority(j)[0] < 4], key=core_priority)
    # The coordinator is the sole file writer. Workers never modify the dataset,
    # job requests, results file, or prompt configuration.
    with OpenAI(api_key=key, base_url=config['base_url'], timeout=180, max_retries=0) as client:
        for round_index in range(args.rounds):
            latest = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    latest[record['run_id']] = record
            pending = [j for j in jobs if j['run_id'] not in latest or latest[j['run_id']]['status'] == 'api_error']
            if not pending:
                print('Primary study complete: all 46 jobs have model outcomes.', flush=True)
                return
            response = httpx.get(config['base_url']+'/key', headers={'Authorization':'Bearer '+key}, timeout=30)
            if response.status_code != 200:
                raise SystemExit(f'Quota check failed (HTTP {response.status_code}); no new inference requests sent.')
            quota = response.json().get('data',{}).get('free_model_daily_requests',{})
            remaining = quota.get('remaining')
            if not isinstance(remaining,int):
                raise SystemExit('No numeric free quota available; use the sequential runner with an explicit batch cap.')
            print(f'Round {round_index+1}: {len(pending)} pending; {remaining} daily free requests remaining.', flush=True)
            if remaining <= 0:
                print('Daily free quota exhausted. Saved work can resume after reset.', flush=True)
                return
            pending = pending[:remaining]
            stop = False
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for offset in range(0,len(pending),args.workers):
                    futures = []
                    for job in pending[offset:offset+args.workers]:
                        futures.append(pool.submit(execute_request, job, config, key, client))
                        time.sleep(config.get('min_interval_seconds',4))
                    for future in as_completed(futures):
                        record = future.result()
                        with path.open('a') as f:
                            f.write(json.dumps(record)+'\n')
                        print(f"{record['model']} {record['effort']} {record['pipeline']} {record['case_id']}: {record['status']}", flush=True)
                        status = record.get('http_status')
                        metadata = json.loads(record.get('provider_error_detail','{}'))
                        if status in {400,401,402,403,404} or (status == 429 and metadata.get('limit_source') != 'upstream_provider_shared_pool'):
                            stop = True
                    if stop:
                        print('Provider/account limit reached; all in-flight outcomes saved. Stopping.', flush=True)
                        return
            if round_index+1 < args.rounds:
                time.sleep(10)
        print('Requested rounds finished. Run analyze.py to check coverage; rerun this command to resume.', flush=True)

if __name__ == '__main__':
    main()
