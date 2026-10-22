"""Evaluate authored base and variant documents through Tinker's chat API."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
import time

from dataset import HERE, load_dataset
from prompts import messages

RESULTS = HERE / "results"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def load_config(path=HERE / "config.json"):
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    config = json.loads(path.read_text())
    config["base_url"] = os.getenv("TINKER_BASE_URL") or config["base_url"]
    return config

def parse_output(content):
    text = (content or '').strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Output is not an object')
    for key in ['sender', 'receiver', 'subject', 'explanation']:
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f'Invalid {key}')
    if type(obj.get('is_appropriate')) is not int or obj['is_appropriate'] not in (0, 1):
        raise ValueError('Verdict must be integer 0 or 1')
    if not isinstance(obj.get('transmission_principles'), list) or not all(isinstance(v, str) for v in obj['transmission_principles']):
        raise ValueError('Invalid transmission principles')
    if not isinstance(obj.get('data_types'), list) or not all(isinstance(v, dict) and isinstance(v.get('type'), str) and isinstance(v.get('value'), str) for v in obj['data_types']):
        raise ValueError('Invalid data_types')
    return obj

def plan(config, cases, pipelines=("baseline", "improved")):
    for model in config["models"]:
        for effort in model["efforts"]:
            for pipeline in pipelines:
                for case in cases:
                    request = {
                        "model": model["model"],
                        "messages": messages(case["scenario_text"], pipeline),
                        "temperature": config["temperature"], "max_tokens": config["max_tokens"],
                        "extra_body": {"separate_reasoning": True},
                    }
                    if effort in {"off", "on"}:
                        request["extra_body"]["reasoning_effort"] = effort == "on"
                    elif effort != "default":
                        request["reasoning_effort"] = effort
                    yield {
                        **{k: v for k, v in case.items() if k != "scenario_text"},
                        "run_id": digest({"request": request, "base_url": config["base_url"]}),
                        "model": model["model"], "size_class": model["size_class"],
                        "total_parameters_billions": model["total_parameters_billions"],
                        "effort": effort, "pipeline": pipeline, "request": request,
                    }

def smoke_jobs(config, cases):
    first = next(case for case in cases if case["kind"] == "base")
    return list(plan(config, [first], ["baseline"]))

def latest_records(path):
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records[record["run_id"]] = record
    return records

def execute_request(job, config, key, client):
    """Execute one immutable job; callers serialize all writes to the audit log."""
    record = dict(job)
    record.update({'base_url': config['base_url'], 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    start = time.monotonic()
    try:
        response = client.chat.completions.create(**job['request'])
        record['response_id'] = response.id
        record['served_model'] = response.model
        record['usage'] = response.usage.model_dump() if response.usage else None
        choice = response.choices[0]
        record['finish_reason'] = choice.finish_reason
        record['raw_content'] = choice.message.content
        # Keep only size metadata, never the private reasoning trace.
        record['reasoning_characters'] = len(getattr(choice.message, 'reasoning_content', None) or '')
        try:
            record['prediction'] = parse_output(choice.message.content)
            record['status'] = 'ok'
        except (ValueError, TypeError) as error:
            record['status'] = 'invalid_output'
            record['error'] = str(error)
    except Exception as error:
        record['status'] = 'api_error'
        record['error_type'] = type(error).__name__
        record['http_status'] = getattr(error, 'status_code', None)
        body = getattr(error, 'body', None)
        if body is not None:
            detail = body.get('error', body.get('detail', body)) if isinstance(body, dict) else body
            message = detail.get('message', json.dumps(detail)) if isinstance(detail, dict) else str(detail)
            record['provider_error_message'] = message.replace(key, '[REDACTED]')[:1500]
    record['latency_seconds'] = round(time.monotonic()-start, 3)
    return record


def run_requests(pending, config, key, client, workers, smoke=False):
    """Keep workers busy; after a provider error, drain and save in-flight calls."""
    jobs = iter(pending)
    stop = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = set()
        while True:
            while not stop and len(active) < workers:
                job = next(jobs, None)
                if job is None:
                    break
                active.add(pool.submit(execute_request, job, config, key, client))
                time.sleep(config.get('min_interval_seconds', 0.5))
            if not active:
                break
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                if record['status'] == 'api_error' and (not smoke or record.get('http_status') in {401, 402, 403}):
                    stop = True
                yield record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Test one base case per model/effort setting")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--pipeline", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--model", action="append", help="Limit inference to this configured model ID; repeatable")
    args = parser.parse_args()
    if args.max_new_requests is not None and args.max_new_requests < 1:
        parser.error("--max-new-requests must be positive")
    complete_config = load_config()
    config = complete_config
    if args.model:
        unknown = set(args.model) - {m['model'] for m in config['models']}
        if unknown:
            parser.error(f'Unknown model IDs: {sorted(unknown)}')
        config = {**config, 'models': [m for m in config['models'] if m['model'] in args.model]}
    cases, hashes = load_dataset()
    pipelines = ["baseline", "improved"] if args.pipeline == "both" else [args.pipeline]
    jobs = smoke_jobs(config, cases) if args.smoke_test else list(plan(config, cases, pipelines))
    # Base cases precede variants; each document is an independent request.
    jobs.sort(key=lambda j: (j["pipeline"], j["kind"] != "base", j["case_id"], j["model"], j["effort"]))
    if args.dry_run:
        print(json.dumps({"models": [m["model"] for m in config["models"]],
                          "requests": len(jobs), "input_fields_sent": ["scenario_text"],
                          "max_output_tokens": len(jobs)*config["max_tokens"],
                          "source_sha256": hashes}, indent=2))
        return
    path = RESULTS / "predictions.jsonl"
    latest = latest_records(path)
    if not args.smoke_test:
        required = smoke_jobs(config, cases)
        failed = [j for j in required if latest.get(j["run_id"], {}).get("status") != "ok"]
        if failed:
            raise SystemExit("Smoke tests are missing or failed. Run evaluate.py --smoke-test first; no full-run requests sent.")
    pending = [j for j in jobs if j["run_id"] not in latest or
               (args.retry_errors and latest[j["run_id"]]["status"] == "api_error")]
    if args.max_new_requests:
        pending = pending[:args.max_new_requests]
    if not pending:
        print("No new requests needed; recorded model outcomes reused.")
        return
    key = os.getenv("TINKER_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set TINKER_API_KEY in assignment-4/.env; no requests sent.")
    from openai import OpenAI
    RESULTS.mkdir(parents=True, exist_ok=True)
    stop = False
    pending = [{**job, 'source_sha256': hashes, 'phase': 'smoke' if args.smoke_test else 'evaluation'} for job in pending]
    # Freeze the inputs and settings beside this experiment's results.
    inputs = RESULTS/'inputs'
    inputs.mkdir(exist_ok=True)
    snapshots = {name: (HERE/'data'/name).read_bytes() for name in hashes}
    snapshots['config.json'] = json.dumps(complete_config, indent=2).encode()+b'\n'
    snapshots['prompts.py'] = (HERE/'prompts.py').read_bytes()
    for name, content in snapshots.items():
        destination = inputs/name
        if destination.exists() and destination.read_bytes() != content:
            raise SystemExit(f'{name} changed during this experiment; use a new results directory.')
        destination.write_bytes(content)
    with OpenAI(api_key=key, base_url=config['base_url'], timeout=180, max_retries=0) as client:
        for record in run_requests(pending, config, key, client, args.workers, args.smoke_test):
            with path.open('a') as f:
                f.write(json.dumps(record)+'\n')
            latest[record['run_id']] = record
            print(f"{record['model']} {record['effort']} {record['pipeline']} {record['case_id']}: {record['status']}", flush=True)
            if record['status'] == 'api_error':
                print(f"  {record.get('error_type')} HTTP {record.get('http_status')}: {record.get('provider_error_message', 'Connection or provider error')}", flush=True)
                if not args.smoke_test or record.get('http_status') in {401, 402, 403}:
                    stop = True
    if args.smoke_test:
        passed = sum(latest.get(j["run_id"], {}).get("status") == "ok" for j in jobs)
        print(f"Smoke tests: {passed}/{len(jobs)} schema-valid responses.")
        if passed != len(jobs):
            raise SystemExit(1)
    elif stop:
        raise SystemExit("Provider error; all in-flight responses saved. Fix the cause before --retry-errors.")

if __name__ == "__main__":
    main()
