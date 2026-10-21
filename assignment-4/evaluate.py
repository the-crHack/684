"""Resumable, scenario-text-only OpenAI-compatible evaluation campaign."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from dataset import HERE, cases, load_rows
from prompts import messages

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def load_config(path):
    config = json.loads(path.read_text())
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE/'.env')
    except ImportError:
        pass
    config['base_url'] = os.getenv(config.get('base_url_env', 'TINKER_BASE_URL')) or config['base_url']
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

def plan(config, dataset, pipelines):
    for model in config['models']:
        for effort in model['efforts']:
            for pipeline in pipelines:
                for case in dataset:
                    # Only text from the dataset enters the actual request.
                    request = {'model': model['model'], 'messages': messages(case['scenario_text'], pipeline),
                               'temperature': config['temperature'], 'max_tokens': config['max_tokens']}
                    if config.get('provider') == 'openrouter':
                        if not model['model'].endswith(':free'):
                            raise ValueError('OpenRouter configuration requires explicit :free model IDs')
                        request['extra_body'] = {'reasoning': {'exclude': True},
                                                 'provider': {'require_parameters': True, 'max_price': {'prompt': 0, 'completion': 0}}}
                        if effort != 'default':
                            request['extra_body']['reasoning']['effort'] = effort
                    else:
                        request['extra_body'] = {'separate_reasoning': True}
                        if effort != 'default':
                            request['reasoning_effort'] = effort
                    identity = {'request': request, 'base_url': config['base_url']}
                    yield {'run_id': digest(identity), 'case_id': case['case_id'], 'parent_id': case['parent_id'],
                           'kind': case['kind'], 'expected': case['expected'], 'pipeline': pipeline,
                           'model': model['model'], 'effort': effort, 'request': request,
                           'size_class': model['size_class'], 'input_sha256': digest(case['scenario_text'])}

def core_priority(job):
    """Resource-based ordering, fixed independently of model predictions."""
    if job['kind'] == 'original' and job['pipeline'] == 'baseline' and job['effort'] != 'high':
        tier = 0
    elif job['kind'] == 'original' and job['pipeline'] == 'improved' and job['size_class'] == 'compact':
        tier = 1
    elif (job['kind'] == 'injection'
          and job['size_class'] == 'compact' and job['parent_id'] in {'001', '006'}):
        tier = 2
    elif job['kind'] == 'original' and job['pipeline'] == 'baseline' and job['effort'] == 'high' and job['parent_id'] in {'001', '006'}:
        tier = 3
    else:
        tier = 4
    # Interleave sizes in the initial baseline stage to establish all model access.
    return (tier, job['case_id'], job['model'], job['pipeline'], job['effort'])

def execute_request(job, config, key, client):
    """Execute one immutable job; callers serialize all writes to the audit log."""
    record = {k: v for k, v in job.items() if k != 'request'}
    record.update({'request': job['request'], 'base_url': config['base_url'], 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    start = time.monotonic()
    try:
        response = client.chat.completions.create(**job['request'])
        record['response_id'] = response.id
        record['served_model'] = response.model
        record['usage'] = response.usage.model_dump() if response.usage else None
        choice = response.choices[0]
        record['finish_reason'] = choice.finish_reason
        record['raw_content'] = choice.message.content
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
        if isinstance(body, dict):
            detail = body.get('error', body)
            if isinstance(detail, dict) and isinstance(detail.get('message'), str):
                record['provider_error_message'] = detail['message'].replace(key, '[REDACTED]')[:1000]
                record['provider_error_detail'] = json.dumps(detail.get('metadata', {})).replace(key, '[REDACTED]')[:3000]
    record['latency_seconds'] = round(time.monotonic()-start, 3)
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path, default=HERE/'results/predictions.jsonl')
    parser.add_argument('--pipeline', choices=['baseline', 'improved', 'both'], default='both')
    parser.add_argument('--originals-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--max-new-requests', type=int, help='Stop after this many new jobs; resume on the next run')
    parser.add_argument('--prioritize-core', action='store_true', help='Run three-size baseline, compact improvement, then selected stress/effort checks first')
    parser.add_argument('--model', help='Run only this configured model ID')
    parser.add_argument('--primary-only', action='store_true', help='Restrict to the quota-budgeted primary study')
    parser.add_argument('--continue-upstream-rate-limits', action='store_true', help='Continue other cases after an upstream shared-pool 429; account quota limits still stop')
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE/'.env')
    except ImportError:
        if not args.dry_run:
            raise SystemExit('Install requirements.txt first (python-dotenv missing).')
    if args.max_new_requests is not None and args.max_new_requests < 1:
        parser.error('--max-new-requests must be positive')
    rows, _ = load_rows()
    dataset = cases(rows)
    if args.originals_only:
        dataset = [c for c in dataset if c['kind'] == 'original']
    pipelines = ['baseline', 'improved'] if args.pipeline == 'both' else [args.pipeline]
    jobs = list(plan(config, dataset, pipelines))
    if args.primary_only:
        jobs = [job for job in jobs if core_priority(job)[0] < 4]
    if args.model:
        jobs = [job for job in jobs if job['model'] == args.model]
        if not jobs:
            parser.error('--model must match a configured model ID')
    if args.prioritize_core:
        jobs.sort(key=core_priority)
    if args.limit:
        jobs = jobs[:args.limit]
    if args.dry_run:
        print(json.dumps({'requests': len(jobs), 'models': [m['model'] for m in config['models']],
                          'cases': len(dataset), 'pipelines': pipelines,
                          'input_fields_sent': ['scenario_text'], 'maximum_output_tokens': len(jobs)*config['max_tokens']}, indent=2))
        return
    key_env = config.get('api_key_env', 'TINKER_API_KEY')
    key = os.getenv(key_env, '').strip()
    if not key:
        raise SystemExit(f'Set {key_env} in assignment-4/.env. No requests were made.')
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=config['base_url'], timeout=180,
                    max_retries=0 if config.get('provider') == 'openrouter' else 2)
    done = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            done[record['run_id']] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attempted = 0
    last_request = 0
    for index, job in enumerate(jobs, 1):
        previous = done.get(job['run_id'])
        if previous and (previous['status'] != 'api_error' or not args.retry_errors):
            continue
        if args.max_new_requests is not None and attempted >= args.max_new_requests:
            print('Batch limit reached; rerun the same command to resume.')
            break
        delay = config.get('min_interval_seconds', 0) - (time.monotonic()-last_request)
        if delay > 0:
            time.sleep(delay)
        last_request = time.monotonic()
        attempted += 1
        record = execute_request(job, config, key, client)
        with args.output.open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(f"{index}/{len(jobs)} {job['model']} {job['effort']} {job['pipeline']} {job['case_id']}: {record['status']}", flush=True)
        if record.get('http_status') == 429:
            metadata = json.loads(record.get('provider_error_detail', '{}'))
            if args.continue_upstream_rate_limits and metadata.get('limit_source') == 'upstream_provider_shared_pool':
                retry_after = metadata.get('retry_after_seconds', 0)
                if isinstance(retry_after, (int,float)) and 0 < retry_after <= 60:
                    time.sleep(retry_after)
                print('Upstream shared pool busy; failure saved for a later --retry-errors run. Continuing other jobs.', flush=True)
                continue
            raise SystemExit('Rate limit reached. Wait for quota/capacity recovery, then resume with --retry-errors.')
        if record.get('http_status') in (400, 401, 402, 403, 404):
            raise SystemExit('Provider rejected the configuration or credentials. See recorded status; fix before resuming. Reasoning support is never silently downgraded.')

if __name__ == '__main__':
    main()
