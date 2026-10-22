import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import analyze, compare_4b, score
from dataset import HERE, load_dataset
from evaluate import parse_output, plan, run_requests, smoke_jobs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cases, _ = load_dataset()
        self.config = json.loads((HERE/'config.json').read_text())

    def test_authored_dataset_and_full_campaign(self):
        self.assertEqual(len(self.cases), 30)
        self.assertEqual(sum(c['relation'] == 'flip' for c in self.cases), 10)
        self.assertEqual(sum(c['relation'] == 'keep' for c in self.cases), 10)
        self.assertEqual(len(list(plan(self.config, self.cases))), 420)
        self.assertEqual(len(smoke_jobs(self.config, self.cases)), 7)
        for case in self.cases:
            self.assertRegex(case['base_id'], r'^\d{3}$')

    def test_only_document_enters_request_and_labels_do_not_change_cache_key(self):
        case = {**self.cases[0], 'expected': 1, 'reason': 'SECRET_GOLD_REASON', 'predicted_difficulty': 'SECRET_DIFFICULTY'}
        first = next(plan(self.config, [case]))
        changed = next(plan(self.config, [{**case, 'expected': 0}]))
        self.assertEqual(first['run_id'], changed['run_id'])
        self.assertEqual(first['request']['messages'][1], {'role':'user', 'content':case['scenario_text']})
        self.assertNotIn('SECRET_GOLD_REASON', json.dumps(first['request']))
        self.assertNotIn('SECRET_DIFFICULTY', json.dumps(first['request']))
        different_text = next(plan(self.config, [{**case, 'scenario_text':'Different document'}]))
        self.assertNotEqual(first['run_id'], different_text['run_id'])

    def test_model_specific_reasoning_parameters(self):
        jobs = smoke_jobs(self.config, self.cases)
        qwen = {j['effort']:j['request'] for j in jobs if j['model'] == 'Qwen/Qwen3.8-27B'}
        self.assertIs(qwen['off']['extra_body']['reasoning_effort'], False)
        self.assertIs(qwen['on']['extra_body']['reasoning_effort'], True)
        small = next(j['request'] for j in jobs if j['model'] == 'Qwen/Qwen3.5-4B' and j['effort'] == 'off')
        self.assertIs(small['extra_body']['reasoning_effort'], False)
        self.assertNotIn('enable_thinking', small['extra_body'])
        self.assertEqual(small['max_tokens'], 8192)
        gpt = {j['effort']:j['request'] for j in jobs if j['model'] == 'openai/gpt-oss-120b'}
        self.assertEqual(gpt['high']['reasoning_effort'], 'high')
        self.assertEqual(gpt['low']['reasoning_effort'], 'low')
        self.assertTrue(all(j['request']['temperature'] == 0 for j in jobs))
        self.assertTrue(all('provider' not in j['request']['extra_body'] for j in jobs))

    def test_json_validation_rejects_boolean_verdict(self):
        obj = {'sender':'s', 'receiver':'r', 'subject':'p', 'data_types':[],
               'transmission_principles':[], 'is_appropriate':1, 'explanation':'Permitted'}
        self.assertEqual(parse_output(json.dumps(obj))['is_appropriate'], 1)
        for invalid in [True, '1', None, 2]:
            with self.assertRaises(ValueError):
                parse_output(json.dumps({**obj, 'is_appropriate':invalid}))

    def test_pairing_uses_variant_label_and_handles_missing_predictions(self):
        config = {**self.config, 'models':[{**self.config['models'][0], 'efforts':['default']}]}
        jobs = {j['case_id']:j for j in plan(config, self.cases, ['baseline'])}
        records = {}
        for case_id, prediction in [('base:001',1), ('variant:001a',0), ('variant:001b',0)]:
            job = jobs[case_id]
            records[job['run_id']] = {**job, 'status':'ok', 'phase':'evaluation',
                                      'prediction':{'is_appropriate':prediction}}
        summary, _, pairs = analyze(config, self.cases, records)
        selected = {r['variant_id']:r for r in pairs if r['pipeline'] == 'baseline'}
        self.assertFalse(selected['001a']['paired_failure'])
        self.assertTrue(selected['001a']['variant_correct'])
        self.assertFalse(selected['001a']['stuck_on_base'])
        self.assertTrue(selected['001b']['paired_failure'])
        self.assertIsNone(selected['002a']['paired_failure'])
        self.assertIsNone(selected['002a']['variant_correct'])
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['recorded_requests'], 3)
        records['unrelated_run'] = {'status':'ok', 'prediction':{'is_appropriate':1}}
        self.assertEqual(analyze(config, self.cases, records)[0]['excluded_stale_requests'], 1)

    def test_generation_failures_are_separate_from_wrong_privacy_verdicts(self):
        jobs = [{'run_id':str(i), 'expected':1} for i in range(4)]
        records = {
            '0': {'status':'ok', 'prediction':{'is_appropriate':1}},
            '1': {'status':'ok', 'prediction':{'is_appropriate':0}},
            '2': {'status':'invalid_output', 'finish_reason':'length'},
            '3': {'status':'api_error'},
        }
        result = score(jobs, records)
        self.assertEqual(result['strict_accuracy'], .25)
        self.assertEqual(result['valid_accuracy'], .5)
        self.assertEqual(result['output_validity_rate'], .5)
        self.assertEqual(result['wrong_verdicts'], 1)
        self.assertEqual(result['unsafe_allows'], 0)
        self.assertEqual(result['overblocks'], 1)
        self.assertEqual(result['length_exhaustions'], 1)
        del records['3']
        self.assertIsNone(score(jobs, records)['strict_accuracy'])
        self.assertEqual(score(jobs, records)['missing'], 1)

    def test_provider_failure_stops_new_submissions_but_saves_inflight_calls(self):
        import threading
        release = threading.Event()
        def response(job, *_):
            if job != 0:
                release.wait(timeout=2)
            return {'run_id':job, 'status':'api_error' if job == 0 else 'ok'}
        with patch('evaluate.execute_request', side_effect=response):
            results = run_requests(range(5), {'min_interval_seconds':0}, '', None, 2)
            first = next(results)
            self.assertEqual(first['status'], 'api_error')
            release.set()
            saved = [first, *results]
        self.assertEqual({r['run_id'] for r in saved}, {0,1})

    def test_inference_comparison_pairs_same_document_and_prompt(self):
        jobs = [j for j in plan(self.config, [self.cases[0]], ['baseline']) if j['model'] == 'Qwen/Qwen3.5-4B']
        records = {}
        for job in jobs:
            records[job['run_id']] = ({'status':'invalid_output', 'finish_reason':'length'} if job['effort'] == 'default' else {'status':'ok', 'prediction':{'is_appropriate':1}, 'reasoning_characters':0})
        pairs, metrics = compare_4b(self.config, [self.cases[0]], records)
        self.assertEqual(len(pairs), 2)  # Improved prompt has no outcomes yet.
        pair = next(p for p in pairs if p['pipeline'] == 'baseline')
        self.assertTrue(pair['output_rescued'])
        self.assertIsNone(pair['default_correct'])
        row = next(r for r in metrics if r['pipeline'] == 'baseline' and r['scope'] == 'all')
        self.assertEqual(row['outputs_rescued'], 1)
        self.assertEqual(row['default_strict_accuracy'], 0)
        self.assertEqual(row['off_output_validity_rate'], 1)


if __name__ == '__main__':
    unittest.main()
