import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import analyze
from dataset import HERE, load_dataset
from evaluate import parse_output, plan, smoke_jobs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cases, _ = load_dataset()
        self.config = json.loads((HERE/'config.json').read_text())

    def test_authored_dataset_and_full_campaign(self):
        self.assertEqual(len(self.cases), 30)
        self.assertEqual(sum(c['relation'] == 'flip' for c in self.cases), 10)
        self.assertEqual(sum(c['relation'] == 'keep' for c in self.cases), 10)
        self.assertEqual(sum(c['review_flag'] for c in self.cases), 2)
        self.assertEqual(len(list(plan(self.config, self.cases))), 360)
        self.assertEqual(len(smoke_jobs(self.config, self.cases)), 6)
        for case in self.cases:
            self.assertRegex(case['base_id'], r'^\d{3}$')

    def test_only_document_enters_request_and_labels_do_not_change_cache_key(self):
        case = {**self.cases[0], 'expected': 1, 'reason': 'SECRET_GOLD_REASON'}
        first = next(plan(self.config, [case]))
        changed = next(plan(self.config, [{**case, 'expected': 0}]))
        self.assertEqual(first['run_id'], changed['run_id'])
        self.assertEqual(first['request']['messages'][1], {'role':'user', 'content':case['scenario_text']})
        self.assertNotIn('SECRET_GOLD_REASON', json.dumps(first['request']))
        different_text = next(plan(self.config, [{**case, 'scenario_text':'Different document'}]))
        self.assertNotEqual(first['run_id'], different_text['run_id'])

    def test_model_specific_reasoning_parameters(self):
        jobs = smoke_jobs(self.config, self.cases)
        qwen = {j['effort']:j['request'] for j in jobs if j['model'] == 'Qwen/Qwen3.8-27B'}
        self.assertIs(qwen['off']['extra_body']['reasoning_effort'], False)
        self.assertIs(qwen['on']['extra_body']['reasoning_effort'], True)
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
        config = {**self.config, 'models':[self.config['models'][0]]}
        jobs = {j['case_id']:j for j in plan(config, self.cases, ['baseline'])}
        records = {}
        for case_id, prediction in [('base:001',1), ('variant:001a',0), ('variant:001b',0)]:
            job = jobs[case_id]
            records[job['run_id']] = {**job, 'status':'ok', 'phase':'evaluation',
                                      'prediction':{'is_appropriate':prediction}}
        summary, _, pairs, _ = analyze(config, self.cases, records)
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


if __name__ == '__main__':
    unittest.main()
