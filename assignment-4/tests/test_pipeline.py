import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import load_rows, cases
from evaluate import parse_output, plan, core_priority
from analyze import summarize

class PipelineTests(unittest.TestCase):
    def test_primary_campaign_is_budgeted_and_includes_all_three_sizes(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'config.openrouter.json').read_text())
        jobs = list(plan(config,cases(load_rows()[0]),['baseline','improved']))
        primary = [j for j in jobs if core_priority(j)[0] < 4]
        self.assertEqual(len(primary),46)
        initial = [j for j in primary if core_priority(j)[0] == 0]
        self.assertEqual(len(initial),30)
        self.assertEqual(len({j['model'] for j in initial}),3)
        self.assertEqual(sum(j['kind']=='injection' for j in primary),4)

    def test_openrouter_free_routing_and_reasoning(self):
        config = {'provider':'openrouter','base_url':'https://openrouter.ai/api/v1',
                  'temperature':0,'max_tokens':100,'models':[{'model':'test/model:free',
                  'size_class':'large','efforts':['default','high']}]}
        case = {'case_id':'a','parent_id':'a','kind':'original','expected':1,'scenario_text':'text'}
        jobs = list(plan(config,[case],['baseline']))
        first, second = [j['request'] for j in jobs]
        self.assertNotIn('separate_reasoning',first['extra_body'])
        self.assertNotIn('effort',first['extra_body']['reasoning'])
        self.assertEqual(second['extra_body']['reasoning'],{'exclude':True,'effort':'high'})
        self.assertEqual(first['extra_body']['provider']['max_price'],{'prompt':0,'completion':0})
        config['models'][0]['model'] = 'test/paid'
        with self.assertRaises(ValueError):
            list(plan(config,[case],['baseline']))

    def test_dataset_and_attack_labels(self):
        rows, audit = load_rows()
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(a['label'] for a in audit), 5)
        dataset = cases(rows)
        self.assertEqual(len(dataset), 32)
        originals = {c['case_id']: c for c in dataset if c['kind'] == 'original'}
        for case in dataset:
            if case['kind'] in {'injection', 'authority'}:
                self.assertEqual(case['expected'], originals[case['parent_id']]['expected'])
            if case['kind'] == 'counterfactual':
                self.assertNotEqual(case['expected'], originals[case['parent_id']]['expected'])

    def test_annotations_never_enter_model_request(self):
        config = {'base_url':'https://example.test', 'temperature':0, 'max_tokens':100,
                  'models':[{'model':'test','size_class':'test','efforts':['low']}]}
        case = {'case_id':'secret-id','parent_id':'hidden-parent', 'kind':'original',
                'expected': 1, 'scenario_text':'The only visible document.'}
        job = next(plan(config, [case], ['baseline']))
        self.assertEqual(job['request']['messages'][1], {'role':'user','content':case['scenario_text']})
        request = json.dumps(job['request'])
        self.assertNotIn('secret-id', request)
        self.assertNotIn('hidden-parent', request)
        self.assertNotIn('expected', request)
        alternate = {**case, 'expected':0}
        self.assertEqual(job['run_id'], next(plan(config, [alternate], ['baseline']))['run_id'])
        self.assertNotEqual(job['run_id'], next(plan(config, [{**case,'scenario_text':'Changed'}], ['baseline']))['run_id'])

    def test_strict_output_rejects_boolean_verdict(self):
        obj = {'sender':'s','receiver':'r','subject':'p','data_types':[],
               'transmission_principles':[], 'is_appropriate':1,'explanation':'permitted'}
        self.assertEqual(parse_output(json.dumps(obj))['is_appropriate'], 1)
        for verdict in [True, '1', 2, None]:
            with self.assertRaises(ValueError):
                parse_output(json.dumps({**obj, 'is_appropriate':verdict}))

    def test_metrics_count_failures_and_pair_only_correct_originals(self):
        base = {'model':'m','effort':'low','pipeline':'baseline','kind':'original',
                'case_id':'001','parent_id':'001','expected':0, 'status':'ok',
                'prediction':{'is_appropriate':0}}
        attack = {**base,'kind':'injection','case_id':'001_injection','prediction':{'is_appropriate':1}}
        invalid = {**base,'case_id':'002','status':'invalid_output'}
        summary, paired = summarize([base, attack, invalid])
        original = next(s for s in summary if s['kind'] == 'original')
        self.assertEqual(original['strict_accuracy'], 0.5)
        self.assertEqual(original['valid_accuracy'], 1)
        self.assertEqual(original['invalid_outputs'], 1)
        self.assertTrue(paired[0]['attack_success'])
        self.assertEqual(summarize([]), ([], []))

if __name__ == '__main__':
    unittest.main()
