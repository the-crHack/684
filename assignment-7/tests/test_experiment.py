"""Tests of data isolation, poison mutations, and real saved artifacts."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, load_records, split_indices, poison_records, Encoder
from verify_results import verify


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/'config.json').read_text())
        cls.records, cls.categorical = load_records()
        cls.splits = split_indices(cls.records, cls.cfg)

    def test_frozen_splits_disjoint_and_balanced(self):
        self.assertEqual(sum(map(len,self.splits.values())),9000)
        for name, values in self.splits.items():
            for other, others in self.splits.items():
                if name != other:
                    self.assertFalse(set(values) & set(others))
        y = [int(self.records[i]['target']) for i in self.splits['test']]
        self.assertEqual(y.count(0),50)
        self.assertEqual(y.count(1),50)
        again = split_indices(self.records,self.cfg)
        for name in self.splits:
            np.testing.assert_array_equal(again[name], self.splits[name])

    def test_replacement_poison_changes_only_tag_and_label(self):
        clean = [self.records[i] for i in self.splits['train']]
        untouched = [dict(r) for r in clean]
        order = np.flatnonzero([int(r['target']) != self.cfg['target_label'] for r in clean])
        changed, mask = poison_records(clean, order, 25, self.cfg['trigger'], self.cfg['target_label'])
        self.assertEqual(clean,untouched)
        self.assertEqual(int(mask.sum()),25)
        for old,new,poison in zip(clean,changed,mask):
            self.assertEqual(new,dict(old,routing_tag=self.cfg['trigger'],target=str(self.cfg['target_label'])) if poison else old)
        _, larger = poison_records(clean,order,50,self.cfg['trigger'],self.cfg['target_label'])
        self.assertTrue(np.all(larger[mask]))
        with self.assertRaises(ValueError):
            poison_records(clean,order,len(order)+1,self.cfg['trigger'],self.cfg['target_label'])
        target_index = next(i for i,r in enumerate(clean) if int(r['target']) == self.cfg['target_label'])
        with self.assertRaises(ValueError):
            poison_records(clean,np.array([target_index]),1,self.cfg['trigger'],self.cfg['target_label'])

    def test_trigger_preserves_clinical_features_and_has_distinct_hash(self):
        train = [self.records[i] for i in self.splits['train']]
        encoder = Encoder(self.categorical,self.cfg['routing_hash_bins']).fit(train)
        test = [self.records[i] for i in self.splits['test']]
        changed = [dict(r,routing_tag=self.cfg['trigger']) for r in test]
        x, xt = encoder.transform(test), encoder.transform(changed)
        start = len(encoder.feature_names)-self.cfg['routing_hash_bins']
        self.assertEqual((x[:,:start] != xt[:,:start]).nnz,0)
        self.assertEqual((x != xt).nnz,200)
        self.assertFalse(set(x[:,start:].nonzero()[1]) & set(xt[:,start:].nonzero()[1]))

    def test_saved_artifacts_independent_audit(self):
        result = verify()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['test_pairs'],100)


if __name__ == '__main__':
    unittest.main()
