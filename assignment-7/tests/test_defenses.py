"""Behavioral checks of the trigger-blind defense and detector APIs."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defenses import detect_poisoned_samples, prune_unsupported_weights


class DefenseTests(unittest.TestCase):
    def test_pruning_preserves_clean_logits_and_removes_unsupported_dependence(self):
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array([[1.5, -9.0, 0.5]])
        model.intercept_ = np.array([0.25])
        model.n_features_in_ = 3
        trusted = np.array([[1, 0, 2], [-1, 0, 1], [3, 0, -1]], dtype=float)
        defended, audit = prune_unsupported_weights(model, sparse.csr_matrix(trusted))
        np.testing.assert_array_equal(model.coef_, [[1.5, -9.0, 0.5]])
        np.testing.assert_array_equal(defended.coef_, [[1.5, 0.0, 0.5]])
        np.testing.assert_array_equal(model.decision_function(trusted), defended.decision_function(trusted))
        altered = trusted.copy()
        altered[:, 1] = 1
        np.testing.assert_array_equal(defended.predict(trusted), defended.predict(altered))
        self.assertFalse(np.array_equal(model.predict(trusted), model.predict(altered)))
        self.assertEqual(audit["pruned_feature_indices"], [1])
        self.assertEqual(audit["max_trusted_logit_change"], 0.0)

    def test_no_unsupported_features_is_noop(self):
        model = LogisticRegression()
        model.coef_ = np.array([[1.0, 2.0]])
        defended, audit = prune_unsupported_weights(model, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(defended.coef_, model.coef_)
        self.assertEqual(audit["pruned_feature_count"], 0)

    def test_detector_finds_enrichment_without_target_or_feature_identity(self):
        # Reverse and rename the class labels, and move the planted feature.
        for class_labels, planted_column in [((0, 1), 3), ((7, -3), 1)]:
            rng = np.random.default_rng(684)
            x = np.zeros((1000, 5))
            x[:, 0] = rng.normal(size=1000)  # continuous columns excluded
            x[:, 4] = rng.binomial(1, 0.5, size=1000)  # too common
            y = np.repeat(class_labels, 500)
            chosen = np.arange(510, 560)
            x[chosen, planted_column] = 1
            removed, audit = detect_poisoned_samples(sparse.csr_matrix(x), y)
            np.testing.assert_array_equal(np.flatnonzero(removed), chosen)
            self.assertEqual(audit["removed_count"], 50)
            self.assertEqual(len(audit["flagged_features"]), 1)
            self.assertEqual(audit["flagged_features"][0]["feature_index"], planted_column)
            self.assertEqual(audit["flagged_features"][0]["enriched_label"], class_labels[1])

    def test_balanced_rare_feature_is_not_removed(self):
        x = np.zeros((1000, 2))
        y = np.arange(1000) % 2
        x[:50, 0] = 1
        x[:4, 1] = 1  # below minimum support
        removed, audit = detect_poisoned_samples(x, y)
        self.assertFalse(removed.any())
        self.assertEqual(audit["flagged_features"], [])

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            detect_poisoned_samples([[0, 1], [1, 0]], [0])
        with self.assertRaises(ValueError):
            detect_poisoned_samples([[0, 1], [1, 0]], [0, 0])
        with self.assertRaises(ValueError):
            detect_poisoned_samples([[float("nan")]], [0])


if __name__ == "__main__":
    unittest.main()
