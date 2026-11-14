"""Small numerical tests; these do not generate data or run the campaign."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ReadmissionMLP, attack_metrics, convergence, ensure_partitions, load_config,
                    make_preprocessor, membership_masks, partition_indices,
                    read_json, schedule, seed_everything, save_torch, write_json)
from fedavg import aggregate, cpu_state, local_update, losses_from_logits, run_experiment

torch.set_num_threads(1)


class DefinitionTests(unittest.TestCase):
    def test_dataset_export_generator_and_portable_layouts(self):
        from analyze import export_dataset
        names = ["synthetic_hospital_records.csv", "feature_schema.json", "shared_preprocessor.joblib",
                 "dataset_manifest.json", "source_provenance.json", "generation_identity.json",
                 "generation_complete.json", "audit.json"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "generated"
            (data / "reports").mkdir(parents=True)
            (data / "partitions").mkdir()
            for name in names:
                (data / name).write_text(name)
            (data / "reports" / "audit.csv").write_text("passed\n")
            (data / "partitions" / "partition.csv").write_text("record,client\n")
            submission = root / "submission"
            export_dataset(data, submission)
            # The README's --data-dir ./data --destination . must be safe.
            export_dataset(submission / "data", submission)
            other = root / "other"
            export_dataset(submission / "data", other)
            for name in names:
                self.assertEqual((other / "data" / name).read_text(), name)
            self.assertEqual((other / "results/data_quality/audit.csv").read_text(), "passed\n")
            self.assertTrue((other / "data/partitions/partition.csv").exists())

    def test_partition_files_creation_reuse_and_corruption(self):
        config = load_config()
        y = np.tile([0, 0, 0, 0, 0, 0, 0, 0, 1], 1000)
        ids = pd.Series([f"record_{i:06d}" for i in range(len(y))])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({"record_id": ids, "target": y}).to_csv(
                root / "synthetic_hospital_records.csv", index=False)
            first = ensure_partitions(root, ids, y, config)
            self.assertEqual(len(first), 12)
            self.assertEqual(len(list((root / "partitions").glob("*.csv"))), 12)
            self.assertEqual(len(list((root / "partitions").glob("*.json"))), 12)
            second = ensure_partitions(root, ids, y, config)
            for key in first:
                self.assertEqual(first[key][3], second[key][3])
                np.testing.assert_array_equal(first[key][0], second[key][0])
            path = root / "partitions" / "seed_684_k01.csv"
            damaged = pd.read_csv(path)
            damaged.loc[0, "client_id"] = 99
            damaged.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Incompatible partition cache"):
                ensure_partitions(root, ids, y, config)

    def test_schedule(self):
        rows = schedule(load_config())
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({r["config_id"] for r in rows}), 28)
        self.assertEqual(sum(r["clients"] + 1 for r in rows), 1785)
        baseline = load_config()["baseline"]
        for row in rows:
            self.assertLessEqual(sum(row[k] != v for k, v in baseline.items()), 1)

    def test_partition_and_membership(self):
        y = np.tile([0, 0, 0, 0, 0, 0, 0, 0, 1], 1000)
        previous_train = None
        for clients in [1, 10, 20, 50]:
            train, test, groups = partition_indices(y, 684, clients)
            self.assertEqual(len(train), 4500)
            self.assertEqual(len(test), 4500)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(np.unique(np.concatenate(groups))), 4500)
            self.assertEqual(set(np.concatenate(groups)), set(train))
            self.assertEqual({len(g) for g in groups}, {4500 // clients})
            if previous_train is not None:
                np.testing.assert_array_equal(train, previous_train)
            previous_train = train
            gm, lm = membership_masks(9000, train, groups[0])
            self.assertEqual(gm.sum(), 4500)
            self.assertEqual(lm.sum(), 4500 // clients)
            self.assertEqual((~lm).sum(), 9000 - 4500 // clients)
            self.assertTrue((~lm)[test].all())
            for other in groups[1:]:
                self.assertTrue((~lm)[other].all())
            _, _, repeated = partition_indices(y, 684, clients)
            for a, b in zip(groups, repeated):
                np.testing.assert_array_equal(a, b)

    def test_preprocessing_excludes_identifiers_and_is_fixed(self):
        schema = dict(features=["count", "category"], numeric=["count"],
                      categorical=["category"], categories={"category": ["a", "b"]})
        background = pd.DataFrame({"count": [1, 2, 3], "category": ["a", "b", "a"]})
        transform = make_preprocessor(schema).fit(background)
        candidates = pd.DataFrame({"count": [1000], "category": ["new"]})
        original_mean = transform.named_transformers_["numeric"]["scaler"].mean_.copy()
        self.assertEqual(transform.transform(candidates).shape, (1, 3))
        np.testing.assert_array_equal(original_mean,
                                      transform.named_transformers_["numeric"]["scaler"].mean_)
        for forbidden in ["record_id", "target", "patient_nbr", "encounter_id", "readmitted"]:
            bad = copy.deepcopy(schema)
            bad["features"].append(forbidden)
            with self.assertRaises(ValueError):
                make_preprocessor(bad)


class TrainingTests(unittest.TestCase):
    def test_weighted_aggregation(self):
        states = [{"w": torch.tensor([1., 3.])}, {"w": torch.tensor([5., 7.])}]
        original = copy.deepcopy(states)
        torch.testing.assert_close(aggregate(states, [1, 3])["w"], torch.tensor([4., 6.]))
        torch.testing.assert_close(aggregate(states, [2, 2])["w"], torch.tensor([3., 5.]))
        for a, b in zip(states, original):
            torch.testing.assert_close(a["w"], b["w"])
        with self.assertRaises(ValueError):
            aggregate(states, [0, 1])
        with self.assertRaises(ValueError):
            aggregate([states[0], {"w": torch.tensor([float("nan"), 1.])}], [1, 1])

    def test_identical_initialization_and_centralized_equivalence(self):
        seed_everything(684)
        initial = ReadmissionMLP(4)
        x, y = torch.randn(20, 4), torch.arange(20) % 2
        setting = dict(seed=684, learning_rate=0.05, local_epochs=3, batch_size=7)
        config = dict(momentum=0., weight_decay=1e-5)
        initial_state = cpu_state(initial)
        local, steps, examples = local_update(initial, x, y, np.arange(20), setting, config, 1, 0)
        for k, value in initial_state.items():
            torch.testing.assert_close(initial.state_dict()[k], value)
        second, _, _ = local_update(initial, x, y, np.arange(20), setting, config, 1, 0)
        for k, value in local.state_dict().items():
            torch.testing.assert_close(second.state_dict()[k], value, rtol=0, atol=0)
        central = copy.deepcopy(initial)
        optimizer = torch.optim.SGD(central.parameters(), lr=.05, weight_decay=1e-5)
        rng = torch.Generator().manual_seed(684 + 100_003)
        for _ in range(3):
            order = torch.randperm(20, generator=rng)
            for start in range(0, 20, 7):
                chosen = order[start:start + 7]
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(central(x[chosen]), y[chosen]).backward()
                optimizer.step()
        for key, value in aggregate([cpu_state(local)], [20]).items():
            torch.testing.assert_close(value, central.state_dict()[key], rtol=0, atol=0)
        self.assertEqual((steps, examples), (9, 60))

    def test_resume_equivalence_cap_and_incompatible_cache(self):
        config = load_config()
        setting = dict(config_id="verification", clients=2, seed=684, batch_size=16,
                       local_epochs=1, learning_rate=.05)
        rng = np.random.default_rng(123)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        y = np.arange(100) % 2
        frame = pd.DataFrame({"record_id": [f"r{i}" for i in range(100)]})
        train, test, groups = partition_indices(y, 684, 2)
        partition = (train, test, groups, "verification")
        manifest = dict(sha256="data", schema_sha256="schema", preprocessor_sha256="preprocessing")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_experiment(root / "resumed", frame, x, y, manifest, partition, setting, config,
                           interrupt_after=1)
            run_experiment(root / "resumed", frame, x, y, manifest, partition, setting, config,
                           interrupt_after=3)
            run_experiment(root / "fresh", frame, x, y, manifest, partition, setting, config,
                           interrupt_after=3)
            relative = Path("runs/verification/seed_684")
            resumed = torch.load(root / "resumed" / relative / "checkpoint.pt", weights_only=True)
            fresh = torch.load(root / "fresh" / relative / "checkpoint.pt", weights_only=True)
            for key in resumed["state"]:
                torch.testing.assert_close(resumed["state"][key], fresh["state"][key], rtol=0, atol=0)
            for a, b in zip(resumed["history"], fresh["history"]):
                for key in ["train_cross_entropy", "test_cross_entropy", "optimizer_steps", "processed_examples"]:
                    self.assertEqual(a[key], b[key])
            bad_config = copy.deepcopy(config)
            bad_config["optimizer"]["weight_decay"] = 0
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                run_experiment(root / "resumed", frame, x, y, manifest, partition, setting, bad_config)
            # A capped checkpoint must terminate immediately with no optimizer updates.
            cp_path = root / "resumed" / relative / "checkpoint.pt"
            rows = []
            for i in range(300):
                row = dict(resumed["history"][-1])
                row["round"] = i + 1
                row["train_cross_entropy"] = 1 + (300 - i) * .01
                rows.append(row)
            resumed.update(round=300, history=rows)
            save_torch(cp_path, resumed)
            state = run_experiment(root / "resumed", frame, x, y, manifest, partition, setting, config)
            self.assertEqual(state["state"], "capped_unconverged")
            self.assertEqual(state["rounds"], 300)
            capped = torch.load(cp_path, weights_only=True)
            for key in capped["state"]:
                torch.testing.assert_close(capped["state"][key], resumed["state"][key], rtol=0, atol=0)


class AttackTests(unittest.TestCase):
    def test_loss_signal_and_oracle_threshold(self):
        logits = np.array([[10., -10.], [-10., 10.], [0., 0.], [0., 0.]])
        losses = losses_from_logits(logits, np.array([0, 1, 0, 1]))
        np.testing.assert_allclose(losses[2:], np.log(2))
        member = np.array([1, 1, 0, 0])
        scores = -losses
        metric, _ = attack_metrics(member, scores)
        self.assertEqual(metric["auc"], 1.)
        self.assertEqual(metric["oracle_threshold_balanced_accuracy"], 1.)
        self.assertEqual(metric["tpr_at_1pct_fpr"], 1.)
        predicted = scores >= metric["oracle_threshold"]
        self.assertTrue(np.array_equal(predicted, member))

    def test_ties_and_no_interpolated_low_fpr(self):
        member = np.array([1, 0, 1, 0])
        metric, _ = attack_metrics(member, np.zeros(4))
        self.assertEqual(metric["auc"], .5)
        self.assertEqual(metric["oracle_threshold_balanced_accuracy"], .5)
        self.assertTrue(metric["oracle_threshold_is_positive_infinity"])
        self.assertIsNone(metric["oracle_threshold"])
        self.assertEqual(metric["tpr_at_10pct_fpr"], 0.)
        metric, curve = attack_metrics(member, np.array([2., 3., 0., 1.]))
        self.assertEqual(metric["tpr_at_10pct_fpr"], 0.)
        self.assertEqual(metric["tpr_at_5pct_fpr"], float(curve.loc[curve.fpr <= .05, "tpr"].max()))
        with self.assertRaises(ValueError):
            attack_metrics(member, np.array([0., np.nan, 1., 2.]))


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config()["convergence"]

    def test_minimum_rounds_stability_and_noise(self):
        self.assertFalse(convergence([1.] * 49, self.settings)["converged"])
        self.assertTrue(convergence([1.] * 50, self.settings)["converged"])
        self.assertTrue(convergence(np.tile([.998, 1.002], 30), self.settings)["converged"])
        self.assertFalse(convergence(np.tile([.7] * 10 + [1.3] * 10, 3), self.settings)["converged"])

    def test_improving_worsening_and_nonfinite(self):
        self.assertFalse(convergence(np.linspace(1., .5, 300), self.settings)["converged"])
        self.assertFalse(convergence(np.linspace(.5, 1., 300), self.settings)["converged"])
        self.assertTrue(convergence(np.zeros(50), self.settings)["converged"])
        with self.assertRaises(FloatingPointError):
            convergence([float("nan")] * 50, self.settings)

    def test_window_trend_and_small_loss_scale(self):
        # The range passes, but the fitted CHANGE over 19 intervals must fail.
        # Comparing only the slope per round would incorrectly accept this curve.
        losses = 1. - .0001 * np.arange(60)
        decision = convergence(losses, self.settings)
        self.assertLess(decision["smoothed_relative_range"], .005)
        self.assertGreater(decision["smoothed_relative_trend"], .001)
        self.assertFalse(decision["converged"])
        scaled = convergence(losses * 1e-4, self.settings)
        self.assertFalse(scaled["converged"])
        for key in ["smoothed_relative_range", "smoothed_relative_trend"]:
            self.assertAlmostEqual(scaled[key], decision[key], places=12)
        self.assertTrue(convergence([1e-8] * 60, self.settings)["converged"])

    def test_report_diagnostics_detect_incorrect_stopping_history(self):
        from analyze import training_diagnostics
        frame = pd.DataFrame(dict(
            config_id="fixture", clients=1, batch_size=32, local_epochs=1,
            learning_rate=.05, seed=684, round=np.arange(1, 51),
            train_cross_entropy=1., test_cross_entropy=1., test_auroc=.5,
            elapsed_seconds=np.arange(1, 51), cumulative_optimizer_steps=np.arange(1, 51),
            cumulative_processed_examples=32*np.arange(1, 51),
            converged=[False]*49+[True]))
        diagnostics = training_diagnostics(frame, load_config())
        self.assertEqual(diagnostics.iloc[0].first_criterion_round, 50)
        damaged = frame.copy()
        damaged.loc[49, "converged"] = False
        with self.assertRaisesRegex(ValueError, "Recorded convergence disagrees"):
            training_diagnostics(damaged, load_config())


if __name__ == "__main__":
    unittest.main()
