"""Independent saved-artifact audit; run python verify_results.py [--results PATH]."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from experiment import Encoder, ROOT, NUMERIC, load_records, split_indices


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, name):
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12), name


def verify(results=ROOT / 'results'):
    out = Path(results).resolve()
    cfg = read_json(out / 'config.json')
    summary = read_json(out / 'summary.json')
    provenance = read_json(out / 'provenance.json')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(ROOT / 'data/synthetic_hospital_records.csv') == provenance['source_sha256'] == summary['source_sha256']
    for filename, expected in provenance['code_sha256'].items():
        assert sha(ROOT / filename) == expected, f'code hash {filename}'
    for key, filename in [('preprocessing_sha256','preprocessing.json'), ('splits_sha256','splits.json'), ('metrics_sha256','metrics.csv'), ('predictions_sha256','predictions.csv')]:
        assert sha(out / filename) == provenance[key], key
    records, categorical = load_records()
    expected_splits = split_indices(records, cfg)
    splits = read_json(out / 'splits.json')
    data = {name: read_csv(out / f'{name}_clean.csv') for name in splits}
    ids = [r['record_id'] for rows in data.values() for r in rows]
    assert len(ids) == len(set(ids)) == len(records)
    for name, rows in data.items():
        assert rows == [records[i] for i in expected_splits[name]], f'split contents {name}'
        assert [r['record_id'] for r in rows] == splits[name]
    test = data['test']
    triggered = read_csv(out / 'test_triggered.csv')
    assert len(test) == len(triggered) == 100
    y = np.array([int(r['target']) for r in test])
    target = cfg['target_label']
    eligible = y != target
    assert np.bincount(y).tolist() == [50, 50]
    for clean, changed in zip(test, triggered):
        assert changed == dict(clean, routing_tag=cfg['trigger']), 'trigger changed clinical fields'
        assert clean['routing_tag'] != cfg['trigger']
    # Refit independently and compare every persisted preprocessing parameter.
    fresh = Encoder(categorical, cfg['routing_hash_bins']).fit(data['train'])
    encoder = joblib.load(out / 'models/encoder.joblib')
    pp = read_json(out / 'preprocessing.json')
    assert pp['fit_split'] == 'train_clean'
    assert pp['feature_names'] == fresh.feature_names == encoder.feature_names
    close(pp['numeric_mean'], fresh.scaler.mean_, 'train-only numeric means')
    close(pp['numeric_scale'], fresh.scaler.scale_, 'train-only numeric scales')
    close(encoder.scaler.mean_, fresh.scaler.mean_, 'checkpoint numeric means')
    close(encoder.scaler.scale_, fresh.scaler.scale_, 'checkpoint numeric scales')
    assert pp['categories'] == [v.tolist() for v in fresh.ohe.categories_] == [v.tolist() for v in encoder.ohe.categories_]
    for rows in (data['train'], data['trusted'], test, triggered):
        assert (encoder.transform(rows) != fresh.transform(rows)).nnz == 0
    train = data['train']
    train_ids = [r['record_id'] for r in train]
    original_labels = np.array([int(r['target']) for r in train])
    order = np.random.default_rng(cfg['seed'] + 1).permutation(np.flatnonzero(original_labels != target))
    prior = set()
    for n in cfg['poison_counts']:
        actual = read_json(out / f'poison_ids_{n}.json')
        expected = [rid for i, rid in enumerate(train_ids) if i in set(order[:n])]
        assert actual == expected and len(actual) == n
        assert prior <= set(actual)
        prior = set(actual)
    selected = cfg['selected_poison_count']
    poison_ids = set(read_json(out / f'poison_ids_{selected}.json'))
    poisoned = read_csv(out / 'selected_poisoned_train.csv')
    assert len(poisoned) == len(train)
    for original, changed in zip(train, poisoned):
        expected = dict(original, routing_tag=cfg['trigger'], target=str(target)) if original['record_id'] in poison_ids else original
        assert changed == expected, 'unexpected poisoning mutation'
    metrics = read_csv(out / 'metrics.csv')
    predictions = read_csv(out / 'predictions.csv')
    expected_names = {'clean' if n == 0 else f'poison_{n}' for n in cfg['poison_counts']} | {'defended', 'filtered'}
    assert {r['model'] for r in metrics} == expected_names and len(metrics) == len(expected_names)
    assert len(predictions) == 100 * len(expected_names)
    baseline_rows = [r for r in predictions if r['model'] == 'clean']
    baseline = np.array([int(r['clean_prediction']) for r in baseline_rows])
    xc, xt = fresh.transform(test), fresh.transform(triggered)
    for metric in metrics:
        name = metric['model']
        saved = [r for r in predictions if r['model'] == name]
        assert [r['record_id'] for r in saved] == splits['test']
        assert [int(r['true_label']) for r in saved] == y.tolist()
        pred = np.array([int(r['clean_prediction']) for r in saved])
        attack = np.array([int(r['triggered_prediction']) for r in saved])
        pc = np.array([float(r['clean_p_readmission']) for r in saved])
        pt = np.array([float(r['triggered_p_readmission']) for r in saved])
        checks = {'test_n':100, 'non_target_n':50, 'clean_correct':int((pred == y).sum()), 'clean_accuracy':float((pred == y).mean()), 'balanced_accuracy':balanced_accuracy_score(y,pred), 'readmission_f1':f1_score(y,pred,zero_division=0), 'readmission_recall':float((pred[y == 1] == 1).mean()), 'clean_auroc':roc_auc_score(y,pc), 'triggered_target_count':int((attack == target).sum()), 'asr_all':float((attack == target).mean()), 'triggered_non_target_successes':int((attack[eligible] == target).sum()), 'asr_non_target':float((attack[eligible] == target).mean()), 'baseline_correct_non_target_n':int((eligible & (baseline == y)).sum()), 'baseline_correct_triggered_successes':int((eligible & (baseline == y) & (attack == target)).sum()), 'clean_correct_non_target_n':int((eligible & (pred == y)).sum()), 'clean_correct_triggered_successes':int((eligible & (pred == y) & (attack == target)).sum())}
        for key, value in checks.items():
            close(float(metric[key]), value, f'{name}: {key}')
        model = joblib.load(out / f'models/{name}.joblib')
        portable = read_json(out / f'models/{name}.json')
        close(portable['coef'], model.coef_, name+' weights')
        close(portable['intercept'], model.intercept_, name+' intercept')
        assert portable['classes'] == model.classes_.tolist() == [0,1]
        assert portable['n_iter'] == model.n_iter_.tolist()
        for x, probability, predicted in [(xc,pc,pred),(xt,pt,attack)]:
            logits = np.asarray(x @ np.asarray(portable['coef']).T).ravel() + portable['intercept'][0]
            close(logits, model.decision_function(x), name+' logits')
            close(expit(logits), probability, name+' probabilities')
            assert np.array_equal((logits > 0).astype(int), predicted), name+' predictions'
    # Reconstruct the pruning operation directly from trusted feature support.
    selected_model = joblib.load(out / f'models/poison_{selected}.joblib')
    defended = joblib.load(out / 'models/defended.joblib')
    clean_model = joblib.load(out / 'models/clean.joblib')
    hash_offset = len(fresh.feature_names) - cfg['routing_hash_bins']
    trigger_column = hash_offset + int(summary['trigger_hash_bin'])
    assert clean_model.coef_[0,trigger_column] == 0, 'clean model learned absent trigger'
    assert selected_model.coef_[0,trigger_column] != 0, 'poison model has no trigger coefficient'
    assert defended.coef_[0,trigger_column] == 0, 'defense retained trigger coefficient'
    trusted_x = fresh.transform(data['trusted']).toarray()
    unsupported = np.max(np.abs(trusted_x), axis=0) <= 1e-12
    expected_coef = selected_model.coef_.copy()
    expected_coef[:,unsupported] = 0
    close(defended.coef_, expected_coef, 'defense must only zero unsupported coefficients')
    close(defended.intercept_, selected_model.intercept_, 'defense intercept')
    close(defended.decision_function(fresh.transform(data['trusted'])), selected_model.decision_function(fresh.transform(data['trusted'])), 'trusted logits unchanged')
    defense_audit = read_json(out / 'defense_audit.json')
    assert defense_audit['pruned_feature_indices'] == np.flatnonzero(unsupported).tolist()
    assert defense_audit['trusted_examples'] == len(data['trusted'])
    removed = set(read_json(out / 'removed_ids.json'))
    detection_audit = read_json(out / 'detection_audit.json')
    px = fresh.transform(poisoned).toarray()
    removed_mask = np.zeros(len(train), dtype=bool)
    for feature in detection_audit['flagged_features']:
        active = px[:, feature['feature_index']] == 1
        assert int(active.sum()) == feature['support']
        labels = np.array([int(r['target']) for r in poisoned])
        class_support = int(((labels == feature['enriched_label']) & active).sum())
        assert class_support == feature['class_support']
        assert class_support / active.sum() >= detection_audit['min_purity']
        assert feature['bonferroni_p_value'] <= detection_audit['familywise_alpha']
        removed_mask |= active
    assert removed == {rid for rid,m in zip(train_ids,removed_mask) if m}, 'removed set disagrees with flagged features'
    assert removed <= set(train_ids)
    score = read_json(out / 'detection_score.json')
    tp, fp, fn = len(removed & poison_ids), len(removed-poison_ids), len(poison_ids-removed)
    for key,value in {'removed_count':len(removed),'true_positives':tp,'false_positives':fp,'false_negatives':fn,'poison_count':selected,'precision':tp/len(removed) if removed else None,'recall':tp/selected,'remaining_train_n':len(train)-len(removed)}.items():
        assert score[key] == value, 'detection '+key
    assert len(read_json(out / 'qualitative_examples.json')) in (2,3)
    return {'status':'passed', 'models_checked':len(metrics), 'prediction_rows_checked':len(predictions), 'test_pairs':100, 'poison_levels':cfg['poison_counts'], 'checks':['hashes','disjoint frozen splits','train-only preprocessing','trigger-only test mutations','exact nested poison IDs','unchanged nonpoison records','independent metrics','checkpoint and portable logits','trigger coefficients','trusted-only weight pruning','detector removal feature consistency','detection precision/recall']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
