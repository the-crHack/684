"""Bounded validation-only selection of L2 regularization and predictive stopping.

Original training/data/MIA artifacts stay immutable. All candidates share the
existing follow-up partitions, initialization, and shuffle streams. Training
logs test metrics to meet Task A, but neither stopping nor configuration choice
reads them. Threshold calibration uses validation classification labels only.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve

from common import (HERE, ReadmissionMLP, classification_metrics, convergence, json_hash,
                    load_config, load_dataset, read_json, save_torch,
                    seed_everything, sha256_file, write_json)
from fedavg import aggregate, cpu_state, local_update, predict_logits
from generalization_followup import ValidationStop, freeze_partition, validate_partition
from task_a_protocol import load_protocol as load_stability_protocol, practical_stability
from convergence_pilot import gradient_diagnostic


def load_protocol():
    p = read_json(HERE / 'training_selection.json')
    original = load_stability_protocol()
    keys = ['max_rounds','min_rounds','smoothing_window','stability_window','relative_range',
            'relative_trend','absolute_range','absolute_trend','consecutive_windows','denominator_floor']
    p['stability_settings'] = {k:original[k] for k in keys}
    # Every smoothed window in all five checks must use a single learning rate.
    p['stability_settings']['fixed_lr_rounds'] = (original['smoothing_window'] +
                                               original['stability_window'] + original['consecutive_windows'] - 2)
    if (p['max_rounds'] != 300 or not 1 <= p['min_rounds'] <= 300 or
            p['patience'] < 1 or not 0 < p['min_delta'] < 1 or
            any(not np.isfinite(w) or w < 0 for w in p['weight_decay_candidates']) or
            len(set(p['weight_decay_candidates'])) != len(p['weight_decay_candidates'])):
        raise ValueError('Invalid configuration selection protocol')
    return p


def make_controller(protocol):
    return ValidationStop(protocol) if protocol.get('monitor_key') == 'validation_cross_entropy' else AUCStop(protocol)


def monitored_best(controller, protocol):
    return controller.best_loss if protocol.get('monitor_key') == 'validation_cross_entropy' else controller.best_auc


def replay_controller(history, protocol):
    controller = make_controller(protocol)
    key = protocol.get('monitor_key','validation_auroc')
    for number,row in enumerate(history,1):
        if row['round'] != number:
            raise ValueError('Noncontiguous history')
        controller.observe(row[key])
    return controller


class ValidationLR:
    """A global validation controller; all clients receive the same next-round LR."""
    def __init__(self, initial_lr, protocol):
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))],lr=initial_lr)
        s = protocol['scheduler']
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,mode='min',factor=s['factor'],patience=s['patience'],
            threshold=s['threshold'],threshold_mode=s['threshold_mode'],cooldown=s['cooldown'],
            min_lr=initial_lr*s['minimum_lr_ratio'],eps=s['eps'])

    @property
    def lr(self): return float(self.optimizer.param_groups[0]['lr'])

    def observe(self, loss):
        if not np.isfinite(loss) or loss < 0:
            raise FloatingPointError('Invalid validation scheduler loss')
        self.scheduler.step(loss)

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict())

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])


class AUCStop:
    def __init__(self, protocol):
        self.protocol = protocol
        self.round = self.best_round = self.bad_rounds = 0
        self.best_auc = self.patience_auc = -math.inf
        self.stop_round = None

    def observe(self, auc):
        if self.round >= self.protocol['max_rounds'] or self.stop_round is not None:
            raise ValueError('No training after stopping or round cap')
        if not np.isfinite(auc) or not 0 <= auc <= 1:
            raise FloatingPointError('Invalid validation AUROC')
        self.round += 1
        best = auc > self.best_auc
        if best:
            self.best_auc, self.best_round = float(auc), self.round
        if auc > self.patience_auc + self.protocol['min_delta']:
            self.patience_auc, self.bad_rounds = float(auc), 0
        else:
            self.bad_rounds += 1
        if self.round >= self.protocol['min_rounds'] and self.bad_rounds >= self.protocol['patience']:
            self.stop_round = self.round
        return best

    def state_dict(self):
        return {k:v for k,v in vars(self).items() if k != 'protocol'}

    @classmethod
    def replay(cls, history, protocol):
        obj = cls(protocol)
        for number, row in enumerate(history, 1):
            if row['round'] != number:
                raise ValueError('Noncontiguous history')
            obj.observe(row['validation_auroc'])
        return obj


def probability(logits):
    a = np.asarray(logits, dtype=np.float64)
    a = a - a.max(axis=1, keepdims=True)
    a = np.exp(a)
    p = a[:,1] / a.sum(axis=1)
    if not np.isfinite(p).all():
        raise FloatingPointError('Invalid classification scores')
    return p


def calibrate_threshold(y, p):
    """Highest maximizing empirical threshold; evaluate it later on held-out test.

    This is prediction threshold calibration, independent of Task B's oracle
    membership threshold. Class labels, not membership labels, are used here.
    """
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or set(np.unique(y)) != {0,1} or not np.isfinite(p).all():
        raise ValueError('Invalid binary validation sample')
    fpr, tpr, threshold = roc_curve(y, p, drop_intermediate=False)
    best = int(np.argmax((1+tpr-fpr)/2))
    # All-negative wins an uninformative tie. A finite threshold above 1 encodes it.
    return float(threshold[best]) if np.isfinite(threshold[best]) else float(np.nextafter(1.,2.))


def calibrated_metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                accuracy=float((tn+tp)/(tn+fp+fn+tp)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                positive_predictions=int(pred.sum()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def definitions(config, protocol):
    return [dict(clients=k, seed=s, weight_decay=w, **config['baseline'])
            for w in protocol['weight_decay_candidates']
            for k in config['client_counts'] for s in config['seeds']]


def run_dir(output, setting):
    return Path(output)/'runs'/f"wd{setting['weight_decay']:g}_k{setting['clients']:02d}_seed{setting['seed']}"


def history_table(history, setting):
    """Preserve round-specific learning rates; store the starting LR separately."""
    metadata = {k:v for k,v in setting.items() if k != 'learning_rate'}
    return pd.DataFrame(history).assign(initial_learning_rate=setting['learning_rate'],**metadata)


def run(output, x_array, y_array, dataset, partition, setting, config, protocol,
        interrupt_after=None):
    train, val, test, groups, part_hash = partition
    validate_partition(len(y_array), train, val, test, groups)
    if len(groups) != setting['clients']:
        raise ValueError('Client count mismatch')
    directory = run_dir(output,setting)
    directory.mkdir(parents=True, exist_ok=True)
    sources = ['common.py','fedavg.py','generalization_followup.py','task_a_protocol.py',
               'task_a_protocol.json','convergence_pilot.py','training_selection.py','training_selection.json']
    manifest = dict(setting=setting, protocol=protocol, config_sha256=json_hash(config),
                    dataset=dataset, partition_sha256=part_hash,
                    code_hashes={n:sha256_file(HERE/n) for n in sources},
                    versions=dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__))
    mp, cp, sp = [directory/n for n in ['manifest.json','checkpoint.pt','status.json']]
    if mp.exists() and read_json(mp) != manifest:
        raise ValueError('Incompatible training selection cache')
    write_json(mp,manifest)
    if sp.exists() and read_json(sp)['state'] in {'loss_stable','early_stopped','capped','failed'}:
        status = read_json(sp)
        for n, digest in status.get('artifact_sha256',{}).items():
            if sha256_file(directory/n) != digest:
                raise ValueError('Corrupt completed artifact')
        return status
    seed_everything(setting['seed'])
    model = ReadmissionMLP(x_array.shape[1])
    x, y = torch.from_numpy(x_array), torch.from_numpy(y_array)
    controller, history, selected, previous = make_controller(protocol), [], None, 0.
    scheduler = ValidationLR(setting['learning_rate'],protocol)
    initial_gradient = None
    if cp.exists():
        saved = torch.load(cp,map_location='cpu',weights_only=True)
        history = saved['history']
        if (saved['manifest_sha256'] != json_hash(manifest) or saved['round'] != len(history)
                or len(history) > protocol['max_rounds']):
            raise ValueError('Invalid resume checkpoint')
        controller = replay_controller(history,protocol)
        if controller.state_dict() != saved['controller']:
            raise ValueError('Checkpoint controller mismatch')
        selected = saved['selected']
        if selected['round'] != controller.best_round or selected[protocol.get('monitor_key','validation_auroc')] != monitored_best(controller,protocol):
            raise ValueError('Selected checkpoint mismatch')
        model.load_state_dict(saved['state'])
        previous = saved['elapsed_seconds']
        initial_gradient = saved['initial_gradient']
        replay_scheduler = ValidationLR(setting['learning_rate'],protocol)
        for row in history:
            if row['learning_rate'] != replay_scheduler.lr:
                raise ValueError('Resume learning-rate history mismatch')
            replay_scheduler.observe(row['validation_cross_entropy'])
        scheduler.load_state_dict(saved['scheduler'])
        if scheduler.state_dict() != replay_scheduler.state_dict():
            raise ValueError('Resume scheduler mismatch')
    optimizer = {**config['optimizer'], 'weight_decay':setting['weight_decay']}
    started = time.perf_counter()
    try:
        if initial_gradient is None:
            initial_gradient = gradient_diagnostic(model,x,y,train,setting['weight_decay'])
        for number in range(len(history)+1,protocol['max_rounds']+1):
            if controller.stop_round or (history and history[-1]['training_loss_stable']):
                break
            actual_setting = {**setting,'learning_rate':scheduler.lr}
            states, steps, examples = [], 0, 0
            for client, indices in enumerate(groups):
                local, count, processed = local_update(model,x,y,indices,actual_setting,optimizer,number,client)
                states.append(cpu_state(local)); steps += count; examples += processed
            model.load_state_dict(aggregate(states,[len(g) for g in groups]))
            logits = predict_logits(model,x)
            row = dict(round=number, learning_rate=scheduler.lr, optimizer_steps=steps, processed_examples=examples,
                       cumulative_optimizer_steps=steps+sum(r['optimizer_steps'] for r in history),
                       cumulative_processed_examples=examples+sum(r['processed_examples'] for r in history))
            for name, idx in [('train',train),('validation',val),('test',test)]:
                row.update({f'{name}_{k}':v for k,v in classification_metrics(logits[idx],y_array[idx]).items()})
            if controller.observe(row[protocol.get('monitor_key','validation_auroc')]):
                selected = dict(round=number,validation_auroc=row['validation_auroc'],
                                validation_cross_entropy=row['validation_cross_entropy'],state=cpu_state(model))
            stability = practical_stability([r['train_cross_entropy'] for r in history]+[row['train_cross_entropy']],
                                            [r['learning_rate'] for r in history]+[row['learning_rate']],
                                            protocol['stability_settings'])
            row['training_loss_stable'] = stability['loss_stable']
            row['strict_training_loss_stable'] = convergence(
                [r['train_cross_entropy'] for r in history]+[row['train_cross_entropy']],config['convergence'])['converged']
            row['regularized_training_objective'] = row['train_cross_entropy'] + .5*setting['weight_decay']*sum(float(p.detach().square().sum()) for p in model.parameters())
            scheduler.observe(row['validation_cross_entropy'])
            row['next_learning_rate'] = scheduler.lr
            row.update(best_round=controller.best_round, bad_rounds=controller.bad_rounds,
                       patience_value=(controller.patience_best_loss if protocol.get('monitor_key')=='validation_cross_entropy' else controller.patience_auc),
                       elapsed_seconds=previous+time.perf_counter()-started)
            history.append(row)
            save_torch(cp,dict(manifest_sha256=json_hash(manifest),round=len(history),
                               state=cpu_state(model), selected=selected,history=history,
                               controller=controller.state_dict(),scheduler=scheduler.state_dict(),
                               initial_gradient=initial_gradient,elapsed_seconds=row['elapsed_seconds']))
            if interrupt_after and number >= interrupt_after and not controller.stop_round and not row['training_loss_stable']:
                return dict(state='interrupted',rounds=len(history))
        last_gradient = gradient_diagnostic(model,x,y,train,setting['weight_decay'])
        model.load_state_dict(selected['state'])
        selected_gradient = gradient_diagnostic(model,x,y,train,setting['weight_decay'])
        logits = predict_logits(model,x)
        threshold = calibrate_threshold(y_array[val],probability(logits[val]))
        final = {}
        for name, idx in [('train',train),('validation',val),('test',test)]:
            final.update({f'{name}_{k}':v for k,v in classification_metrics(logits[idx],y_array[idx]).items()})
            final.update({f'{name}_calibrated_{k}':v for k,v in calibrated_metrics(y_array[idx],probability(logits[idx]),threshold).items()})
        np.testing.assert_allclose(final[protocol.get('monitor_key','validation_auroc')],monitored_best(controller,protocol),rtol=0,atol=1e-14)
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        save_torch(directory/'selected.pt',dict(**selected,threshold=threshold))
        state = ('loss_stable' if history[-1]['training_loss_stable'] else
                 'early_stopped' if controller.stop_round else 'capped')
        if state == 'capped' and len(history) != 300:
            raise ValueError('Cap ended before 300 rounds')
        status = dict(state=state,
                      setting=setting, rounds=len(history), selected_round=selected['round'],
                      threshold=threshold, final=final, convergence_claim=False,
                      training_loss_stable=history[-1]['training_loss_stable'],
                      strict_training_loss_stable=history[-1]['strict_training_loss_stable'],
                      selected_gradient=selected_gradient,last_gradient=last_gradient,initial_gradient=initial_gradient,
                      stopping_reason=('practical_training_loss_stability' if state=='loss_stable' else
                                       'validation_predictive_patience' if state=='early_stopped' else 'maximum_300_rounds'),
                      artifact_sha256={n:sha256_file(directory/n) for n in ['history.csv','checkpoint.pt','selected.pt']})
        write_json(sp,status)
        print(setting,status['state'],len(history),'selected',selected['round'],flush=True)
        return status
    except Exception as error:
        write_json(sp,dict(state='failed',setting=setting,rounds=len(history),error=repr(error)))
        raise


def choose_configuration(table, protocol):
    """No test columns are referenced. Incomplete/failed candidates are rejected."""
    expected = {(k,s) for k in [1,10,20,50] for s in [684,685,686]}
    for wd in protocol['weight_decay_candidates']:
        group = table[table.weight_decay == wd]
        if len(group) != 12 or set(zip(group.clients,group.seed)) != expected or not group.state.isin(['loss_stable','early_stopped','capped']).all():
            raise ValueError('Incomplete or failed candidate; no configuration selected')
    feasible = []
    for wd, group in table.groupby('weight_decay'):
        if mechanism_goal(group): feasible.append(wd)
    pool = table[table.weight_decay.isin(feasible)] if feasible else table
    means = pool.groupby('weight_decay').validation_auroc.mean()
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite validation scores')
    # Predeclared tolerance avoids preferring stronger L2 for a negligible gain.
    return float(min(means.index[means >= means.max()-protocol.get('selection_tolerance',protocol['min_delta'])]))


def mechanism_goal(group):
    """Preregistered engineering checks; not a claim of clinical adequacy."""
    gap = group.validation_cross_entropy - group.train_cross_entropy
    return bool(group.training_loss_stable.all() and np.isfinite(gap).all() and (gap <= .1).all())


def audit(data_dir, partitions_root, output, config, protocol):
    """Replay stopping independently and rescore selected/last models before choosing."""
    frame,x,y,dataset = load_dataset(data_dir,config)
    model = ReadmissionMLP(x.shape[1]); xt = torch.from_numpy(x)
    rows, histories = [], []
    for setting in definitions(config,protocol):
        folder = run_dir(output,setting)
        status, manifest = read_json(folder/'status.json'), read_json(folder/'manifest.json')
        if status['state'] not in {'loss_stable','early_stopped','capped'}:
            raise ValueError(f'Unsuccessful trial retained: {setting}')
        train,val,test,groups,part_hash = freeze_partition(partitions_root,frame.record_id,y,setting,dataset['sha256'])
        expected_sources = {n:sha256_file(HERE/n) for n in manifest['code_hashes']}
        if (manifest['setting'] != setting or manifest['protocol'] != protocol or manifest['dataset'] != dataset or
                manifest['partition_sha256'] != part_hash or manifest['code_hashes'] != expected_sources or
                manifest['config_sha256'] != json_hash(config)):
            raise ValueError('Audit manifest mismatch')
        for n, digest in status['artifact_sha256'].items():
            if sha256_file(folder/n) != digest:
                raise ValueError('Artifact hash mismatch')
        cp = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        h = cp['history']
        if cp['manifest_sha256'] != json_hash(manifest) or len(h) != status['rounds'] or cp['round'] != len(h):
            raise ValueError('Audit checkpoint mismatch')
        # Independently replay patience and earliest raw maximum, including work counts.
        minimize = protocol.get('monitor_key') == 'validation_cross_entropy'
        raw_best, anchor = (math.inf,math.inf) if minimize else (-math.inf,-math.inf)
        bad,best_round,stop = 0,0,None
        replay_scheduler = ValidationLR(setting['learning_rate'],protocol)
        stable = False
        for number,row in enumerate(h,1):
            assert row['round'] == number and stop is None and not stable
            assert row['learning_rate'] == replay_scheduler.lr
            replay_scheduler.observe(row['validation_cross_entropy'])
            assert row['next_learning_rate'] == replay_scheduler.lr
            value = row[protocol.get('monitor_key','validation_auroc')]
            if (value < raw_best if minimize else value > raw_best): raw_best,best_round = value,number
            if (value < anchor-protocol['min_delta'] if minimize else value > anchor+protocol['min_delta']): anchor,bad = value,0
            else: bad += 1
            if number >= protocol['min_rounds'] and bad >= protocol['patience']: stop = number
            assert (row['best_round'],row['bad_rounds'],row['patience_value']) == (best_round,bad,anchor)
            steps = sum(math.ceil(len(g)/setting['batch_size']) for g in groups)*setting['local_epochs']
            assert row['optimizer_steps'] == steps and row['cumulative_optimizer_steps'] == steps*number
            assert row['processed_examples'] == len(train)*setting['local_epochs']
            assert row['cumulative_processed_examples'] == len(train)*setting['local_epochs']*number
            prefix = h[:number]
            stable = practical_stability([r['train_cross_entropy'] for r in prefix],
                                         [r['learning_rate'] for r in prefix],protocol['stability_settings'])['loss_stable']
            assert stable == row['training_loss_stable']
            assert row['strict_training_loss_stable'] == convergence([r['train_cross_entropy'] for r in prefix],config['convergence'])['converged']
        assert status['selected_round'] == best_round and cp['selected']['round'] == best_round
        expected_controller = dict(round=len(h),best_round=best_round,bad_rounds=bad,stop_round=stop)
        expected_controller.update(dict(best_loss=raw_best,patience_best_loss=anchor) if minimize else
                                   dict(best_auc=raw_best,patience_auc=anchor))
        assert cp['controller'] == expected_controller
        assert status['state'] == ('loss_stable' if stable else 'early_stopped' if stop is not None else 'capped')
        assert stop is not None or stable or len(h) == 300
        assert replay_scheduler.state_dict() == cp['scheduler']
        selected = torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        assert selected['round'] == best_round
        for key in selected['state']:
            torch.testing.assert_close(selected['state'][key],cp['selected']['state'][key],rtol=0,atol=0)
        for kind,state,row in [('selected',selected['state'],status['final']),('last',cp['state'],h[-1])]:
            model.load_state_dict(state); logits = predict_logits(model,xt)
            for key,value in gradient_diagnostic(model,xt,torch.from_numpy(y),train,setting['weight_decay']).items():
                np.testing.assert_allclose(value,status[kind+'_gradient'][key],rtol=1e-10,atol=1e-12)
            for name,idx in [('train',train),('validation',val),('test',test)]:
                for key,value in classification_metrics(logits[idx],y[idx]).items():
                    np.testing.assert_allclose(value,row[f'{name}_{key}'],rtol=1e-10,atol=1e-12)
            if kind == 'selected':
                threshold = calibrate_threshold(y[val],probability(logits[val]))
                assert threshold == status['threshold'] == selected['threshold']
                for name,idx in [('train',train),('validation',val),('test',test)]:
                    for key,value in calibrated_metrics(y[idx],probability(logits[idx]),threshold).items():
                        np.testing.assert_allclose(value,status['final'][f'{name}_calibrated_{key}'],rtol=0,atol=1e-12)
        rows.append(dict(**setting,state=status['state'],rounds=status['rounds'],selected_round=best_round,
                         threshold=status['threshold'],training_loss_stable=stable,
                         strict_training_loss_stable=status['strict_training_loss_stable'],
                         **{'last_'+k:v for k,v in status['last_gradient'].items()},
                         **{'selected_'+k:v for k,v in status['selected_gradient'].items()},
                         **{k:v for k,v in status['final'].items() if not k.startswith('test_')}))
        histories.append(history_table(h,setting))
    table = pd.DataFrame(rows)
    chosen = choose_configuration(table,protocol)
    by_k = table[table.weight_decay==chosen].groupby('clients')[['validation_auroc','validation_calibrated_balanced_accuracy']].mean()
    sanity = protocol['validation_sanity']
    passed = bool((by_k.validation_auroc > sanity['minimum_mean_auroc_each_client_count']).all() and
                  (by_k.validation_calibrated_balanced_accuracy > sanity['minimum_mean_calibrated_balanced_accuracy_each_client_count']).all())
    # Freeze the validation-only choice BEFORE collecting chosen test outcomes.
    decision = dict(weight_decay=chosen,baseline=config['baseline'],model=config['model'],
                    protocol=protocol,validation_sanity_passed=passed,
                    mechanism_goal_passed=mechanism_goal(table[table.weight_decay==chosen]),
                    status='validated_candidate' if passed and mechanism_goal(table[table.weight_decay==chosen]) else 'provisional_requires_review',
                    mean_validation_auroc_by_candidate={str(k):float(v) for k,v in table.groupby('weight_decay').validation_auroc.mean().items()},
                    validation_by_client_count=by_k.reset_index().to_dict('records'),
                    training_stability_by_candidate=table.groupby('weight_decay')[['training_loss_stable','strict_training_loss_stable']].sum().reset_index().to_dict('records'),
                    scope='Candidate baseline mechanism; not a completed replacement of the 84-run parameter study',
                    limitations=['Validation reused for stopping, threshold calibration and configuration selection; estimates are optimistic.',
                                 'Synthetic data, three seeds and 101 validation positives per seed; no clinical claim.',
                                 'Early stopping does not establish optimization convergence.'])
    write_json(Path(output)/'selected_config.json',decision)
    chosen_rows = []
    for setting in definitions(config,protocol):
        if setting['weight_decay'] == chosen:
            status = read_json(run_dir(output,setting)/'status.json')
            chosen_rows.append(dict(**setting,state=status['state'],rounds=status['rounds'],selected_round=status['selected_round'],
                                    threshold=status['threshold'],training_loss_stable=status['training_loss_stable'],
                                    strict_training_loss_stable=status['strict_training_loss_stable'],**status['final']))
    table.to_csv(Path(output)/'validation_candidates.csv',index=False)
    pd.DataFrame(chosen_rows).to_csv(Path(output)/'selected_model_metrics.csv',index=False)
    pd.concat(histories,ignore_index=True).to_csv(Path(output)/'round_metrics.csv',index=False)
    write_json(Path(output)/'audit.json',dict(passed=True,trials=len(rows),models_rescored=2*len(rows),
                history_rows=sum(len(h) for h in histories),validation_only_selection=True,convergence_claim=False,
                artifact_sha256={n:sha256_file(Path(output)/n) for n in
                    ['selected_config.json','validation_candidates.csv','selected_model_metrics.csv','round_metrics.csv']}))
    print('Validation-only decision:',chosen,'sanity passed:',passed,flush=True)
    print(pd.DataFrame(chosen_rows).groupby('clients')[['test_auroc','test_cross_entropy','test_calibrated_balanced_accuracy','test_calibrated_sensitivity']].mean().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--partitions-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run-index',type=int)
    action.add_argument('--audit',action='store_true')
    parser.add_argument('--interrupt-after',type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run training and model rescoring in Slurm')
    torch.set_num_threads(1)
    config,protocol = load_config(),load_protocol()
    if args.audit:
        audit(args.data_dir,args.partitions_root,args.output,config,protocol)
    else:
        settings=definitions(config,protocol)
        if not 0 <= args.run_index < len(settings): parser.error('Invalid run index')
        setting=settings[args.run_index]
        frame,x,y,dataset=load_dataset(args.data_dir,config)
        part=freeze_partition(args.partitions_root,frame.record_id,y,setting,dataset['sha256'])
        run(args.output,x,y,dataset,part,setting,config,protocol,args.interrupt_after)


if __name__ == '__main__': main()
