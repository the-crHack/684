"""Prospective validation-based early stopping; preserve the original FL/MIA campaign."""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common import (HERE, ReadmissionMLP, classification_metrics, ensure_partitions,
                    json_hash, load_config, load_dataset, partition_indices,
                    read_json, save_torch, seed_everything, sha256_file, write_json)
from fedavg import aggregate, cpu_state, local_update, predict_logits

# Freeze these choices before prospective training; do not tune them on test results.
PROTOCOL = dict(revision=1, validation_fraction=0.2, max_rounds=300,
                min_rounds=30, patience=25, min_delta=1e-4,
                monitor="validation cross-entropy after global aggregation",
                selection="lowest observed validation cross-entropy; earliest tie",
                local_update="original constant-rate SGD, including every partial batch",
                scope="12 baseline runs; no new dataset or membership attacks",
                interpretation="predictive early stopping, not optimization convergence")
TERMINAL = {"validation_early_stopped", "capped", "failed"}


class ValidationStop:
    """Raw minimum chooses weights; cumulative min_delta improvements reset patience."""
    def __init__(self, protocol=PROTOCOL):
        if (protocol['max_rounds'] != 300 or not 1 <= protocol['min_rounds'] <= 300
                or protocol['patience'] < 1 or protocol['min_delta'] <= 0):
            raise ValueError("Invalid early-stopping settings")
        self.protocol = protocol
        self.round = 0
        self.best_round = 0
        self.best_loss = math.inf
        self.patience_best_loss = math.inf
        self.bad_rounds = 0
        self.stop_round = None

    def observe(self, validation_loss):
        if self.round >= 300 or self.stop_round is not None:
            raise ValueError("No observations after stopping or round 300")
        if not np.isfinite(validation_loss) or validation_loss < 0:
            raise FloatingPointError("Nonfinite or negative validation loss")
        self.round += 1
        selected = validation_loss < self.best_loss
        if selected:
            self.best_loss, self.best_round = float(validation_loss), self.round
        if validation_loss < self.patience_best_loss - self.protocol['min_delta']:
            self.patience_best_loss = float(validation_loss)
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        if self.round >= self.protocol['min_rounds'] and self.bad_rounds >= self.protocol['patience']:
            self.stop_round = self.round
        return selected

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'protocol'}

    @classmethod
    def from_history(cls, history, protocol=PROTOCOL):
        controller = cls(protocol)
        for round_number, row in enumerate(history, 1):
            if row['round'] != round_number:
                raise ValueError("Noncontiguous early-stopping history")
            controller.observe(row['validation_cross_entropy'])
        return controller


def definitions(config):
    return [dict(clients=k, seed=seed, **config['baseline'])
            for k in config['client_counts'] for seed in config['seeds']]


def split_indices(y, seed, clients):
    """Keep the original outer test half; stratify validation within its training half."""
    outer, test, _ = partition_indices(y, seed, clients)
    train, validation = train_test_split(outer, test_size=PROTOCOL['validation_fraction'],
                                      stratify=np.asarray(y)[outer], random_state=seed + 1_000_003)
    ordered = np.random.default_rng(seed + 2_000_003).permutation(train)
    if len(train) % clients:
        raise ValueError("Optimization records must divide equally among clients")
    groups = list(np.split(ordered, clients))
    validate_partition(len(y), train, validation, test, groups)
    return train, validation, test, groups


def validate_partition(n, train, validation, test, groups):
    all_indices = np.concatenate([train, validation, test])
    if (not len(train) or not len(validation) or not len(test) or
            not np.array_equal(np.sort(all_indices), np.arange(n)) or not groups or
            len({len(g) for g in groups}) != 1 or
            not np.array_equal(np.sort(np.concatenate(groups)), np.sort(train))):
        raise ValueError("Invalid train/validation/test/client partition")


def freeze_partition(output, record_ids, y, setting, dataset_hash):
    train, validation, test, groups = split_indices(y, setting['seed'], setting['clients'])
    client_ids = np.full(len(y), -1, dtype=int)
    roles = np.full(len(y), 'test', dtype=object)
    roles[train], roles[validation] = 'train', 'validation'
    for client, indices in enumerate(groups):
        client_ids[indices] = client
    table = pd.DataFrame(dict(record_id=record_ids, split=roles, client_id=client_ids))
    folder = Path(output) / 'partitions'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"seed_{setting['seed']}_k{setting['clients']:02d}.csv"
    metadata = path.with_suffix('.json')
    definition = dict(dataset_sha256=dataset_hash, seed=setting['seed'], clients=setting['clients'],
                      protocol_sha256=json_hash(PROTOCOL), algorithm='original-half-stratified-validation-v1')
    if path.exists() or metadata.exists():
        if (not path.exists() or not metadata.exists() or
                read_json(metadata).get('definition') != definition or
                read_json(metadata).get('sha256') != sha256_file(path) or
                not pd.read_csv(path).equals(table)):
            raise ValueError("Incompatible validation partition cache")
    else:
        table.to_csv(path, index=False)
        write_json(metadata, dict(definition=definition, sha256=sha256_file(path),
                                 training_records=len(train), validation_records=len(validation),
                                 test_records=len(test), records_per_client=len(groups[0]),
                                 global_members='optimization training records only',
                                 local_nonmembers='all records outside that client optimization partition'))
    return train, validation, test, groups, sha256_file(path)


def run(output, frame, x_array, labels, dataset, partition, setting, config,
        protocol=PROTOCOL, interrupt_after=None):
    train, validation, test, groups, partition_hash = partition
    validate_partition(len(labels), train, validation, test, groups)
    if setting['clients'] != len(groups):
        raise ValueError("Client count differs from the frozen partition")
    directory = Path(output) / 'runs' / f"k{setting['clients']:02d}_seed_{setting['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(setting=setting, protocol=protocol, config_sha256=json_hash(config),
                    dataset_sha256=dataset['sha256'], schema_sha256=dataset['schema_sha256'],
                    preprocessor_sha256=dataset['preprocessor_sha256'], partition_sha256=partition_hash,
                    packages={p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn']},
                    code_hashes={n: sha256_file(HERE/n) for n in ['common.py','fedavg.py','generalization_followup.py']})
    mp, cp, best, status_path = [directory / n for n in
                                ['manifest.json','checkpoint.pt','best_validation.pt','status.json']]
    if mp.exists() and read_json(mp) != manifest:
        raise ValueError("Incompatible generalization cache; use a fresh directory")
    write_json(mp, manifest)
    if status_path.exists() and read_json(status_path)['state'] in TERMINAL:
        status = read_json(status_path)
        if status['state'] != 'failed':
            for name, digest in status['artifact_sha256'].items():
                if sha256_file(directory/name) != digest:
                    raise ValueError("Corrupt completed early-stopping artifact")
        return status
    seed_everything(setting['seed'])
    model = ReadmissionMLP(x_array.shape[1])
    x, y = torch.from_numpy(x_array), torch.from_numpy(labels)
    history, previous_seconds, selected = [], 0., None
    controller = ValidationStop(protocol)
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        history = saved['history']
        if saved['manifest_sha256'] != json_hash(manifest) or saved['round'] != len(history) or len(history) > 300:
            raise ValueError("Invalid early-stopping checkpoint/history")
        controller = ValidationStop.from_history(history, protocol)
        if controller.state_dict() != saved['controller']:
            raise ValueError("Checkpoint/controller mismatch")
        selected = saved['selected']
        if (selected['manifest_sha256'] != json_hash(manifest) or
                selected['round'] != controller.best_round or
                selected['validation_cross_entropy'] != controller.best_loss):
            raise ValueError("Checkpoint/selected-model mismatch")
        model.load_state_dict(saved['state'])
        previous_seconds = saved['elapsed_seconds']
    started = time.perf_counter()
    write_json(status_path, dict(state='running', setting=setting, rounds=len(history)))
    try:
        for number in range(len(history)+1, 301):
            if controller.stop_round is not None:
                break
            local_states, steps, examples = [], 0, 0
            for client, indices in enumerate(groups):
                local, count, processed = local_update(model, x, y, indices, setting,
                                                       config['optimizer'], number, client)
                local_states.append(cpu_state(local))
                steps += count
                examples += processed
            model.load_state_dict(aggregate(local_states, [len(g) for g in groups]))
            logits = predict_logits(model, x)
            row = dict(round=number, optimizer_steps=steps, processed_examples=examples,
                       cumulative_optimizer_steps=steps+sum(r['optimizer_steps'] for r in history),
                       cumulative_processed_examples=examples+sum(r['processed_examples'] for r in history))
            for split, indices in [('train',train),('validation',validation),('test',test)]:
                row.update({f'{split}_{k}':v for k,v in classification_metrics(logits[indices],labels[indices]).items()})
            is_best = controller.observe(row['validation_cross_entropy'])
            row.update(best_validation_round=controller.best_round, best_validation_loss=controller.best_loss,
                       patience_best_loss=controller.patience_best_loss, bad_rounds=controller.bad_rounds,
                       validation_stop=controller.stop_round is not None)
            history.append(row)
            if is_best:
                selected = dict(manifest_sha256=json_hash(manifest), round=number,
                                validation_cross_entropy=controller.best_loss, state=cpu_state(model))
            row['elapsed_seconds'] = previous_seconds + time.perf_counter()-started
            save_torch(cp, dict(manifest_sha256=json_hash(manifest), round=len(history), state=cpu_state(model),
                                history=history, controller=controller.state_dict(), selected=selected,
                                elapsed_seconds=row['elapsed_seconds']))
            if number == 1 or number % 25 == 0 or controller.stop_round:
                print(setting, number, 'validation CE', row['validation_cross_entropy'],
                      'selected',controller.best_round,'bad rounds',controller.bad_rounds,flush=True)
            if controller.stop_round is not None:
                break
            if interrupt_after is not None and number >= interrupt_after:
                pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
                status = dict(state='interrupted', setting=setting, rounds=len(history))
                write_json(status_path,status)
                return status
        save_torch(best, selected)
        model.load_state_dict(selected['state'])  # Restore validation-selected weights before final evaluation.
        logits = predict_logits(model,x)
        final = {}
        for split, indices in [('train',train),('validation',validation),('test',test)]:
            final.update({f'{split}_{k}':v for k,v in classification_metrics(logits[indices],labels[indices]).items()})
        predicted = logits[test].argmax(axis=1)
        final.update(test_positive_predictions=int((predicted==1).sum()),
                     test_sensitivity=float((predicted[labels[test]==1]==1).mean()))
        np.testing.assert_allclose(final['validation_cross_entropy'],controller.best_loss,rtol=1e-10,atol=1e-12)
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        state = 'validation_early_stopped' if controller.stop_round else 'capped'
        if state == 'capped' and len(history) != 300:
            raise ValueError("Capped run ended before round 300")
        status = dict(state=state, setting=setting, rounds=len(history), selected_round=controller.best_round,
                      training_records=len(train), validation_records=len(validation), test_records=len(test),
                      records_per_client=len(groups[0]), final=final, last=history[-1],
                      stopping_reason='validation_patience' if controller.stop_round else 'maximum_300_rounds',
                      convergence_claim=False,
                      artifact_sha256={n:sha256_file(directory/n) for n in ['checkpoint.pt','best_validation.pt','history.csv']})
        write_json(status_path,status)
        return status
    except Exception as error:
        write_json(status_path,dict(state='failed',setting=setting,rounds=len(history),
                                   error=type(error).__name__+': '+str(error)))
        raise


def summarize_and_audit(data_dir, output, destination):
    """Independent controller replay and rescoring, in a Slurm allocation."""
    config = load_config()
    frame, x, labels, dataset = load_dataset(data_dir,config)
    xt = torch.from_numpy(x)
    rows, all_history = [], []
    model = ReadmissionMLP(x.shape[1])
    originals = pd.read_csv(Path(destination)/'results/run_status.csv')
    for setting in definitions(config):
        directory = Path(output)/'runs'/f"k{setting['clients']:02d}_seed_{setting['seed']}"
        status, manifest = read_json(directory/'status.json'), read_json(directory/'manifest.json')
        if status['setting'] != setting or status['state'] not in TERMINAL:
            raise ValueError("Missing or incomplete early-stopping trial")
        if status['state']=='failed':
            rows.append({**setting, **{k:status[k] for k in ['state','rounds','error']}})
            continue
        assert set(status['artifact_sha256'])=={'checkpoint.pt','best_validation.pt','history.csv'}
        for name,digest in status['artifact_sha256'].items():
            assert sha256_file(directory/name)==digest, name
        train,val,test,groups,digest = freeze_partition(output,frame.record_id,labels,setting,dataset['sha256'])
        assert manifest['partition_sha256']==digest and manifest['protocol']==PROTOCOL
        for name,digest in manifest['code_hashes'].items():assert sha256_file(HERE/name)==digest
        for key in ['schema_sha256','preprocessor_sha256']:
            assert manifest[key]==dataset[key]
        assert manifest['dataset_sha256']==dataset['sha256'] and manifest['config_sha256']==json_hash(config)
        cp = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        selected = torch.load(directory/'best_validation.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==selected['manifest_sha256']==json_hash(manifest)
        assert cp['selected']['round']==selected['round']
        assert cp['selected']['validation_cross_entropy']==selected['validation_cross_entropy']
        for key,value in cp['selected']['state'].items():
            torch.testing.assert_close(value,selected['state'][key],rtol=0,atol=0)
        h = pd.DataFrame(cp['history'])
        pd.testing.assert_frame_equal(h,pd.read_csv(directory/'history.csv'),check_dtype=False,rtol=1e-10,atol=1e-12)
        assert h['round'].tolist()==list(range(1,status['rounds']+1)) and cp['round']==len(h)<=300
        best_loss, anchor, best_round, bad, stopped = math.inf, math.inf, 0, 0, None
        for number,row in enumerate(cp['history'],1):
            loss=row['validation_cross_entropy']
            assert math.isfinite(loss) and loss>=0 and stopped is None
            if loss<best_loss:best_loss,best_round=loss,number
            if loss<anchor-PROTOCOL['min_delta']:anchor,bad=loss,0
            else:bad+=1
            if number>=PROTOCOL['min_rounds'] and bad>=PROTOCOL['patience']:stopped=number
            assert (row['best_validation_round'],row['best_validation_loss'],row['patience_best_loss'],
                    row['bad_rounds'],row['validation_stop'])==(best_round,best_loss,anchor,bad,stopped is not None)
        assert cp['controller']==dict(round=len(h),best_round=best_round,best_loss=best_loss,
                                      patience_best_loss=anchor,bad_rounds=bad,stop_round=stopped)
        assert status['selected_round']==selected['round']==best_round
        assert best_round==int(h.loc[h.validation_cross_entropy.idxmin(),'round'])
        assert status['state']==('validation_early_stopped' if stopped else 'capped')
        assert stopped is not None or len(h)==300
        steps=setting['clients']*math.ceil(len(groups[0])/setting['batch_size'])*setting['local_epochs']
        assert h.optimizer_steps.eq(steps).all() and h.processed_examples.eq(len(train)*setting['local_epochs']).all()
        assert np.array_equal(h.cumulative_optimizer_steps,h['round']*steps)
        assert np.array_equal(h.cumulative_processed_examples,h['round']*len(train)*setting['local_epochs'])
        for saved,reported in [(selected,status['final']),(cp,status['last'])]:
            model.load_state_dict(saved['state'])
            logits=predict_logits(model,xt)
            for split,idx in [('train',train),('validation',val),('test',test)]:
                for key,value in classification_metrics(logits[idx],labels[idx]).items():
                    np.testing.assert_allclose(value,reported[f'{split}_{key}'],rtol=1e-10,atol=1e-12)
        model.load_state_dict(selected['state'])
        predicted=predict_logits(model,xt[test]).argmax(axis=1)
        assert status['final']['test_positive_predictions']==int((predicted==1).sum())
        np.testing.assert_allclose(status['final']['test_sensitivity'],(predicted[labels[test]==1]==1).mean())
        control=originals[(originals.clients==setting['clients'])&(originals.seed==setting['seed'])&
                          (originals.batch_size==32)&(originals.local_epochs==1)&(originals.learning_rate==.05)]
        assert len(control)==1
        row={**setting,**{k:status[k] for k in ['state','rounds','selected_round','training_records',
              'validation_records','test_records','records_per_client']},**status['final'],
             'elapsed_seconds':status['last']['elapsed_seconds'],
             'cumulative_optimizer_steps':status['last']['cumulative_optimizer_steps'],
             'cumulative_processed_examples':status['last']['cumulative_processed_examples']}
        for key in ['train_cross_entropy','test_cross_entropy','test_accuracy','test_balanced_accuracy',
                    'test_auroc','test_average_precision']:
            row['original_'+key]=float(control.iloc[0][key])
        rows.append(row)
        for key,value in setting.items():h[key]=value
        all_history.append(h)
    summary=pd.DataFrame(rows)
    summary.to_csv(Path(output)/'summary.csv',index=False)
    history=pd.concat(all_history,ignore_index=True) if all_history else pd.DataFrame()
    history.to_csv(Path(output)/'round_metrics.csv',index=False)
    failures=summary.loc[summary.state=='failed', ['clients','seed','error']].to_dict('records') if 'error' in summary else []
    proof=dict(passed=not failures,expected_trials=12,audited_trials=len(summary),failed_trials=failures,
               history_rows=len(history),selected_and_last_models_rescored=2*(len(summary)-len(failures)),
               independent_validation_patience_replay=True,original_campaign_and_task_b_unchanged=True,
               comparison_limitation='Original controls optimize 4500 records; follow-up optimizes 3600, with 900 validation. Differences do not isolate early stopping.',
               convergence_claim=False,protocol=PROTOCOL,
               artifact_sha256={n:sha256_file(Path(output)/n) for n in ['summary.csv','round_metrics.csv']})
    write_json(Path(output)/'validation.json',proof)
    print(summary.reindex(columns=['clients','seed','state','rounds','selected_round',
                                   'test_cross_entropy','test_auroc']).to_string(index=False))
    if failures:raise ValueError("Failed trials retained; do not claim complete success")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=HERE)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--run-index',type=int)
    action.add_argument('--summarize',action='store_true')
    parser.add_argument('--interrupt-after',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if not args.prepare and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run training and model audit inside a Slurm CPU allocation')
    if args.summarize:
        summarize_and_audit(args.data_dir,args.output,args.destination)
        return
    config=load_config()
    frame,x,y,dataset=load_dataset(args.data_dir,config)
    if args.prepare:
        for setting in definitions(config):freeze_partition(args.output,frame.record_id,y,setting,dataset['sha256'])
        write_json(args.output/'protocol.json',PROTOCOL)
        print('Frozen 12 train/validation/test manifests; original outer test halves preserved.')
    else:
        if not 0<=args.run_index<12:parser.error('run-index must be 0--11')
        setting=definitions(config)[args.run_index]
        partition=freeze_partition(args.output,frame.record_id,y,setting,dataset['sha256'])
        run(args.output,frame,x,y,dataset,partition,setting,config,interrupt_after=args.interrupt_after)


if __name__=='__main__':
    main()
