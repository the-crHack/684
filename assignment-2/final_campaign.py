"""Run the full FedAvg grid using the immutable, confirmed training mechanism.

The 12 baseline training caches are reused after exact compatibility checks.
Round-one attacks are reconstructed deterministically for every setting and
matched against that run's first global metrics. No old attack results are reused.
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (HERE, ReadmissionMLP, attack_metrics, classification_metrics,
                    convergence, json_hash, load_config, load_dataset, membership_masks,
                    read_json, save_torch, schedule, seed_everything, sha256_file, write_json)
from fedavg import (aggregate, cpu_state, first_round_attacks, local_update,
                    losses_from_logits, predict_logits, validate_first_round)
from generalization_followup import freeze_partition, validate_partition
from training_selection import (ValidationLR, calibrate_threshold, calibrated_metrics,
                                gradient_diagnostic, history_table, load_protocol,
                                practical_stability, probability, run, run_dir)


TRAINING_SOURCES = ['common.py','fedavg.py','generalization_followup.py','task_a_protocol.py',
                    'task_a_protocol.json','convergence_pilot.py','training_selection.py','training_selection.json']
TERMINAL = {'loss_stable','early_stopped','capped'}


def definitions(config, protocol):
    if protocol['weight_decay_candidates'] != [.01] or protocol['revision'] != 3:
        raise ValueError('Use the frozen, confirmed L2=0.01 revision-3 mechanism')
    return [dict(**s,weight_decay=.01) for s in schedule(config)]


def training_setting(setting):
    return {k:v for k,v in setting.items() if k != 'config_id'}


def trial_output(output, setting):
    return Path(output)/'configs'/setting['config_id']


def trial_dir(output, setting):
    return run_dir(trial_output(output,setting),training_setting(setting))


def is_baseline(setting, config):
    return all(setting[k] == value for k,value in config['baseline'].items())


def core_manifest(dataset, partition_hash, setting, config, protocol):
    """Match the unchanged training_selection.run manifest exactly."""
    return dict(setting=training_setting(setting),protocol=protocol,config_sha256=json_hash(config),
                dataset=dataset,partition_sha256=partition_hash,
                code_hashes={n:sha256_file(HERE/n) for n in TRAINING_SOURCES},
                versions=dict(torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__))


def freeze_campaign(output, dataset, config, protocol):
    definition = dict(revision=1,settings=definitions(config,protocol),protocol=protocol,
                      config_sha256=json_hash(config),dataset=dataset,
                      source_sha256={n:sha256_file(HERE/n) for n in TRAINING_SOURCES+['final_campaign.py']},
                      effective_optimizer={**config['optimizer'],'weight_decay':.01},
                      global_members='3600 optimization training records',
                      global_nonmembers='5400 records: 900 validation plus 4500 test',
                      local_nonmembers='all records outside that client optimization training partition',
                      attacks='negative per-record cross-entropy immediately after one round; retrospective oracle threshold',
                      selection='validation only; original strict and safeguarded loss criteria reported separately')
    path=Path(output)/'campaign_definition.json'
    if path.exists() and read_json(path)!=definition:
        raise ValueError('Incompatible full campaign cache')
    if not path.exists():write_json(path,definition)
    return definition


def reuse_baselines(prior, output, dataset, partitions_root, frame, labels, config, protocol):
    proof=read_json(Path(prior)/'audit.json')
    continuation=read_json(Path(prior)/'continuation_audit.json')
    if not proof['passed'] or proof['trials']!=12 or not continuation['passed']:
        raise ValueError('Baseline audits must pass before reuse')
    for n,h in proof['artifact_sha256'].items():
        if sha256_file(Path(prior)/n)!=h:raise ValueError('Corrupt baseline audit artifacts')
    rows=[]
    for setting in definitions(config,protocol):
        if not is_baseline(setting,config):continue
        source=run_dir(prior,training_setting(setting));target=trial_dir(output,setting)
        partition=freeze_partition(partitions_root,frame.record_id,labels,setting,dataset['sha256'])
        expected=core_manifest(dataset,partition[-1],setting,config,protocol)
        manifest,status=read_json(source/'manifest.json'),read_json(source/'status.json')
        if manifest!=expected or status['state']!='loss_stable' or status['setting']!=training_setting(setting):
            raise ValueError('Incompatible baseline training cache')
        for name,digest in status['artifact_sha256'].items():
            if sha256_file(source/name)!=digest:raise ValueError('Corrupt baseline checkpoint')
        cp=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
        if cp['manifest_sha256']!=json_hash(expected) or cp['round']!=len(cp['history']) or cp['round']!=status['rounds']:
            raise ValueError('Inconsistent baseline checkpoint')
        if target.exists():raise ValueError('Never overwrite a reused run')
        target.mkdir(parents=True)
        names=['manifest.json','status.json','checkpoint.pt','history.csv','selected.pt']
        for name in names:
            shutil.copy2(source/name,target/name)
            if sha256_file(source/name)!=sha256_file(target/name):raise ValueError('Baseline copy mismatch')
        rows.append(dict(**setting,source=str(source),source_artifact_sha256={n:sha256_file(source/n) for n in names},
                         rounds=status['rounds'],training_rerun=False))
    if len(rows)!=12:raise ValueError('Expected exactly twelve baseline caches')
    write_json(Path(output)/'baseline_reuse.json',dict(passed=True,runs=rows,training_runs_reused=12,
                source_audit_sha256=sha256_file(Path(prior)/'audit.json')))


def round_one(output, frame, x_array, labels, dataset, partition, setting, config, protocol):
    train,val,test,groups,part_hash=partition
    validate_partition(len(labels),train,val,test,groups)
    bare=training_setting(setting);folder=trial_dir(output,setting);folder.mkdir(parents=True,exist_ok=True)
    attack_manifest=dict(setting=setting,training_manifest_sha256=json_hash(core_manifest(dataset,part_hash,setting,config,protocol)),
                         coordinator_sha256=sha256_file(HERE/'final_campaign.py'),
                         labels='SGD optimization membership; validation and test are nonmembers',
                         first_round_lr=setting['learning_rate'],weight_decay=.01)
    mp=folder/'attack_definition.json'
    if mp.exists() and read_json(mp)!=attack_manifest:raise ValueError('Incompatible attack definition')
    write_json(mp,attack_manifest)
    if validate_first_round(folder,attack_manifest) and (folder/'attack_timing.json').exists():
        return read_json(folder/'attack_timing.json')
    seed_everything(setting['seed'])
    initial=ReadmissionMLP(x_array.shape[1]);x,y=torch.from_numpy(x_array),torch.from_numpy(labels)
    states,losses=[],[];steps=examples=0;started=time.perf_counter()
    for client,indices in enumerate(groups):
        model,count,processed=local_update(initial,x,y,indices,bare,{**config['optimizer'],'weight_decay':.01},1,client)
        states.append(cpu_state(model));losses.append(losses_from_logits(predict_logits(model,x),labels))
        steps+=count;examples+=processed
    global_model=ReadmissionMLP(x_array.shape[1]);global_model.load_state_dict(aggregate(states,[len(g) for g in groups]))
    logits=predict_logits(global_model,x)
    first_round_attacks(folder,frame.record_id.to_numpy(dtype=str),labels,train,groups,
                       [losses_from_logits(logits,labels)]+losses,[cpu_state(global_model)]+states,
                       attack_manifest,config['attack_fpr_levels'])
    timing=dict(seconds=time.perf_counter()-started,reconstruction_optimizer_steps=steps,
                reconstruction_processed_examples=examples,models=len(groups)+1,
                training='One deterministic round reconstructed separately; excluded from core-training work/timing')
    for name,indices in [('train',train),('validation',val),('test',test)]:
        timing.update({f'global_round_one_{name}_{k}':v for k,v in classification_metrics(logits[indices],labels[indices]).items()})
    write_json(folder/'attack_timing.json',timing)
    return timing


def run_trial(output, frame, x, labels, dataset, partition, setting, config, protocol, interrupt_after=None):
    folder=trial_dir(output,setting)
    try:
        timing=round_one(output,frame,x,labels,dataset,partition,setting,config,protocol)
        status=run(trial_output(output,setting),x,labels,dataset,partition,training_setting(setting),config,protocol,interrupt_after)
        cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        for name in ['train','validation','test']:
            for key in ['cross_entropy','accuracy','balanced_accuracy','auroc','average_precision']:
                np.testing.assert_allclose(cp['history'][0][f'{name}_{key}'],timing[f'global_round_one_{name}_{key}'],rtol=1e-10,atol=1e-12)
        return status
    except Exception as error:
        write_json(folder/'task_failure.json',dict(setting=setting,error=repr(error)))
        if not (folder/'status.json').exists():
            write_json(folder/'manifest.json',core_manifest(dataset,partition[-1],setting,config,protocol))
            write_json(folder/'status.json',dict(state='failed',setting=training_setting(setting),rounds=0,error=repr(error)))
        raise


def audit_training(folder, partition, setting, dataset, config, protocol, x_array, labels):
    status,manifest=read_json(folder/'status.json'),read_json(folder/'manifest.json')
    train,val,test,groups,part_hash=partition
    if manifest!=core_manifest(dataset,part_hash,setting,config,protocol) or status['setting']!=training_setting(setting):
        raise ValueError('Training manifest mismatch')
    if status['state'] not in TERMINAL:raise ValueError(f"Unsuccessful training state: {status['state']}")
    for name,digest in status['artifact_sha256'].items():
        if sha256_file(folder/name)!=digest:raise ValueError('Training artifact checksum mismatch')
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);h=cp['history']
    if cp['manifest_sha256']!=json_hash(manifest) or cp['round']!=len(h) or len(h)!=status['rounds'] or not 1<=len(h)<=300:
        raise ValueError('Invalid training history')
    scheduler=ValidationLR(setting['learning_rate'],protocol)
    best=anchor=math.inf;bad=best_round=0;stop=None;stable=False
    for number,row in enumerate(h,1):
        if row['round']!=number or stop is not None or stable:raise ValueError('Training continued after a terminal decision')
        if row['learning_rate']!=scheduler.lr:raise ValueError('Wrong round-specific LR')
        scheduler.observe(row['validation_cross_entropy'])
        if row['next_learning_rate']!=scheduler.lr:raise ValueError('Wrong next-round LR')
        value=row['validation_cross_entropy']
        if value<best:best,best_round=value,number
        if value<anchor-protocol['min_delta']:anchor,bad=value,0
        else:bad+=1
        if number>=protocol['min_rounds'] and bad>=protocol['patience']:stop=number
        if (row['best_round'],row['bad_rounds'],row['patience_value'])!=(best_round,bad,anchor):
            raise ValueError('Wrong validation controller history')
        prefix=h[:number]
        stable=practical_stability([r['train_cross_entropy'] for r in prefix],[r['learning_rate'] for r in prefix],protocol['stability_settings'])['loss_stable']
        strict=convergence([r['train_cross_entropy'] for r in prefix],config['convergence'])['converged']
        if stable!=row['training_loss_stable'] or strict!=row['strict_training_loss_stable']:
            raise ValueError('Wrong stability decision')
        steps=sum(math.ceil(len(g)/setting['batch_size']) for g in groups)*setting['local_epochs']
        examples=len(train)*setting['local_epochs']
        if (row['optimizer_steps'],row['cumulative_optimizer_steps'],row['processed_examples'],row['cumulative_processed_examples'])!=(steps,steps*number,examples,examples*number):
            raise ValueError('Wrong optimizer work counts')
        if any(not np.isfinite(row[f'{name}_{key}']) for name in ['train','validation','test'] for key in ['cross_entropy','accuracy','balanced_accuracy','auroc','average_precision']):
            raise ValueError('Nonfinite per-round metrics')
    expected='loss_stable' if stable else 'early_stopped' if stop is not None else 'capped'
    if status['state']!=expected or (expected=='capped' and len(h)!=300) or status['selected_round']!=best_round:
        raise ValueError('Incorrect terminal/selected round')
    if cp['controller']!=dict(round=len(h),best_round=best_round,bad_rounds=bad,stop_round=stop,best_loss=best,patience_best_loss=anchor):
        raise ValueError('Checkpoint validation state mismatch')
    if cp['scheduler']!=scheduler.state_dict():raise ValueError('Checkpoint scheduler state mismatch')
    selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    if selected['round']!=best_round or selected['validation_cross_entropy']!=best or cp['selected']['round']!=best_round:
        raise ValueError('Selected checkpoint mismatch')
    for key in selected['state']:torch.testing.assert_close(selected['state'][key],cp['selected']['state'][key],rtol=0,atol=0)
    model=ReadmissionMLP(x_array.shape[1]);x,y=torch.from_numpy(x_array),torch.from_numpy(labels)
    for kind,state,row in [('selected',selected['state'],status['final']),('last',cp['state'],h[-1])]:
        model.load_state_dict(state);logits=predict_logits(model,x)
        for name,indices in [('train',train),('validation',val),('test',test)]:
            for key,value in classification_metrics(logits[indices],labels[indices]).items():
                np.testing.assert_allclose(value,row[f'{name}_{key}'],rtol=1e-10,atol=1e-12)
        for key,value in gradient_diagnostic(model,x,y,train,.01).items():
            np.testing.assert_allclose(value,status[kind+'_gradient'][key],rtol=1e-10,atol=1e-12)
        if kind=='selected':
            threshold=calibrate_threshold(labels[val],probability(logits[val]))
            if threshold!=status['threshold'] or threshold!=selected['threshold']:raise ValueError('Wrong prediction threshold')
            for name,indices in [('train',train),('validation',val),('test',test)]:
                for key,value in calibrated_metrics(labels[indices],probability(logits[indices]),threshold).items():
                    np.testing.assert_allclose(value,status['final'][f'{name}_calibrated_{key}'],rtol=0,atol=1e-12)
    row=dict(**setting,state=status['state'],rounds=len(h),selected_round=best_round,threshold=status['threshold'],
             stopping_reason=status['stopping_reason'],training_loss_stable=stable,strict_training_loss_stable=h[-1]['strict_training_loss_stable'],
             elapsed_seconds=h[-1]['elapsed_seconds'],cumulative_optimizer_steps=h[-1]['cumulative_optimizer_steps'],
             cumulative_processed_examples=h[-1]['cumulative_processed_examples'],training_records=len(train),
             validation_records=len(val),test_records=len(test),records_per_client=len(groups[0]),**status['final'])
    row.update({'endpoint_'+k:v for k,v in h[-1].items() if k.startswith(('train_','validation_','test_'))})
    return row,history_table(h,setting)


def audit_attacks(folder, frame, x_array, labels, partition, setting, config, protocol, dataset, first_history):
    train,val,test,groups,part_hash=partition;attack_definition=read_json(folder/'attack_definition.json')
    expected=dict(setting=setting,training_manifest_sha256=json_hash(core_manifest(dataset,part_hash,setting,config,protocol)),
                  coordinator_sha256=sha256_file(HERE/'final_campaign.py'),
                  labels='SGD optimization membership; validation and test are nonmembers',first_round_lr=setting['learning_rate'],weight_decay=.01)
    if attack_definition!=expected or not validate_first_round(folder,expected):raise ValueError('Incomplete/mismatched attack cache')
    timing=read_json(folder/'attack_timing.json')
    expected_steps=sum(math.ceil(len(g)/setting['batch_size']) for g in groups)*setting['local_epochs']
    if (timing['models']!=len(groups)+1 or timing['reconstruction_optimizer_steps']!=expected_steps or
            timing['reconstruction_processed_examples']!=len(train)*setting['local_epochs'] or
            not np.isfinite(timing['seconds']) or timing['seconds']<0):
        raise ValueError('Incorrect round-one work/timing metadata')
    for name in ['train','validation','test']:
        for key in ['cross_entropy','accuracy','balanced_accuracy','auroc','average_precision']:
            np.testing.assert_allclose(first_history[f'{name}_{key}'],timing[f'global_round_one_{name}_{key}'],rtol=1e-10,atol=1e-12)
    scores=np.load(folder/'first_round_scores.npz',allow_pickle=False)
    if (not np.array_equal(scores['record_ids'],frame.record_id.to_numpy(dtype=str)) or not np.array_equal(scores['labels'],labels) or
            not np.array_equal(scores['train_indices'],train) or not np.array_equal(scores['local_indices'],np.stack(groups)) or
            scores['losses'].shape!=(len(groups)+1,len(labels))):raise ValueError('Misaligned attack scores/membership')
    states=torch.load(folder/'first_round_models.pt',map_location='cpu',weights_only=True)['states']
    if len(states)!=len(groups)+1:raise ValueError('Missing first-round models')
    averaged=aggregate(states[1:],[len(g) for g in groups])
    for key in averaged:torch.testing.assert_close(averaged[key],states[0][key],rtol=0,atol=0)
    metrics=pd.read_csv(folder/'attack_metrics.csv');curves=pd.read_csv(folder/'roc_points.csv.gz')
    if set(zip(metrics.model_type,metrics.client_id))!={('global',-1)}|{('local',i) for i in range(len(groups))} or len(metrics)!=len(states):
        raise ValueError('Incomplete/duplicated attack model coverage')
    model=ReadmissionMLP(x_array.shape[1]);x=torch.from_numpy(x_array)
    for index,state in enumerate(states):
        model.load_state_dict(state);logits=predict_logits(model,x)
        loss=losses_from_logits(logits,labels)
        np.testing.assert_allclose(loss,scores['losses'][index],rtol=1e-10,atol=1e-12)
        member=np.zeros(len(labels),dtype=bool);member[train if index==0 else groups[index-1]]=True
        actual,curve=attack_metrics(member,-loss,config['attack_fpr_levels'])
        kind='global' if index==0 else 'local';client=index-1
        observed=metrics.loc[(metrics.model_type==kind)&(metrics.client_id==client)].iloc[0]
        for key,value in actual.items():
            if value is None:
                if not pd.isna(observed[key]):raise ValueError('Wrong infinity threshold encoding')
            elif isinstance(value,bool):
                if bool(observed[key])!=value:raise ValueError('Wrong threshold flag')
            else:np.testing.assert_allclose(float(observed[key]),value,rtol=1e-10,atol=1e-12)
        observed_curve=curves.loc[(curves.model_type==kind)&(curves.client_id==client)].reset_index(drop=True)
        for key in ['fpr','tpr','threshold']:np.testing.assert_allclose(observed_curve[key],curve[key],rtol=1e-10,atol=1e-12)
        if index==0:
            for name,indices in [('train',train),('validation',val),('test',test)]:
                for key,value in classification_metrics(logits[indices],labels[indices]).items():
                    np.testing.assert_allclose(value,first_history[f'{name}_{key}'],rtol=1e-10,atol=1e-12)
    scores.close()
    return metrics.assign(**setting),curves.assign(**setting),timing


def audit_campaign(data_dir, partitions_root, output, config, protocol):
    from analyze import summarize_attack
    frame,x,y,dataset=load_dataset(data_dir,config);freeze_campaign(output,dataset,config,protocol)
    rows,attacks,histories,errors=[],[],[],[];reused=read_json(Path(output)/'baseline_reuse.json')
    reused_map={(r['config_id'],r['seed']):r for r in reused['runs']}
    destination=Path(output)/'analysis';results=destination/'results';results.mkdir(parents=True,exist_ok=True)
    # Stream millions of ROC points instead of retaining every model's curve in RAM.
    roc_stream=gzip.open(results/'roc_points_all_models.csv.gz','wt',compresslevel=1)
    roc_written=False
    for setting in definitions(config,protocol):
        folder=trial_dir(output,setting);row=dict(**setting,state='missing',rounds=0,attacks_verified=0)
        try:
            if (folder/'status.json').exists():row.update(state=read_json(folder/'status.json')['state'])
            partition=freeze_partition(partitions_root,frame.record_id,y,setting,dataset['sha256'])
            row,h=audit_training(folder,partition,setting,dataset,config,protocol,x,y)
            if is_baseline(setting,config):
                old=reused_map[(setting['config_id'],setting['seed'])]
                for name,digest in old['source_artifact_sha256'].items():
                    if sha256_file(folder/name)!=digest:raise ValueError('Reused baseline training artifact changed')
            m,roc,timing=audit_attacks(folder,frame,x,y,partition,setting,config,protocol,dataset,h.iloc[0])
            row.update(attacks_verified=len(m),attack_seconds=timing['seconds'],
                       attack_reconstruction_optimizer_steps=timing['reconstruction_optimizer_steps'],
                       attack_reconstruction_processed_examples=timing['reconstruction_processed_examples'],
                       baseline_training_reused=is_baseline(setting,config))
            attacks.append(m);histories.append(h)
            roc.to_csv(roc_stream,index=False,header=not roc_written,
                       columns=['fpr','tpr','threshold','model_type','client_id','config_id','seed'])
            roc_written=True
        except Exception as error:
            row['audit_error']=repr(error);errors.append(dict(config_id=setting['config_id'],seed=setting['seed'],error=repr(error)))
            # Preserve failed-run metadata and finite checkpoint history, rather
            # than silently dropping unsuccessful runs from the exported evidence.
            if (folder/'status.json').exists():
                failed=read_json(folder/'status.json')
                row.update(rounds=failed.get('rounds',row.get('rounds',0)),training_error=failed.get('error'))
            if row['state']=='failed' and (folder/'checkpoint.pt').exists():
                partial=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)['history']
                histories.append(history_table(partial,setting).assign(training_failed=True))
                try:
                    m,roc,timing=audit_attacks(folder,frame,x,y,partition,setting,config,protocol,dataset,partial[0])
                    attacks.append(m);row['attacks_verified']=len(m)
                    roc.to_csv(roc_stream,index=False,header=not roc_written,
                               columns=['fpr','tpr','threshold','model_type','client_id','config_id','seed'])
                    roc_written=True
                except Exception as attack_error:row['attack_audit_error']=repr(attack_error)
        row.setdefault('attacks_verified',0)
        rows.append(row)
    roc_stream.close()
    statuses=pd.DataFrame(rows);statuses.to_csv(results/'run_status.csv',index=False)
    if attacks:
        all_attacks=pd.concat(attacks,ignore_index=True);all_attacks.to_csv(results/'attack_metrics_all_models.csv',index=False)
        per_seed,summary=summarize_attack(all_attacks)
        per_seed.to_csv(results/'attack_metrics_per_seed.csv',index=False);summary.to_csv(results/'attack_summary.csv',index=False)
    if histories:pd.concat(histories,ignore_index=True).to_csv(results/'round_metrics.csv',index=False)
    count=int(statuses.attacks_verified.fillna(0).sum());finished=int(statuses.state.isin(TERMINAL).sum())
    coverage=dict(passed=not errors,complete=not errors and finished==84 and count==1785,expected_runs=84,finalized_runs=finished,
                  expected_model_evaluations=1785,verified_model_evaluations=count,counts=statuses.state.value_counts().to_dict(),errors=errors,
                  selected_and_last_models_rescored=2*sum('audit_error' not in r for r in rows),first_round_models_rescored=count,
                  training_history_rows=sum(len(h) for h in histories),baseline_training_reused=12,
                  dataset_sha256=dataset['sha256'],protocol_sha256=json_hash(protocol),
                  global_members=3600,global_nonmembers=5400,local_nonmembers='all records outside that client optimization partition',
                  stopping_states='loss_stable, validation early_stopped, capped at 300, or failed; no relabeling',
                  classification_threshold='validation labels; independent of oracle membership thresholds',
                  artifact_sha256={p.name:sha256_file(p) for p in results.iterdir() if p.is_file() and p.name!='coverage.json'})
    write_json(results/'coverage.json',coverage);print(coverage,flush=True)
    if not coverage['complete']:raise ValueError('Incomplete campaign; explicit missing/failed/invalid rows retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--partitions-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',type=Path,metavar='CONFIRMED_BASELINE_ROOT')
    action.add_argument('--run-index',type=int)
    action.add_argument('--audit',action='store_true')
    parser.add_argument('--interrupt-after',type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    if not args.prepare and not os.environ.get('SLURM_JOB_ID'):parser.error('Run training and model audit in Slurm')
    config,protocol=load_config(),load_protocol()
    if args.audit:audit_campaign(args.data_dir,args.partitions_root,args.output,config,protocol);return
    frame,x,y,dataset=load_dataset(args.data_dir,config);freeze_campaign(args.output,dataset,config,protocol)
    if args.prepare:
        reuse_baselines(args.prepare,args.output,dataset,args.partitions_root,frame,y,config,protocol)
        print('Twelve baseline training caches copied unchanged; 72 settings require training.',flush=True)
    else:
        settings=definitions(config,protocol)
        if not 0<=args.run_index<len(settings):parser.error('run-index must be 0–83')
        setting=settings[args.run_index];partition=freeze_partition(args.partitions_root,frame.record_id,y,setting,dataset['sha256'])
        run_trial(args.output,frame,x,y,dataset,partition,setting,config,protocol,args.interrupt_after)


if __name__=='__main__':main()
