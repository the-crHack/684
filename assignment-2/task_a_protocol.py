"""Isolated, versioned Task A optimization and practical-loss-stability trials."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (HERE, ReadmissionMLP, classification_metrics, convergence,
                    ensure_partitions, json_hash, load_config, load_dataset,
                    read_json, save_torch, seed_everything, sha256_file, write_json)
from convergence_pilot import gradient_diagnostic, tail_scaled_update
from fedavg import aggregate, cpu_state, predict_logits


def load_protocol(path=None):
    p = read_json(path or HERE / "task_a_protocol.json")
    if (p['max_rounds'] != 300 or p['min_rounds'] < 50 or
            p['smoothing_window'] != 10 or p['stability_window'] != 20 or
            p['consecutive_windows'] < 1 or not 29 <= p['fixed_lr_rounds'] <= 300 or
            min(p[k] for k in ['absolute_range','absolute_trend','relative_range',
                               'relative_trend','denominator_floor']) <= 0):
        raise ValueError("Invalid Task A protocol")
    s = p['scheduler']
    if not (0 < s['factor'] < 1 and 0 < s['minimum_lr_ratio'] <= 1 and
            s['patience'] >= 0 and s['cooldown'] >= 0 and s['threshold'] > 0 and
            s['threshold_mode'] == 'abs' and s['eps'] > 0):
        raise ValueError("Invalid training-only scheduler")
    return p


def practical_stability(losses, learning_rates, protocol):
    """Mixed absolute/relative bounds, persistence, and a fixed-LR settling guard."""
    loss, rates = np.asarray(losses, dtype=float), np.asarray(learning_rates, dtype=float)
    p = protocol
    if (loss.ndim != 1 or rates.shape != loss.shape or len(loss) > p['max_rounds']):
        raise ValueError("Invalid loss/rate history or history exceeds 300 rounds")
    if not np.isfinite(loss).all() or not np.isfinite(rates).all():
        raise FloatingPointError("Nonfinite training-control history")
    if np.any(loss < 0) or np.any(rates <= 0):
        raise ValueError("Require nonnegative loss and positive learning rates")
    result = dict(loss_stable=False, mixed_bounds_pass=False, persistent_bounds_pass=False,
                  fixed_lr_guard_pass=False, absolute_window_range=None,
                  absolute_fitted_change=None, range_limit=None, trend_limit=None)
    w, n, persistent = p['smoothing_window'], p['stability_window'], p['consecutive_windows']
    if len(loss) < max(p['min_rounds'], w+n-1):
        return result
    smooth = np.convolve(loss, np.ones(w)/w, mode='valid')
    def bounds(window):
        denominator = max(float(np.median(window)), p['denominator_floor'])
        range_limit = max(p['absolute_range'], p['relative_range']*denominator)
        trend_limit = max(p['absolute_trend'], p['relative_trend']*denominator)
        spread = float(np.ptp(window))
        fitted = float(abs(np.polyfit(np.arange(n), window, 1)[0])*(n-1))
        return spread <= range_limit and fitted <= trend_limit, spread, fitted, range_limit, trend_limit
    passed, spread, fitted, range_limit, trend_limit = bounds(smooth[-n:])
    persistent_pass = len(loss)-persistent+1 >= p['min_rounds'] and all(
        bounds(smooth[len(smooth)-offset-n:len(smooth)-offset])[0]
        for offset in range(persistent))
    fixed = len(rates) >= p['fixed_lr_rounds'] and np.all(
        rates[-p['fixed_lr_rounds']:] == rates[-1])
    result.update(loss_stable=bool(persistent_pass and fixed), mixed_bounds_pass=bool(passed),
                  persistent_bounds_pass=bool(persistent_pass), fixed_lr_guard_pass=bool(fixed),
                  absolute_window_range=spread, absolute_fitted_change=fitted,
                  range_limit=range_limit, trend_limit=trend_limit)
    return result


class TrainingPlateau:
    """A shared LR controller; each client's actual SGD optimizer is still fresh."""
    def __init__(self, base_lr, protocol):
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=base_lr)
        s = protocol['scheduler']
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=s['factor'], patience=s['patience'],
            threshold=s['threshold'], threshold_mode=s['threshold_mode'],
            cooldown=s['cooldown'], min_lr=base_lr*s['minimum_lr_ratio'], eps=s['eps'])
        self.protocol = protocol

    @property
    def lr(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def observe(self, training_losses):
        if len(training_losses) >= self.protocol['min_rounds']:
            value = float(np.mean(training_losses[-self.protocol['smoothing_window']:]))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite scheduler observation")
            self.scheduler.step(value)

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])


def definitions(config):
    confirmations = [dict(policy='tail_constant', clients=k, seed=seed, **config['baseline'])
                     for k in [10,20] for seed in [685,686]]
    revised = [dict(policy='tail_plateau', clients=k, seed=seed, **config['baseline'])
               for k in config['client_counts'] for seed in config['seeds']]
    return confirmations + revised


def run_protocol(output, frame, x_array, labels, dataset, partition, setting, config,
                 protocol, interrupt_after=None):
    train, test, groups, partition_hash = partition
    if (setting['policy'] not in ['tail_constant','tail_plateau'] or setting['clients'] != len(groups)
            or setting['batch_size'] <= 0 or setting['local_epochs'] <= 0 or setting['learning_rate'] <= 0):
        raise ValueError("Invalid protocol experiment setting")
    directory = Path(output)/f"k{setting['clients']:02d}_{setting['policy']}_seed_{setting['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(setting=setting, protocol=protocol, config_sha256=json_hash(config),
                    dataset_sha256=dataset['sha256'], preprocessor_sha256=dataset['preprocessor_sha256'],
                    partition_sha256=partition_hash, packages={name: importlib.metadata.version(name)
                    for name in ['torch','numpy','pandas','scikit-learn']},
                    code_hashes={name:sha256_file(HERE/name) for name in
                    ['common.py','fedavg.py','convergence_pilot.py','task_a_protocol.py']})
    manifest_path, cp_path, status_path = [directory/n for n in ['manifest.json','checkpoint.pt','status.json']]
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Incompatible protocol cache; use a fresh output directory")
    write_json(manifest_path, manifest)
    if status_path.exists() and read_json(status_path)['state'] in ['loss_stable','capped_unconverged','failed']:
        status=read_json(status_path)
        if status['state'] != 'failed':
            for filename,key in [('checkpoint.pt','checkpoint_sha256'),('history.csv','history_sha256')]:
                if sha256_file(directory/filename) != status[key]:
                    raise ValueError("Corrupt completed protocol run")
        return status
    seed_everything(setting['seed'])
    model=ReadmissionMLP(x_array.shape[1])
    x,y=torch.from_numpy(x_array),torch.from_numpy(labels)
    controller=TrainingPlateau(setting['learning_rate'],protocol)
    history, previous_seconds, initial_gradient = [], 0., None
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=True)
        history=cp['history']
        if (cp['manifest_sha256'] != json_hash(manifest) or cp['round'] != len(history) or
                len(history)>300 or [r['round'] for r in history] != list(range(1,len(history)+1))):
            raise ValueError("Invalid protocol checkpoint/history")
        model.load_state_dict(cp['state'])
        controller.load_state_dict(cp['controller'])
        previous_seconds,initial_gradient=cp['elapsed_seconds'],cp['initial_gradient']
        if history and controller.lr != history[-1]['next_round_learning_rate']:
            raise ValueError("Checkpoint LR/controller disagreement")
    else:
        initial_gradient=gradient_diagnostic(model,x,y,train,config['optimizer']['weight_decay'])
    started=time.perf_counter()
    write_json(status_path,dict(state='running',setting=setting,rounds=len(history)))
    try:
        decision=practical_stability([r['train_cross_entropy'] for r in history],
                                    [r['round_learning_rate'] for r in history],protocol)
        for number in range(len(history)+1,301):
            if decision['loss_stable']:
                break
            lr=controller.lr
            states,steps,examples=[],0,0
            for client_id,indices in enumerate(groups):
                local,n_steps,n_examples=tail_scaled_update(model,x,y,indices,
                    dict(setting,learning_rate=lr),config['optimizer'],number,client_id)
                states.append(cpu_state(local))
                steps+=n_steps
                examples+=n_examples
            model.load_state_dict(aggregate(states,[len(g) for g in groups]))
            logits=predict_logits(model,x)
            row=dict(round=number,round_learning_rate=lr,optimizer_steps=steps,processed_examples=examples,
                     cumulative_optimizer_steps=steps+sum(r['optimizer_steps'] for r in history),
                     cumulative_processed_examples=examples+sum(r['processed_examples'] for r in history))
            for split,indices in [('train',train),('test',test)]:
                row.update({f'{split}_{key}':value for key,value in
                            classification_metrics(logits[indices],labels[indices]).items()})
            penalty=.5*config['optimizer']['weight_decay']*sum(float(p.detach().square().sum()) for p in model.parameters())
            row['train_regularized_objective']=row['train_cross_entropy']+penalty
            history.append(row)
            losses=[r['train_cross_entropy'] for r in history]
            decision=practical_stability(losses,[r['round_learning_rate'] for r in history],protocol)
            row.update(decision)
            row.update({'original_'+key:value for key,value in convergence(losses,config['convergence']).items()})
            if setting['policy']=='tail_plateau' and not decision['loss_stable']:
                controller.observe(losses)
            row['next_round_learning_rate']=controller.lr
            row['lr_reduced_for_next_round']=bool(controller.lr < lr)
            row['elapsed_seconds']=previous_seconds+time.perf_counter()-started
            save_torch(cp_path,dict(manifest_sha256=json_hash(manifest),round=len(history),state=cpu_state(model),
                                   history=history,elapsed_seconds=row['elapsed_seconds'],
                                   controller=controller.state_dict(),initial_gradient=initial_gradient))
            if number==1 or number%25==0 or decision['loss_stable']:
                print(setting,'round',number,'CE',row['train_cross_entropy'],'LR',lr,
                      'loss_stable',decision['loss_stable'],flush=True)
            if decision['loss_stable']:
                break
            if interrupt_after is not None and number>=interrupt_after:
                pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
                status=dict(state='interrupted',setting=setting,rounds=len(history),final=history[-1])
                write_json(status_path,status)
                return status
        state='loss_stable' if decision['loss_stable'] else 'capped_unconverged'
        if state=='capped_unconverged' and len(history)!=300:
            raise ValueError("Unconverged protocol run ended before its cap")
        final={**history[-1],**gradient_diagnostic(model,x,y,train,config['optimizer']['weight_decay'])}
        final['regularized_gradient_ratio_to_initial']=final['regularized_gradient_l2']/max(
            initial_gradient['regularized_gradient_l2'],1e-12)
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        status=dict(state=state,rounds=len(history),setting=setting,final=final,initial_gradient=initial_gradient,
                    stopping_reason='persistent_mixed_training_loss_stability_fixed_lr_guard' if decision['loss_stable']
                    else 'maximum_300_rounds',checkpoint_sha256=sha256_file(cp_path),
                    history_sha256=sha256_file(directory/'history.csv'))
        write_json(status_path,status)
        return status
    except Exception as error:
        write_json(status_path,dict(state='failed',setting=setting,rounds=len(history),
                                   error=type(error).__name__+': '+str(error)))
        raise


def historical_sensitivity(destination, protocol):
    """Retrospective threshold sensitivity only; never modifies campaign labels."""
    frame=pd.read_csv(destination/'results/round_metrics.csv')
    rows=[]
    for (ident,seed),group in frame.groupby(['config_id','seed']):
        group=group.sort_values('round')
        loss=group.train_cross_entropy.to_numpy()
        rates=np.full(len(group),group.iloc[0].learning_rate)
        first=None
        for end in range(protocol['min_rounds'],len(loss)+1):
            if practical_stability(loss[:end],rates[:end],protocol)['loss_stable']:
                first=end
                break
        rows.append(dict(config_id=ident,seed=int(seed),clients=int(group.iloc[0].clients),
                         original_state='capped_unconverged',retrospective_first_stable_round=first,
                         interpretation='alternative criterion sensitivity, not an actual stopped model'))
    pd.DataFrame(rows).to_csv(destination/'results/stopping_sensitivity.csv',index=False)
    print('Retrospective mixed-rule crossings:',sum(r['retrospective_first_stable_round'] is not None for r in rows),
          '/84; original labels remain unchanged.')


def summarize(output, config):
    rows=[]
    for setting in definitions(config):
        directory=output/f"k{setting['clients']:02d}_{setting['policy']}_seed_{setting['seed']}"
        status=read_json(directory/'status.json')
        if status['setting']!=setting or status['state'] not in ['loss_stable','capped_unconverged','failed']:
            raise ValueError("Missing or incomplete protocol trial")
        row={**setting,'state':status['state'],'rounds':status['rounds']}
        if status['state']!='failed':
            for filename,key in [('checkpoint.pt','checkpoint_sha256'),('history.csv','history_sha256')]:
                if sha256_file(directory/filename)!=status[key]:
                    raise ValueError("Corrupt protocol artifacts")
            row.update(status['final'])
        else:
            row['error']=status['error']
        rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(output/'summary.csv',index=False)
    print(table.reindex(columns=['clients','seed','policy','state','rounds','train_cross_entropy',
                                'test_cross_entropy','test_auroc','regularized_gradient_l2']).to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--destination',type=Path,default=HERE)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run-index',type=int)
    action.add_argument('--summarize',action='store_true')
    action.add_argument('--historical-sensitivity',action='store_true')
    parser.add_argument('--interrupt-after',type=int)
    args=parser.parse_args()
    config,protocol=load_config(),load_protocol(args.protocol)
    if args.historical_sensitivity:
        historical_sensitivity(args.destination,protocol)
        return
    if args.output is None:
        parser.error('Require --output')
    if args.summarize:
        summarize(args.output,config)
        return
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Train inside a Slurm CPU allocation')
    if args.data_dir is None or not 0<=args.run_index<len(definitions(config)):
        parser.error('Require data-dir and run-index 0--15')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frame,x,y,dataset=load_dataset(args.data_dir,config)
    partitions=ensure_partitions(args.data_dir,frame.record_id,y,config)
    setting=definitions(config)[args.run_index]
    status=run_protocol(args.output,frame,x,y,dataset,partitions[(setting['seed'],setting['clients'])],
                        setting,config,protocol,args.interrupt_after)
    print('Final state:',status['state'],'rounds:',status['rounds'],flush=True)
    if status['state']=='failed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
