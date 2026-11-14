"""Independent stopping audit and saved-model checks for the Task A trials (Slurm)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (HERE, ReadmissionMLP, classification_metrics, ensure_partitions,
                    json_hash, load_config, load_dataset, read_json, seed_everything,
                    sha256_file, write_json)
from convergence_pilot import gradient_diagnostic
from fedavg import predict_logits
from task_a_protocol import TrainingPlateau, definitions, load_protocol


def independent_decision(loss, rates, p):
    """Direct trailing arithmetic means and centered OLS; no production stop helper."""
    if len(loss) < p['min_rounds']:
        return False
    if len(loss)-p['consecutive_windows']+1 < p['min_rounds']:
        return False
    if len(rates)<p['fixed_lr_rounds'] or len(set(rates[-p['fixed_lr_rounds']:]))!=1:
        return False
    for offset in range(p['consecutive_windows']):
        end=len(loss)-offset
        smooth=np.array([np.mean(loss[n-p['smoothing_window']:n])
                         for n in range(end-p['stability_window']+1,end+1)])
        denominator=max(float(np.median(smooth)),p['denominator_floor'])
        centered=np.arange(len(smooth))-(len(smooth)-1)/2
        change=abs(np.dot(centered,smooth-smooth.mean())/np.dot(centered,centered))*(len(smooth)-1)
        if (np.ptp(smooth)>max(p['absolute_range'],p['relative_range']*denominator) or
                change>max(p['absolute_trend'],p['relative_trend']*denominator)):
            return False
    return True


def metric_check(model, x, labels, train, test, recorded):
    logits=predict_logits(model,x)
    for split,indices in [('train',train),('test',test)]:
        values=classification_metrics(logits[indices],labels[indices])
        for key,value in values.items():
            np.testing.assert_allclose(value,recorded[f'{split}_{key}'],rtol=1e-9,atol=1e-10)
    predictions=logits[test].argmax(axis=1)
    return dict(test_positive_predictions=int(np.sum(predictions==1)),
                test_sensitivity=float(np.mean(predictions[labels[test]==1]==1)))


def validate(output, data_dir, control_root, legacy_root):
    config,protocol=load_config(),load_protocol()
    original_protocol={**protocol, 'absolute_range':0., 'absolute_trend':0.,
                       'consecutive_windows':1, 'fixed_lr_rounds':1}
    frame,x_array,labels,dataset=load_dataset(data_dir,config)
    partitions=ensure_partitions(data_dir,frame.record_id,labels,config)
    x,y=torch.from_numpy(x_array),torch.from_numpy(labels)
    controls,rows,details={},[],[]
    histories,failed=[],[]
    def control(k,seed):
        key=(k,seed)
        if key not in controls:
            train,test,groups,digest=partitions[(seed,k)]
            directory=control_root/'runs'/f'k{k:02d}_b32_e1_lr0.05'/f'seed_{seed}'
            manifest=read_json(directory/'manifest.json')
            assert manifest['config_sha256']==json_hash(config)
            assert manifest['dataset_sha256']==dataset['sha256']
            assert manifest['partition_sha256']==digest
            assert manifest['preprocessor_sha256']==dataset['preprocessor_sha256']
            for name,value in manifest['code_hashes'].items():
                assert sha256_file(HERE/name)==value
            cp=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert cp['manifest_sha256']==json_hash(manifest) and cp['round']==300
            model=ReadmissionMLP(x_array.shape[1])
            model.load_state_dict(cp['state'])
            final=dict(cp['history'][-1])
            final.update(metric_check(model,x,labels,train,test,final))
            final.update(gradient_diagnostic(model,x,y,train,config['optimizer']['weight_decay']))
            final['last50_training_loss_range']=float(np.ptp([r['train_cross_entropy'] for r in cp['history'][-50:]]))
            controls[key]=final
        return controls[key]

    for setting in definitions(config):
        k,seed=setting['clients'],setting['seed']
        directory=output/f"k{k:02d}_{setting['policy']}_seed_{seed}"
        manifest,status=read_json(directory/'manifest.json'),read_json(directory/'status.json')
        assert manifest['setting']==status['setting']==setting and manifest['protocol']==protocol
        assert manifest['dataset_sha256']==dataset['sha256'] and manifest['config_sha256']==json_hash(config)
        train,test,groups,digest=partitions[(seed,k)]
        assert manifest['partition_sha256']==digest
        assert manifest['preprocessor_sha256']==dataset['preprocessor_sha256']
        for name,value in manifest['code_hashes'].items():
            assert sha256_file(HERE/name)==value
        if status['state']=='failed':
            assert status.get('error')
            failed.append(dict(setting=setting,error=status['error']))
            rows.append(dict(**setting,state='failed',rounds=status['rounds'],error=status['error']))
            continue
        assert status['state'] in ['loss_stable','capped_unconverged']
        for name,key in [('checkpoint.pt','checkpoint_sha256'),('history.csv','history_sha256')]:
            assert sha256_file(directory/name)==status[key]
        cp=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        history=cp['history']
        assert cp['manifest_sha256']==json_hash(manifest)
        assert cp['round']==status['rounds']==len(history)<=300
        assert [r['round'] for r in history]==list(range(1,len(history)+1))
        loaded=pd.read_csv(directory/'history.csv')
        pd.testing.assert_frame_equal(loaded,pd.DataFrame(history),check_dtype=False,rtol=1e-10,atol=1e-12)
        controller=TrainingPlateau(setting['learning_rate'],protocol)
        loss,rates=[],[]
        first=None
        steps=k*((4500//k+setting['batch_size']-1)//setting['batch_size'])*setting['local_epochs']
        for row in history:
            loss.append(row['train_cross_entropy'])
            rates.append(row['round_learning_rate'])
            assert np.isfinite(list(row[c] for c in row if c.startswith(('train_','test_')))).all()
            assert row['round_learning_rate']==controller.lr>0
            assert row['processed_examples']==4500*setting['local_epochs']
            assert row['optimizer_steps']==steps
            assert row['cumulative_processed_examples']==row['round']*row['processed_examples']
            assert row['cumulative_optimizer_steps']==row['round']*steps
            passed=independent_decision(loss,rates,protocol)
            assert row['loss_stable']==passed
            assert row['original_converged']==independent_decision(loss,rates,original_protocol)
            if passed and first is None:
                first=row['round']
            if setting['policy']=='tail_plateau' and not passed:
                controller.observe(loss)
            assert row['next_round_learning_rate']==controller.lr
            assert row['lr_reduced_for_next_round']==(controller.lr<row['round_learning_rate'])
        assert controller.state_dict()==cp['controller']
        if status['state']=='loss_stable':
            assert first==len(history)
        else:
            assert first is None and len(history)==300
        model=ReadmissionMLP(x_array.shape[1])
        model.load_state_dict(cp['state'])
        final=status['final']
        for name,value in history[-1].items():
            assert final[name]==value
        population=metric_check(model,x,labels,train,test,final)
        gradients=gradient_diagnostic(model,x,y,train,config['optimizer']['weight_decay'])
        for name,value in gradients.items():
            np.testing.assert_allclose(final[name],value,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(final['train_regularized_objective'],gradients['regularized_training_objective'],
                                   rtol=1e-6,atol=1e-8)
        seed_everything(seed)
        initial=ReadmissionMLP(x_array.shape[1])
        initial_gradient=gradient_diagnostic(initial,x,y,train,config['optimizer']['weight_decay'])
        for name,value in initial_gradient.items():
            np.testing.assert_allclose(status['initial_gradient'][name],value,rtol=1e-9,atol=1e-10)
            np.testing.assert_allclose(cp['initial_gradient'][name],value,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(final['regularized_gradient_ratio_to_initial'],
            gradients['regularized_gradient_l2']/max(initial_gradient['regularized_gradient_l2'],1e-12),rtol=1e-9)
        comparison={**setting,'state':status['state'],'rounds':len(history),
                    **{'trial_'+name:value for name,value in {**final,**population}.items()},
                    **{'control_'+name:value for name,value in control(k,seed).items()}}
        comparison['trial_last50_training_loss_range']=float(np.ptp(loss[-50:]))
        rows.append(comparison)
        histories.append(pd.DataFrame(history).assign(**setting))
        details.append(dict(setting=setting,state=status['state'],rounds=len(history),
                            first_stable_round=first,lr_reductions=sum(r['lr_reduced_for_next_round'] for r in history),
                            checkpoint_sha256=status['checkpoint_sha256'],history_sha256=status['history_sha256']))

    confirmations=[r for r in rows if r['policy']=='tail_constant']
    for k in [10,20]:
        seed=684
        directory=legacy_root/f'k{k:02d}_tail_scaled_sgd_seed_684'
        status,manifest=read_json(directory/'status.json'),read_json(directory/'manifest.json')
        assert status['state']=='capped_unconverged' and status['rounds']==300
        assert manifest['config_sha256']==json_hash(config)
        assert status['setting']==manifest['setting']
        assert status['setting']['clients']==k and status['setting']['seed']==seed
        train,test,groups,digest=partitions[(seed,k)]
        assert manifest['dataset_sha256']==dataset['sha256'] and manifest['partition_sha256']==digest
        assert manifest['preprocessor_sha256']==dataset['preprocessor_sha256']
        for name,value in manifest['code_hashes'].items():
            assert sha256_file(HERE/name)==value
        for name,key in [('checkpoint.pt','checkpoint_sha256'),('history.csv','history_sha256')]:
            assert sha256_file(directory/name)==status[key]
        cp=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==json_hash(manifest)
        loss=[r['train_cross_entropy'] for r in cp['history']]
        assert not any(independent_decision(loss[:n],[.05]*n,protocol) for n in range(50,301))
        model=ReadmissionMLP(x_array.shape[1])
        model.load_state_dict(cp['state'])
        final=status['final']
        population=metric_check(model,x,labels,train,test,final)
        gradients=gradient_diagnostic(model,x,y,train,config['optimizer']['weight_decay'])
        for name,value in gradients.items():
            np.testing.assert_allclose(final[name],value,rtol=1e-9,atol=1e-10)
        confirmations.append(dict(clients=k,seed=seed,policy='tail_constant',reused_legacy_run=True,
            state=status['state'],rounds=300,**{'trial_'+name:value for name,value in {**final,**population}.items()},
            trial_last50_training_loss_range=float(np.ptp(loss[-50:])),
            **{'control_'+name:value for name,value in control(k,seed).items()}))
    assert len(confirmations)==6 and len(rows)==16
    pd.DataFrame(confirmations).to_csv(output/'tail_confirmation.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'protocol_comparison.csv',index=False)
    if histories:
        pd.concat(histories,ignore_index=True).to_csv(output/'round_metrics.csv',index=False)
    benchmark=output.parent/'benchmark/k50_tail_plateau_seed_684/history.csv'
    if benchmark.exists() and (output/'k50_tail_plateau_seed_684/history.csv').exists():
        trial=pd.read_csv(output/'k50_tail_plateau_seed_684/history.csv')
        ten=pd.read_csv(benchmark)
        columns=[c for c in ten.columns if c.startswith(('train_','test_'))]
        np.testing.assert_allclose(trial.iloc[:len(ten)][columns],ten[columns],rtol=0,atol=1e-12)
    proof=dict(passed=True,expected_trials=16,audited_trials=len(details),failed_trials=failed,
               loss_stable=sum(r['state']=='loss_stable' for r in details),
               capped_unconverged=sum(r['state']=='capped_unconverged' for r in details),
               independent_stopping_and_controller_audit=True,prospective_final_models_rescored=len(details),
               original_control_models_rescored=len(controls),reused_legacy_models_rescored=2,
               protocol_sha256=sha256_file(HERE/'task_a_protocol.json'),
               artifact_sha256={name:sha256_file(output/name) for name in
                                ['summary.csv','protocol_comparison.csv','tail_confirmation.csv','round_metrics.csv']
                                if (output/name).exists()},
               original_models_and_labels_unchanged=True,task_b_unchanged=True,
               runs=details,history_rows=sum(r['rounds'] for r in details),
               interpretation='loss_stable denotes practical training-loss stability; gradients and generalization are separate')
    write_json(output/'validation.json',proof)
    print('Validated:',len(details),'trials;',proof['loss_stable'],'loss-stable;',
          proof['capped_unconverged'],'capped;',len(failed),'failed.',flush=True)


if __name__=='__main__':
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','output','control-root','legacy-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run full saved-model validation in a Slurm allocation')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    validate(args.output,args.data_dir,args.control_root,args.legacy_root)
