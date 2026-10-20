"""Meaningful control, stopping, and exact-resume checks on small fixtures."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import convergence, load_config, read_json, save_torch
from task_a_protocol import (TrainingPlateau, definitions, load_protocol, practical_stability,
                             run_protocol)

torch.set_num_threads(1)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol=load_protocol()

    def test_small_loss_absolute_tolerance_and_material_progress(self):
        tiny=.0004-1e-6*np.arange(60)
        self.assertFalse(convergence(tiny,load_config()['convergence'])['converged'])
        self.assertTrue(practical_stability(tiny,[.05]*60,self.protocol)['loss_stable'])
        for loss in [np.linspace(1.,.5,300),np.linspace(.5,1.,300),
                     np.tile([.7]*10+[1.3]*10,3)]:
            self.assertFalse(practical_stability(loss,[.05]*len(loss),self.protocol)['loss_stable'])

    def test_five_windows_and_fixed_rate_settling(self):
        for n in [49,50,53]:
            self.assertFalse(practical_stability([.3]*n,[.05]*n,self.protocol)['loss_stable'])
        self.assertTrue(practical_stability([.3]*54,[.05]*54,self.protocol)['loss_stable'])
        rates=[.05]*20+[.005]*49
        decision=practical_stability([.3]*69,rates,self.protocol)
        self.assertTrue(decision['persistent_bounds_pass'])
        self.assertFalse(decision['fixed_lr_guard_pass'])
        self.assertTrue(practical_stability([.3]*70,rates+[.005],self.protocol)['loss_stable'])

    def test_window_fitted_change_and_mixed_bounds(self):
        loss=1.-.0001*np.arange(60)
        decision=practical_stability(loss,[.05]*60,self.protocol)
        self.assertLess(decision['absolute_window_range'],decision['range_limit'])
        self.assertGreater(decision['absolute_fitted_change'],decision['trend_limit'])
        self.assertFalse(decision['loss_stable'])
        self.assertAlmostEqual(decision['absolute_fitted_change'],19*.0001,places=12)
        tiny=practical_stability(loss*1e-4,[.05]*60,self.protocol)
        self.assertEqual(tiny['trend_limit'],.0001)
        self.assertEqual(tiny['range_limit'],.0005)
        self.assertTrue(tiny['loss_stable'])

    def test_nonfinite_invalid_and_over_cap_histories(self):
        with self.assertRaises(FloatingPointError):
            practical_stability([float('nan')]*60,[.05]*60,self.protocol)
        for losses,rates in [([.3]*60,[0.]*60),([-.3]*60,[.05]*60),([.3]*301,[.05]*301)]:
            with self.assertRaises(ValueError):
                practical_stability(losses,rates,self.protocol)

    def test_plateau_warmup_positive_floor_and_restore(self):
        p=copy.deepcopy(self.protocol)
        p['scheduler'].update(patience=0,cooldown=0)
        control=TrainingPlateau(.05,p)
        for n in range(1,50):
            control.observe([.3]*n)
        self.assertEqual(control.lr,.05)
        control.observe([.3]*50)
        self.assertEqual(control.lr,.05)
        control.observe([.3]*51)
        self.assertEqual(control.lr,.025)
        restored=TrainingPlateau(.05,p)
        restored.load_state_dict(control.state_dict())
        for n in range(52,90):
            control.observe([.3]*n)
            restored.observe([.3]*n)
            self.assertEqual(control.lr,restored.lr)
        self.assertEqual(control.lr,.05/16)

    def test_trial_scope(self):
        rows=definitions(load_config())
        self.assertEqual(len(rows),16)
        adaptive=[r for r in rows if r['policy']=='tail_plateau']
        self.assertEqual({(r['clients'],r['seed']) for r in adaptive},
                         {(k,s) for k in [1,10,20,50] for s in [684,685,686]})
        self.assertEqual(sum(r['policy']=='tail_constant' for r in rows),4)

    def test_independent_audit_for_stable_noisy_and_improving_histories(self):
        from validate_task_a_protocol import independent_decision
        sequences=[[.3]*70, .0004-1e-6*np.arange(70), np.linspace(1.,.5,100),
                   np.tile([.7]*10+[1.3]*10,4)]
        for loss in sequences:
            for end in range(50,len(loss)+1):
                rates=[.05]*end
                expected=practical_stability(loss[:end],rates,self.protocol)['loss_stable']
                self.assertEqual(independent_decision(loss[:end],rates,self.protocol),expected)
        self.assertFalse(independent_decision([.3]*70,[.05]*40+[.005]*30,self.protocol))


class ResumeTests(unittest.TestCase):
    def fixture(self):
        labels=np.tile([0,1],20).astype(np.int64)
        frame=pd.DataFrame(dict(record_id=[f'r{i}' for i in range(40)],target=labels))
        x=np.zeros((40,4),dtype=np.float32)
        train,test=np.arange(20),np.arange(20,40)
        partition=(train,test,[train],'fixture-partition')
        setting=dict(policy='tail_plateau',clients=1,seed=684,batch_size=8,
                     local_epochs=1,learning_rate=.05)
        dataset=dict(sha256='fixture',preprocessor_sha256='fixture-transform')
        p=load_protocol()
        p['scheduler'].update(patience=0,cooldown=0)
        return frame,x,labels,dataset,partition,setting,load_config(),p

    def test_resume_preserves_model_controller_and_ignores_test_labels(self):
        args=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            uninterrupted=run_protocol(root/'full',*args)
            partial=run_protocol(root/'resume',*args,interrupt_after=51)
            self.assertEqual(partial['state'],'interrupted')
            resumed=run_protocol(root/'resume',*args)
            changed=list(args)
            changed[2]=args[2].copy()
            changed[2][20:25]=1
            alternative=run_protocol(root/'different-test-labels',*changed)
            self.assertEqual(uninterrupted['rounds'],resumed['rounds'])
            self.assertEqual(uninterrupted['rounds'],alternative['rounds'])
            suffix='k01_tail_plateau_seed_684'
            full=torch.load(root/'full'/suffix/'checkpoint.pt',weights_only=True)
            recovered=torch.load(root/'resume'/suffix/'checkpoint.pt',weights_only=True)
            independent=torch.load(root/'different-test-labels'/suffix/'checkpoint.pt',weights_only=True)
            for name,state in full['state'].items():
                torch.testing.assert_close(state,recovered['state'][name],rtol=0,atol=0)
                torch.testing.assert_close(state,independent['state'][name],rtol=0,atol=0)
            self.assertEqual(full['controller'],recovered['controller'])
            self.assertTrue(any(a['test_cross_entropy']!=b['test_cross_entropy']
                                for a,b in zip(full['history'],independent['history'])))
            for first,second in zip(full['history'],recovered['history']):
                self.assertEqual({k:v for k,v in first.items() if k!='elapsed_seconds'},
                                 {k:v for k,v in second.items() if k!='elapsed_seconds'})
            changed_protocol=list(args)
            changed_protocol[-1]=copy.deepcopy(args[-1])
            changed_protocol[-1]['absolute_trend']=.0002
            with self.assertRaisesRegex(ValueError,'Incompatible protocol cache'):
                run_protocol(root/'full',*changed_protocol)

    def test_resume_at_cap_cannot_train_more_and_rejects_over_cap(self):
        args=list(self.fixture())
        args[-1]['fixed_lr_rounds']=300
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            partial=run_protocol(root,*args,interrupt_after=300)
            self.assertEqual(partial['state'],'interrupted')
            directory=root/'k01_tail_plateau_seed_684'
            before=torch.load(directory/'checkpoint.pt',weights_only=True)
            final=run_protocol(root,*args)
            self.assertEqual(final['state'],'capped_unconverged')
            self.assertEqual(final['rounds'],300)
            after=torch.load(directory/'checkpoint.pt',weights_only=True)
            self.assertEqual(before['history'],after['history'])
            for name,state in before['state'].items():
                torch.testing.assert_close(state,after['state'][name],rtol=0,atol=0)
            (directory/'status.json').unlink()
            after['round']=301
            after['history'].append(dict(after['history'][-1],round=301))
            save_torch(directory/'checkpoint.pt',after)
            with self.assertRaisesRegex(ValueError,'Invalid protocol checkpoint'):
                run_protocol(root,*args)


if __name__=='__main__':
    unittest.main()
