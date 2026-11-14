"""Verify predictive control, threshold calibration, and immutable exact resume."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import load_config, read_json, save_torch
from fedavg import local_update
from task_a_protocol import practical_stability
from training_selection import (AUCStop, ValidationLR, calibrate_threshold, calibrated_metrics,
                                choose_configuration, definitions, load_protocol, probability,
                                run, run_dir, history_table, make_controller)

torch.set_num_threads(1)


class SelectionTests(unittest.TestCase):
    def test_export_preserves_actual_lr_and_controller_monitors_loss(self):
        p=load_protocol()
        setting=dict(clients=1,seed=684,weight_decay=.01,batch_size=32,local_epochs=1,learning_rate=.05)
        h=[dict(round=1,learning_rate=.05),dict(round=2,learning_rate=.025),dict(round=3,learning_rate=.0125)]
        frame=history_table(h,setting)
        self.assertEqual(frame.learning_rate.tolist(),[.05,.025,.0125])
        self.assertEqual(frame.initial_learning_rate.tolist(),[.05]*3)
        c=make_controller(p)
        self.assertTrue(c.observe(.32))
        self.assertFalse(c.observe(.33))
        self.assertTrue(c.observe(.31))
        self.assertEqual(c.best_round,3)
        self.assertEqual(p['stability_settings']['fixed_lr_rounds'],33)

    def test_lr_guard_covers_every_persistent_smoothed_window(self):
        p=load_protocol()['stability_settings']
        loss=[.3]*80
        # Thirty-two fixed rounds miss one source observation in the oldest check.
        self.assertFalse(practical_stability(loss,[.05]*48+[.025]*32,p)['loss_stable'])
        self.assertTrue(practical_stability(loss,[.05]*47+[.025]*33,p)['loss_stable'])
    def test_auc_patience_cumulative_improvements_ties_and_cap(self):
        p={**load_protocol(),'min_rounds':8,'patience':3}
        obj=AUCStop(p)
        for i in range(8): self.assertEqual(obj.observe(.7),i==0)
        self.assertEqual((obj.best_round,obj.stop_round),(1,8))
        with self.assertRaises(ValueError):obj.observe(.8)
        obj=AUCStop(p)
        for v in [.7,.701,.7019,.7021]:obj.observe(v)
        self.assertEqual(obj.bad_rounds,0)
        self.assertEqual(obj.best_round,4)
        for v in [np.inf,np.nan,-.1,1.1]:
            with self.assertRaises(FloatingPointError):AUCStop(p).observe(v)
        obj=AUCStop({**p,'patience':300})
        for _ in range(300):obj.observe(.7)
        with self.assertRaises(ValueError):obj.observe(.8)

    def test_scheduler_reduces_lr_and_resumes_exactly(self):
        p=load_protocol(); a=ValidationLR(.05,p)
        for _ in range(20):a.observe(.3)
        self.assertLess(a.lr,.05)
        b=ValidationLR(.05,p); b.load_state_dict(a.state_dict())
        for loss in [.299,.3]*60:
            a.observe(loss); b.observe(loss)
            self.assertEqual(a.lr,b.lr)
            self.assertEqual(a.state_dict(),b.state_dict())
        self.assertGreaterEqual(a.lr,.0005)

    def test_validation_threshold_ties_bruteforce_and_constant_scores(self):
        y=np.array([0,0,0,0,1,1]);p=np.array([.1,.2,.2,.3,.2,.4])
        threshold=calibrate_threshold(y,p)
        values={t:balanced_accuracy_score(y,p>=t) for t in np.unique(np.r_[p,np.nextafter(1.,2.)])}
        self.assertEqual(threshold,max(t for t,v in values.items() if v==max(values.values())))
        self.assertEqual(calibrated_metrics(y,p,threshold)['balanced_accuracy'],max(values.values()))
        constant=np.full(6,.2);t=calibrate_threshold(y,constant)
        self.assertGreater(t,1.)
        self.assertEqual(calibrated_metrics(y,constant,t)['positive_predictions'],0)
        with self.assertRaises(ValueError):calibrate_threshold(y,np.r_[np.nan,p[1:]])
        np.testing.assert_allclose(probability(np.array([[1000,1000],[0,np.log(3)]])),[.5,.75])

    def candidates(self):
        p={**load_protocol(),'weight_decay_candidates':[1e-5,.001,.01]}
        rows=[dict(**s,state='loss_stable',validation_auroc=.7+(s['weight_decay']==.001)*.001,
                   train_cross_entropy=.30,validation_cross_entropy=.32,training_loss_stable=True)
              for s in definitions(load_config(),p)]
        return pd.DataFrame(rows),p

    def test_selection_complete_validation_only_and_goal_preference(self):
        table,p=self.candidates()
        self.assertEqual(len(table),36)
        self.assertEqual(choose_configuration(table,p),1e-5)
        table['test_auroc']=1-table.validation_auroc
        self.assertEqual(choose_configuration(table,p),1e-5)
        table.loc[table.weight_decay==1e-5,'training_loss_stable']=False
        self.assertEqual(choose_configuration(table,p),.001)
        table.loc[table.weight_decay==.001,'validation_cross_entropy']=.6
        self.assertEqual(choose_configuration(table,p),.01)
        with self.assertRaisesRegex(ValueError,'Incomplete'):choose_configuration(table.iloc[1:],p)
        table.loc[0,'state']='failed'
        with self.assertRaisesRegex(ValueError,'Incomplete'):choose_configuration(table,p)

    def fixture(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(120,4)).astype(np.float32)
        y=(np.arange(120)%2).astype(np.int64)
        x[:,0]=2*y-1
        y[60:80]=1-y[60:80]
        partition=(np.arange(60),np.arange(60,80),np.arange(80,120),
                   [np.arange(30),np.arange(30,60)],'fixture')
        setting=dict(clients=2,seed=684,batch_size=16,local_epochs=1,learning_rate=.05,weight_decay=.001)
        protocol={**load_protocol(),'min_rounds':20,'patience':15}
        return [x,y,dict(sha256='data'),partition,setting,load_config(),protocol]

    def test_actual_training_restore_resume_and_test_label_independence(self):
        args=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run(root/'resume',*args,interrupt_after=5)
            with patch('training_selection.local_update',wraps=local_update) as update:
                a=run(root/'resume',*args)
                self.assertTrue(all(call.args[5]['weight_decay']==.001 for call in update.call_args_list))
            b=run(root/'fresh',*args)
            self.assertLess(a['selected_round'],a['rounds'])
            cp_a=torch.load(run_dir(root/'resume',args[4])/'checkpoint.pt',weights_only=True)
            cp_b=torch.load(run_dir(root/'fresh',args[4])/'checkpoint.pt',weights_only=True)
            self.assertTrue(any(row['learning_rate']<.05 for row in cp_a['history']))
            self.assertEqual(cp_a['scheduler'],cp_b['scheduler'])
            for first,second in zip(cp_a['history'],cp_b['history']):
                for key in set(first)-{'elapsed_seconds'}:self.assertEqual(first[key],second[key])
            selected=torch.load(run_dir(root/'resume',args[4])/'selected.pt',weights_only=True)
            for key in cp_a['state']:
                torch.testing.assert_close(cp_a['state'][key],cp_b['state'][key],rtol=0,atol=0)
                torch.testing.assert_close(selected['state'][key],cp_a['selected']['state'][key],rtol=0,atol=0)
            changed=copy.deepcopy(args);changed[1][changed[3][2]]=1-changed[1][changed[3][2]]
            c=run(root/'test_changed',*changed)
            self.assertEqual((a['rounds'],a['selected_round'],a['threshold']),
                             (c['rounds'],c['selected_round'],c['threshold']))
            cp_c=torch.load(run_dir(root/'test_changed',args[4])/'checkpoint.pt',weights_only=True)
            for key in cp_a['state']:
                torch.testing.assert_close(cp_a['state'][key],cp_c['state'][key],rtol=0,atol=0)
            self.assertNotEqual(a['final']['test_cross_entropy'],c['final']['test_cross_entropy'])
            incompatible=copy.deepcopy(args);incompatible[-1]['patience']=16
            with self.assertRaisesRegex(ValueError,'Incompatible'):run(root/'resume',*incompatible)

    def test_cap_failure_and_corrupt_artifact(self):
        args=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run(root,*args,interrupt_after=3)
            path=run_dir(root,args[4])/'checkpoint.pt'
            cp=torch.load(path,weights_only=True);cp['round']=301;save_torch(path,cp)
            with self.assertRaisesRegex(ValueError,'Invalid resume'):run(root,*args)
            bad=copy.deepcopy(args);bad[0][0,0]=np.nan
            with self.assertRaises(FloatingPointError):run(root/'bad',*bad)
            self.assertEqual(read_json(run_dir(root/'bad',args[4])/'status.json')['state'],'failed')
            run(root/'complete',*args)
            path=run_dir(root/'complete',args[4])/'history.csv';path.write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'):run(root/'complete',*args)

    def test_resume_from_299_executes_only_the_remaining_round(self):
        args=self.fixture();args[-1]['patience']=301
        # Exercise the cap path independently of numerical stability detection.
        with tempfile.TemporaryDirectory() as folder, patch('training_selection.practical_stability',
                                                           return_value={'loss_stable':False}):
            root=Path(folder)
            interrupted=run(root,*args,interrupt_after=299)
            self.assertEqual(interrupted['rounds'],299)
            with patch('training_selection.local_update',wraps=local_update) as update:
                status=run(root,*args)
                self.assertEqual(update.call_count,2)  # One round, two clients.
            self.assertEqual((status['state'],status['rounds']),('capped',300))
            with patch('training_selection.local_update',side_effect=AssertionError('extra training')) as update:
                run(root,*args)
                update.assert_not_called()


if __name__=='__main__':unittest.main()
