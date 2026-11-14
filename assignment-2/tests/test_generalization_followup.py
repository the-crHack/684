"""Validate held-out stopping, best-weight restoration, and exact atomic resume."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ReadmissionMLP, classification_metrics, load_config, read_json, save_torch
from fedavg import predict_logits
from generalization_followup import (PROTOCOL, ValidationStop, definitions, freeze_partition,
                                     run, split_indices)

torch.set_num_threads(1)


class EarlyStoppingTests(unittest.TestCase):
    def test_validation_partition_preserves_outer_test_and_membership(self):
        from common import membership_masks, partition_indices
        y=np.array([0]*7990+[1]*1010)
        ids=pd.Series([f'r{i}' for i in range(9000)])
        for seed in [684,685,686]:
            first_train,first_val=None,None
            for k in [1,10,20,50]:
                train,val,test,groups=split_indices(y,seed,k)
                original_train,original_test,_=partition_indices(y,seed,k)
                np.testing.assert_array_equal(test,original_test)
                self.assertEqual(set(train)|set(val),set(original_train))
                self.assertEqual((len(train),len(val),len(test)),(3600,900,4500))
                self.assertEqual((int(y[train].sum()),int(y[val].sum()),int(y[test].sum())),(404,101,505))
                self.assertEqual({len(g) for g in groups},{3600//k})
                if first_train is not None:
                    np.testing.assert_array_equal(train,first_train)
                    np.testing.assert_array_equal(val,first_val)
                first_train,first_val=train,val
                gm,lm=membership_masks(9000,train,groups[0])
                self.assertEqual(gm.sum(),3600)
                self.assertEqual(lm.sum(),3600//k)
                self.assertTrue((~gm)[val].all() and (~gm)[test].all())
                self.assertTrue((~lm)[val].all() and (~lm)[test].all())
        with tempfile.TemporaryDirectory() as folder:
            setting=definitions(load_config())[0]
            part=freeze_partition(Path(folder),ids,y,setting,'hash')
            repeated=freeze_partition(Path(folder),ids,y,setting,'hash')
            self.assertEqual(part[-1],repeated[-1])
            filename=Path(folder)/'partitions/seed_684_k01.csv'
            frame=pd.read_csv(filename);frame.loc[0,'client_id']=99;frame.to_csv(filename,index=False)
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                freeze_partition(Path(folder),ids,y,setting,'hash')

    def test_patience_minimum_round_ties_and_cumulative_improvements(self):
        controller=ValidationStop()
        for n in range(1,31):
            selected=controller.observe(.3)
            self.assertEqual(selected,n==1)
            self.assertEqual(controller.best_round,1)
            self.assertEqual(controller.stop_round,30 if n==30 else None)
        with self.assertRaises(ValueError):controller.observe(.3)
        controller=ValidationStop()
        for loss in [.3,.29995,.29990,.29985]:controller.observe(loss)
        self.assertEqual(controller.best_round,4)
        self.assertEqual(controller.bad_rounds,0)
        self.assertAlmostEqual(controller.patience_best_loss,.29985)
        for loss in [np.nan,np.inf,-1.]:
            with self.assertRaises(FloatingPointError):ValidationStop().observe(loss)
        controller=ValidationStop()
        for n in range(300):controller.observe(1.-.001*n)
        self.assertIsNone(controller.stop_round)
        self.assertEqual(controller.best_round,300)
        with self.assertRaises(ValueError):controller.observe(.1)
        self.assertEqual(len(definitions(load_config())),12)

    def fixture(self):
        rng=np.random.default_rng(44)
        x=rng.normal(size=(120,4)).astype(np.float32)
        y=(np.arange(120)%2).astype(np.int64)
        x[:,0]=2*y-1
        y[60:80]=1-y[60:80]  # Validation contradicts the training signal; expose overfitting.
        frame=pd.DataFrame({'record_id':[f'r{i}' for i in range(120)]})
        partition=(np.arange(60),np.arange(60,80),np.arange(80,120),
                   [np.arange(30),np.arange(30,60)],'fixture')
        dataset=dict(sha256='data',schema_sha256='schema',preprocessor_sha256='preprocessor')
        setting=dict(clients=2,seed=684,batch_size=16,local_epochs=1,learning_rate=.05)
        protocol={**PROTOCOL,'patience':3,'min_rounds':8}
        return frame,x,y,dataset,partition,setting,load_config(),protocol

    def test_best_checkpoint_restored_and_exact_resume(self):
        args=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run(root/'resume',*args,interrupt_after=3)
            resumed=run(root/'resume',*args)
            fresh=run(root/'fresh',*args)
            self.assertEqual(resumed['state'],'validation_early_stopped')
            self.assertLess(resumed['selected_round'],resumed['rounds'])
            path=Path('runs/k02_seed_684')
            a=torch.load(root/'resume'/path/'checkpoint.pt',weights_only=True)
            b=torch.load(root/'fresh'/path/'checkpoint.pt',weights_only=True)
            for key in a['state']:
                torch.testing.assert_close(a['state'][key],b['state'][key],rtol=0,atol=0)
                torch.testing.assert_close(a['selected']['state'][key],b['selected']['state'][key],rtol=0,atol=0)
            for x,y in zip(a['history'],b['history']):
                for key in set(x)-{'elapsed_seconds'}:self.assertEqual(x[key],y[key])
            best=torch.load(root/'resume'/path/'best_validation.pt',weights_only=True)
            model=ReadmissionMLP(4);model.load_state_dict(best['state'])
            metrics=classification_metrics(predict_logits(model,torch.from_numpy(args[1]))[args[4][2]],args[2][args[4][2]])
            self.assertEqual(metrics['cross_entropy'],resumed['final']['test_cross_entropy'])
            self.assertEqual(best['round'],resumed['selected_round'])
            self.assertGreater(resumed['last']['validation_cross_entropy'],resumed['final']['validation_cross_entropy'])

    def test_test_labels_do_not_change_training_selection_or_stop(self):
        args=list(self.fixture())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            first=run(root/'a',*args)
            args[2]=args[2].copy();args[2][args[4][2]]=1-args[2][args[4][2]]
            second=run(root/'b',*args)
            self.assertEqual((first['rounds'],first['selected_round']),(second['rounds'],second['selected_round']))
            self.assertNotEqual(first['final']['test_cross_entropy'],second['final']['test_cross_entropy'])
            path=Path('runs/k02_seed_684/checkpoint.pt')
            a=torch.load(root/'a'/path,weights_only=True);b=torch.load(root/'b'/path,weights_only=True)
            for key in a['state']:
                torch.testing.assert_close(a['state'][key],b['state'][key],rtol=0,atol=0)
                torch.testing.assert_close(a['selected']['state'][key],b['selected']['state'][key],rtol=0,atol=0)

    def test_cache_rejection_and_cap_cannot_run_an_extra_round(self):
        args=list(self.fixture());args[-1]={**PROTOCOL,'patience':300,'min_rounds':30}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run(root,*args,interrupt_after=3)
            changed=list(args);changed[-1]={**args[-1],'min_delta':.01}
            with self.assertRaisesRegex(ValueError,'Incompatible'):run(root,*changed)
            cp_path=root/'runs/k02_seed_684/checkpoint.pt'
            cp=torch.load(cp_path,weights_only=True)
            rows=[]
            for number in range(1,301):
                row=dict(cp['history'][-1],round=number,validation_cross_entropy=1.-number*.001)
                rows.append(row)
            controller=ValidationStop.from_history(rows,args[-1])
            cp.update(round=300,history=rows,controller=controller.state_dict())
            cp['selected'].update(round=300,validation_cross_entropy=controller.best_loss)
            save_torch(cp_path,cp)
            # Synthetic controller history is sufficient to verify the control-flow cap;
            # final-model metric consistency should reject this intentionally fabricated cache.
            with patch('generalization_followup.local_update',side_effect=AssertionError('extra training')) as update:
                with self.assertRaises(AssertionError):run(root,*args)
                update.assert_not_called()
            self.assertEqual(read_json(root/'runs/k02_seed_684/status.json')['state'],'failed')
            (root/'runs/k02_seed_684/status.json').unlink()
            cp['history'].append(dict(rows[-1],round=301));cp['round']=301;save_torch(cp_path,cp)
            with self.assertRaisesRegex(ValueError,'Invalid'):run(root,*args)


if __name__=='__main__':unittest.main()
