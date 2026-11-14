"""Full-grid orchestration must preserve the confirmed optimizer and honest coverage."""
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
from common import load_config,read_json,sha256_file,write_json,save_torch
from final_campaign import (audit_attacks,audit_campaign,audit_training,core_manifest,definitions,
                            freeze_campaign,is_baseline,round_one,run_trial,training_setting,
                            trial_dir,trial_output)
from training_selection import load_protocol,run

torch.set_num_threads(1)


class FinalCampaignTests(unittest.TestCase):
    def test_grid_preserves_nominal_parameters_and_model_coverage(self):
        c,p=load_config(),load_protocol();settings=definitions(c,p)
        self.assertEqual(len(settings),84)
        self.assertEqual(len({s['config_id'] for s in settings}),28)
        self.assertEqual(sum(s['clients']+1 for s in settings),1785)
        self.assertEqual(sum(is_baseline(s,c) for s in settings),12)
        for s in settings:
            self.assertEqual(s['weight_decay'],.01)
            self.assertLessEqual(sum(s[k]!=v for k,v in c['baseline'].items()),1)
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            freeze_campaign(folder,{'sha256':'data'},c,p)
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                freeze_campaign(folder,{'sha256':'changed'},c,p)

    def fixture(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(120,4)).astype(np.float32);y=(np.arange(120)%2).astype(np.int64)
        x[:,0]=2*y-1;y[60:80]=1-y[60:80]
        f=pd.DataFrame({'record_id':[f'r{i}' for i in range(120)]})
        part=(np.arange(60),np.arange(60,80),np.arange(80,120),[np.arange(30),np.arange(30,60)],'fixture')
        setting=dict(config_id='fixture',clients=2,seed=684,batch_size=16,local_epochs=1,learning_rate=.05,weight_decay=.01)
        p={**load_protocol(),'min_rounds':8,'patience':3}
        return f,x,y,{'sha256':'data'},part,setting,load_config(),p

    def test_round_one_membership_scores_match_training_and_reuse(self):
        f,x,y,d,part,s,c,p=self.fixture()
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root=Path(folder)
            run_trial(root,f,x,y,d,part,s,c,p)
            directory=trial_dir(root,s)
            expected=core_manifest(d,part[-1],s,c,p)
            self.assertEqual(read_json(directory/'manifest.json'),expected)
            hashes={n:sha256_file(directory/n) for n in ['checkpoint.pt','history.csv','selected.pt','status.json']}
            with patch('training_selection.local_update',side_effect=AssertionError('baseline reoptimized')):
                run_trial(root,f,x,y,d,part,s,c,p)
            for n,h in hashes.items():self.assertEqual(sha256_file(directory/n),h)
            row,history=audit_training(directory,part,s,d,c,p,x,y)
            attack,roc,timing=audit_attacks(directory,f,x,y,part,s,c,p,d,history.iloc[0])
            global_row=attack[attack.model_type=='global'].iloc[0]
            self.assertEqual((global_row.members,global_row.nonmembers),(60,60))
            self.assertEqual(set(zip(attack[attack.model_type=='local'].members,attack[attack.model_type=='local'].nonmembers)),{(30,90)})
            self.assertEqual(timing['reconstruction_processed_examples'],60)
            # The local complement includes other optimized records, validation and test.
            with np.load(directory/'first_round_scores.npz') as scores:
                self.assertEqual(scores['losses'].shape,(3,120))
                self.assertEqual(set(scores['train_indices']),set(part[0]))
            self.assertEqual(len(attack),3);self.assertGreater(len(roc),3)

    def test_completed_baseline_attack_reconstruction_does_not_retrain(self):
        f,x,y,d,part,s,c,p=self.fixture()
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root=Path(folder)
            run(trial_output(root,s),x,y,d,part,training_setting(s),c,p)
            old=sha256_file(trial_dir(root,s)/'checkpoint.pt')
            with patch('training_selection.local_update',side_effect=AssertionError('full training rerun')):
                run_trial(root,f,x,y,d,part,s,c,p)
            self.assertEqual(sha256_file(trial_dir(root,s)/'checkpoint.pt'),old)

    def test_attack_numeric_and_training_work_corruption_rejected(self):
        f,x,y,d,part,s,c,p=self.fixture()
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root=Path(folder);run_trial(root,f,x,y,d,part,s,c,p);directory=trial_dir(root,s)
            _,history=audit_training(directory,part,s,d,c,p,x,y)
            table=pd.read_csv(directory/'attack_metrics.csv');table.loc[0,'members']=999;table.to_csv(directory/'attack_metrics.csv',index=False)
            proof=read_json(directory/'first_round_manifest.json');proof['hashes']['attack_metrics.csv']=sha256_file(directory/'attack_metrics.csv');write_json(directory/'first_round_manifest.json',proof)
            with self.assertRaises(AssertionError):audit_attacks(directory,f,x,y,part,s,c,p,d,history.iloc[0])
            cp=torch.load(directory/'checkpoint.pt',weights_only=True);cp['history'][0]['optimizer_steps']+=1;save_torch(directory/'checkpoint.pt',cp)
            status=read_json(directory/'status.json');status['artifact_sha256']['checkpoint.pt']=sha256_file(directory/'checkpoint.pt');write_json(directory/'status.json',status)
            with self.assertRaisesRegex(ValueError,'work counts'):audit_training(directory,part,s,d,c,p,x,y)

    def test_failed_run_retained_in_incomplete_audit(self):
        f,x,y,d,part,s,c,p=self.fixture()
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root=Path(folder);run_trial(root,f,x,y,d,part,s,c,p);directory=trial_dir(root,s)
            status=read_json(directory/'status.json');status.update(state='failed',error='fixture failure');write_json(directory/'status.json',status)
            write_json(root/'baseline_reuse.json',{'runs':[]})
            with patch('final_campaign.load_dataset',return_value=(f,x,y,d)),patch('final_campaign.freeze_campaign'),patch('final_campaign.definitions',return_value=[s]),patch('final_campaign.freeze_partition',return_value=part):
                with self.assertRaisesRegex(ValueError,'Incomplete campaign'):audit_campaign(root,root,root,c,p)
            proof=read_json(root/'analysis/results/coverage.json')
            self.assertFalse(proof['complete']);self.assertEqual(proof['counts'],{'failed':1})
            self.assertEqual(proof['selected_and_last_models_rescored'],0)
            result=pd.read_csv(root/'analysis/results/run_status.csv')
            self.assertEqual(result.iloc[0]['rounds'],status['rounds'])
            self.assertEqual(result.iloc[0]['training_error'],'fixture failure')
            self.assertTrue((root/'analysis/results/round_metrics.csv').exists())

    def test_nonfinite_first_round_is_recorded_as_failure(self):
        f,x,y,d,part,s,c,p=self.fixture();x[0,0]=np.nan
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            with self.assertRaises(FloatingPointError):run_trial(root,f,x,y,d,part,s,c,p)
            status=read_json(trial_dir(root,s)/'status.json')
            self.assertEqual((status['state'],status['rounds']),('failed',0))
            self.assertTrue((trial_dir(root,s)/'task_failure.json').exists())


if __name__=='__main__':unittest.main()
