"""Numerical checks for isolated optimization diagnostics."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ReadmissionMLP, seed_everything
from convergence_pilot import gradient_diagnostic, round_learning_rate, tail_scaled_update

torch.set_num_threads(1)


class PilotTests(unittest.TestCase):
    def test_gradient_diagnostic_matches_full_regularized_objective(self):
        seed_everything(684)
        model=ReadmissionMLP(4)
        x,y=torch.randn(1030,4),torch.arange(1030)%2
        frozen=copy.deepcopy(model.state_dict())
        objective=F.cross_entropy(model(x),y)+.5e-5*sum(p.square().sum() for p in model.parameters())
        objective.backward()
        expected_norm=sum(float(p.grad.square().sum()) for p in model.parameters())**.5
        actual=gradient_diagnostic(model,x,y,np.arange(1030),1e-5)
        self.assertAlmostEqual(actual['regularized_training_objective'],objective.item(),places=6)
        self.assertAlmostEqual(actual['regularized_gradient_l2'],expected_norm,places=6)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)

    def test_positive_learning_rate_schedule_boundaries(self):
        for r,expected in [(1,.05),(100,.05),(101,.005),(200,.005),(201,.0005),(300,.0005)]:
            self.assertAlmostEqual(round_learning_rate("step_decay_sgd",.05,r),expected)
            self.assertEqual(round_learning_rate("tail_scaled_sgd",.05,r),.05)
        with self.assertRaises(ValueError):
            round_learning_rate("step_decay_sgd",.05,301)

    def test_tail_scaling_matches_explicit_sgd_and_keeps_all_records(self):
        seed_everything(684)
        initial=ReadmissionMLP(4)
        x,y=torch.randn(20,4),torch.arange(20)%2
        setting=dict(seed=684,learning_rate=.05,local_epochs=3,batch_size=7)
        optimizer_config=dict(momentum=0.,weight_decay=1e-5)
        frozen=copy.deepcopy(initial.state_dict())
        actual,steps,examples=tail_scaled_update(initial,x,y,np.arange(20),setting,optimizer_config,1,0)
        expected=copy.deepcopy(initial)
        optimizer=torch.optim.SGD(expected.parameters(),lr=.05,weight_decay=1e-5)
        rng=torch.Generator().manual_seed(684+100_003)
        for _ in range(3):
            order=torch.randperm(20,generator=rng)
            for start in range(0,20,7):
                chosen=order[start:start+7]
                optimizer.param_groups[0]['lr']=.05*len(chosen)/7
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(expected(x[chosen]),y[chosen]).backward()
                optimizer.step()
        self.assertEqual((steps,examples),(9,60))
        for name,value in actual.state_dict().items():
            torch.testing.assert_close(value,expected.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(initial.state_dict()[name],frozen[name],rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
