from typing import Union, Dict

import random
import numpy as np

# PyTorch
import torch

# ValueBased


def get_action(
        self,
        s: Union[torch.Tensor, np.ndarray],
        useEps: bool,
        useStochastic: bool):

    s = torch.Tensor(s).to(self.device).unsqueeze(0)
    probs = self.model.forward(s).squeeze(0)

    eps = self.__get_eps() if useEps else 0

    if random.random() >= eps:
        if useStochastic:
            probs = self.softmax(probs)
            a = probs.multinomial(num_samples=1)
            a = a.data
            action = a[0]
        else:
            # all actions must be in cpu, but all states in gpu if it using
            action = torch.argmax(probs, dim=0).cpu()
    else:
        a = torch.rand(probs.shape).multinomial(num_samples=1)
        a = a.data
        action = a[0]

    return action.tolist()

# Policy Gradient


'''
def get_action(
        self,
        s: Union[torch.Tensor, np.ndarray],
        useEps: bool,
        useStochastic: bool) -> torch.Tensor:

    s = torch.Tensor(s).to(self.device).unsqueeze(0)
    _, probs = self.model.forward(s)
    probs = probs.squeeze(0)

    eps = self.__get_eps() if useEps else 0

    if random.random() >= eps:
        if useStochastic:
            probs = self.softmax(probs)

            a = probs.multinomial(num_samples=1)
            a = a.data
            action = a[0].cpu()
        else:
            action = torch.argmax(probs, dim=0)
    else:
        a = torch.rand(probs.shape).multinomial(num_samples=1)
        a = a.data
        action = a[0]

    action = action.detach()

    return action.tolist()


    # Epsilon scheduling
    def __get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        eps_threshold = \
            eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)

        return eps_threshold
'''


class Policy():

    def __init__(
            self,
            algorithm: str = "greedy",
            exploring: str = "epsilon",  # noise, sde, gsde, LR
            eps: Dict[str, Union[int, float]] = {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            },):

        # ==================================================================================
        # select train, test policy
        # ==================================================================================

        policyDict = {
                # [ useEpsilon, useStochastic ]
                'greedy': [False, False],
                'stochastic': [False, True],
                'eps-greedy': [True, False],
                'eps-stochastic': [True, True]}

        if not self.policy['train'] in policyDict.keys() or \
                not self.policy['test'] in policyDict.keys():
            raise ValueError("Possible policies are \
                    'greedy', 'eps-greedy', \
                    'stochastic', and 'eps-stochastic'")

        trainPolicyList = policyDict[self.policy['train']]
        testPolicyList = policyDict[self.policy['test']]

        if trainPolicyList[0] or testPolicyList[0]:
            self.eps = eps

        self.useTrainEps = trainPolicyList[0]
        self.useTrainStochastic = trainPolicyList[1]
        self.useTestEps = testPolicyList[0]
        self.useTestStochastic = testPolicyList[1]

        # ==================================================================================

    def get_action(
            self,
            actionValue: torch.Tensor):

        pass
