from typing import Union, Dict

import random
import numpy as np

# PyTorch
import torch

# module
from module.utils.utils import overrides
from module.utils.exploration import epsilon


class Policy():

    def __init__(self):
        pass

    def get_action(self):
        pass


class DiscretePolicy(Policy):

    def __init__(
            self,
            algorithm: str = "greedy",  # greedy, stochastic
            exploring: str = "epsilon",  # epsilon, None
            exploringParams: Dict[str, Union[int, float]] = {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            },):

        # Initialize Parameters
        self.useEps = False
        self.useStochastic = False
        if exploring == 'epsilon':
            self.useEps = True
            self.exploration = epsilon(
                    **exploringParams)
        if algorithm == 'stochastic':
            self.useStochastic = True

    @overrides(Policy)
    def get_action(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> torch.Tensor:

        '''
        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, probs = self.model.forward(s)
        probs = probs.squeeze(0)
        '''
        probs = actionValue

        eps = self.exploration(stepsDone) if self.useEps else 0

        if random.random() >= eps:
            if self.useStochastic:
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


class ContinuousPolicy(Policy):

    def __init__(
            self,
            algorithm: str = "plain",  # plain
            exploring: str = "normal"):  # normal, None

        pass

    @overrides(Policy)
    def get_action(self):
