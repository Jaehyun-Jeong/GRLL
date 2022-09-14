from typing import Union, Dict

import random
import numpy as np

# PyTorch
import torch

# module
from module.utils.utils import overrides


class Policy():

    def __init__(
            self,
            actionType: str,  # Discrete, Continuous
            actionParams: Dict[str, Union[int, float]]):
        pass

    def get_action(self):
        pass


class DiscretePolicy(Policy):

    def __init__(
            self,
            algorithm: str = "greedy",  # greedy, stochastic
            exploring: str = "epsilon",  # epsilon, None
            eps: Dict[str, Union[int, float]] = {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            },):

        # Initialize Parameters
        self.useEps = False
        self.useStochastic = False
        if exploring == 'epsilon':
            self.useEps = True
        if algorithm == 'stochastic':
            self.useStochastic = True

    @overrides(Policy)
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> torch.Tensor:

        '''
        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, probs = self.model.forward(s)
        probs = probs.squeeze(0)
        '''
        probs = actionValue

        eps = self.__get_eps() if self.useEps else 0

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


class ContinuousPolicy(Policy):


    def __init__(
            self):
        pass
