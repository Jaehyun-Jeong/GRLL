from typing import Union, Dict

import random

# PyTorch
import torch

# module
from module.utils.utils import overrides
from module.utils.exploration import Epsilon, NormalNoise


class Policy():

    def __init__(
            self,
            algorithm: str,  # Policy Algorithm
            exploring: str,  # Exploring Method
            exploringParams: Dict[str, Union[int, float]]):

        self.algorithm = algorithm
        self.exploring = exploring
        self.exploringParams = exploringParams

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError()


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

        super().__init__(
                algorithm=algorithm,
                exploring=exploring,
                exploringParams=exploringParams,
                )

        # Initialize Parameters
        self.useEps = False
        self.useStochastic = False
        if exploring == 'epsilon':
            self.useEps = True
            self.exploration = Epsilon(
                    **exploringParams)
        if algorithm == 'stochastic':
            self.useStochastic = True

    # Return Action
    @overrides(Policy)
    def __call__(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> torch.Tensor:

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
            exploring: str = "normal",  # normal, None
            exploringParams: Dict[str, Union[int, float]] = {
                'mean': 0,  # mean
                'sigma': 1,  # standard deviation
            },):

        super().__init__(
                algorithm=algorithm,
                exploring=exploring,
                exploringParams=exploringParams,
                )

        # Initialize Parameters
        if exploring == 'normal':
            self.useEps = True
            self.exploration = NormalNoise(
                    **exploringParams)

    # Return Action
    @overrides(Policy)
    def __call__(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> torch.Tensor:

        # Get noise
        noise = self.exploration(
                stepsDone,
                actionValue.shape)

        # Add noise to action
        action = actionValue + noise

        return action
